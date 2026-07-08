"""Synthetic log -> summary corpus generator.

    python -m model_forge.summarizer.corpus --pairs 4000

The corpus is generated, not collected: a seeded template grammar over 14 common
infrastructure failure domains (containers, databases, TLS, DNS, queues, disks,
auth, config, rate limits, cron, cache, ...). Each domain renders through several
log *shapes* — terse key-value lines, bracketed logger output, prose sentences —
while the paired summary always restates component, cause, and code in fluent
prose. That mapping (many surface forms -> one normalized statement) is exactly
the job the fine-tune has to learn.

Because every byte is produced by this file from a fixed seed, provenance is
trivial: there is no source dataset, no scrubbing story, nothing to leak.
Regenerating with the same seed reproduces the corpus byte-for-byte.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

SERVICES = [
    "order-api", "cart-service", "payment-worker", "inventory-sync", "auth-gateway",
    "search-indexer", "media-encoder", "email-dispatcher", "report-builder",
    "session-store", "feed-aggregator", "geo-resolver", "webhook-relay",
    "ledger-service", "notify-hub", "catalog-api", "backup-agent", "thumbnail-worker",
]
HOSTS = [f"node-{i:02d}" for i in range(1, 13)] + [f"worker-{c}" for c in "abcdef"]
FUNCTIONS = [
    "startContainer", "connectDb", "runMigration", "handleRequest", "processBatch",
    "flushBuffer", "renewLease", "fetchUpstream", "decodePayload", "writeSnapshot",
    "resolveHost", "openChannel", "acquireLock", "buildIndex", "syncReplica",
]
TABLES = ["orders", "invoices", "sessions", "products", "events", "customers", "payments"]
QUEUES = ["orders.created", "emails.outbound", "media.encode", "webhooks.delivery",
          "reports.nightly", "payments.retry"]
ENDPOINTS = ["/api/v1/orders", "/api/v1/checkout", "/api/v1/search", "/api/v1/reports",
             "/api/v1/uploads", "/api/v1/sessions"]
ROLES = ["reporting_ro", "app_writer", "migrator", "readonly", "batch_runner"]
ENV_VARS = ["DATABASE_URL", "QUEUE_URL", "CACHE_HOST", "API_BASE_URL", "MAX_WORKERS",
            "TLS_CERT_PATH", "LOG_LEVEL"]
JOBS = ["nightly-report", "session-cleanup", "index-rebuild", "invoice-rollup",
        "backup-snapshot", "metrics-flush"]

# Each domain: (log shapes, summary shapes, extra slot factories).
# Log and summary render from the SAME slot fill, so they always agree on facts.
DOMAINS: list[tuple[list[str], list[str], dict[str, callable]]] = [
    (  # container OOM
        ["Container {service} on {host} terminated unexpectedly. Exit code 137. "
         "Process {fn} exceeded the memory limit of {mem}Mi.",
         "ERROR service={service} host={host} oom_killed=true limit={mem}Mi function={fn} exit=137",
         "[{level}] container runtime killed {service} on {host}: memory limit {mem}Mi "
         "reached inside {fn}, exit code 137"],
        ["The {service} container on {host} was killed with exit code 137 after {fn} "
         "exceeded its {mem}Mi memory limit.",
         "{service} on {host} ran out of memory: {fn} crossed the {mem}Mi limit and the "
         "container exited with code 137.",
         "An out-of-memory kill terminated {service} on {host} (exit 137); the {fn} "
         "process breached the {mem}Mi cap."],
        {"mem": lambda r: r.choice([256, 512, 1024, 2048])},
    ),
    (  # image pull failure
        ["Failed to pull image for {service}: registry returned {code} while fetching "
         "tag {tag}. Retrying with backoff.",
         "ERROR service={service} phase=image_pull code={code} tag={tag} msg=pull backoff",
         "[{level}] {service} deployment stalled: image tag {tag} could not be pulled, "
         "registry answered {code}"],
        ["Deploying {service} failed because image tag {tag} could not be pulled; the "
         "registry responded with {code} and the pull is backing off.",
         "The {service} image pull for tag {tag} failed with registry error {code}, "
         "leaving the deployment in backoff.",
         "Registry error {code} blocked {service} from pulling image tag {tag}."],
        {"code": lambda r: r.choice([401, 403, 404, 502, 503]),
         "tag": lambda r: f"v{r.randint(1, 9)}.{r.randint(0, 20)}.{r.randint(0, 9)}"},
    ),
    (  # database connection
        ["{service} could not connect to the database on port {port}. Error {code} in "
         "{fn}: connection refused after {ms}ms.",
         "ERROR service={service} fn={fn} db_port={port} code={code} msg=connection refused "
         "elapsed_ms={ms}",
         "[{level}] database unreachable from {service}: {fn} got error {code} "
         "(connection refused, port {port})"],
        ["{service} failed to reach the database: {fn} received a connection-refused "
         "error (code {code}) on port {port} after {ms}ms.",
         "A database connection failure hit {service}; error {code} in {fn} indicates "
         "the server on port {port} refused the connection.",
         "The database on port {port} refused {service}'s connection attempt in {fn}, "
         "failing with code {code}."],
        {"port": lambda r: r.choice([5432, 3306, 6379, 9042]),
         "code": lambda r: r.choice([61, 111, 2002, 2013]),
         "ms": lambda r: r.randint(30, 9000)},
    ),
    (  # database auth / permission
        ["Access denied for role {role} when {service} executed {fn} against table "
         "{table}. Error {code}.",
         "ERROR service={service} role={role} table={table} code={code} msg=permission denied",
         "[{level}] {service}: permission denied for {role} on {table} during {fn} "
         "(error {code})"],
        ["{service} was denied access to table {table}: role {role} lacks the required "
         "grant, producing error {code} in {fn}.",
         "Database permissions blocked {service} — role {role} cannot use table {table}, "
         "so {fn} failed with error {code}.",
         "Error {code} in {fn}: the {role} role used by {service} has no access to {table}."],
        {"code": lambda r: r.choice([1044, 1045, 42501])},
    ),
    (  # slow query / lock
        ["Query on {table} from {service} exceeded {ms}ms and was cancelled. Statement "
         "timed out inside {fn}.",
         "WARN service={service} table={table} elapsed_ms={ms} fn={fn} msg=statement timeout",
         "[{level}] lock wait on {table}: {service} transaction in {fn} aborted after {ms}ms"],
        ["A statement from {service} against {table} ran past the {ms}ms timeout and was "
         "cancelled in {fn}.",
         "{service}'s query on {table} was aborted after {ms}ms — the statement timeout "
         "fired during {fn}.",
         "The {table} query issued by {service} timed out at {ms}ms inside {fn}."],
        {"ms": lambda r: r.choice([1000, 3000, 5000, 15000, 30000])},
    ),
    (  # TLS certificate
        ["TLS handshake with {service} failed: certificate expired {days} days ago. "
         "{fn} aborted the connection.",
         "ERROR service={service} fn={fn} tls=handshake_failed reason=certificate_expired "
         "expired_days={days}",
         "[{level}] certificate for {service} is no longer valid (expired {days}d); "
         "handshake rejected in {fn}"],
        ["Connections to {service} are failing because its TLS certificate expired {days} "
         "days ago; {fn} rejects the handshake.",
         "The TLS certificate presented by {service} lapsed {days} days ago, so the "
         "handshake in {fn} was aborted.",
         "{service} has an expired TLS certificate ({days} days past validity), causing "
         "handshake failures in {fn}."],
        {"days": lambda r: r.randint(1, 90)},
    ),
    (  # DNS
        ["Name resolution failed for upstream {dep}: {reason}. {service} cannot reach it "
         "from {host}.",
         "ERROR service={service} host={host} upstream={dep} dns={reason}",
         "[{level}] {service} on {host}: lookup of {dep} returned {reason}"],
        ["{service} on {host} cannot resolve its upstream {dep} — DNS returned {reason}.",
         "DNS resolution of {dep} failed with {reason}, leaving {service} on {host} "
         "unable to connect.",
         "The {dep} hostname would not resolve ({reason}), blocking {service} on {host}."],
        {"dep": lambda r: r.choice(SERVICES),
         "reason": lambda r: r.choice(["NXDOMAIN", "SERVFAIL", "timeout"])},
    ),
    (  # upstream 5xx / circuit breaker
        ["Upstream {dep} returned {code} for {endpoint} after {ms}ms. Circuit breaker in "
         "{service} is now open.",
         "ERROR service={service} upstream={dep} endpoint={endpoint} code={code} "
         "elapsed_ms={ms} breaker=open",
         "[{level}] {service}: {count} consecutive {code} responses from {dep} on "
         "{endpoint}; breaker opened"],
        ["{service} opened its circuit breaker after {dep} answered {endpoint} with "
         "{code} (latest attempt took {ms}ms).",
         "Repeated {code} errors from {dep} on {endpoint} tripped {service}'s circuit "
         "breaker.",
         "The {dep} upstream is failing {endpoint} with {code}, so {service} stopped "
         "sending traffic (breaker open)."],
        {"dep": lambda r: r.choice(SERVICES),
         "code": lambda r: r.choice([500, 502, 503, 504]),
         "ms": lambda r: r.randint(100, 30000),
         "count": lambda r: r.randint(3, 50)},
    ),
    (  # queue lag / dead letter
        ["Message on {queue} was redelivered {count} times and moved to the dead-letter "
         "queue. Consumer {service} raised {error} in {fn}.",
         "WARN service={service} queue={queue} redeliveries={count} action=dead_letter "
         "error={error}",
         "[{level}] consumer lag on {queue}: {service} is {count} messages behind; "
         "oldest message failed with {error}"],
        ["A message on {queue} exhausted {count} redeliveries and was dead-lettered "
         "after {service} kept failing with {error} in {fn}.",
         "{service} could not process a {queue} message — {error} in {fn} across {count} "
         "attempts sent it to the dead-letter queue.",
         "After {count} failed deliveries to {service} ({error}), the {queue} message "
         "landed in the dead-letter queue."],
        {"count": lambda r: r.randint(3, 25),
         "error": lambda r: r.choice(["ValidationError", "TimeoutError", "JSONDecodeError",
                                      "KeyError"])},
    ),
    (  # disk space
        ["No space left on device: {service} on {host} failed to write during {fn}. "
         "Volume {mount} is at {pct}% capacity.",
         "ERROR service={service} host={host} mount={mount} used_pct={pct} fn={fn} "
         "msg=no space left on device",
         "[{level}] {host}: volume {mount} at {pct}%; writes from {service} are failing "
         "in {fn}"],
        ["Writes from {service} on {host} are failing because volume {mount} is {pct}% "
         "full — {fn} hit 'no space left on device'.",
         "Volume {mount} on {host} filled to {pct}%, so {service} can no longer write "
         "({fn} failed).",
         "{host} ran out of disk on {mount} ({pct}% used), breaking {service}'s {fn}."],
        {"mount": lambda r: r.choice(["/var/lib/data", "/var/log", "/tmp", "/srv/uploads"]),
         "pct": lambda r: r.randint(95, 100)},
    ),
    (  # auth token / permission
        ["Request to {endpoint} rejected with {code}: the session token presented to "
         "{service} expired {mins} minutes ago.",
         "WARN service={service} endpoint={endpoint} code={code} reason=token_expired "
         "expired_min={mins}",
         "[{level}] {service}: {code} on {endpoint} — expired credentials ({mins}m past "
         "validity)"],
        ["{service} returned {code} for {endpoint} because the caller's token expired "
         "{mins} minutes earlier.",
         "An expired token ({mins} minutes past validity) caused {service} to reject "
         "{endpoint} with {code}.",
         "The {endpoint} call failed with {code}: {service} saw credentials that lapsed "
         "{mins} minutes ago."],
        {"code": lambda r: r.choice([401, 403]),
         "mins": lambda r: r.randint(1, 720)},
    ),
    (  # config / startup
        ["{service} exited during startup: required environment variable {var} is not "
         "set. {fn} aborted with status 1.",
         "FATAL service={service} fn={fn} missing_env={var} exit=1",
         "[{level}] startup failure in {service}: {var} unset, {fn} cannot continue"],
        ["{service} failed to start because the {var} environment variable is missing; "
         "{fn} exited with status 1.",
         "Startup of {service} aborted in {fn} — the required {var} setting was never "
         "provided.",
         "A missing {var} value stopped {service} from booting ({fn} exited 1)."],
        {"var": lambda r: r.choice(ENV_VARS)},
    ),
    (  # rate limiting
        ["Client exceeded {count} requests per minute on {endpoint}; {service} is "
         "responding 429 with retry-after {secs}s.",
         "WARN service={service} endpoint={endpoint} limit_per_min={count} code=429 "
         "retry_after_s={secs}",
         "[{level}] throttling active on {endpoint}: {service} rejecting excess traffic "
         "with 429 (limit {count}/min, retry after {secs}s)"],
        ["{service} is throttling {endpoint}: the {count}-per-minute limit was exceeded, "
         "so callers get 429 with a {secs}s retry-after.",
         "Traffic to {endpoint} passed {count} requests per minute and {service} began "
         "returning 429 (retry after {secs}s).",
         "Rate limiting kicked in on {endpoint} — over {count}/min — and {service} now "
         "answers 429 with retry-after {secs}s."],
        {"count": lambda r: r.choice([60, 120, 300, 600]),
         "secs": lambda r: r.choice([5, 15, 30, 60])},
    ),
    (  # scheduled job
        ["Scheduled job {job} failed after {secs}s with exit status 1. Last step was "
         "{fn} on {host}.",
         "ERROR job={job} host={host} elapsed_s={secs} exit=1 last_step={fn}",
         "[{level}] {job} did not complete: exit 1 after {secs}s (step {fn}, {host})"],
        ["The {job} scheduled job failed on {host}: it exited 1 after {secs}s while "
         "running {fn}.",
         "{job} aborted with exit status 1 on {host}; the failure happened {secs}s in, "
         "during {fn}.",
         "On {host}, the {job} job died in step {fn} after {secs}s (exit 1)."],
        {"job": lambda r: r.choice(JOBS),
         "secs": lambda r: r.randint(5, 3600)},
    ),
]


def generate(pairs: int = 4000, seed: int = 1337) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    attempts = 0
    while len(out) < pairs and attempts < pairs * 60:
        attempts += 1
        logs, summaries, extra = DOMAINS[rng.randrange(len(DOMAINS))]
        fill = {
            "service": rng.choice(SERVICES),
            "host": rng.choice(HOSTS),
            "fn": rng.choice(FUNCTIONS),
            "table": rng.choice(TABLES),
            "queue": rng.choice(QUEUES),
            "endpoint": rng.choice(ENDPOINTS),
            "role": rng.choice(ROLES),
            "level": rng.choice(["ERROR", "CRIT", "ALERT"]),
        }
        for key, factory in extra.items():
            fill[key] = factory(rng)
        log = rng.choice(logs).format(**fill)
        summary = rng.choice(summaries).format(**fill)
        if log in seen or log == summary:
            continue
        seen.add(log)
        out.append((log, summary))
    if len(out) < pairs:
        raise RuntimeError(f"template space exhausted at {len(out)} unique pairs")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, default=Path("data/log_summary_pairs.csv"))
    args = parser.parse_args()

    rows = generate(args.pairs, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["log", "summary"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} pairs to {args.out} (seed {args.seed})")


if __name__ == "__main__":
    main()
