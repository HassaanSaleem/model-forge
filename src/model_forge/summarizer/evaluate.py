"""Evaluate the fine-tune: base t5-small vs fine-tuned, ROUGE on the held-out split.

    python -m model_forge.summarizer.evaluate --model models/log-summarizer-t5

Uses the same seeded 90/10 split as training, so the eval set here is exactly the
set the trainer validated on and never trained on. Writes rouge.json plus an
examples.md with side-by-side generations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import SummarizationModel
from .train import SEED, load_pairs


def rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}
    for pred, ref in zip(predictions, references, strict=True):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    n = max(len(predictions), 1)
    return {key: round(value / n, 4) for key, value in totals.items()}


def generate_all(model: SummarizationModel, logs: list[str], batch_size: int = 16) -> list[str]:
    out: list[str] = []
    for i in range(0, len(logs), batch_size):
        out.extend(model.summarize_batch(logs[i:i + batch_size]))
        print(f"\r  {min(i + batch_size, len(logs))}/{len(logs)}", end="", flush=True)
    print()
    return out


def evaluate(model_dir: Path, corpus: Path = Path("data/log_summary_pairs.csv"),
             base_model: str = "google-t5/t5-small",
             outdir: Path = Path("docs/summarizer"), sample: int | None = None) -> dict:
    _, eval_df = load_pairs(corpus, seed=SEED)
    if sample:
        eval_df = eval_df.head(sample)
    logs = eval_df["log"].tolist()
    references = eval_df["summary"].tolist()
    print(f"evaluating on {len(logs)} held-out pairs")

    print(f"base model ({base_model}):")
    base_predictions = generate_all(SummarizationModel(base_model), logs)
    print(f"fine-tuned ({model_dir}):")
    tuned_predictions = generate_all(SummarizationModel(str(model_dir)), logs)

    result = {
        "eval_pairs": len(logs),
        "base": rouge_scores(base_predictions, references),
        "fine_tuned": rouge_scores(tuned_predictions, references),
    }

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "rouge.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# Held-out examples: base vs fine-tuned\n"]
    for i in range(min(5, len(logs))):
        lines += [
            f"### Example {i + 1}",
            f"**Log:** {logs[i]}",
            f"**Reference:** {references[i]}",
            f"**Base t5-small:** {base_predictions[i]}",
            f"**Fine-tuned:** {tuned_predictions[i]}",
            "",
        ]
    (outdir / "examples.md").write_text("\n".join(lines))

    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/log-summarizer-t5"))
    parser.add_argument("--corpus", type=Path, default=Path("data/log_summary_pairs.csv"))
    parser.add_argument("--base-model", default="google-t5/t5-small")
    parser.add_argument("--outdir", type=Path, default=Path("docs/summarizer"))
    parser.add_argument("--sample", type=int, default=None,
                        help="evaluate on the first N pairs only (quick check)")
    args = parser.parse_args()
    evaluate(args.model, args.corpus, args.base_model, args.outdir, args.sample)


if __name__ == "__main__":
    main()
