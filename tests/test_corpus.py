import re

from model_forge.summarizer.corpus import generate


def test_deterministic():
    assert generate(200, seed=42) == generate(200, seed=42)


def test_different_seeds_differ():
    assert generate(200, seed=1) != generate(200, seed=2)


def test_count_and_uniqueness():
    rows = generate(1000, seed=7)
    assert len(rows) == 1000
    assert len({log for log, _ in rows}) == 1000


def test_pairs_share_facts():
    # log and summary render from the same slot fill, so numbers must agree
    for log, summary in generate(300, seed=11):
        for number in re.findall(r"\b\d{2,4}\b", summary):
            assert number in log, (number, log, summary)


def test_nothing_credential_shaped():
    text = " ".join(log + " " + summary for log, summary in generate(2000, seed=13))
    assert not re.search(r"[0-9a-fA-F]{12,}", text)   # no long hex
    assert not re.search(r"[0-9]{7,}", text)          # no long digit runs
    assert not re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text)  # no emails
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)  # no IPs
