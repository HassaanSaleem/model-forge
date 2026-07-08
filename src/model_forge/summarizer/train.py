"""Fine-tune google-t5/t5-small on the log -> summary corpus.

    python -m model_forge.summarizer.train --epochs 5 --outdir models/log-summarizer-t5

The recipe: every log line is prefixed with "summarize: " (T5 is a text-to-text
model — the prefix *is* the task selector), inputs truncate at 256 tokens and
targets at 128, and training runs with linear LR decay from 5e-5, 500 warmup
steps, weight decay 0.01, per-epoch evaluation, best-checkpoint restoration on
eval loss, and early stopping after 3 flat epochs. Pad tokens in the labels are
masked to -100 so they don't contribute to the loss. Runs on CUDA, Apple MPS, or
CPU — whatever torch finds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MAX_INPUT_TOKENS = 256
MAX_TARGET_TOKENS = 128
SEED = 1337


def load_pairs(csv_path: Path, seed: int = SEED):
    """Load the corpus and return a deterministic 90/10 train/eval split."""
    import pandas as pd

    df = pd.read_csv(csv_path).dropna()
    eval_df = df.sample(frac=0.1, random_state=seed)
    train_df = df.drop(eval_df.index)
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


class PairDataset:
    """Torch-style dataset of tokenized (log, summary) pairs."""

    def __init__(self, frame, tokenizer):
        from .model import TASK_PREFIX

        inputs = tokenizer(
            [TASK_PREFIX + t for t in frame["log"].tolist()],
            padding="max_length", truncation=True, max_length=MAX_INPUT_TOKENS,
        )
        targets = tokenizer(
            frame["summary"].tolist(),
            padding="max_length", truncation=True, max_length=MAX_TARGET_TOKENS,
        )
        pad_id = tokenizer.pad_token_id
        labels = [
            [(token if token != pad_id else -100) for token in row]
            for row in targets["input_ids"]
        ]
        self.encodings = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        import torch

        return {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}


def train(corpus: Path = Path("data/log_summary_pairs.csv"),
          base_model: str = "google-t5/t5-small",
          outdir: Path = Path("models/log-summarizer-t5"),
          epochs: int = 5, batch_size: int = 16, learning_rate: float = 5e-5) -> Path:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    train_df, eval_df = load_pairs(corpus)
    print(f"corpus: {len(train_df)} train / {len(eval_df)} eval pairs")

    args = TrainingArguments(
        output_dir=str(outdir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=3,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=PairDataset(train_df, tokenizer),
        eval_dataset=PairDataset(eval_df, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    outdir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(outdir))
    tokenizer.save_pretrained(str(outdir))
    print(f"saved fine-tuned model to {outdir}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/log_summary_pairs.csv"))
    parser.add_argument("--base-model", default="google-t5/t5-small")
    parser.add_argument("--outdir", type=Path, default=Path("models/log-summarizer-t5"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()
    train(args.corpus, args.base_model, args.outdir, args.epochs,
          args.batch_size, args.learning_rate)


if __name__ == "__main__":
    main()
