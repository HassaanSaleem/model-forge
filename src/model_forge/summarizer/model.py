"""T5 summarization model wrapper.

One class wraps tokenizer + model + generation so the pool and the API never touch
transformers directly. The "summarize: " task prefix is applied here, at inference,
because training applied it — a T5 fine-tune only answers the question it was asked
during training, and forgetting the prefix at serving time is the classic way to
quietly lose most of the fine-tune's benefit.
"""

from __future__ import annotations

TASK_PREFIX = "summarize: "
MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 64


class SummarizationModel:
    def __init__(self, model_dir: str = "google-t5/t5-small"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def summarize(self, text: str) -> str:
        return self.summarize_batch([text])[0]

    def summarize_batch(self, texts: list[str]) -> list[str]:
        import torch

        prefixed = [TASK_PREFIX + t for t in texts]
        inputs = self.tokenizer(
            prefixed, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_INPUT_TOKENS,
        )
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
            )
        return [
            self.tokenizer.decode(o, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True).strip()
            for o in outputs
        ]
