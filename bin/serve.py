"""Serve the summarizer: a pool of model replicas behind FastAPI.

    MODEL_DIR=models/log-summarizer-t5 POOL_SIZE=2 python bin/serve.py
"""

import os

import uvicorn

from model_forge.summarizer.api import create_app
from model_forge.summarizer.model import SummarizationModel
from model_forge.summarizer.pool import ModelPool


def main() -> None:
    model_dir = os.environ.get("MODEL_DIR", "models/log-summarizer-t5")
    pool_size = int(os.environ.get("POOL_SIZE", "2"))
    port = int(os.environ.get("PORT", "8000"))

    pool = ModelPool(pool_size, lambda: SummarizationModel(model_dir))
    app = create_app(pool)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
