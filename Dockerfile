FROM python:3.12-slim

WORKDIR /app

# CPU-only torch keeps the image an order of magnitude smaller than the default wheel
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[summarizer]"

COPY bin ./bin

ENV MODEL_DIR=models/log-summarizer-t5 \
    POOL_SIZE=2 \
    PORT=8000

EXPOSE 8000
CMD ["python", "bin/serve.py"]
