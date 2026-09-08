# Atlantic Policy RAG API — cu124 PyTorch (same family as host conda).
# CUDA wheels still run on CPU when no GPU is injected; image is large.
# Pin tags for reproducible builds.
FROM python:3.11.11-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (uid 1000); HF cache volume mounts under /home/app/.cache
RUN useradd --create-home --uid 1000 app

COPY pyproject.toml README.md ./
COPY backend/ backend/
COPY src/ src/
COPY data/raw/ data/raw/
COPY docker-entrypoint.sh /docker-entrypoint.sh

# CUDA 12.4 wheels — use GPU when Compose injects nvidia; else CPU fallback
# via torch.cuda.is_available() in src/indexing/embeddings.py.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -e . \
    && chmod +x /docker-entrypoint.sh \
    && chown -R app:app /app /home/app

ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/health || exit 1

# Entrypoint runs as root briefly to chown the hf_cache volume, then drops to app.
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
