# Atlantic Policy RAG API

Stateless policy Q&A API for the Atlantic Ventures AI Developer Challenge.
Answers grounded in corporate remote-work and IT hardware policies, with citations.

## Stack

- **LLM (generation):** Google Gemini `gemini-2.5-flash` (Step 3+)
- **Embeddings (local GPU):** `BAAI/bge-m3` via sentence-transformers (CUDA), **1024-d** dense; FastEmbed BM25 for sparse
- **Vector store:** Qdrant (Docker) with named vectors `dense` + `bm25`
- **Retrieval:** Hybrid BM25 + dense, fused with Qdrant RRF
- **API:** FastAPI `POST /query` + `GET /health` (OpenAPI at `/docs`)

## Layout

- `data/raw/` — policy corpus (`.txt`)
- `data/eval/` — deterministic benchmark JSON
- `src/ingestion/` — parse + chunk
- `src/indexing/` — embeddings + Qdrant hybrid index
- `src/generation/` — RAG prompt assembly
- `src/guardrails/` — refusal / injection defenses
- `backend/` — HTTP API

## Setup (conda + GPU)

Use a **project-specific** conda env (`atlantic-rag`). Do not reuse the broken base Anaconda torch install.

```bash
conda env create -f environment.yml
conda activate atlantic-rag

# RTX 3080 / CUDA 12.x driver — GPU wheels
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -e ".[dev]"
cp .env.example .env   # GEMINI_API_KEY only needed for Step 3 generation
docker compose up -d qdrant

# Sanity check
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

First embedding run downloads BGE-M3 from HuggingFace (~2 GB; do not commit the cache).

## Index the corpus (Step 2)

```bash
conda activate atlantic-rag
docker compose up -d qdrant
python -m src.indexing.ingest              # upsert by stable chunk id
python -m src.indexing.ingest --recreate   # wipe collection + reindex (dev)
pytest tests/indexing/ -q
```

## Run the API (Step 3)

Host process (GPU embeddings) + Docker Qdrant:

```bash
conda activate atlantic-rag
docker compose up -d qdrant
python -m src.indexing.ingest   # if collection empty
uvicorn backend.main:app --reload
# OpenAPI: http://localhost:8000/docs
curl -sf http://localhost:8000/health
curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d "{\"query\": \"What is the laptop allowance for an engineer in Germany?\"}"
pytest tests/ -q
```

Pipeline: `hybrid_search` → **drop superseded legacy** → injection / low-score refuse (skip LLM) → Gemini Flash → citations from kept hits.

## Decision Log

| Decision | Options considered | Choice | Why |
|----------|-------------------|--------|-----|
| Dense embeddings | Gemini embed API; FastEmbed custom ONNX; **sentence-transformers BGE-M3** | **`BAAI/bge-m3` via sentence-transformers on CUDA, 1024-d** | Reference M3 checkpoint (not a FastEmbed registry workaround). Dedicated conda env + cu124 torch so GPU works; FastEmbed kept only for BM25. |
| Sparse / lexical | M3 learned sparse / ColBERT; **classic BM25** | **FastEmbed `Qdrant/bm25` + Qdrant `Modifier.IDF`** | Identifier-safe (`€1,800`, `IT-SPEC-2025-A`). Independent of the dense net. Brief asks for BM25. |
| Fusion | Weighted score blend; hand-rolled RRF; **Qdrant Fusion.RRF** | **Qdrant `Fusion.RRF`** (prefetch ~20 / return ~8) | Dense cosine and BM25 scores are not on one scale; ranks fuse without alpha tuning. |
| Point ids | String chunk id; uint hash; **UUID5** | **`uuid5(namespace, chunk.id)`** + payload `chunk_id` | Qdrant accepts UUID/uint64 only; UUID5 keeps upserts idempotent. |
| Temporal drop | Drop legacy at index time; prompt-only; intent regex lift; **index both, drop at assembly always** | **Strict A+D at assembly** — 2024 never in the prompt | Recency is metadata (`status`/`supersedes`), not embeddings. Demo can still show both in Qdrant. Historical compare is a **v1 limit**; v2 would be an explicit `mode`/`as_of` request field, not English NLU. |
| LLM | OpenAI; local LLM; **Gemini Flash** | **`gemini-2.5-flash`** (`GEMINI_API_KEY` generation only) | Free tier; retrieval stays local so Google quota does not break search. |
| Refusal / skip LLM | Always call model; **empty or top fused RRF &lt; `MIN_FUSED_SCORE` (default 0.02)**; injection patterns | **Skip Gemini** on empty/low retrieval and injection; grounded refuse for off-topic-but-retrieved (e.g. parental leave) | Cheaper and safer than asking the model to be humble on failed retrieval. Score is not a topic classifier — parental leave still hits the LLM under the grounded prompt. |
| Python env | Base Anaconda; venv; **project conda env** | **`atlantic-rag` (conda + cu124 torch)** | Isolates GPU torch from broken base-env DLLs. |