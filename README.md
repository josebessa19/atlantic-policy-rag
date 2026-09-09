# Atlantic Policy RAG API

Stateless policy Q&A API for the **Atlantic Ventures AI Developer Challenge (Part B)**.
Answers are grounded in corporate remote-work and IT hardware policies, with citations.

**Architecture one-liner:** Hybrid BM25 + BGE-M3 in Qdrant (RRF) → drop superseded 2024 at assembly → guardrails → Gemini Flash → citations `{document_id, section, relevance_score}`.

## Stack

- **LLM (generation):** Google Gemini `gemini-2.5-flash` (`GEMINI_API_KEY` only)
- **Embeddings (local):** `BAAI/bge-m3` via sentence-transformers, **1024-d** dense; FastEmbed BM25 for sparse
- **Vector store:** Qdrant with named vectors `dense` + `bm25`
- **Retrieval:** Hybrid BM25 + dense, fused with Qdrant RRF
- **API:** FastAPI `POST /query` + `GET /health` (OpenAPI at `/docs`)

## Layout

- `data/raw/` — policy corpus (`.txt`)
- `data/eval/` — deterministic benchmark JSON (+ gitignored RAGAS reports)
- `src/ingestion/` — parse + chunk
- `src/indexing/` — embeddings + Qdrant hybrid index
- `src/generation/` — RAG prompt assembly
- `src/guardrails/` — refusal / injection defenses
- `src/eval/` — benchmark helpers + RAGAS report CLI
- `backend/` — HTTP API
- `docs/DEMO.md` — ~20 min live demo script

## How to run (Docker Compose — reviewer path)

Requires Docker Desktop (or equivalent) and a `.env` with `GEMINI_API_KEY` (for `/query`).

```bash
cp .env.example .env   # set GEMINI_API_KEY
docker compose up --build
# First start: downloads BGE-M3 (~2 GB) into the hf_cache volume, then AUTO_INGEST indexes data/raw
curl -sf http://localhost:8000/health
curl -s http://localhost:8000/query -H "Content-Type: application/json" \
  -d "{\"query\":\"What is the laptop allowance for an engineer in Germany?\"}"
```

OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)

**GPU in Docker (optional, NVIDIA + Docker Desktop WSL2):** same CUDA image; overlay injects the device. Default compose does **not** require NVIDIA so CPU-only reviewers still boot.

```bash
# Sanity: docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
# Look for: Embedding device: cuda
```

Without the overlay (or without a GPU), expect `Embedding device: cpu`. CUDA wheels still run on CPU; the image is large because of those libs.

Manual ingest (if you disabled `AUTO_INGEST`):

```bash
docker compose exec api python -m src.indexing.ingest --if-empty
# or wipe + reindex:
docker compose exec api python -m src.indexing.ingest --recreate
```

## How to run (host conda — eval / GPU iterate)

Use project env `atlantic-rag` (cu124 torch). Do not reuse a broken base Anaconda torch install.

```bash
conda env create -f environment.yml
conda activate atlantic-rag
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
cp .env.example .env
docker compose up -d qdrant
python -m src.indexing.ingest              # upsert by stable chunk id
# python -m src.indexing.ingest --if-empty  # no-op if already populated
# python -m src.indexing.ingest --recreate  # wipe + reindex (dev)
uvicorn backend.main:app --reload
```

Pipeline: `hybrid_search` → **drop superseded legacy** → injection / low-score refuse (skip LLM) → Gemini Flash → citations from kept hits.

## Evaluate

Deterministic **pytest** is the gate. **RAGAS** is a separate dashboard (faithfulness + context recall on post-temporal-drop contexts).

```bash
conda activate atlantic-rag
docker compose up -d qdrant
python -m src.indexing.ingest --if-empty
pip install -e ".[dev,eval]"

pytest tests/ -q

# Live golden benchmark (Qdrant + Gemini) — TEST-01..05
LIVE=1 pytest tests/eval/test_benchmark.py -q
# PowerShell:  $env:LIVE=1; pytest tests/eval/test_benchmark.py -q

python -m src.eval.ragas_report
# → data/eval/ragas_report.md (+ .json)
```


| Flag / command                    | Needs                              | What it proves                                 |
| --------------------------------- | ---------------------------------- | ---------------------------------------------- |
| `pytest tests/ -q`                | nothing live                       | Unit + offline assertion helpers               |
| `LIVE=1 pytest tests/eval/…`      | Qdrant ingested + `GEMINI_API_KEY` | Five JSON categories against the real pipeline |
| `python -m src.eval.ragas_report` | same + `[eval]` extra              | Faithfulness + context recall dashboard        |


**Known flakiness:** free-tier Gemini RPM/RPD can 429 on live tests or RAGAS. Retry after a pause. LLM-as-judge scores vary; **do not retune the pipeline for RAGAS** — pytest citation/refusal contracts are authoritative.

## Decision Log


| Decision           | Options considered                                                        | Choice                                                     | Why                                                                                               |
| ------------------ | ------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Parser             | Docling / OCR / PDF stack; **hand** `.txt` **parser**                     | `policy_txt`                                               | Corpus is three born-digital `.txt` files; no layout engine needed.                               |
| Chunking           | Fixed windows; LLM chunking; **structure-aware**                          | **Structure walk; tables atomic; ~500 tok / 50 overlap**   | Keeps Germany matrix rows intact for TEST-02; sections stay citable.                              |
| Dense embeddings   | Gemini embed API; FastEmbed custom ONNX; **sentence-transformers BGE-M3** | `BAAI/bge-m3` **via sentence-transformers, 1024-d**        | Reference M3 checkpoint. FastEmbed kept only for BM25.                                            |
| Sparse / lexical   | M3 learned sparse / ColBERT; **classic BM25**                             | **FastEmbed** `Qdrant/bm25` **+ Qdrant** `Modifier.IDF`    | Identifier-safe (`€1,800`, `IT-SPEC-2025-A`). Brief asks for BM25.                                |
| Fusion / rerank    | Weighted blend; hand-rolled RRF; cross-encoder now; **Qdrant Fusion.RRF** | **Qdrant** `Fusion.RRF` (prefetch ~20 / return ~8)         | Dense and BM25 scores are not on one scale. Cross-encoder is a later pool over top-k, not Part B. |
| Point ids          | String chunk id; uint hash; **UUID5**                                     | `uuid5(namespace, chunk.id)` + payload `chunk_id`          | Qdrant accepts UUID/uint64 only; UUID5 keeps upserts idempotent.                                  |
| Temporal drop      | Drop 2024 at index time; prompt-only; **index both, drop at assembly**    | **Strict drop at assembly** — 2024 never in the prompt     | Recency is metadata (`status`/`supersedes`). Demo can still show both docs in Qdrant.             |
| LLM                | OpenAI; local LLM; **Gemini Flash**                                       | `gemini-3.5-flash-lite` (`GEMINI_API_KEY` generation only) | Free tier; retrieval stays local so Google quota does not break search.                           |
| Refusal / skip LLM | Always call model; **empty / low fused RRF / injection**                  | **Skip Gemini** on empty/low retrieval and injection       | Cheaper and safer than asking the model to be humble on failed retrieval.                         |
| Demo client        | Telegram / Streamlit; **OpenAPI**                                         | `/docs` **+ curl**                                         | Deliverable is an API, not a chat widget.                                                         |
| Docker GPU         | CPU-only image; GPU-mandatory compose; **cu124 image + optional overlay** | **Same CUDA image; default compose no nvidia device**      | Reviewers without NVIDIA still `compose up`. Overlay injects GPU on Docker Desktop WSL2.          |
| Python env         | Base Anaconda; venv; **project conda**                                    | `atlantic-rag` **(conda + cu124 torch)**                   | Isolates GPU torch; host path for pytest/RAGAS without image rebuilds.                            |
| Eval split         | RAGAS-only; pytest-only; **both**                                         | **Pytest gate + RAGAS dashboard**                          | Pytest locks IDs/refusal/injection; RAGAS speaks metric language in interviews.                   |
| RAGAS metrics      | +precision; +relevancy; **faithfulness + context recall**                 | **Faithfulness + context recall**                          | Avoid Google embeddings via RAGAS. Contexts are post-drop texts.                                  |
| Live gate          | Auto-detect Qdrant; always live; `LIVE=1` **opt-in**                      | `LIVE=1` **for golden tests**                              | Keeps default `pytest` offline-green.                                                             |                                                  |


## Limitations

- **Single-tenant** — no auth, no `tenant_id` filter, no ReBAC.
- **Gemini is not a BAA** — fine for a challenge PoC; not production PHI/HR legal cover.
- **Tiny corpus** — three `.txt` files; not a production document lake.
- **LLM-judge variance** — RAGAS scores move; pytest is the contract.
- **No cross-encoder reranker** — RRF ranks only.
- **CUDA Docker image is large**; default compose does not inject a GPU (CPU fallback is slower on first ingest/query).

## Scale to 10M+ docs (talking points only — not implemented)

- **HNSW / RAM:** 10M × 1024-d float32 ≈ 40GB vectors; graph overhead often ~1.5–2× → tens of GB **per shard**. This PoC is one small collection.
- **Sharding:** payload `tenant_id` + Qdrant sharding; this API has neither.
- **Incremental upsert:** UUID5 ids already make upserts idempotent; 10M needs CDC from object storage, not `parse_dir` on boot.
- **Rerank pool:** cross-encoder on a **GPU pool over the fused top-k** (e.g. 50), not over 10M docs.
- **BM25 distributed:** Qdrant sparse IDF is fine here; at 10M split lexical (OpenSearch/Elastic) from dense ANN.
- **Eval sampling:** pytest on a labeled slice; RAGAS on a sample — never full-corpus LLM-judge.
- `index_version` **cache:** key answers by `(query_hash, index_version)` so a reindex busts cache.

## What I would do next

- Cross-encoder **rerank pool** over fused top-k
- **Langfuse** (or similar) request traces
- Payload `tenant_id` **filter** at retrieval (still not fake multi-tenant ReBAC)
- PDF / layout parser when the corpus leaves `.txt`

None of these are in the running Part B code.

## Live demo

See `[docs/DEMO.md](docs/DEMO.md)` (~20 minutes, OpenAPI + curl against the five benchmark cases).