# Atlantic Policy RAG API

Stateless policy Q&A API for the Atlantic Ventures AI Developer Challenge.
Answers grounded in corporate remote-work and IT hardware policies, with citations.

## Stack (planned)

- **LLM / embeddings:** Google Gemini (`gemini-2.5-flash`, `gemini-embedding-001`)
- **Vector store:** Qdrant (Docker)
- **Retrieval:** Hybrid BM25 + dense, fused with RRF
- **API:** FastAPI `POST /query` (later)

## Layout

- `data/raw/` — policy corpus (`.txt`)
- `data/eval/` — deterministic benchmark JSON
- `src/ingestion/` — parse + chunk
- `src/indexing/` — embeddings + Qdrant
- `src/generation/` — RAG prompt assembly
- `src/guardrails/` — refusal / injection defenses
- `backend/` — HTTP API

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set GEMINI_API_KEY
docker compose up -d qdrant
```

Ingestion, indexing, and the query API land in later steps.
