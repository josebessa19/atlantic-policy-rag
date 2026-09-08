"""CLI: parse data/raw → upsert into Qdrant hybrid index.

Usage:
  python -m src.indexing.ingest
  python -m src.indexing.ingest --data data/raw --recreate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ingestion.pipeline import parse_dir
from src.indexing.config import qdrant_collection, qdrant_url
from src.indexing.store import get_client, index_chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest policy chunks into Qdrant (dense BGE-M3 + BM25)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw"),
        help="Directory of policy .txt files (default: data/raw)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection (dev schema changes)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=f"Qdrant collection name (default: {qdrant_collection()})",
    )
    args = parser.parse_args(argv)

    raw_dir = args.data
    if not raw_dir.is_dir():
        raise SystemExit(f"Data directory not found: {raw_dir}")

    chunks = parse_dir(raw_dir)
    print(f"Parsed {len(chunks)} chunks from {raw_dir}")

    client = get_client(qdrant_url())
    n = index_chunks(
        chunks,
        client=client,
        collection=args.collection,
        recreate=args.recreate,
    )
    coll = args.collection or qdrant_collection()
    print(f"Upserted {n} points into collection {coll!r} at {qdrant_url()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
