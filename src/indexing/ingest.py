"""CLI: parse data/raw → upsert into Qdrant hybrid index.

Usage:
  python -m src.indexing.ingest
  python -m src.indexing.ingest --data data/raw --recreate
  python -m src.indexing.ingest --if-empty
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.ingestion.pipeline import parse_dir
from src.indexing.config import qdrant_collection, qdrant_url
from src.indexing.store import get_client, index_chunks

logger = logging.getLogger(__name__)


def collection_is_empty(
    *,
    client=None,
    collection: str | None = None,
) -> bool:
    """True when the collection is missing or has zero points."""
    client = client or get_client()
    name = collection or qdrant_collection()
    if not client.collection_exists(name):
        return True
    info = client.get_collection(name)
    return int(info.points_count or 0) == 0


def ingest_if_empty(
    *,
    data_dir: Path | str = Path("data/raw"),
    client=None,
    collection: str | None = None,
) -> int:
    """Parse and upsert only when the collection is empty. Returns points upserted (0 if skipped)."""
    client = client or get_client()
    name = collection or qdrant_collection()
    raw_dir = Path(data_dir)

    if not collection_is_empty(client=client, collection=name):
        logger.info("Collection %r already populated — skip ingest", name)
        return 0

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {raw_dir}")

    logger.info("Collection %r empty — ingesting %s", name, raw_dir)
    chunks = parse_dir(raw_dir)
    logger.info("Parsed %d chunks from %s", len(chunks), raw_dir)
    n = index_chunks(chunks, client=client, collection=name, recreate=False)
    logger.info("Upserted %d points into collection %r", n, name)
    return n


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
        "--if-empty",
        action="store_true",
        help="Only ingest when the collection is missing or has zero points",
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

    if args.if_empty and args.recreate:
        raise SystemExit("Use either --if-empty or --recreate, not both")

    client = get_client(qdrant_url())
    coll = args.collection or qdrant_collection()

    if args.if_empty:
        n = ingest_if_empty(data_dir=raw_dir, client=client, collection=coll)
        if n == 0:
            print(f"Collection {coll!r} already populated — skipped ingest")
        else:
            print(f"Upserted {n} points into collection {coll!r} at {qdrant_url()}")
        return 0

    chunks = parse_dir(raw_dir)
    print(f"Parsed {len(chunks)} chunks from {raw_dir}")

    n = index_chunks(
        chunks,
        client=client,
        collection=args.collection,
        recreate=args.recreate,
    )
    print(f"Upserted {n} points into collection {coll!r} at {qdrant_url()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
