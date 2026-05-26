#!/usr/bin/env python3
"""CLI wrapper around GET /ask-rag logic."""

import argparse
import sys
from pathlib import Path
from textwrap import shorten

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.container import init_app_container  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Qdrant knowledge base.")
    parser.add_argument("--query", required=True, help="Search text")
    parser.add_argument("-k", "--limit", type=int, default=5, help="Final results count.")
    parser.add_argument("--fetch-k", type=int, default=None, help="Candidates before rerank.")
    parser.add_argument("--doc-id", default=None, help="Restrict search to one document id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    container = init_app_container()
    result = container.rag_service.get_answer(
        args.query,
        limit=args.limit,
        fetch_k=args.fetch_k,
        doc_id=args.doc_id,
    )

    print(f"Query: {result.query!r}\n")
    print(f"Answer:\n{shorten(result.answer, width=800, placeholder=' ...')}\n")
    for rank, source in enumerate(result.sources, start=1):
        print(
            f"--- #{rank}  rerank={source.rerank_score:.4f}  "
            f"vector={source.vector_score:.4f}  doc={source.doc_id}  chunk={source.chunk_index}"
        )
        if source.title:
            print(f"Title: {shorten(source.title, width=120)}")
        print(shorten(source.text, width=500, placeholder=" ..."))
        print()


if __name__ == "__main__":
    main()
