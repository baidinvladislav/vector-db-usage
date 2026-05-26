#!/usr/bin/env python3
"""CLI wrapper around POST /init-knowledge logic."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.container import init_app_container  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Qdrant knowledge base from text files.")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection.")
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Process only the first N documents (overrides DOCS_LIMIT).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    container = init_app_container()

    if args.limit_docs is not None:
        container.rag_service.chunker_service.docs_limit = args.limit_docs

    result = container.rag_service.init_knowledge_base(recreate=args.recreate)
    print(
        f"Done. documents={result.documents}, chunks={result.chunks}, points={result.points}"
    )


if __name__ == "__main__":
    main()
