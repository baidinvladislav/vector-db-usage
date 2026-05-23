import argparse
import sys
from pathlib import Path
from textwrap import shorten

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.embeddings import EmbeddingService  # noqa: E402
from src.qdrant_store import QdrantStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Qdrant knowledge base.")

    parser.add_argument(
        "--query",
        default="Какая длина реки Нева?",
        help="Search text, e.g. 'река Нева длина'",
    )
    parser.add_argument(
        "-k",
        "--limit",
        type=int,
        default=5,
        help="Number of chunks to return (default: 5).",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Search only within one document (file stem, e.g. 101).",
    )

    return parser.parse_args()


def main() -> None:
    """ Поиск релевантного ответа на запрос пользователя в векторной БД """
    args = parse_args()
    settings = Settings.from_env()

    store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
    )
    if store.points_count() == 0:
        raise SystemExit(
            f"Collection '{settings.collection_name}' is empty. "
            "Run: python scripts/prepare_knowledge_base.py"
        )

    embedder = EmbeddingService(settings.embedding_model)
    query_vector = embedder.embed_query(args.query)
    hits = store.search(
        query_vector,
        limit=args.limit,
        doc_id=args.doc_id,
    )

    print(f"Query: {args.query!r}\n")
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        title = payload.get("title", "")
        doc_id = payload.get("doc_id", "")
        chunk_index = payload.get("chunk_index", "")
        text = payload.get("text", "")
        print(f"--- #{rank}  score={hit.score:.4f}  doc={doc_id}  chunk={chunk_index}")
        if title:
            print(f"Title: {shorten(str(title), width=120)}")
        print(shorten(str(text), width=500, placeholder=" ..."))
        print()


if __name__ == "__main__":
    main()
