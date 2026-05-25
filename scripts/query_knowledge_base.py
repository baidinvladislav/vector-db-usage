import argparse
import sys
from pathlib import Path
from textwrap import shorten

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppSettings  # noqa: E402
from src.embeddings import EmbeddingService  # noqa: E402
from src.qdrant_store import QdrantRepository  # noqa: E402
from src.reranker import RerankerService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Qdrant knowledge base.")

    parser.add_argument(
        "--query",
        required=True,
        help="Search text",
    )
    parser.add_argument(
        "-k",
        "--limit",
        type=int,
        default=5,
        help="Final number of chunks after reranking (default: 5).",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=None,
        help="Candidates before rerank (default: from .env, auto-scaled by index size).",
    )
    parser.add_argument("--doc-id", default=None, help="Restrict search to one document id.")
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Vector search only (no keyword merge).",
    )

    return parser.parse_args()


def _default_fetch_k(settings: AppSettings, points_count: int) -> int:
    base = settings.rerank_fetch_k if settings.rerank_fetch_k > 0 else 20
    if points_count > 10_000:
        return max(base, 50)
    if points_count > 1_000:
        return max(base, 30)
    return base


def main() -> None:
    """ Поиск релевантного ответа на запрос пользователя в векторной БД """
    args = parse_args()
    settings = AppSettings.from_env()

    store = QdrantRepository(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
    )
    points_count = store.points_count()
    if points_count == 0:
        raise SystemExit(
            f"Collection '{settings.collection_name}' is empty. "
            "Run: python scripts/prepare_knowledge_base.py"
        )

    use_hybrid = settings.hybrid_search_enabled and not args.no_hybrid
    fetch_k = args.fetch_k or _default_fetch_k(settings, points_count)
    if fetch_k < args.limit:
        fetch_k = args.limit

    store.ensure_text_index()

    embedder = EmbeddingService(settings.embedder_model)
    query_vector = embedder.embed_query(args.query)

    if use_hybrid:
        hits = store.search_hybrid(
            args.query,
            query_vector,
            limit=fetch_k,
            doc_id=args.doc_id,
        )
        retrieval = "hybrid (vector + keyword)"
    else:
        hits = store.search(query_vector, limit=fetch_k, doc_id=args.doc_id)
        retrieval = "vector"

    print(f"Query: {args.query!r}")
    print(f"Index: {points_count} points  |  retrieval: {retrieval}  |  candidates: {len(hits)}")

    if len(hits) < fetch_k:
        print(
            f"Note: only {len(hits)} candidates returned (requested {fetch_k}). "
            "Try --fetch-k 80 or a more specific query."
        )

    reranker = RerankerService(settings.reranker_model)
    results = reranker.rerank(args.query, hits, top_k=args.limit)
    print(f"Reranker: {reranker.model_name}\n")

    for rank, item in enumerate(results, start=1):
        payload = item.point.payload or {}
        _print_hit(
            rank=rank,
            doc_id=payload.get("doc_id", ""),
            chunk_index=payload.get("chunk_index", ""),
            title=payload.get("title", ""),
            text=payload.get("text", ""),
            score_label=f"rerank={item.rerank_score:.4f}  vector={item.vector_score:.4f}",
        )


def _print_hit(
    rank: int,
    *,
    doc_id: str,
    chunk_index: str | int,
    title: str,
    text: str,
    score_label: str,
) -> None:
    print(f"--- #{rank}  {score_label}  doc={doc_id}  chunk={chunk_index}")
    if title:
        print(f"Title: {shorten(str(title), width=120)}")
    print(shorten(str(text), width=500, placeholder=" ..."))
    print()


if __name__ == "__main__":
    main()
