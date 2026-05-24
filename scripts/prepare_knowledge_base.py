import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.documents import build_chunks  # noqa: E402
from src.embeddings import EmbeddingService  # noqa: E402
from src.qdrant_store import QdrantStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Qdrant knowledge base from text files.")

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingest.",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Process only the first N documents (for quick tests).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    recreate = settings.recreate_collection or args.recreate

    if not settings.docs_dir.is_dir():
        raise SystemExit(f"Docs directory not found: {settings.docs_dir}")

    print(f"Loading documents from {settings.docs_dir} ...")
    all_chunks = build_chunks(
        settings.docs_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        limit=args.limit_docs,
    )
    if not all_chunks:
        raise SystemExit(f"No .txt documents found in {settings.docs_dir}")

    doc_count = len({c.doc_id for c in all_chunks})
    print(f"Documents: {doc_count}, chunks: {len(all_chunks)}")

    embedder = EmbeddingService(settings.embedding_model)
    store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
    )
    store.ensure_collection(
        vector_size=embedder.vector_size(),
        recreate=recreate,
    )

    total_upserted = 0
    batch_size = settings.batch_size
    for start in tqdm(
        range(0, len(all_chunks), batch_size),
        desc="Indexing",
        unit="batch",
    ):
        batch = all_chunks[start : start + batch_size]
        vectors = embedder.embed_passages([c.text for c in batch])
        total_upserted += store.upsert_chunks(
            batch,
            vectors,
            embedding_model=embedder.model_name,
        )

    info = store.collection_info()
    points = info.points_count if info else total_upserted
    print(
        f"Done. Collection '{settings.collection_name}' at {settings.qdrant_url} "
        f"— points: {points}"
    )


if __name__ == "__main__":
    main()
