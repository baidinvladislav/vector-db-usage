from dataclasses import dataclass

from tqdm import tqdm

from src.services.embedder_service import EmbedderService
from src.services.chunker_service import ChunkerService
from src.repositories.qdrant_repository import QdrantRepository


@dataclass
class RagService:
    chunker_service: ChunkerService
    embedder_service: EmbedderService
    qdrant_repository: QdrantRepository
    reranker_service: RerankerService
    retrieval_service: RetrievalService

    def init_knowledge_base(self) -> None:
        args = parse_args()
        # settings = AppSettings.from_env()
        recreate = settings.recreate_collection or args.recreate

        if not settings.docs_dir.is_dir():
            raise SystemExit(f"Docs directory not found: {settings.docs_dir}")

        print(f"Loading documents from {settings.docs_dir} ...")
        all_chunks = self.chunker_service.build_chunks()
        if not all_chunks:
            raise SystemExit(f"No .txt documents found in {settings.docs_dir}")

        doc_count = len({c.doc_id for c in all_chunks})
        print(f"Documents: {doc_count}, chunks: {len(all_chunks)}")

        self.qdrant_repository.ensure_collection(
            vector_size=self.embedder_service.vector_size(),
            recreate=recreate,
        )

        total_upserted = 0
        batch_size = settings.batch_size
        for start in tqdm(
            range(0, len(all_chunks), batch_size),
            desc="Indexing",
            unit="batch",
        ):
            batch = all_chunks[start: start + batch_size]
            vectors = self.embedder_service.embed_passages([c.text for c in batch])
            total_upserted += self.qdrant_repository.upsert_chunks(
                chunks=batch,
                vectors=vectors,
                embedding_model=self.embedder_service.model_name,
            )

        info = self.qdrant_repository.collection_info()
        points = info.points_count if info else total_upserted
        print(
            f"Done. Collection '{settings.collection_name}' at {settings.qdrant_url} "
            f"— points: {points}"
        )

    def get_answer(self, query: str) -> str:
        pass
