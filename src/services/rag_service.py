from dataclasses import dataclass

from tqdm import tqdm

from src.domain.models import InitKnowledgeResult, RagAnswer, SourceChunk
from src.repositories.qdrant_repository import QdrantRepository
from src.services.chunker_service import ChunkerService
from src.services.embedder_service import EmbedderService
from src.services.reranker_service import RerankerService
from src.shared.settings import AppSettings


@dataclass
class RagService:
    settings: AppSettings
    chunker_service: ChunkerService
    embedder_service: EmbedderService
    qdrant_repository: QdrantRepository
    reranker_service: RerankerService

    def init_knowledge_base(self, *, recreate: bool | None = None) -> InitKnowledgeResult:
        should_recreate = (
            self.settings.recreate_collection if recreate is None else recreate
        )

        if not self.settings.docs_dir.is_dir():
            raise FileNotFoundError(f"Docs directory not found: {self.settings.docs_dir}")

        all_chunks = self.chunker_service.build_chunks()
        if not all_chunks:
            raise ValueError(f"No .txt documents found in {self.settings.docs_dir}")

        doc_count = len({chunk.doc_id for chunk in all_chunks})

        self.qdrant_repository.ensure_collection(
            vector_size=self.embedder_service.vector_size(),
            recreate=should_recreate,
        )

        total_upserted = 0
        batch_size = self.settings.batch_size
        for start in tqdm(
            range(0, len(all_chunks), batch_size),
            desc="Indexing",
            unit="batch",
        ):
            batch = all_chunks[start : start + batch_size]
            vectors = self.embedder_service.embed_passages([chunk.text for chunk in batch])
            total_upserted += self.qdrant_repository.upsert_chunks(
                chunks=batch,
                vectors=vectors,
                embedding_model=self.embedder_service.model_name,
            )

        info = self.qdrant_repository.collection_info()
        points = info.points_count if info else total_upserted

        return InitKnowledgeResult(
            documents=doc_count,
            chunks=len(all_chunks),
            points=points,
        )

    def get_answer(
        self,
        query: str,
        *,
        limit: int | None = None,
        fetch_k: int | None = None,
        doc_id: str | None = None,
    ) -> RagAnswer:
        if self.qdrant_repository.points_count() == 0:
            raise RuntimeError(
                "Knowledge base is empty. Call POST /init-knowledge first."
            )

        top_k = limit or self.settings.result_limit
        candidate_k = fetch_k or self._default_fetch_k()
        if candidate_k < top_k:
            candidate_k = top_k

        self.qdrant_repository.ensure_text_index()

        query_vector = self.embedder_service.embed_query(query)
        if self.settings.hybrid_search_enabled:
            hits = self.qdrant_repository.search_hybrid(
                query,
                query_vector,
                limit=candidate_k,
                doc_id=doc_id,
            )
        else:
            hits = self.qdrant_repository.search(
                query_vector,
                limit=candidate_k,
                doc_id=doc_id,
            )

        ranked = self.reranker_service.rerank(query, hits, top_k=top_k)
        sources = [self._to_source(item) for item in ranked]
        answer = sources[0].text if sources else "No relevant documents found."

        return RagAnswer(query=query, answer=answer, sources=sources)

    def _default_fetch_k(self) -> int:
        points_count = self.qdrant_repository.points_count()
        base = self.settings.rerank_fetch_k if self.settings.rerank_fetch_k > 0 else 20
        if points_count > 10_000:
            return max(base, 50)
        if points_count > 1_000:
            return max(base, 30)
        return base

    @staticmethod
    def _to_source(item) -> SourceChunk:
        payload = item.point.payload or {}
        return SourceChunk(
            doc_id=str(payload.get("doc_id", "")),
            chunk_index=int(payload.get("chunk_index", 0)),
            title=str(payload.get("title", "")),
            text=str(payload.get("text", "")),
            rerank_score=item.rerank_score,
            vector_score=item.vector_score,
        )
