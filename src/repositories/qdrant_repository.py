import uuid
from dataclasses import dataclass
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import ScoredPoint

from src.documents import DocumentChunk
from src.services.retrieval_service import RetrievalService


def _point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{chunk_index}"))


@dataclass
class QdrantRepository:
    collection_name: str
    client: QdrantClient
    retrieval_service: RetrievalService

    def ensure_collection(self, *, vector_size: int, recreate: bool) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=vector_size,
                    distance=rest.Distance.COSINE,
                ),
            )

        self.ensure_text_index()

    def ensure_text_index(self) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=rest.TextIndexParams(
                    type=rest.TextIndexType.TEXT,
                    tokenizer=rest.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=30,
                    lowercase=True,
                ),
            )
        except Exception:
            pass

    def upsert_chunks(
        self,
        chunks: Iterable[DocumentChunk],
        vectors: list[list[float]],
        *,
        embedding_model: str,
    ) -> int:
        chunk_list = list(chunks)
        if len(chunk_list) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")

        points: list[rest.PointStruct] = []
        for chunk, vector in zip(chunk_list, vectors):
            points.append(
                rest.PointStruct(
                    id=_point_id(chunk.doc_id, chunk.chunk_index),
                    vector=vector,
                    payload={
                        "doc_id": chunk.doc_id,
                        "source_path": chunk.source_path,
                        "title": chunk.title,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "embedding_model": embedding_model,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def _doc_filter(self, doc_id: str | None) -> rest.Filter | None:
        if doc_id is None:
            return None
        return rest.Filter(
            must=[rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=doc_id))]
        )

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 30,
        doc_id: str | None = None,
    ) -> list[ScoredPoint]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._doc_filter(doc_id),
            limit=limit,
            with_payload=True,
        )
        return response.points

    def keyword_search(
        self,
        query_text: str,
        *,
        limit: int,
        doc_id: str | None = None,
    ) -> list[ScoredPoint]:
        keywords = self.retrieval_service.extract_keywords(query_text)
        if not keywords:
            return []

        should = [
            rest.FieldCondition(key="text", match=rest.MatchText(text=word))
            for word in keywords
        ]
        scroll_filter = rest.Filter(
            must=self._doc_filter(doc_id).must if doc_id else [],
            should=should,
        )

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            ScoredPoint(
                id=record.id,
                version=getattr(record, "version", 0) or 0,
                score=1.0 - rank * 0.01,
                payload=record.payload,
            )
            for rank, record in enumerate(records)
        ]

    def search_hybrid(
        self,
        query_text: str,
        query_vector: list[float],
        *,
        limit: int,
        doc_id: str | None = None,
    ) -> list[ScoredPoint]:
        vector_hits = self.search(query_vector, limit=limit, doc_id=doc_id)
        keyword_hits = self.keyword_search(query_text, limit=limit, doc_id=doc_id)
        if not keyword_hits:
            return vector_hits
        return self.retrieval_service.reciprocal_rank_fusion([vector_hits, keyword_hits], limit=limit)

    def collection_info(self) -> rest.CollectionInfo | None:
        if not self.client.collection_exists(self.collection_name):
            return None
        return self.client.get_collection(self.collection_name)

    def points_count(self) -> int:
        info = self.collection_info()
        return info.points_count if info else 0
