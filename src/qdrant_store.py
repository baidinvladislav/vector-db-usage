import uuid
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import ScoredPoint

from src.documents import DocumentChunk


def _point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{chunk_index}"))


class QdrantStore:
    def __init__(self, *, url: str, api_key: str | None, collection_name: str) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)

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

    def upsert_chunks(self, chunks: Iterable[DocumentChunk], vectors: list[list[float]], *, embedding_model: str) -> int:
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

    def search(self, query_vector: list[float], *, limit: int = 5, doc_id: str | None = None) -> list[ScoredPoint]:
        """ Ищет релевантные чанки в векторной БД по эмбеддингу запроса пользователя """
        query_filter = None
        if doc_id is not None:
            query_filter = rest.Filter(
                must=[rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=doc_id))]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return response.points

    def collection_info(self) -> rest.CollectionInfo | None:
        if not self.client.collection_exists(self.collection_name):
            return None

        return self.client.get_collection(self.collection_name)

    def points_count(self) -> int:
        info = self.collection_info()
        return info.points_count if info else 0
