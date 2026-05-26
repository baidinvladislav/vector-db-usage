from dataclasses import dataclass

from qdrant_client.http.models import ScoredPoint


@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    source_path: str
    title: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RankedHit:
    point: ScoredPoint
    vector_score: float
    rerank_score: float


@dataclass(frozen=True)
class SourceChunk:
    doc_id: str
    chunk_index: int
    title: str
    text: str
    rerank_score: float
    vector_score: float


@dataclass(frozen=True)
class InitKnowledgeResult:
    documents: int
    chunks: int
    points: int


@dataclass(frozen=True)
class RagAnswer:
    query: str
    answer: str
    sources: list[SourceChunk]
