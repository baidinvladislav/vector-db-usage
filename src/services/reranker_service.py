from dataclasses import dataclass

from qdrant_client.http.models import ScoredPoint
from sentence_transformers import CrossEncoder


@dataclass(frozen=True)
class RankedHit:
    point: ScoredPoint
    vector_score: float
    rerank_score: float


@dataclass
class RerankerService:
    model: CrossEncoder
    model_name: str

    @property
    def model_name(self) -> str:
        return self.model_name

    def rerank(
        self,
        query: str,
        hits: list[ScoredPoint],
        *,
        top_k: int,
    ) -> list[RankedHit]:
        if not hits:
            return []

        pairs = [[query, (hit.payload or {}).get("text", "")] for hit in hits]
        scores = self.model.predict(pairs)

        reranked = sorted(
            zip(hits, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]

        return [
            RankedHit(
                point=hit,
                vector_score=float(hit.score),
                rerank_score=float(score),
            )
            for hit, score in reranked
        ]
