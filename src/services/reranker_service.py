from dataclasses import dataclass

from qdrant_client.http.models import ScoredPoint
from sentence_transformers import CrossEncoder

from src.domain.models import RankedHit


@dataclass
class RerankerService:
    model: CrossEncoder
    model_name: str

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
