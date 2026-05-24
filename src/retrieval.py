import re

from qdrant_client.http.models import ScoredPoint

_RU_STOPWORDS = {
    "как",
    "какая",
    "какой",
    "какие",
    "какое",
    "что",
    "где",
    "когда",
    "почему",
    "зачем",
    "который",
    "которая",
    "которые",
    "это",
    "этот",
    "эта",
    "эти",
    "для",
    "при",
    "над",
    "под",
    "или",
    "ли",
    "не",
    "ни",
    "в",
    "во",
    "на",
    "по",
    "из",
    "к",
    "ко",
    "о",
    "об",
    "от",
    "до",
    "и",
    "а",
    "но",
    "же",
    "бы",
    "the",
    "is",
    "are",
    "was",
    "were",
}


def extract_keywords(query: str, *, max_keywords: int = 6) -> list[str]:
    words = re.findall(r"[\w\d]+", query.lower(), flags=re.UNICODE)
    keywords: list[str] = []
    seen: set[str] = set()
    for word in words:
        if len(word) < 3 or word in _RU_STOPWORDS:
            continue
        if word in seen:
            continue
        seen.add(word)
        keywords.append(word)
        if len(keywords) >= max_keywords:
            break
    return keywords


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredPoint]],
    *,
    limit: int,
    k: int = 60,
) -> list[ScoredPoint]:
    scores: dict[str | int, float] = {}
    points: dict[str | int, ScoredPoint] = {}

    for ranked in ranked_lists:
        for rank, point in enumerate(ranked):
            point_id = point.id
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank + 1)
            if point_id not in points:
                points[point_id] = point

    merged_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)[:limit]
    merged: list[ScoredPoint] = []
    for point_id in merged_ids:
        point = points[point_id]
        merged.append(
            ScoredPoint(
                id=point.id,
                version=point.version,
                score=scores[point_id],
                payload=point.payload,
                vector=point.vector,
                shard_key=point.shard_key,
            )
        )
    return merged
