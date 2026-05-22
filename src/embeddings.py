from __future__ import annotations

from fastembed import TextEmbedding


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self._model = TextEmbedding(model_name=model_name)
        self._model_name = model_name
        self._uses_e5_prefix = "multilingual-e5" in model_name.lower()

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if self._uses_e5_prefix:
            vectors = list(self._model.passage_embed(texts))
        else:
            vectors = list(self._model.embed(texts))
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        if self._uses_e5_prefix:
            vector = next(self._model.query_embed([text]))
        else:
            vector = next(self._model.embed([text]))
        return vector.tolist()

    def vector_size(self) -> int:
        return TextEmbedding.get_embedding_size(self._model_name)
