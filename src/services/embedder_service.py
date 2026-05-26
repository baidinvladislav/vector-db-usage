from dataclasses import dataclass

from fastembed import TextEmbedding


@dataclass
class EmbedderService:
    model: TextEmbedding
    model_name: str
    uses_e5_prefix: bool

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if self.uses_e5_prefix:
            vectors = list(self.model.passage_embed(texts))
        else:
            vectors = list(self.model.embed(texts))
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        if self.uses_e5_prefix:
            vector = next(self.model.query_embed([text]))
        else:
            vector = next(self.model.embed([text]))
        return vector.tolist()

    def vector_size(self) -> int:
        return TextEmbedding.get_embedding_size(self.model_name)
