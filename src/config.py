import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """ Настройка конфигурации приложения """
    qdrant_url: str
    qdrant_api_key: str | None
    collection_name: str
    embedding_model: str
    docs_dir: Path
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    recreate_collection: bool
    reranker_model: str
    rerank_fetch_k: int
    hybrid_search_enabled: bool

    @classmethod
    def from_env(cls) -> "Settings":
        docs_dir = Path(os.getenv("DOCS_DIR", "files"))
        if not docs_dir.is_absolute():
            docs_dir = PROJECT_ROOT / docs_dir

        return cls(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=os.getenv("QDRANT_COLLECTION", "knowledge_base"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            docs_dir=docs_dir,
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            batch_size=int(os.getenv("BATCH_SIZE", "64")),
            recreate_collection=os.getenv("RECREATE_COLLECTION", "false").lower() in ("1", "true", "yes"),
            reranker_model=os.getenv(
                "RERANKER_MODEL",
                "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            ),
            rerank_fetch_k=int(os.getenv("RERANK_FETCH_K", "50")),
            hybrid_search_enabled=os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() in ("1", "true", "yes"),
        )
