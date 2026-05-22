import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    qdrant_api_key: str | None
    collection_name: str
    embedding_model: str
    docs_dir: Path
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    recreate_collection: bool

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
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            batch_size=int(os.getenv("BATCH_SIZE", "64")),
            recreate_collection=os.getenv("RECREATE_COLLECTION", "false").lower() in ("1", "true", "yes"),
        )
