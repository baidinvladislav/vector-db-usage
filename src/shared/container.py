from dataclasses import dataclass

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from src.services.embedder_service import EmbedderService
from src.services.rag_service import RagService
from src.services.chunker_service import ChunkerService
from src.services.reranker_service import RerankerService
from src.services.retrieval_service import RetrievalService
from src.repositories.qdrant_repository import QdrantRepository
from src.shared.settings import AppSettings


@dataclass
class AppContainer:
    rag_service: RagService
    app_settings: AppSettings


def init_app_container() -> AppContainer:
    app_settings = AppSettings.from_env()

    chunker_service = ChunkerService(
        docs_dir=app_settings.docs_dir,
        chunk_size=app_settings.chunk_size,
        chunk_overlap=app_settings.chunk_overlap,
        docs_limit=app_settings.docs_limit,
    )

    embedder_service = EmbedderService(
        model=TextEmbedding(model_name=app_settings.embedder_model),
        model_name=app_settings.embedder_model,
        uses_e5_prefix="multilingual-e5" in app_settings.embedder_model.lower(),
    )

    retrieval_service = RetrievalService()

    qdrant_repository = QdrantRepository(
        client=QdrantClient(
            url=app_settings.qdrant_url,
            api_key=app_settings.qdrant_api_key,
        ),
        collection_name=app_settings.collection_name,
        retrieval_service=retrieval_service,
    )

    reranker_service = RerankerService(
        model=CrossEncoder(app_settings.reranker_model),
        model_name=app_settings.reranker_model,
    )

    rag_service = RagService(
        chunker_service=chunker_service,
        embedder_service=embedder_service,
        qdrant_repository=qdrant_repository,
        reranker_service=reranker_service,
        retrieval_service=retrieval_service,
    )

    return AppContainer(
        rag_service=rag_service,
        app_settings=app_settings,
    )
