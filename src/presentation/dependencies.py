from src.services.rag_service import RagService
from src.shared.container import init_app_container
from src.shared.settings import AppSettings

container = init_app_container()


def get_rag_service() -> RagService:
    return container.rag_service


def get_app_settings() -> AppSettings:
    return container.app_settings
