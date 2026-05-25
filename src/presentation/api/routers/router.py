from typing import Any

from fastapi import APIRouter, status, Depends

from src.presentation.dependencies import get_rag_service, get_app_settings
from src.services.rag_service import RagService
from src.shared.settings import AppSettings

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
def get_app_info(app_settings: AppSettings = Depends(get_app_settings)) -> dict[str, str]:
    return {
        "app_name": app_settings.app_name,
        "version": app_settings.app_version,
        "llm_model": app_settings.llm_model,
        "embedding_model": app_settings.embedder_model,
    }


@router.post("/init-knowledge", status_code=status.HTTP_201_CREATED)
def create_knowledge_base(rag_service: RagService = Depends(get_rag_service)) -> dict[str, str]:
    rag_service.init_knowledge_base()
    return {}


@router.get("/ask-rag", status_code=status.HTTP_200_OK)
def get_query(query: str, rag_service: RagService = Depends(get_rag_service)) -> dict[str, Any]:
    response = rag_service.get_answer(query)
    return {}
