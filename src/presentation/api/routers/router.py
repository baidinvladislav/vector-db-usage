from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.presentation.api.schemas import (
    AskRagResponse,
    HealthResponse,
    InitKnowledgeRequest,
    InitKnowledgeResponse,
    SourceChunkResponse,
)
from src.presentation.dependencies import get_app_settings, get_rag_service
from src.services.rag_service import RagService
from src.shared.settings import AppSettings

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK, response_model=HealthResponse)
def get_app_info(
    app_settings: AppSettings = Depends(get_app_settings),
    rag_service: RagService = Depends(get_rag_service),
) -> HealthResponse:
    return HealthResponse(
        app_name=app_settings.app_name,
        version=app_settings.app_version,
        llm_model=app_settings.llm_model or "none",
        embedding_model=app_settings.embedder_model,
        collection=app_settings.collection_name,
        points_count=rag_service.qdrant_repository.points_count(),
    )


@router.post(
    "/init-knowledge",
    status_code=status.HTTP_201_CREATED,
    response_model=InitKnowledgeResponse,
)
def create_knowledge_base(
    body: InitKnowledgeRequest,
    rag_service: RagService = Depends(get_rag_service),
) -> InitKnowledgeResponse:
    try:
        result = rag_service.init_knowledge_base(recreate=body.recreate)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return InitKnowledgeResponse(
        documents=result.documents,
        chunks=result.chunks,
        points=result.points,
        message="Knowledge base initialized successfully.",
    )


@router.get("/ask-rag", status_code=status.HTTP_200_OK, response_model=AskRagResponse)
def ask_rag(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=20),
    fetch_k: int | None = Query(default=None, ge=1, le=200),
    doc_id: str | None = Query(default=None),
    rag_service: RagService = Depends(get_rag_service),
) -> AskRagResponse:
    try:
        result = rag_service.get_answer(
            query,
            limit=limit,
            fetch_k=fetch_k,
            doc_id=doc_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return AskRagResponse(
        query=result.query,
        answer=result.answer,
        sources=[
            SourceChunkResponse(
                doc_id=source.doc_id,
                chunk_index=source.chunk_index,
                title=source.title,
                text=source.text,
                rerank_score=source.rerank_score,
                vector_score=source.vector_score,
            )
            for source in result.sources
        ],
    )
