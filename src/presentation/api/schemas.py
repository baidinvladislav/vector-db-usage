from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    app_name: str
    version: str
    llm_model: str
    embedding_model: str
    collection: str
    points_count: int


class InitKnowledgeRequest(BaseModel):
    recreate: bool = False


class InitKnowledgeResponse(BaseModel):
    documents: int
    chunks: int
    points: int
    message: str


class SourceChunkResponse(BaseModel):
    doc_id: str
    chunk_index: int
    title: str
    text: str
    rerank_score: float
    vector_score: float


class AskRagResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceChunkResponse]


class AskRagQueryParams(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    fetch_k: int | None = Field(default=None, ge=1, le=200)
    doc_id: str | None = None
