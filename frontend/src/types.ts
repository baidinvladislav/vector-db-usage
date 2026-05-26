export interface HealthResponse {
  app_name: string;
  version: string;
  llm_model: string;
  embedding_model: string;
  collection: string;
  points_count: number;
}

export interface InitKnowledgeResponse {
  documents: number;
  chunks: number;
  points: number;
  message: string;
}

export interface SourceChunk {
  doc_id: string;
  chunk_index: number;
  title: string;
  text: string;
  rerank_score: number;
  vector_score: number;
}

export interface AskRagResponse {
  query: string;
  answer: string;
  sources: SourceChunk[];
}
