import type { AskRagResponse, HealthResponse, InitKnowledgeResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function initKnowledge(recreate = false): Promise<InitKnowledgeResponse> {
  return request<InitKnowledgeResponse>("/init-knowledge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ recreate }),
  });
}

export function askRag(
  query: string,
  limit = 5,
  fetchK?: number,
): Promise<AskRagResponse> {
  const params = new URLSearchParams({ query, limit: String(limit) });
  if (fetchK) {
    params.set("fetch_k", String(fetchK));
  }
  return request<AskRagResponse>(`/ask-rag?${params.toString()}`);
}
