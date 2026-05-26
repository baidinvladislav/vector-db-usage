import { FormEvent, useCallback, useEffect, useState } from "react";
import { askRag, fetchHealth, initKnowledge } from "./api";
import type { AskRagResponse, HealthResponse } from "./types";

const EXAMPLE_QUERIES = [
  "Какая длина реки Нева?",
  "История Санкт-Петербурга",
  "Площадь бассейна Ладожского озера",
];

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<AskRagResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const loadHealth = useCallback(async () => {
    try {
      setHealth(await fetchHealth());
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    loadHealth();
  }, [loadHealth]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await askRag(trimmed, 5, 50);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleInitKnowledge(recreate: boolean) {
    setIndexing(true);
    setError(null);

    try {
      const response = await initKnowledge(recreate);
      setToast(
        `Indexed ${response.documents} docs → ${response.points} points in Qdrant`,
      );
      await loadHealth();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Indexing failed");
    } finally {
      setIndexing(false);
    }
  }

  return (
    <div className="page">
      <div className="glow glow-a" />
      <div className="glow glow-b" />

      <header className="header">
        <div>
          <p className="eyebrow">Wikipedia RAG</p>
          <h1>Ask your knowledge base</h1>
          <p className="subtitle">
            Hybrid search + cross-encoder rerank over indexed wiki articles.
          </p>
        </div>

        <div className="status-card">
          <span className={`dot ${health ? "online" : "offline"}`} />
          <div>
            <strong>{health?.app_name ?? "API offline"}</strong>
            <p>
              {health
                ? `${health.points_count.toLocaleString()} points · ${health.collection}`
                : "Start backend on :8000"}
            </p>
          </div>
        </div>
      </header>

      <section className="panel search-panel">
        <form onSubmit={handleSubmit} className="search-form">
          <label htmlFor="query">Your question</label>
          <div className="search-row">
            <input
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Например: Какая длина реки Нева?"
              disabled={loading}
            />
            <button type="submit" disabled={loading || !query.trim()}>
              {loading ? "Searching…" : "Ask"}
            </button>
          </div>
        </form>

        <div className="chips">
          {EXAMPLE_QUERIES.map((example) => (
            <button
              key={example}
              type="button"
              className="chip"
              onClick={() => setQuery(example)}
            >
              {example}
            </button>
          ))}
        </div>

        <div className="admin-row">
          <button
            type="button"
            className="ghost"
            disabled={indexing}
            onClick={() => handleInitKnowledge(false)}
          >
            {indexing ? "Indexing…" : "Index documents"}
          </button>
          <button
            type="button"
            className="ghost danger"
            disabled={indexing}
            onClick={() => handleInitKnowledge(true)}
          >
            Rebuild index
          </button>
        </div>
      </section>

      {error && <div className="alert error">{error}</div>}
      {toast && <div className="alert success">{toast}</div>}

      {result && (
        <section className="panel answer-panel">
          <p className="eyebrow">Best match</p>
          <h2>{result.query}</h2>
          <article className="answer">{result.answer}</article>
        </section>
      )}

      {result && result.sources.length > 0 && (
        <section className="sources">
          <h3>Sources ({result.sources.length})</h3>
          <div className="source-grid">
            {result.sources.map((source, index) => (
              <article key={`${source.doc_id}-${source.chunk_index}`} className="source-card">
                <div className="source-head">
                  <span className="rank">#{index + 1}</span>
                  <div>
                    <strong>{source.doc_id}</strong>
                    <p>{source.title}</p>
                  </div>
                </div>
                <p className="source-text">{source.text}</p>
                <div className="scores">
                  <span>rerank {source.rerank_score.toFixed(3)}</span>
                  <span>vector {source.vector_score.toFixed(3)}</span>
                  <span>chunk {source.chunk_index}</span>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
