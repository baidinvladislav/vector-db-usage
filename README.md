# vector-db-usage

RAG knowledge base over text documents in `files/`, stored in [Qdrant](https://qdrant.tech/).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
```

## Prepare knowledge base

Indexes all `.txt` files: splits into overlapping chunks, embeds with a multilingual model, upserts into Qdrant with metadata (`doc_id`, `title`, `source_path`, `text`).

```bash
python scripts/prepare_knowledge_base.py
```

Quick test on a few documents:

```bash
python scripts/prepare_knowledge_base.py --limit-docs 5
```

Recreate the collection from scratch:

```bash
python scripts/prepare_knowledge_base.py --recreate
```

## Query (search)

After indexing, search by meaning (works in Russian and English):

```bash
python scripts/query_knowledge_base.py "длина реки Нева"
python scripts/query_knowledge_base.py "история Санкт-Петербурга" -k 3
python scripts/query_knowledge_base.py "пороги" --doc-id 101
```

Results show similarity score, document id, title, and matching text chunk.

Qdrant UI (optional): open http://localhost:6333/dashboard to inspect the collection.

## Configuration

See `.env.example`. Main variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP API |
| `QDRANT_COLLECTION` | `knowledge_base` | Collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | FastEmbed model (multilingual, good for Russian) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Text splitting |
| `DOCS_DIR` | `files` | Folder with `.txt` documents |
