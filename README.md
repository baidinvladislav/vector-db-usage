# Описание сервиса
RAG базе знаний из документов, хранящихся в директории `files/`, которые были загружены в Qdrant.

## Установка
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
```

## Подготовка базы знаний
1. Находит все`.txt` файлы в директории `files/`.
2. Делит текст на чанки.
2. Векторизует в эмбеддинги с помощью модели эмбеддера.
3. Вставляет в Qdrant с метаданными: (`doc_id`, `title`, `source_path`, `text`).

```bash
python scripts/prepare_knowledge_base.py
```

Быстрый тест на нескольких документах:
```bash
python scripts/prepare_knowledge_base.py --limit-docs 5
```

Пересоздать коллекцию эмбеддингов заново:
```bash
python scripts/prepare_knowledge_base.py --recreate
```

## Запрос (поиск)
После индексации документов можем найти релевантный ответ в базе знаний:

```bash
python scripts/query_knowledge_base.py --query "длина реки Нева"
python scripts/query_knowledge_base.py --query "история Санкт-Петербурга" -k 3
python scripts/query_knowledge_base.py --query "пороги" --doc-id 101
```

Поиск всегда использует **reranker** (cross-encoder): сначала берётся `RERANK_FETCH_K` кандидатов из Qdrant, затем переранжирование моделью `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, в ответе — `rerank` и `vector` score.

```bash
python scripts/query_knowledge_base.py --query "длина реки Нева" --fetch-k 80 -k 5
```

Результат показывает рейтинг, идентификатор документа и соответствующий чанк документа.

Веб-интерфейс базы знаний Qdrant находится по адресу http://localhost:6333/dashboard.

## Конфигурация
Конфиги находятся в файле `.env.example`.

| Variable                       | Default                                                       | Description                     |
|--------------------------------|---------------------------------------------------------------|---------------------------------|
| `QDRANT_URL`                   | `http://localhost:6333`                                       | Адрес Qdrant                    |
| `QDRANT_COLLECTION`            | `knowledge_base`                                              | Название коллекции              |
| `EMBEDDING_MODEL`              | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Модель эмбеддера                |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120`                                                 | Параметры чанкирования          |
| `DOCS_DIR`                     | `files`                                                       | Директория с `.txt` документами |
| `RERANKER_MODEL`               | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`                  | Модель reranker                 |
| `RERANK_FETCH_K`               | `20`                                                          | Кандидатов до rerank            |
