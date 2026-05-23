from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    source_path: str
    title: str
    chunk_index: int
    text: str


def _first_line_title(text: str, fallback: str) -> str:
    """ Создаёт заголовок файла """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:500]

    return fallback


def load_text_files(docs_dir: Path, limit: int = 3) -> list[tuple[str, Path, str]]:
    """ Загружает текста документов """
    entries: list[tuple[str, Path, str]] = []

    for path in sorted(docs_dir.glob("*.txt")):
        if len(entries) >= limit:
            return entries

        doc_id = path.stem
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            entries.append((doc_id, path, text))

    return entries


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """ Режет текст на чанки """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    normalized = "\n".join(line.rstrip() for line in text.splitlines())
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = start + chunk_size
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break

        start = end - chunk_overlap

    return chunks


def build_chunks(docs_dir: Path, *, chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    """ Формируе чанки для векторной БД """
    documents = load_text_files(docs_dir)

    result: list[DocumentChunk] = []
    for doc_id, path, text in documents:
        title = _first_line_title(text, fallback=doc_id)
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for index, chunk in enumerate(chunks):
            result.append(
                DocumentChunk(
                    doc_id=doc_id,
                    source_path=str(path.relative_to(docs_dir.parent)),
                    title=title,
                    chunk_index=index,
                    text=chunk,
                )
            )

    return result
