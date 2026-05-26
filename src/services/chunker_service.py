import re
from dataclasses import dataclass
from pathlib import Path

from src.domain.models import DocumentChunk

_WIKI_SECTION = re.compile(r"\n(?==+ [^=].*? ==\n)")


@dataclass
class ChunkerService:
    docs_dir: Path
    chunk_size: int
    chunk_overlap: int
    docs_limit: int | None = None

    def build_chunks(self) -> list[DocumentChunk]:
        result: list[DocumentChunk] = []
        for doc_id, path, text in self.load_text_files(self.docs_dir, limit=self.docs_limit):
            title = self._first_line_title(text, fallback=doc_id)
            for index, chunk in enumerate(
                self._chunk_for_embedding(
                    text,
                    title=title,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            ):
                result.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        source_path=str(path.relative_to(self.docs_dir.parent)),
                        title=title,
                        chunk_index=index,
                        text=chunk,
                    )
                )
        return result

    def _chunk_for_embedding(
        self,
        text: str,
        *,
        title: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        sections = self._split_wiki_sections(text)
        raw_chunks: list[str] = []
        for section in sections:
            raw_chunks.extend(
                self.chunk_text(section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            )
        if not raw_chunks:
            raw_chunks = self.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        prefixed: list[str] = []
        for chunk in raw_chunks:
            body = chunk if chunk.startswith(title) else f"{title}\n\n{chunk}"
            prefixed.append(body)
        return prefixed

    def _split_wiki_sections(self, text: str) -> list[str]:
        parts = _WIKI_SECTION.split(text)
        if len(parts) <= 1:
            return [text]
        return [part.strip() for part in parts if part.strip()]

    def chunk_text(
        self,
        text: str,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
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

    def load_text_files(
        self,
        docs_dir: Path,
        *,
        limit: int | None = None,
    ) -> list[tuple[str, Path, str]]:
        entries: list[tuple[str, Path, str]] = []
        for path in sorted(docs_dir.glob("*.txt")):
            doc_id = path.stem
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                entries.append((doc_id, path, text))
            if limit is not None and len(entries) >= limit:
                break
        return entries

    def _first_line_title(self, text: str, fallback: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:500]
        return fallback
