"""Plain-text recursive character-based splitter (LangChain-style fallback).

Mirrors packages/core/src/splitter/langchain-splitter.ts.
"""

from __future__ import annotations

from codecontext.core.types import CodeChunk


class Splitter:
    """Abstract base for code splitters."""

    def __init__(self, chunk_size: int = 2500, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, code: str, language: str, file_path: str = "") -> list[CodeChunk]:
        raise NotImplementedError


class TextSplitter(Splitter):
    """Simple recursive character-based splitter (LangChain-style)."""

    _SEPARATORS = ["\n\n", "\n", " ", ""]

    def split(self, code: str, language: str, file_path: str = "") -> list[CodeChunk]:
        raw_chunks = self._recursive_split(code, self._SEPARATORS)
        return self._to_code_chunks(raw_chunks, code, language, file_path)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0] if separators else ""
        rest = separators[1:] if len(separators) > 1 else []

        if not sep:
            chunks: list[str] = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i : i + self.chunk_size])
            return chunks

        parts = text.split(sep)
        chunks = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) > self.chunk_size and current:
                chunks.append(current)
                overlap_start = max(0, len(current) - self.chunk_overlap)
                current = current[overlap_start:] + sep + part
            else:
                current = candidate

        if current:
            if len(current) > self.chunk_size and rest:
                chunks.extend(self._recursive_split(current, rest))
            else:
                chunks.append(current)
        return chunks

    def _to_code_chunks(
        self, raw: list[str], full_code: str, language: str, file_path: str
    ) -> list[CodeChunk]:
        results: list[CodeChunk] = []
        search_from = 0
        for text in raw:
            idx = full_code.find(text, search_from)
            if idx == -1:
                idx = search_from
            start_line = full_code[:idx].count("\n") + 1
            end_line = start_line + text.count("\n")
            results.append(
                CodeChunk(
                    content=text,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    file_path=file_path,
                )
            )
            search_from = idx + len(text)
        return results
