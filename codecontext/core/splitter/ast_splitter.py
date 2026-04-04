"""AST-aware code splitter using Tree-Sitter.

Mirrors packages/core/src/splitter/ast-splitter.ts.
"""

from __future__ import annotations

import logging

from codecontext.core.types import CodeChunk
from .text_splitter import Splitter, TextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language → tree-sitter grammar mapping (lazy-loaded)
# ---------------------------------------------------------------------------

_SPLITTABLE_NODE_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition", "class_definition", "decorated_definition",
        "async_function_definition",
    },
    "javascript": {
        "function_declaration", "class_declaration", "method_definition",
        "arrow_function", "export_statement",
    },
    "typescript": {
        "function_declaration", "class_declaration", "method_definition",
        "arrow_function", "export_statement",
        "interface_declaration", "type_alias_declaration",
    },
    "java": {
        "class_declaration", "method_declaration", "constructor_declaration",
        "interface_declaration", "enum_declaration",
    },
    "go": {
        "function_declaration", "method_declaration", "type_declaration",
    },
    "rust": {
        "function_item", "impl_item", "struct_item", "enum_item",
        "trait_item", "mod_item",
    },
    "c": {
        "function_definition", "struct_specifier", "enum_specifier",
    },
    "cpp": {
        "function_definition", "class_specifier", "struct_specifier",
        "namespace_definition", "template_declaration",
    },
}


def _get_ts_language(lang: str):
    """Return the tree-sitter Language object for *lang*, or None."""
    try:
        import tree_sitter
        if lang == "python":
            import tree_sitter_python as tslang
        elif lang == "javascript":
            import tree_sitter_javascript as tslang
        elif lang == "typescript":
            import tree_sitter_typescript as tslang
            return tree_sitter.Language(tslang.language_typescript())
        elif lang == "java":
            import tree_sitter_java as tslang
        elif lang == "go":
            import tree_sitter_go as tslang
        elif lang == "rust":
            import tree_sitter_rust as tslang
        elif lang in ("c", "cpp"):
            if lang == "c":
                import tree_sitter_c as tslang
            else:
                import tree_sitter_cpp as tslang
        else:
            return None
        return tree_sitter.Language(tslang.language())
    except Exception:
        return None


class AstSplitter(Splitter):
    """
    Tree-sitter based splitter that extracts logical code units
    (functions, classes, methods, …).  Falls back to TextSplitter
    for unsupported languages.
    """

    def __init__(self, chunk_size: int = 2500, chunk_overlap: int = 300):
        super().__init__(chunk_size, chunk_overlap)
        self._fallback = TextSplitter(chunk_size, chunk_overlap)
        self._parsers: dict[str, object] = {}

    def split(self, code: str, language: str, file_path: str = "") -> list[CodeChunk]:
        ts_lang = _get_ts_language(language)
        if ts_lang is None:
            return self._fallback.split(code, language, file_path)

        try:
            import tree_sitter
            parser = tree_sitter.Parser(ts_lang)
            tree = parser.parse(code.encode("utf-8"))
        except Exception as exc:
            logger.warning("tree-sitter parse failed for %s: %s — using fallback", file_path, exc)
            return self._fallback.split(code, language, file_path)

        splittable = _SPLITTABLE_NODE_TYPES.get(language, set())
        raw_chunks = self._extract_nodes(tree.root_node, code, splittable)

        if not raw_chunks:
            return self._fallback.split(code, language, file_path)

        # Split oversized chunks using TS-matching line-by-line strategy
        refined: list[CodeChunk] = []
        for chunk in raw_chunks:
            if len(chunk.content) > self.chunk_size:
                refined.extend(
                    self._split_large_chunk(chunk.content, chunk.start_line, language, file_path)
                )
            else:
                refined.append(chunk)

        for c in refined:
            c.language = language
            c.file_path = file_path

        return refined if refined else self._fallback.split(code, language, file_path)

    def _extract_nodes(self, root, full_code: str, splittable: set[str]) -> list[CodeChunk]:
        """Extract splittable AST nodes (functions, classes, etc.).

        Mirrors TS ast-splitter: only splittable nodes become chunks;
        gap text between nodes is discarded (matches TS behavior).
        """
        chunks: list[CodeChunk] = []

        def walk(node):
            if node.type in splittable:
                text = full_code[node.start_byte : node.end_byte]
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                chunks.append(CodeChunk(content=text, start_line=start_line, end_line=end_line))
            else:
                for child in node.children:
                    walk(child)

        walk(root)
        return chunks

    def _split_large_chunk(
        self, content: str, base_line: int, language: str, file_path: str
    ) -> list[CodeChunk]:
        """Split oversized chunk line-by-line with overlap (matches TS splitLargeChunk)."""
        lines = content.split("\n")
        chunks: list[CodeChunk] = []
        current_lines: list[str] = []
        current_len = 0

        for i, line in enumerate(lines):
            line_len = len(line) + 1  # +1 for newline
            if current_len + line_len > self.chunk_size and current_lines:
                chunk_text = "\n".join(current_lines)
                start = base_line + (i - len(current_lines))
                end = base_line + i - 1
                chunks.append(CodeChunk(
                    content=chunk_text, start_line=start, end_line=end,
                    language=language, file_path=file_path,
                ))
                # Overlap: keep last N characters worth of lines
                overlap_lines: list[str] = []
                overlap_len = 0
                for ol in reversed(current_lines):
                    if overlap_len + len(ol) + 1 > self.chunk_overlap:
                        break
                    overlap_lines.insert(0, ol)
                    overlap_len += len(ol) + 1
                current_lines = overlap_lines
                current_len = overlap_len

            current_lines.append(line)
            current_len += line_len

        if current_lines:
            chunk_text = "\n".join(current_lines)
            start = base_line + (len(lines) - len(current_lines))
            end = base_line + len(lines) - 1
            chunks.append(CodeChunk(
                content=chunk_text, start_line=start, end_line=end,
                language=language, file_path=file_path,
            ))

        return chunks
