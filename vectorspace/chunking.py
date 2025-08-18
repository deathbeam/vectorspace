import logging
from pygments.lexers import get_lexer_for_filename
from tree_sitter_language_pack import get_parser
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

OUTLINE_TYPES = [
    "local_function",
    "function_item",
    "arrow_function",
    "function_definition",
    "function_declaration",
    "method_definition",
    "method_declaration",
    "proc_declaration",
    "template_declaration",
    "macro_declaration",
    "constructor_declaration",
    "class_definition",
    "class_declaration",
    "interface_definition",
    "interface_declaration",
    "record_declaration",
    "type_alias_declaration",
    "atx_heading",
]


def get_language_from_filename(filename: str) -> str:
    try:
        lexer = get_lexer_for_filename(filename)
        return lexer.name.lower()
    except Exception:
        return "text"


def chunk_with_overlap(text: str, chunk_size=256, overlap=32) -> List[Dict[str, Any]]:
    positions = []
    line, col = 0, 0
    for c in text:
        positions.append((line, col))
        if c == "\n":
            line += 1
            col = 0
        else:
            col += 1

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        start_row, start_col = positions[start] if start < len(positions) else (0, 0)
        end_row, end_col = positions[end - 1] if end - 1 < len(positions) else (line, col)
        chunks.append(
            {
                "body": chunk_text,
                "metadata": {
                    "type": "text",
                    "start_row": start_row,
                    "end_row": end_row,
                    "start_col": start_col,
                    "end_col": end_col,
                },
            }
        )
        start += chunk_size - overlap
    return chunks


def semantic_chunk_nodes(text: str, language: str) -> List[Dict[str, Any]]:
    try:
        parser = get_parser(language)  # type: ignore
    except Exception:
        logger.warning(f"Could not get parser for language '{language}', falling back to text chunks.")
        return []

    tree = parser.parse(bytes(text, "utf8"))
    root = tree.root_node
    chunks = []

    def visit(node):
        if node.type in OUTLINE_TYPES and (node.end_byte - node.start_byte) > 32:
            chunk_text = text[node.start_byte : node.end_byte]
            chunks.append(
                {
                    "body": chunk_text,
                    "metadata": {
                        "type": node.type,
                        "start_row": node.start_point[0],
                        "start_col": node.start_point[1],
                        "end_row": node.end_point[0],
                        "end_col": node.end_point[1],
                    },
                }
            )
        for child in node.children:
            visit(child)

    visit(root)
    return chunks


def chunk_file(filename: str, text: str, chunk_size=256, overlap=32) -> List[Dict[str, Any]]:
    language = get_language_from_filename(filename)
    semantic_chunks = semantic_chunk_nodes(text, language)
    out_chunks = []
    if semantic_chunks:
        for chunk in semantic_chunks:
            if len(chunk["body"]) > chunk_size:
                out_chunks.extend(chunk_with_overlap(chunk["body"], chunk_size, overlap))
            else:
                out_chunks.append(chunk)
        return out_chunks
    else:
        return chunk_with_overlap(text, chunk_size, overlap)
