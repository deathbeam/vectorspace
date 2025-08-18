from pygments.lexers import get_lexer_for_filename
from tree_sitter_languages import get_parser
from typing import List, Dict, Any

# List of semantic node types (from your Lua resources.lua)
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
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(
            {
                "body": chunk_text,
                "metadata": {
                    "chunk_start": start,
                    "chunk_end": end,
                    "type": "text",
                },
            }
        )
        start += chunk_size - overlap
    return chunks


def semantic_chunk_nodes(text: str, language: str) -> List[Dict[str, Any]]:
    try:
        parser = get_parser(language)
    except Exception:
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
