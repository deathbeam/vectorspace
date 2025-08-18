import sys
from mcp.server.fastmcp import FastMCP

from .logger import setup_logger
from .core import VectorspaceCore


mcp = FastMCP("vectorspace")
vectorspace_core = VectorspaceCore()


@mcp.tool()
def vectorspace_search(query: str, directory: str, max_results: int = 10) -> str:
    """
    Semantic search through workspace using vector embeddings.

    Args:
        query: The search query to find relevant code snippets
        directory: Directory to search in (defaults to current working directory)
        max_results: Maximum number of results to return (1-50, default 10)
    """

    max_results = min(50, max(1, max_results))

    try:
        results = vectorspace_core.query(directory, query, max_results)

        if not results:
            return "No relevant files found for your query."

        content_parts = []
        for i, result in enumerate(results, 1):
            filename = result["filename"]
            start_line = result.get("start_row", "?")
            end_line = result.get("end_row", "?")
            start_col = result.get("start_col", "?")
            end_col = result.get("end_col", "?")
            score = result.get("score", 0.0)
            content_parts.append(
                f"\n## {i}. {filename} (score: {score:.3f}) [lines {start_line}-{end_line}, cols {start_col}-{end_col}]"
            )
            content_parts.append(result["content"])

        return "\n".join(content_parts)

    except Exception as e:
        return f"Error performing search: {str(e)}"


def main():
    setup_logger()
    mcp.run(transport=str(sys.argv[1]) if len(sys.argv) > 1 else "stdio")  # type: ignore


if __name__ == "__main__":
    main()
