import sys
from mcp.server.fastmcp import FastMCP
from .core import VectorspaceCore


mcp = FastMCP("vectorspace")
vectorspace_core = VectorspaceCore()


@mcp.tool()
def vectorspace_search(
    query: str,
    directory: str,
    max_results: int = 10
) -> str:
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

        # Format results
        content_parts = []
        content_parts.append(f"Found {len(results)} relevant files for query: '{query}'\n")

        for i, result in enumerate(results, 1):
            content_parts.append(f"\n## {i}. {result['filename']} (score: {result['score']:.3f})")
            content_parts.append("```")
            # Truncate content if too long
            content = result['content'][:1000] + "..." if len(result['content']) > 1000 else result['content']
            content_parts.append(content)
            content_parts.append("```")

        return "\n".join(content_parts)

    except Exception as e:
        return f"Error performing search: {str(e)}"


def main():
    mcp.run(transport=str(sys.argv[1]) if len(sys.argv) > 1 else 'stdio') # type: ignore


if __name__ == "__main__":
    main()
