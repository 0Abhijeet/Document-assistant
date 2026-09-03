"""
Exposes the existing pgvector retrieval service as an MCP tool over stdio.
This file is standalone -- it imports the existing service layer directly
and does NOT touch app.py or any existing FastAPI route.

Run directly (e.g. from an MCP client config) with:
    python mcp_server.py
"""
from mcp.server.mcpserver import MCPServer

from src.retrieve import retrieve_relevant_chunks

mcp = MCPServer("rag-document-assistant")


@mcp.tool()
async def search_documents(
    query: str,
    top_k: int = 5,
    doc_id: str | None = None,
) -> list[dict]:
    """
    Search ingested documents for chunks relevant to a query.

    Args:
        query: Natural-language question or search text.
        top_k: Maximum number of chunks to return (default 5).
        doc_id: Optional document ID to restrict the search to a single document.

    Returns:
        A list of matching chunks, each with content, page, document_id,
        filename, and similarity (0-1, higher is more relevant), ordered
        by relevance (most relevant first).
    """
    return await retrieve_relevant_chunks(query, k=top_k, doc_id=doc_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")