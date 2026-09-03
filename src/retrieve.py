import asyncio

from sqlalchemy import select

from src.database import AsyncSessionLocal
from src.models import Chunk, Document
from src.ingest import get_embeddings


async def retrieve_relevant_chunks(query: str, k: int = 3, doc_id: str | None = None):
    """
    Embed the query, return the k nearest chunks by cosine distance.
    Optionally scoped to a single document_id (used by the MCP tool).
    Includes filename and a 0-1 similarity score per chunk (1 - cosine distance).
    """
    embeddings = get_embeddings()
    # fastembed has no async API -- it's ONNX inference, CPU-bound, no I/O to
    # await on. asyncio.to_thread offloads it so it doesn't freeze the event
    # loop for every other in-flight request (see build log for the measured
    # before/after).
    query_vector = await asyncio.to_thread(embeddings.embed_query, query)

    async with AsyncSessionLocal() as db:
        distance = Chunk.embedding.cosine_distance(query_vector)
        stmt = (
            select(Chunk, Document.filename, distance.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .order_by(distance)
            .limit(k)
        )
        if doc_id is not None:
            stmt = stmt.where(Chunk.document_id == doc_id)

        result = await db.execute(stmt)
        rows = result.all()
        return [
            {
                "content": chunk.content,
                "page": chunk.page_number,
                "document_id": str(chunk.document_id),
                "filename": filename,
                "similarity": round(1 - dist, 4),
            }
            for chunk, filename, dist in rows
        ]


if __name__ == "__main__":
    async def _main():
        query = input("Enter your question: ")
        chunks = await retrieve_relevant_chunks(query)
        print("\nTop relevant chunks:\n")
        for i, c in enumerate(chunks, 1):
            print(f"Chunk {i} (page {c['page']}, {c['filename']}, sim={c['similarity']}):")
            print(c["content"])
            print("-" * 50)

    asyncio.run(_main())