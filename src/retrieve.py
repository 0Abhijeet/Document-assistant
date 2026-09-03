import asyncio

from sqlalchemy import select

from src.database import AsyncSessionLocal
from src.models import Chunk
from src.ingest import get_embeddings


async def retrieve_relevant_chunks(query: str, k: int = 3):
    """Embed the query, return the k nearest chunks by cosine distance, across all ingested docs."""
    embeddings = get_embeddings()
    query_vector = await asyncio.to_thread(embeddings.embed_query, query)

    async with AsyncSessionLocal() as db:
        stmt = (
            select(Chunk)
            .order_by(Chunk.embedding.cosine_distance(query_vector))
            .limit(k)
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "content": r.content,
                "page": r.page_number,
                "document_id": str(r.document_id),
            }
            for r in rows
        ]


if __name__ == "__main__":
    async def _main():
        query = input("Enter your question: ")
        chunks = await retrieve_relevant_chunks(query)
        print("\nTop relevant chunks:\n")
        for i, c in enumerate(chunks, 1):
            print(f"Chunk {i} (page {c['page']}):")
            print(c["content"])
            print("-" * 50)

    asyncio.run(_main())