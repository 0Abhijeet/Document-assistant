from src.database import SessionLocal
from src.models import Chunk
from src.ingest import get_embeddings


def retrieve_relevant_chunks(query: str, k: int = 3):
    """Embed the query, return the k nearest chunks by cosine distance, across all ingested docs."""
    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)

    db = SessionLocal()
    try:
        results = (
            db.query(Chunk)
            .order_by(Chunk.embedding.cosine_distance(query_vector))
            .limit(k)
            .all()
        )
        # Detach plain dicts so callers don't hold a session open after db.close()
        return [
            {
                "content": r.content,
                "page": r.page_number,
                "document_id": str(r.document_id),
            }
            for r in results
        ]
    finally:
        db.close()


if __name__ == "__main__":
    query = input("Enter your question: ")
    chunks = retrieve_relevant_chunks(query)
    print("\nTop relevant chunks:\n")
    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i} (page {c['page']}):")
        print(c["content"])
        print("-" * 50)
