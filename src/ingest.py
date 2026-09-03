import asyncio

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding

from src.database import AsyncSessionLocal
from src.models import Document, Chunk


class _FastEmbedWrapper:
    """Wraps fastembed to match the .embed_documents()/.embed_query() interface
    the rest of this codebase expects — keeps call sites unchanged."""

    def __init__(self):
        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text):
        return next(iter(self._model.embed([text]))).tolist()


_embeddings = None


def get_embeddings():
    """Lazy singleton — loading the model is expensive, don't reload it per call.
    Sync on purpose: model loading happens once at process warm-up, not per-request."""
    global _embeddings
    if _embeddings is None:
        _embeddings = _FastEmbedWrapper()
    return _embeddings


async def ingest_documents(file_path: str, filename: str) -> str:
    """
    Ingest a PDF: split into chunks, embed, store in Postgres.
    Multi-document — does NOT wipe previous documents. Returns the new document's id.
    """
    loader = PyPDFLoader(file_path)
    # .aload() offloads the blocking parse via run_in_executor internally
    # (verified against langchain-core source, not assumed) -- no manual
    # to_thread needed here.
    documents = await loader.aload()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # split_documents is pure-Python CPU work, no async equivalent exists ->
    # offload manually.
    chunks = await asyncio.to_thread(splitter.split_documents, documents)

    embeddings = get_embeddings()
    texts = [c.page_content for c in chunks]
    # ONNX inference, CPU-bound, no async API -> offload manually.
    vectors = await asyncio.to_thread(embeddings.embed_documents, texts)

    async with AsyncSessionLocal() as db:
        try:
            doc = Document(filename=filename, status="processing")
            db.add(doc)
            await db.flush()  # assigns doc.id without committing yet

            for chunk, vector in zip(chunks, vectors):
                db.add(Chunk(
                    document_id=doc.id,
                    page_number=chunk.metadata.get("page"),
                    content=chunk.page_content,
                    embedding=vector,
                ))

            doc.status = "ready"
            await db.commit()
            return str(doc.id)
        except Exception:
            await db.rollback()
            raise


if __name__ == "__main__":
    async def _main():
        doc_id = await ingest_documents("data/docs/AR_CITIZEN_Paper.pdf", "AR_CITIZEN_Paper.pdf")
        print(f"Ingested document {doc_id}")

    asyncio.run(_main())