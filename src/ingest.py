from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding

from src.database import SessionLocal
from src.models import Document, Chunk


class _FastEmbedWrapper:
    """Wraps fastembed to match the .embed_documents()/.embed_query() interface
    the rest of this codebase expects — keeps ingest.py/retrieve.py unchanged."""

    def __init__(self):
        # BAAI/bge-small-en-v1.5 outputs 384-dim vectors — matches EMBEDDING_DIM
        # in models.py exactly, so no schema migration needed for this swap.
        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text):
        return next(iter(self._model.embed([text]))).tolist()


_embeddings = None


def get_embeddings():
    """Lazy singleton — loading the model is expensive, don't reload it per call."""
    global _embeddings
    if _embeddings is None:
        _embeddings = _FastEmbedWrapper()
    return _embeddings


def ingest_documents(file_path: str, filename: str) -> str:
    """
    Ingest a PDF: split into chunks, embed, store in Postgres.
    Multi-document — does NOT wipe previous documents. Returns the new document's id.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    db = SessionLocal()
    try:
        doc = Document(filename=filename, status="processing")
        db.add(doc)
        db.flush()  # assigns doc.id without committing yet

        for chunk, vector in zip(chunks, vectors):
            db.add(Chunk(
                document_id=doc.id,
                page_number=chunk.metadata.get("page"),
                content=chunk.page_content,
                embedding=vector,
            ))

        doc.status = "ready"
        db.commit()
        return str(doc.id)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    doc_id = ingest_documents("data/docs/AR_CITIZEN_Paper.pdf", "AR_CITIZEN_Paper.pdf")
    print(f"Ingested document {doc_id}")
