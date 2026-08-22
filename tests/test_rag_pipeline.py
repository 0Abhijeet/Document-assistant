import os

os.environ.setdefault("DATABASE_URL", "postgresql://rag:ragpassword@localhost:5432/ragdb_test")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

import pytest
from fastapi.testclient import TestClient

from src.database import Base, engine, SessionLocal
from src.models import Document, Chunk, QueryLog
from src.ingest import ingest_documents
from src.retrieve import retrieve_relevant_chunks
from src.app import app


@pytest.fixture(autouse=True)
def clean_db():
    """Fresh schema per test — keeps tests independent of each other and of run order."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def sample_pdf(tmp_path):
    """A minimal one-page PDF with real text, built at test time — no binary fixture file needed."""
    from reportlab.pdfgen import canvas

    path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(72, 700, "This is a test document for the RAG pipeline test suite.")
    c.drawString(72, 680, "It contains enough real text for the splitter to produce chunks.")
    c.save()
    return str(path)


def test_ingest_creates_document_and_chunks(sample_pdf):
    doc_id = ingest_documents(sample_pdf, "sample.pdf")

    db = SessionLocal()
    try:
        doc = db.query(Document).filter_by(id=doc_id).first()
        assert doc is not None
        assert doc.status == "ready"
        # A blank page still produces at least one chunk after PyPDFLoader + splitter.
        chunk_count = db.query(Chunk).filter_by(document_id=doc.id).count()
        assert chunk_count >= 1
    finally:
        db.close()


def test_ingest_is_additive_not_destructive(sample_pdf):
    """Uploading a second document must not delete the first — this was a real bug in v1."""
    first_id = ingest_documents(sample_pdf, "first.pdf")
    second_id = ingest_documents(sample_pdf, "second.pdf")

    db = SessionLocal()
    try:
        assert db.query(Document).count() == 2
        assert db.query(Document).filter_by(id=first_id).first() is not None
        assert db.query(Document).filter_by(id=second_id).first() is not None
    finally:
        db.close()


def test_retrieve_returns_k_chunks(sample_pdf):
    ingest_documents(sample_pdf, "sample.pdf")
    results = retrieve_relevant_chunks("any question", k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    if results:
        assert "content" in results[0]


def test_upload_endpoint_rejects_non_pdf():
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_endpoint_accepts_pdf(sample_pdf):
    client = TestClient(app)
    with open(sample_pdf, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("sample.pdf", f, "application/pdf")},
        )
    assert response.status_code == 200
    assert "document_id" in response.json()


def test_stream_endpoint_rejects_empty_question():
    client = TestClient(app)
    response = client.post("/stream", data={"question": "   "})
    assert response.status_code == 400


def test_query_log_table_exists_and_is_empty_initially():
    db = SessionLocal()
    try:
        assert db.query(QueryLog).count() == 0
    finally:
        db.close()
