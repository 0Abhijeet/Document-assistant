import os

os.environ.setdefault("DATABASE_URL", "postgresql://rag:ragpassword@localhost:5432/ragdb_test")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

import pytest
import pytest_asyncio
import httpx
from sqlalchemy import text, select

from src.database import Base, engine, AsyncSessionLocal
from src.models import Document, Chunk, QueryLog
from src.ingest import ingest_documents
from src.retrieve import retrieve_relevant_chunks
from src.app import app


@pytest_asyncio.fixture(autouse=True)
async def clean_db():
    """Fresh schema per test — keeps tests independent of each other and of run order.
    Async now: engine.connect()/Base.metadata.drop_all/create_all all need to run
    through the async engine via run_sync, and the fixture itself needs to be an
    async generator for pytest-asyncio to await it."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


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


@pytest.mark.asyncio
async def test_ingest_creates_document_and_chunks(sample_pdf):
    doc_id = await ingest_documents(sample_pdf, "sample.pdf")

    async with AsyncSessionLocal() as db:
        doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
        assert doc is not None
        assert doc.status == "ready"
        chunks = (await db.execute(select(Chunk).where(Chunk.document_id == doc.id))).scalars().all()
        assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_ingest_is_additive_not_destructive(sample_pdf):
    """Uploading a second document must not delete the first — this was a real bug in v1."""
    first_id = await ingest_documents(sample_pdf, "first.pdf")
    second_id = await ingest_documents(sample_pdf, "second.pdf")

    async with AsyncSessionLocal() as db:
        docs = (await db.execute(select(Document))).scalars().all()
        assert len(docs) == 2
        assert (await db.execute(select(Document).where(Document.id == first_id))).scalar_one_or_none() is not None
        assert (await db.execute(select(Document).where(Document.id == second_id))).scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_retrieve_returns_k_chunks(sample_pdf):
    await ingest_documents(sample_pdf, "sample.pdf")
    results = await retrieve_relevant_chunks("any question", k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    if results:
        assert "content" in results[0]


# NOTE ON THIS CHANGE: the sync `fastapi.testclient.TestClient` used previously
# runs the ASGI app in its own background thread with its own event loop,
# separate from the loop pytest-asyncio uses for the async tests above.
# asyncpg connections are bound to the event loop that created them, so a
# pooled connection created during an async test becomes invalid the moment
# TestClient's different loop tries to reuse it ("attached to a different
# loop" — hit and confirmed via a real failing test run, not assumed).
# httpx.AsyncClient + ASGITransport runs the app on the SAME event loop as
# the test itself, which removes the cross-loop boundary entirely.

@pytest.mark.asyncio
async def test_upload_endpoint_rejects_non_pdf():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/upload",
            files={"file": ("notes.txt", b"just some text", "text/plain")},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_endpoint_accepts_pdf(sample_pdf):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with open(sample_pdf, "rb") as f:
            response = await client.post(
                "/upload",
                files={"file": ("sample.pdf", f, "application/pdf")},
            )
    assert response.status_code == 200
    assert "document_id" in response.json()


@pytest.mark.asyncio
async def test_stream_endpoint_rejects_empty_question():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/stream", data={"question": "   "})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_query_log_table_exists_and_is_empty_initially():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(QueryLog))).scalars().all()
        assert len(rows) == 0