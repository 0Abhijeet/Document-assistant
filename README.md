# RAG Document Assistant

A full-stack Retrieval-Augmented Generation system: upload a PDF, ask questions, get streamed answers grounded in the document's actual content.

**🔗 Live demo:** https://document-assistant-okwx.onrender.com
*(First request may take 30-60s to wake up — free-tier hosting spins down when idle.)*

## What this demonstrates

- Designed and built a RAG pipeline end-to-end: ingestion, chunking, embedding, vector retrieval, and LLM generation
- Replaced a local-disk vector store with **Postgres + pgvector**, enabling persistent, multi-document storage
- Fixed a real concurrency bug — blocking I/O inside async routes — and can explain why it mattered
- Diagnosed and fixed a production memory-limit crash by swapping a torch-based embedding library for a lightweight ONNX-based one, with zero change to the database schema
- Wrote a test suite that runs against a real Postgres instance (not mocked), covering ingestion, retrieval, and API endpoints
- Set up CI (GitHub Actions) that spins up a fresh database and runs the full suite on every push
- Deployed to production (Render + Neon), debugging real infrastructure issues along the way: stale local environments, deprecated model names, environment variable scoping, and out-of-memory crash loops

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Vector storage | PostgreSQL + pgvector (via Neon) |
| Embeddings | fastembed (BAAI/bge-small-en-v1.5, ONNX runtime — no torch dependency) |
| LLM | Groq (Llama-based, streamed responses) |
| ORM / migrations | SQLAlchemy + Alembic |
| Testing | pytest, run against a real Postgres instance |
| CI/CD | GitHub Actions |
| Deployment | Docker → Render |

## Architecture decisions (and why)

| Decision | Reasoning |
|---|---|
| Postgres + pgvector over FAISS | FAISS's local index doesn't survive a redeploy; Postgres gives persistent, queryable, multi-document storage in the same database as the app's metadata |
| Groq over local Ollama | Cloud hosting can't run a local LLM process; Groq's API has a genuine no-card free tier and keeps the app fully stateless |
| Sync (`def`) FastAPI routes, not `async def` | The embedding and DB calls in this app are blocking; wrapping them in `async def` would freeze the event loop for every concurrent request. FastAPI runs sync routes in a thread pool automatically — a small but important correctness fix |
| fastembed over sentence-transformers | sentence-transformers pulls in torch, which alone can exceed a 512MB hosting limit; fastembed uses ONNX runtime and produces the same 384-dim vectors with a fraction of the memory footprint |
| Tests run against real Postgres, not SQLite | pgvector's vector type has no SQLite equivalent — testing against the real database catches schema and query bugs mocks would hide |

## Known limitations

Said out loud on purpose — demonstrating I understand the tradeoffs matters more than pretending they don't exist:

- Thread-pool concurrency, not true async — fine for demo traffic, not production scale
- No auth on the upload endpoint — anyone with the URL can add documents
- CI runs tests on push; deployment is manual, not yet automated
- Embedding model reloads on cold start (free-tier hosting spins down when idle)
- ivfflat index tuning (`lists = 100`) is a reasonable default, not benchmarked against real data volume

## Local development

```bash
cp .env.example .env   # add your own GROQ_API_KEY and DATABASE_URL
docker compose up --build
docker compose exec app alembic upgrade head
```

Run tests:
```bash
docker compose exec app pytest -v
```