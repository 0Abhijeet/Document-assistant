# RAG Document Assistant

FastAPI + pgvector RAG pipeline. Upload PDFs, ask questions, get streamed answers grounded in the uploaded documents.

## What changed from the prototype, and why

| Before | After | Why |
|---|---|---|
| FAISS index on local disk, wiped on every upload | Postgres + pgvector, additive | Local disk doesn't survive a redeploy; wiping meant only one document could exist at a time |
| Ollama (local model) | Groq API | Cloud Run/Render can't host a local LLM; Groq has a genuine free tier, no card required |
| `async def` routes calling blocking code directly | `def` routes | FastAPI runs sync `def` routes in a thread pool automatically — blocking embedding/DB calls no longer freeze the event loop for every other request |
| Raw `file.filename` used in the save path | Sanitized + UUID-prefixed filename | Prevents path traversal via a crafted filename |
| No tests | pytest against a real Postgres+pgvector instance | Proves the actual schema/queries work, not just business logic in isolation |

## Local development

```bash
cp .env.example .env
# edit .env: add your GROQ_API_KEY (free at console.groq.com, no card)

docker compose up --build
```

App: http://localhost:8000
First run applies no migrations automatically — run once:

```bash
docker compose exec app alembic upgrade head
```

## Running tests

```bash
docker compose exec app pytest -v
```

Tests use a fake embedding model (see `tests/conftest.py`) so CI doesn't need to download the real HuggingFace model or hit a live API — only the DB/pipeline logic is under test.

## Deploying by Monday (no GCP)

**Database — Neon (free, permanent, pgvector built in):**
1. Sign up at neon.tech, create a project.
2. Copy the connection string, enable pgvector: run `CREATE EXTENSION IF NOT EXISTS vector;` in Neon's SQL editor (Alembic's migration also does this, redundant is fine).
3. Set `DATABASE_URL` to the Neon string.

**Run migrations against Neon:**
```bash
DATABASE_URL="<your neon url>" alembic upgrade head
```

**App hosting — Render or Railway (free tier, git-push deploy):**
1. Push this repo to GitHub.
2. New Web Service → connect repo → it detects the Dockerfile.
3. Set environment variables: `DATABASE_URL` (Neon string), `GROQ_API_KEY`.
4. Deploy. First cold start will be slow (~30-60s) — the embedding model downloads on first use.

**CI:** GitHub Actions runs pytest against a throwaway Postgres+pgvector container on every push. It does not auto-deploy — deploy manually via Render/Railway's dashboard or CLI until you add that as a deliberate next step. Don't claim auto-deploy on your resume; you didn't build it yet.

## Known limitations (say these out loud in an interview, don't get caught not knowing them)

- Thread-pool concurrency, not true async — fine for demo traffic, not for production scale.
- Embedding model reloads on cold start (Render free tier spins down after inactivity).
- No auth on `/upload` — anyone with the URL can add documents. Fine for a portfolio demo, not for anything real.
- ivfflat index tuning (`lists = 100`) is a reasonable default, not benchmarked against your actual data volume.
