# CLAUDE.md

RAG document assistant: upload a PDF, ask questions, get answers streamed back. Every claim below was checked against the code at the time of writing; if you change the code, update this file.

## Purpose and architecture

FastAPI app (`src/app.py`) with two routes that matter: `POST /upload` ingests a PDF (PyPDFLoader -> RecursiveCharacterTextSplitter 500/50 -> fastembed `BAAI/bge-small-en-v1.5`, 384-dim -> Postgres/pgvector), and `POST /stream` embeds the question, fetches the top-k nearest chunks by cosine distance, drops chunks below a similarity threshold, and streams an LLM answer (Groq by default; Bedrock, OpenAI or Azure optional). Every answered query is written to `query_logs` and traced in Langfuse. A separate stdio MCP server exposes the same retrieval function as a `search_documents` tool. There is no agent framework: the pipeline is a straight function chain, not a graph.

| File | Role |
|---|---|
| `src/app.py` | Routes `/`, `/upload`, `/stream`; inline HTML chat UI; validates provider name |
| `src/generate.py` | `stream_answer()` async generator: retrieve -> threshold filter -> provider dispatch -> log. Groq client, prompt builder, `SIMILARITY_THRESHOLD`, `NO_CONTEXT_ANSWER`, Langfuse spans |
| `src/retrieve.py` | `retrieve_relevant_chunks(query, k=3, doc_id=None)`; returns `content, page, document_id, filename, similarity` (`1 - cosine_distance`) |
| `src/ingest.py` | `ingest_documents()`, lazy fastembed singleton `get_embeddings()`; `embed_documents()` embeds in batches of `EMBED_BATCH_SIZE` (env-overridable, default 8) to bound peak memory — fastembed's own default (256) OOM'd on Render |
| `src/bedrock.py` | Bedrock via raw `boto3` (`invoke_model_with_response_stream` and `converse_stream`) |
| `src/openai_provider.py` | OpenAI direct via native `AsyncOpenAI`; lazy client; `OPENAI_MODEL` default `gpt-4o-mini`; imports `generate._build_prompt` inside the function (circular import otherwise) |
| `src/azure_foundry.py` | Azure AI Foundry via Azure's OpenAI-compatible v1 route: `openai.AsyncOpenAI` with `base_url=<AZURE_OPENAI_ENDPOINT>/openai/v1/` (no api-version); lazy client; `model=` is the *deployment name* (`AZURE_OPENAI_DEPLOYMENT`, default `gpt-4.1-mini`); key from `AZURE_OPENAI_API_KEY`, sent as a Bearer token; uses chat completions (not the Responses API); skips empty-`choices` chunks |
| `src/async_bridge.py` | `sync_iter_to_async`: thread + `asyncio.Queue` bridge so boto3's blocking stream yields incrementally |
| `src/database.py` | Async engine/session, URL rewriting for asyncpg, `Base` |
| `src/models.py` | `Document`, `Chunk`, `QueryLog` |
| `src/tracing.py` | `get_langfuse_client()` wrapper |
| `mcp_server.py` (repo root) | `search_documents(query, top_k=5, doc_id=None)` MCP tool |
| `alembic/` | One migration, `0001_initial_schema.py` |
| `tests/` | `conftest.py` (fake embeddings) and `test_rag_pipeline.py` (16 tests) |
| `project_docs/PROJECT_DETAILS.md` | The build log (debugging history, sections 4, 10-14). Canonical copy. `data/docs/PROJECT_DETAILS.md` is a stale duplicate; do not edit it |

## Tech stack (versions actually installed in `venv/`)

`requirements.txt` is unpinned, so these are simply what is installed now. Docker image and CI use Python 3.11; the local venv is 3.12.6.

- FastAPI 0.141.1, uvicorn 0.52.4
- SQLAlchemy 2.0.52 (async, `select()` + `await session.execute()`) with asyncpg 0.31.0; Alembic 1.19.1 with psycopg2-binary 2.9.12 (sync, migrations only)
- pgvector 0.5.0 (`Vector(384)`, `cosine_distance`); Postgres 16 image `pgvector/pgvector:pg16`
- groq 1.6.0 (`AsyncGroq`), model `openai/gpt-oss-20b`
- boto3 1.43.88, raw client, default model `amazon.nova-micro-v1:0` (override `BEDROCK_MODEL_ID`; region `AWS_REGION`, default `us-east-1`)
- fastembed 0.8.0 (ONNX; no torch), langchain-community 0.4.2 + langchain-text-splitters 1.1.2 (PDF loading and splitting only), pypdf 6.16.1
- openai 3.16.2 (`AsyncOpenAI`, `src/openai_provider.py`; default model `gpt-4o-mini`, override `OPENAI_MODEL`)
- mcp 2.1.1 (v2 SDK: `from mcp.server.mcpserver import MCPServer`), langfuse 4.15.2
- Local `venv/` also contains torch 2.14.0 and transformers 5.15.1: leftovers, not in `requirements.txt`, not used by the app. Do not add them (see below).
- pytest 9.1.1, pytest-asyncio 1.4.0, httpx 0.28.1, reportlab 5.0.1

## Constraints to respect

### Async rules
- **Must be awaited:** `retrieve_relevant_chunks`, `ingest_documents`, `loader.aload()`, every `session.execute/flush/commit/rollback`, `client.chat.completions.create(...)`. `stream_answer()` and the provider functions are async generators: iterate with `async for`, do not `await` them.
- **Run via `asyncio.to_thread` because they are CPU-bound or blocking with no async equivalent:** `embeddings.embed_query` (retrieve.py), `embeddings.embed_documents` and `splitter.split_documents` (ingest.py), `_save_upload_sync` (app.py), and the initial boto3 calls `invoke_model_with_response_stream` / `converse_stream` (bedrock.py). The boto3 response stream is then iterated through `sync_iter_to_async`, never `to_thread(list, stream)`, which would drain the whole stream before yielding anything.
- **Known behavior, not a fix target:** `get_embeddings()` is sync and is called directly on the event loop, so the first call after a cold start loads the ONNX model on the loop (the 11s cold-start trace in the build log).
- pytest runs in `asyncio_mode = strict` with session-scoped loops: async tests need `@pytest.mark.asyncio`, async fixtures need `@pytest_asyncio.fixture`. Async DB fixtures and tests must share one loop.
- `AsyncSessionLocal` uses `expire_on_commit=False`; keep it, or attribute access after commit raises `MissingGreenlet`.

### Database and migrations
- Tables: `documents` (id UUID, filename, uploaded_at, status `processing|ready|failed`), `chunks` (id, document_id FK `ON DELETE CASCADE`, page_number nullable, content, `embedding vector(384)`), `query_logs` (id, question, answer nullable, created_at).
- `EMBEDDING_DIM = 384` is defined in both `models.py` and the migration. Changing the embedding model means a new migration plus a full re-ingest.
- Migration `0001` creates the `vector` extension, an index `ix_chunks_document_id`, and an ivfflat index `ix_chunks_embedding` (`vector_cosine_ops`, `lists = 100`). The ivfflat index exists only in the migration, not in `models.py`, so `Base.metadata.create_all` (used by the tests) does not create it.
- `uploaded_at` and `created_at` are `NOT NULL` with no server default; the defaults are Python-side (`datetime.utcnow`). Raw SQL inserts must supply them.
- `alembic/env.py` uses a sync engine on the raw `DATABASE_URL` (psycopg2), separate from the async app engine. A fresh database must have `alembic upgrade head` run; `create_all` does not create the extension.
- `database.py` rewrites the URL to `postgresql+asyncpg://` and drops the query string. SSL is set via `connect_args` only if the URL contains `sslmode=require`; a `-pooler` hostname (Neon PgBouncer) sets `statement_cache_size=0`. A URL without `sslmode=require` (e.g. RDS as configured) connects without SSL.
- `load_dotenv()` is called in `database.py`, `generate.py`, and `alembic/env.py`. It does not override variables already in the shell environment.

### Retrieval guard
- `SIMILARITY_THRESHOLD` (env-overridable, default `0.35`, not benchmarked) lives in `generate.py`, not `retrieve.py`. `retrieve_relevant_chunks` also backs the MCP tool and must keep returning unfiltered top-k.
- If no chunk clears the threshold (including an empty result set), `stream_answer` yields `NO_CONTEXT_ANSWER`, skips the provider call entirely, and still writes `query_logs` and closes the Langfuse span. Only the filtered `relevant_chunks` go into the prompt.

### Cost-threshold gate / critic node / LangGraph
**None of this exists.** There is no LangGraph dependency, no critic node, and no cost field or threshold anywhere in the code, and none in any commit on any branch (checked with `git log -S` on `langgraph`, `StateGraph`, `critic`). If a task refers to them, the premise is wrong for this repo; ask before inventing one.

### Don't reintroduce (from the build log; each is a bug that already happened)
- **Don't call blocking code inside `async def`.** Naive `async def` around fastembed blocked the loop for 814 ms vs 20-24 ms with `to_thread`. Same class as the original FAISS-era bug.
- **Don't add torch-based dependencies** (e.g. `sentence-transformers`). Render's 512 MB limit OOM-crash-looped the app. Keep fastembed.
- **Don't call fastembed's `.embed()` without an explicit `batch_size`.** It defaults to 256, so every chunk of an uploaded PDF gets embedded in one padded ONNX forward pass — a second, distinct OOM on Render, unrelated to torch. Measured: a 71-chunk PDF took RSS from ~223MB to ~669MB at batch_size=256, vs ~283MB at batch_size=8; thread count and the ONNX memory arena made no measurable difference. `embed_documents()` now passes `EMBED_BATCH_SIZE` (default 8); see build log §17. Not yet confirmed fixed against the live Render instance.
- **Don't wipe existing documents on ingest.** Ingestion is additive; `test_ingest_is_additive_not_destructive` guards it.
- **Don't use `fastapi.testclient.TestClient`.** It runs the app on its own loop and asyncpg raises "attached to a different loop". Use `httpx.AsyncClient` with `ASGITransport(app=app)`.
- **Don't run bare `pytest`.** Use `python -m pytest` (adds the repo root to `sys.path`; CI uses the same form).
- **Don't rely on `create_all` for the pgvector extension.** `conftest.py` runs `CREATE EXTENSION IF NOT EXISTS vector` itself; keep that.
- **Don't decorate async generators with Langfuse `@observe`** (upstream bug langfuse#7226), and don't put tracing inside `retrieve.py`; that would silently trace the MCP tool's separate process too. Manual spans in `generate.py` only.
- **Don't build test embeddings from constant vectors.** `[0.9]*384` and `[0.01]*384` have cosine distance 0. Vary components by index.
- **Don't hardcode model IDs and assume they last.** Groq deprecated `llama-3.1-8b-instant`; Bedrock Titan Text reached end-of-life; Claude and gpt-oss on Bedrock hit account-level gates. Nova Micro is the current default.
- **Don't trust "container running" as proof of the right backend.** `docker-compose.yml` still hardcodes `DATABASE_URL: postgresql://rag:ragpassword@db:5432/ragdb` under `app.environment`, which beats `.env`, and `depends_on: db` auto-starts the local db. The build log says this was fixed to `${DATABASE_URL}` with `--no-deps` on the EC2 host, but the repo's compose file was never updated. Verify which database the app is actually talking to.
- **Don't assume a shell `DATABASE_URL` is harmless.** A stale one silently overrides `.env`.
- **MCP SDK v2 gotchas:** the low-level server attribute is `_lowlevel_server`; `list_tools()` is genuinely async; Python fields are snake_case (`is_error`, `input_schema`) while the wire format is camelCase.
- **Windows Claude Desktop config** lives under `%LOCALAPPDATA%\Packages\Claude_<id>\LocalCache\Roaming\Claude\`, not `%APPDATA%\Claude\`; editing the wrong file fails silently.
- The Neon database once had no schema applied at all; `alembic current` showed nothing. Run `alembic upgrade head` against any new target.

## Running things

**Tests.** `python -m pytest -v` from the repo root; 16 tests, all against real Postgres + pgvector. `conftest.py` sets `DATABASE_URL` with `setdefault` to `postgresql://rag:ragpassword@localhost:5432/ragdb_test` and monkeypatches fastembed with a deterministic fake (no model download).
- **`clean_db` runs `drop_all` before and after every test on whatever `DATABASE_URL` points at.** Never run the suite with a Neon/RDS/dev URL in the environment. Use only a dedicated test database.
- On this machine a local Postgres owns port 5432 and rejects the `rag` user, so the default fails with `InvalidPasswordError`. The compose db container is published on **15432** and already contains a `ragdb_test` database:
  `DATABASE_URL=postgresql://rag:ragpassword@localhost:15432/ragdb_test python -m pytest -v`
  (start Docker Desktop first; `docker compose up -d db`). Do not point tests at the container's `ragdb`, which is the dev database.
- The fake embedding patches `ingest._embeddings`. `retrieve.py` binds `get_embeddings` by import, so it only picks up the fake because the real function returns the already-set `_embeddings` global. Keep that in mind when refactoring embedding access.
- `test_upload_endpoint_accepts_pdf` writes a `*_sample.pdf` into `data/docs/` on every run and does not clean it up. Delete strays; `data/docs/` also holds tracked sample PDFs.
- CI (`.github/workflows/ci.yml`): Python 3.11, `pgvector/pgvector:pg16` service on 5432 with db `ragdb_test`, `python -m pytest -v`. It does not run migrations and does not deploy.

**App.** Needs `DATABASE_URL` and `GROQ_API_KEY` (from the environment or `.env`; `generate.py` reads `os.environ["GROQ_API_KEY"]` at import, so the app will not import without it). Optional: `SIMILARITY_THRESHOLD`, `EMBED_BATCH_SIZE` (default 8; batch size for `ingest.py::embed_documents`, bounds peak memory during ingestion), `BEDROCK_MODEL_ID`, `AWS_REGION`, `OPENAI_API_KEY` (required only for `provider=openai`; read lazily), `OPENAI_MODEL` (default `gpt-4o-mini`), `OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` (both required only for `provider=azure`; read lazily), `AZURE_OPENAI_DEPLOYMENT`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (without Langfuse keys tracing is a safe no-op).
```
alembic upgrade head
uvicorn src.app:app --reload        # Dockerfile CMD uses --host 0.0.0.0 --port 8000, no reload
```
`/stream` takes form fields `question` and `provider` (`groq` | `bedrock_invoke` | `bedrock_converse` | `openai` | `azure`); the provider set is validated in both `app.py` and `generate.py` and must be kept in sync. If behavior looks impossible given the code, restart the process before suspecting the logic (`--reload` has missed changes before). `docker compose up --build` runs the app plus a local db; see the compose caveat above.

**MCP server.** Standalone process, not mounted on the FastAPI app: `python mcp_server.py` (stdio transport), normally launched by an MCP client config (Claude Desktop, or `npx @modelcontextprotocol/inspector python mcp_server.py`). It imports `src.retrieve` directly, so it needs `DATABASE_URL` but not `GROQ_API_KEY`. Its default `top_k` is 5 (the RAG path uses `k=3`). Do not add Langfuse or threshold filtering to this path.

**Prompt duplication.** The prompt text exists twice: `_build_prompt` in `generate.py` (Groq) and `_build_messages` in `bedrock.py`. Change both or the providers diverge.

## Not implemented / not verified live

- **`provider=openai` has never called the real OpenAI service**: unverified due to pricing concerns (a separate paid OpenAI account would be needed; none was set up, so there is no key). Verified only against a local fake SSE server via `OPENAI_BASE_URL` (incremental deltas, `query_logs` row) and unit tests. Langfuse traces for it are unverified.
- **The `EMBED_BATCH_SIZE` fix for the Render upload OOM (build log §17) is not yet confirmed against the live Render instance.** Diagnosed and fixed against the real fastembed model locally (measured RSS before/after, confirmed batch size doesn't change embedding output), but the real pytest suite hasn't run against this change (Docker Desktop wasn't running at fix time) and no upload has been done against the hosted app since.

- **Bedrock has never made a successful live call.** Every invocation, even a bare synchronous `boto3.converse()` outside the app, fails with `ValidationException: Operation not allowed`: an AWS account-level restriction (support case open). What is verified: the async bridge (real incremental delivery), event parsing against documented schemas, provider dispatch, and `/stream` provider selection. Bedrock Langfuse traces are unverified for the same reason. Treat `bedrock_invoke` and `bedrock_converse` as untested against the real service.
- **Azure: one real live call succeeded (2026-09-19)** through `POST /stream` with `provider=azure` against a real gpt-4.1-mini deployment (resource `...-4286-resource`, v1 route, chat completions): a short grounded answer came back and a `query_logs` row was written. Not verified: incremental token arrival on the real service (the answer was one short line; incremental delivery was only verified against a fake server), long answers, and Langfuse traces for Azure. Key and endpoint must come from the *same* Azure resource that holds the deployment, or Azure returns 401.
- **No tests exist** for `mcp_server.py`, `bedrock.py`, or `async_bridge.py` in the repo; the build log's verification of those was done in ad hoc sandbox scripts that were not committed.
- `SIMILARITY_THRESHOLD = 0.35` is a starting guess, not tuned on labeled data. The ivfflat `lists = 100` is also untuned.
- No authentication, rate limiting, or per-user document scoping. Deployment (Render, and the AWS EC2 + RDS exercise) is manual; CI does not deploy. The AWS deployment is an interview artifact meant to be torn down, with an SSH security-group rule still open at `0.0.0.0/0`.
- `async_bridge` has no cancellation: if the consumer stops early, the producer thread keeps draining the stream, and the queue is unbounded.
- README is partly stale: it references `.env.example` (does not exist), calls the Groq model "Llama-based" (it is `openai/gpt-oss-20b`), and its test command assumes the compose `app` container.
