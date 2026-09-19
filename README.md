# RAG Document Assistant
A full-stack Retrieval-Augmented Generation system: upload a PDF, ask questions, get streamed answers grounded in the document's actual content.
**🔗 Live demo:** https://document-assistant-okwx.onrender.com
*(First request may take 30-60s to wake up — free-tier hosting spins down when idle.)*
## What this demonstrates
- Designed and built a RAG pipeline end-to-end: ingestion, chunking, embedding, vector retrieval, and LLM generation
- Replaced a local-disk vector store with **Postgres + pgvector**, enabling persistent, multi-document storage
- Fixed a real concurrency bug — blocking I/O inside async routes — and can explain why it mattered
- Converted the pipeline to fully async (SQLAlchemy/asyncpg, async Groq client), including diagnosing and fixing a real event-loop-blocking regression risk and a testing-framework/event-loop incompatibility along the way
- Diagnosed and fixed a production memory-limit crash by swapping a torch-based embedding library for a lightweight ONNX-based one, with zero change to the database schema
- Wrote a test suite that runs against a real Postgres instance (not mocked), covering ingestion, retrieval, and API endpoints
- Set up CI (GitHub Actions) that spins up a fresh database and runs the full suite on every push
- Deployed to production (Render + Neon), debugging real infrastructure issues along the way: stale local environments, deprecated model names, environment variable scoping, and out-of-memory crash loops
- Separately deployed the same application to **AWS (EC2 + RDS)** to gain hands-on cloud infrastructure experience — VPC networking, IAM, security groups, and cross-provider database migration (see AWS deployment section below)
- Wrapped the RAG pipeline's retrieval as an MCP (Model Context Protocol) tool — `search_documents`, with top-k and per-document filtering — verified against a live production database via the MCP Inspector and Claude Desktop
- Added AWS Bedrock as an alternate LLM provider alongside Groq — both the legacy `InvokeModel` API and the unified `Converse` API — including a custom async bridge to stream boto3's fully synchronous responses without blocking the event loop, and diagnosed a real AWS account-level access restriction down to an isolated, minimal reproduction (see PROJECT_DETAILS.md §12)
- Instrumented the pipeline with Langfuse (OpenTelemetry-based manual tracing) across all three LLM providers — nested retrieval and generation spans under one trace per request, and used the trace data to catch and confirm a real cold-start latency spike (11s → ~1s after warm-up) *(Previous scope: the three providers that existed at that step. OpenAI and Azure were added afterwards; their traces are unverified.)*
- Added a similarity-threshold guard (`SIMILARITY_THRESHOLD`, default 0.35, not benchmarked): when no retrieved chunk is relevant enough, the app skips the LLM call, returns a fixed "no relevant context" answer, and still logs the query and closes the trace (see PROJECT_DETAILS.md §14)
- Grew the provider abstraction to five per-request-selectable providers (Groq, Bedrock `InvokeModel`, Bedrock `Converse`, OpenAI direct, **Azure AI Foundry**), all raw SDKs with no framework wrapper. Azure (gpt-4.1-mini via Azure's OpenAI-compatible v1 endpoint) was confirmed with a real live call, after checking billing risk up front and diagnosing a 401 caused by a key from the wrong Azure resource. OpenAI direct is wired and unit-tested but has never called the real service (see §15–16)
- Wrote a Claude Code skill (`add-llm-provider`) and packaged it as an installable plugin (`rag-provider-tools`), then used the installed plugin to re-run a full provider addition; the bugs found along the way are logged in PROJECT_DETAILS.md §15.1
## Tech stack
| Layer | Technology |
|---|---|
| API | FastAPI (async) |
| Vector storage | PostgreSQL + pgvector (via Neon; also deployed on AWS RDS — see below) |
| Embeddings | fastembed (BAAI/bge-small-en-v1.5, ONNX runtime — no torch dependency) |
| LLM | Groq (async client, streamed responses), model `openai/gpt-oss-20b`. *Previous: Llama-based (`llama-3.1-8b-instant`), since deprecated by Groq.* |
| ORM / migrations | SQLAlchemy 2.0 (async) + asyncpg; Alembic (sync, unaffected) |
| Testing | pytest, pytest-asyncio, run against a real Postgres instance |
| CI/CD | GitHub Actions |
| Deployment | Docker → Render (production); Docker → AWS EC2 + RDS (infrastructure exercise) |
| Agent tooling | Model Context Protocol (MCP) — stdio transport, official `mcp` SDK |
| LLM (alternate) | AWS Bedrock (Amazon Nova) — via `boto3`, both `InvokeModel` and `Converse` APIs |
| Observability | Langfuse (Cloud, free tier) — manual OpenTelemetry-based tracing |
| LLM (alternate) | OpenAI direct — `openai` SDK, `AsyncOpenAI` (not verified against the real service) |
| LLM (alternate) | Azure AI Foundry — gpt-4.1-mini through Azure's OpenAI-compatible v1 endpoint, `openai` SDK `AsyncOpenAI` (verified live once) |
| Dev tooling | Claude Code skill + plugin (`rag-provider-tools`) for adding providers |
## Architecture decisions (and why)
| Decision | Reasoning |
|---|---|
| Postgres + pgvector over FAISS | FAISS's local index doesn't survive a redeploy; Postgres gives persistent, queryable, multi-document storage in the same database as the app's metadata |
| Groq over local Ollama | Cloud hosting can't run a local LLM process; Groq's API has a genuine no-card free tier and keeps the app fully stateless |
| Fully async (`async def` routes, asyncpg, AsyncGroq) | Async SQLAlchemy/asyncpg and an async LLM client are explicitly named in target JDs. CPU-bound calls with no async equivalent (fastembed, PDF parsing) are offloaded via `asyncio.to_thread` — naively awaiting them inline would reintroduce the exact event-loop-blocking bug this project already fixed once (see PROJECT_DETAILS.md §10) |
| fastembed over sentence-transformers | sentence-transformers pulls in torch, which alone can exceed a 512MB hosting limit; fastembed uses ONNX runtime and produces the same 384-dim vectors with a fraction of the memory footprint |
| Tests run against real Postgres, not SQLite | pgvector's vector type has no SQLite equivalent — testing against the real database catches schema and query bugs mocks would hide |
| RDS security group referenced by ID, not IP | EC2's traffic to RDS originates from its security group identity inside the VPC, not from any external IP — an IP-based rule can never match it regardless of which IP is used |
| Separate `mcp_server.py`, not a route on the existing app | Keeps the MCP tool decoupled from the FastAPI app's lifecycle and imports the existing service layer (`src.retrieve`) directly — no duplication, no changes to existing routes |
| Custom async bridge for boto3 streaming | `boto3` has no async API; naively wrapping its blocking stream in a single `asyncio.to_thread` call would drain it entirely before yielding anything, defeating streaming. A background-thread-to-asyncio-Queue bridge preserves genuine incremental delivery — verified experimentally, not assumed |
| Similarity threshold lives in `generate.py`, not `retrieve.py` | `retrieve_relevant_chunks` also backs the MCP `search_documents` tool, where a caller may legitimately want low-similarity results. Filtering only where an answer is generated keeps the MCP tool's behavior unchanged. The 0.35 default is a starting guess, not benchmarked |
| Raw provider SDKs, no LangChain wrapper | boto3 for Bedrock (blocking, needs the async bridge), the `openai` SDK for OpenAI and Azure (natively async). Same position across all providers so the integration approaches can be compared on equal footing |
| Azure via the v1 endpoint with `AsyncOpenAI` | The portal's own snippet for the deployment uses the v1 route with no `api-version`, which removes an unverifiable version guess. *Previous approach (first implementation, replaced before any live call):* `AsyncAzureOpenAI` on the legacy `/openai/deployments/<name>/...?api-version=` route with a guessed default `2024-10-21` |
| Manual Langfuse instrumentation, not `@observe` | `stream_answer` is an async generator with no single return value — `@observe`'s automatic output-capture has a known, currently-open upstream bug for async generators specifically (langfuse/langfuse#7226). Manual placement also keeps tracing scoped to the HTTP path only, since instrumenting inside `retrieve.py` would silently trace the MCP tool's separate call path into the same function |
## Known limitations
Said out loud on purpose — demonstrating I understand the tradeoffs matters more than pretending they don't exist:
- No auth on the upload endpoint — anyone with the URL can add documents
- CI runs tests on push; deployment is manual, not yet automated
- Embedding model reloads on cold start (free-tier hosting spins down when idle)
- ivfflat index tuning (`lists = 100`) is a reasonable default, not benchmarked against real data volume
- AWS Bedrock integration is code-complete and verified against mocked responses, but live end-to-end verification is currently blocked by an AWS account-level restriction (open AWS Support case) — see PROJECT_DETAILS.md §12
- OpenAI direct has never called the real OpenAI service (no key); it is verified against unit tests and a local fake server only
- Azure was confirmed with one real short call; incremental token arrival on the real service, long answers, and Langfuse traces for it are not verified
- `SIMILARITY_THRESHOLD` (0.35) is not tuned on labeled data
- A provider error after streaming has started reaches the client as an empty HTTP 200 (the error surfaces after headers are sent), and failed generations are not written to `query_logs`
## Local development

### Current
There is no `.env.example`; create `.env` by hand with `DATABASE_URL` and `GROQ_API_KEY` (the app will not import without a Groq key). Optional variables per provider (`OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_DEPLOYMENT`, `AWS_REGION` / `BEDROCK_MODEL_ID`, `SIMILARITY_THRESHOLD`, Langfuse keys) are listed in `CLAUDE.md`.
```bash
alembic upgrade head              # against the database DATABASE_URL points at
uvicorn src.app:app --reload
```
`docker compose up --build` also works, but `docker-compose.yml` hardcodes `DATABASE_URL` to the local db container, which overrides `.env`; check which database the app is really using.

Run tests (16 tests, real Postgres + pgvector):
```bash
DATABASE_URL=postgresql://rag:ragpassword@localhost:15432/ragdb_test python -m pytest -v
```
The compose db container is published on 15432 (a local Postgres may own 5432); CI uses 5432. The suite drops every table in the database `DATABASE_URL` points at, so only ever use a dedicated test database.

### Previous (kept for reference; partly stale)
The original instructions, from before the async rework and the multi-provider work. `.env.example` does not exist, and the test command assumes the compose `app` container.
```bash
cp .env.example .env   # add your own GROQ_API_KEY and DATABASE_URL
docker compose up --build
docker compose exec app alembic upgrade head
```
Run tests:
```bash
docker compose exec app pytest -v
```

---

## AWS deployment (infrastructure exercise)

The live demo above (Render + Neon) is the permanent, always-on version of this project. Separately, I deployed the same application to **AWS EC2 + RDS** — not as a second production environment, but specifically to build and demonstrate hands-on experience with core AWS services (EC2, RDS, IAM, VPC/security groups).

**Decision:** migrated the database to RDS specifically, rather than keeping Neon and only using EC2 — RDS is explicitly named in target job descriptions, and the VPC/security-group work involved is the transferable skill, not just a keyword match.

### Architecture
- **EC2** (`t2.micro`, Ubuntu 22.04) — app running in Docker, `restart: unless-stopped`, verified to survive a real instance reboot with no manual intervention
- **RDS PostgreSQL** (`db.t3.micro`, PG 16.x) — pgvector-enabled, migrated from Neon via `pg_dump`/`pg_restore`
- **Networking** — RDS has no public access; its only inbound rule references the EC2 security group directly (SG-to-SG), not an IP range, so only the app server can reach the database
- **Elastic IP** — keeps the demo URL stable across EC2 stop/start cycles
- **IAM** — dedicated console-access user, MFA on root, cost budget with alert threshold configured

### Skills demonstrated
- EC2 provisioning: AMI selection, security groups, both key-based and browser-based (EC2 Instance Connect) access
- RDS provisioning: engine/version selection for extension compatibility, instance-class/storage sizing within free tier
- VPC networking: security-group-to-security-group referencing as the correct pattern for private service-to-service access, vs. IP allowlisting
- IAM fundamentals: least-privilege console access, separating root from daily use
- Docker deployment on a persistent host (vs. Render's managed platform): env var handling via `.env` (`chmod 600`, not committed), restart policies, and verifying container identity against the actual backend rather than trusting "container running" as proof
- Cross-provider data migration: `pg_dump`/`pg_restore` between managed Postgres providers, including `--no-owner`/`--no-privileges` role handling

### AWS deployment (infrastructure exercise)

Same app deployed separately to AWS EC2 + RDS to build hands-on cloud infrastructure experience (VPC networking, IAM, security groups, cross-provider DB migration). Full write-up, architecture, and debugging log: [`project_docs/PROJECT_DETAILS.md`]