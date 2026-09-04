# RAG Document Assistant — Full Project Documentation

**Repo:** https://github.com/0Abhijeet/Document-assistant
**Live demo:** https://document-assistant-okwx.onrender.com
**Built:** August 2026, over a weekend (Saturday–Monday)

---

## 1. What it is

A web application where a user uploads a PDF and asks natural-language questions about it. The system retrieves the most relevant sections of the document and uses a language model to generate an answer grounded in that content, streamed back to the browser token by token.

This is a Retrieval-Augmented Generation (RAG) system — the standard pattern for building question-answering tools over private documents that a general-purpose LLM was never trained on.

---

## 2. Why it was built this way

### The starting point
The original prototype was a single-file FastAPI app using:
- FAISS for vector storage, saved to local disk
- Ollama running a local LLM
- No database, no tests, no deployment story

### The problems with that starting point
1. **FAISS's local index was wiped and rebuilt on every upload** — meaning only one document could exist in the system at a time. This is a correctness bug, not just a scaling limitation.
2. **Ollama requires a local model process** — incompatible with any serverless or container-based cloud host (Cloud Run, Render, Railway), which are stateless and don't persist multi-gigabyte model weights across restarts.
3. **`async def` FastAPI routes were calling blocking code directly** (embedding generation, disk I/O, DB calls) — which freezes the entire event loop for every other concurrent request.
4. **No persistence layer** — chunk metadata and query history had nowhere to live.
5. **No tests, no CI, no deployment.**

### The rebuild
| Component | Chosen solution | Reasoning |
|---|---|---|
| Vector storage | PostgreSQL + pgvector | One database for both document metadata and embeddings — simpler than running a separate vector DB, and it's the same database already needed for query logs |
| Database hosting | Neon | Free tier is *permanent*, not a time-boxed trial; pgvector supported out of the box; no VPC/networking complexity like Cloud SQL |
| LLM | Groq API | Free tier with no credit card, generous rate limits (30 req/min, 14,400/day), OpenAI-compatible client — near drop-in replacement for the local Ollama calls |
| Embeddings | fastembed (BAAI/bge-small-en-v1.5) | Initially used `sentence-transformers`, which pulls in `torch` — this later caused a production memory crash (see Section 4). Swapped to `fastembed`, which uses ONNX runtime instead of torch, at a fraction of the memory cost, with the same 384-dimension output so no schema changes were needed |
| App hosting | Render | Free tier, git-push deploy, auto-detects a Dockerfile — no IAM/VPC ceremony required for a portfolio project |
| Migrations | Alembic | Standard, versioned schema management — first migration also enables the `pgvector` Postgres extension |
| Testing | pytest against real Postgres (not SQLite) | pgvector's `Vector` column type has no SQLite equivalent, so testing against a real Postgres instance (via a local Docker container, and later a fresh instance in CI) was the only way to genuinely validate the schema and queries |
| CI | GitHub Actions | Spins up a throwaway Postgres+pgvector service container and runs the full test suite on every push |

---

## 3. Architecture

```
┌─────────────┐     upload PDF      ┌──────────────┐
│   Browser   │ ──────────────────▶ │   FastAPI    │
│  (chat UI)  │                     │   (Render)   │
└─────────────┘ ◀────────────────── └──────┬───────┘
    ask question, get                       │
    streamed answer                         │
                                             ▼
                              ┌──────────────────────────┐
                              │  ingest.py                │
                              │  - PyPDFLoader             │
                              │  - RecursiveCharacterSplit │
                              │  - fastembed (384-dim)     │
                              └──────────┬─────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────────┐
                              │  Postgres + pgvector       │
                              │  (Neon)                    │
                              │  - documents                │
                              │  - chunks (with embeddings) │
                              │  - query_logs                │
                              └──────────┬─────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────────┐
                              │  retrieve.py               │
                              │  cosine similarity search   │
                              └──────────┬─────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────────┐
                              │  generate.py                │
                              │  Groq API (streamed)        │
                              └──────────────────────────┘
```

### Database schema
- **`documents`** — id, filename, uploaded_at, status (processing/ready/failed)
- **`chunks`** — id, document_id (FK, cascade delete), page_number, content, embedding (`vector(384)`)
- **`query_logs`** — id, question, answer, created_at

Ingestion is **additive**: uploading a new document never deletes existing ones. This directly fixes the original FAISS bug and is covered by a dedicated regression test (`test_ingest_is_additive_not_destructive`).

---

## 4. The debugging log — real problems hit and fixed

This is the part worth remembering for interviews. Almost nothing shipped on the first attempt; every fix here came from reading the actual error, not guessing.

1. **`.env` not loading** — `os.environ["DATABASE_URL"]` raised `KeyError` because nothing was calling `load_dotenv()`. Fixed by adding `python-dotenv` and loading it explicitly in `database.py` and `alembic/env.py`. Root cause: `.env` files are never auto-loaded by plain Python — something has to read them.

2. **Groq model 404** — `llama-3.1-8b-instant` returned "model does not exist." Groq had deprecated it since. Fixed by switching to `openai/gpt-oss-20b`. Lesson: third-party model names are not stable long-term.

3. **Retrieval "not working" (actually a stale process)** — after fixing a bug in `retrieve.py`, the answer still looked wrong. Root cause: `uvicorn --reload` didn't fully pick up the change; a full process restart fixed it. Lesson: when behavior looks impossible given the code, suspect the running process before the logic.

4. **pytest `ModuleNotFoundError: No module named 'src'`** — running `pytest` directly doesn't add the project root to `sys.path`; `python -m pytest` does. This same fix had to be applied twice: once locally, once again in the GitHub Actions workflow file, since the CI YAML still had the old invocation.

5. **Docker Desktop not running** — `npipe` connection errors meant the Docker engine process itself wasn't started, not a PATH or install problem.

6. **Postgres port conflict** — a local Postgres install was already listening on 5432, causing a misleading "password authentication failed" error against a fresh container. Fixed by running the test container on a different port.

7. **`pgvector` extension not enabled on a fresh database** — `Base.metadata.create_all()` doesn't run `CREATE EXTENSION`; only the Alembic migration did. Fixed by adding an explicit `CREATE EXTENSION IF NOT EXISTS vector` call inside the test fixture itself, so any fresh test database (local or CI) self-heals.

8. **CI failing after "it works locally"** — the local fix for #7 was applied and tested against an *already-patched* local test database, so the test file's actual fixture code was never verified to contain the fix. CI caught it because it always starts from a genuinely blank database. Lesson: "passes locally" isn't proof if your local environment has leftover state a fresh environment won't.

9. **`git push` typo in CI** — a copy-paste error left the workflow running `pytest -m pytest -v` (a marker filter) instead of `python -m pytest -v` (correct module invocation). CI logs showed the literal command run, which made the typo immediately obvious.

10. **Render deployment crash-looping (OOM)** — the app deployed and briefly showed "live," but every real request either hung indefinitely or the instance silently failed and restarted. Render's free tier caps memory at 512MB; `sentence-transformers` pulls in `torch`, which alone can approach that limit. Diagnosed via Render's "Instance failed" event log showing repeated silent restarts. Fixed by replacing `sentence-transformers` with `fastembed` (ONNX-based, no torch), producing the same 384-dimension vectors with a much smaller memory footprint — no schema migration required.

11. **Stale `DATABASE_URL` environment variable** — after manually setting `$env:DATABASE_URL` in PowerShell to point at a test container, the app run afterward *in the same terminal session* silently used that same test database instead of `.env`'s real one, since PowerShell session variables take priority. Fixed by using a fresh terminal window.

---

## 5. Testing strategy

7 tests, all run against a real (not mocked) Postgres+pgvector instance:

- `test_ingest_creates_document_and_chunks` — proves ingestion actually writes to the DB
- `test_ingest_is_additive_not_destructive` — regression test for the original FAISS wipe-on-upload bug
- `test_retrieve_returns_k_chunks` — proves the pgvector similarity query runs and returns results
- `test_upload_endpoint_rejects_non_pdf` — validation
- `test_upload_endpoint_accepts_pdf` — full HTTP → ingest → DB path, not just the function in isolation
- `test_stream_endpoint_rejects_empty_question` — input validation on the chat endpoint
- `test_query_log_table_exists_and_is_empty_initially` — sanity check on the logging table

A fake embedding model (`tests/conftest.py`) is used in place of the real one so CI runs in seconds without downloading model weights or calling a live API — only the database and pipeline logic are actually under test.

---

## 6. CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Triggers on every push and pull request to `main`
- Spins up a `pgvector/pgvector:pg16` Postgres service container
- Installs dependencies, runs the full pytest suite against that fresh database
- Does **not** auto-deploy — deployment to Render is manual, a deliberate scope cut given the project timeline

---

## 7. Known limitations (own these, don't hide them)

- **Thread-pool concurrency, not true async.** FastAPI runs the app's blocking `def` routes in a worker thread pool automatically, which avoids freezing the event loop — but the thread pool has a fixed size, so this doesn't scale to heavy concurrent load. Fine for a demo, not for production traffic.
- **No authentication.** Anyone with the URL can upload documents or query them. Acceptable for a portfolio demo; would need real auth for anything handling actual private data.
- **Deployment is manual**, not part of CI. A deliberate scope cut, not an oversight — mentioned explicitly so it's never misrepresented as "full CI/CD."
- **Cold starts.** Render's free tier spins the service down after inactivity; first request after idle time is slow (30-60s).
- **ivfflat index tuning** (`lists = 100`) is a reasonable default, not benchmarked against real data volume — would need tuning if the document corpus grew significantly.

---

## 8. What I'd do differently at scale

(Good material for a "how would you scale this" interview question)

- Move to true async DB/embedding calls (e.g., `asyncpg`, async-compatible embedding inference) instead of thread-pool concurrency
- Add authentication and per-user document scoping
- Automate deployment in CI (currently a deliberate scope cut)
- Benchmark and tune the ivfflat index parameters against real data volume, or evaluate HNSW indexing
- Add rate limiting on the upload/query endpoints
- Add structured logging/observability beyond the basic `query_logs` table

---

## 9. AWS deployment (resume/interview artifact)

The production demo above (Render) is the permanent, always-on version of this project. Separately, the same application was deployed to AWS to gain hands-on experience with core AWS services listed in target job descriptions. This deployment is not intended to run indefinitely — it exists to produce real, defensible experience and documentation, not as a second production environment.

**Decision:** migrated the database from Neon to RDS specifically (rather than keeping Neon and only using EC2), since RDS is explicitly named in target JDs and the VPC/security-group work involved is itself the transferable skill, not just a keyword match.

### Architecture
- **EC2** (`t2.micro`, Ubuntu 22.04) runs the FastAPI application in a Docker container, restart policy `unless-stopped`, verified to survive a real instance reboot with no manual intervention
- **RDS PostgreSQL** (`db.t3.micro`, PostgreSQL 16.x) runs the pgvector-enabled database, migrated from the original Neon instance via `pg_dump`/`pg_restore`
- **Networking**: RDS has no public access. The only inbound rule on its security group references the EC2 instance's security group directly (SG-to-SG), not an IP range — so only the application server can reach the database, and nothing on the open internet can
- **Elastic IP** allocated and associated so the demo URL survives EC2 stop/start cycles
- **IAM**: dedicated IAM user created for console access rather than using root, MFA enabled on root, a $5 cost budget with an alert threshold configured

### What this demonstrates
- EC2 provisioning: AMI selection, security group configuration, both key-based and browser-based (EC2 Instance Connect) access
- RDS provisioning: engine/version selection for extension compatibility (pgvector requires PostgreSQL 15.2+), instance-class and storage sizing within free tier
- VPC networking: security-group-to-security-group referencing as the correct pattern for private service-to-service access, versus IP allowlisting
- IAM fundamentals: least-privilege console access, separating root from day-to-day use
- Docker deployment on a persistent host (vs. Render's managed platform): environment variable handling via `.env` (not committed to git, `chmod 600`), restart policies, and verifying container identity against the actual intended backend rather than trusting "container running" as sufficient proof
- Cross-provider data migration: `pg_dump`/`pg_restore` between managed Postgres providers, including `--no-owner`/`--no-privileges` handling for role mismatches

### The debugging log — AWS deployment

1. **Wrong AMI selected on first EC2 launch** — "Microsoft SQL Server is not supported for the instance type 't2.micro'" error at launch. Root cause: an AMI search for "ubuntu 22.04" returned a Marketplace SQL-Server-bundled image as the top/only Quick Start result, not the plain Canonical base image. Fixed by clearing a stuck search filter and using the default (no search term) Quick Start AMI list instead. Lesson: check the AMI description for bundled software before selecting, especially when a search returns exactly one hit.

2. **EC2 Instance Connect failed to connect** — "Failed to connect to your instance." Two compounding causes: the key pair had been created as `.ppk` (incompatible with plain OpenSSH/PowerShell), and separately, the EC2 security group's SSH rule was scoped to "My IP," which blocks AWS's own Instance Connect IP range (it doesn't originate from the user's home/office IP). Abandoned the local SSH client path entirely in favor of EC2 Instance Connect (browser-based, no key file needed), and temporarily widened the SSH rule to unblock it. Lesson: Instance Connect traffic comes from AWS's infrastructure, not the user's own IP — an "SSH from My IP only" rule that's correct for a local SSH client silently breaks the browser-based connect method.

3. **`docker-compose.yml` silently connected to the wrong database** — the app container started successfully with no errors, but was actually talking to a throwaway local Postgres container instead of the intended external database (Neon, then RDS). Root cause: `DATABASE_URL` was hardcoded inline in the compose file under `app.environment`, which takes precedence over the same key set in `.env`; separately, `depends_on: db: condition: service_healthy` auto-starts the local `db` service regardless of which service is named in `docker compose up`. Fixed by changing the compose file's `DATABASE_URL` to `${DATABASE_URL}` and using `--no-deps` to prevent the local db from auto-starting. Lesson: a container reporting "running"/"healthy" is not proof it's talking to the intended backend — always verify against the actual expected data source, not just process status.

4. **RDS unreachable from EC2 — "Connection timed out"** — `psql` hung and failed despite RDS showing "Available" and both instances in the same VPC. Root cause: RDS's security group inbound rule was scoped to a raw IP address (first "My IP," briefly and incorrectly widened to `0.0.0.0/0` as a shortcut) instead of referencing the EC2 security group. EC2's traffic to RDS originates from its security group identity within the VPC, not from any external IP, so an IP-based rule can never match it regardless of which IP is used. Fixed by deleting the IP-based rule and adding a new one with the source set to the EC2 security group directly (selected from autocomplete, not typed as an IP). Lesson: SG-to-SG referencing and IP-based rules are mutually exclusive on a single AWS security group rule; also — briefly opening a *database's* inbound rule to the whole internet to "just get it working" is a materially worse mistake than doing the same on SSH, since it's direct data exposure rather than a login prompt.

5. **App crashed after cutover to RDS — "Could not parse SQLAlchemy URL"** — the `.env` line had accidentally become `DATABASE_URL=DATABASE_URL=postgresql://...` (the key typed twice while editing in nano), so SQLAlchemy tried to parse a value that started with the literal string "DATABASE_URL=postgresql://...". Fixed by rewriting the line as a single clean assignment. Lesson: after any `.env` edit, `cat` it back and read it before restarting the container — a malformed env line fails silently at the shell level and only surfaces once the app actually tries to use the value.

### Explicit scope decisions
- **Nginx/HTTPS termination was scoped out.** It demonstrates general reverse-proxy/webserver skills rather than AWS-specific ones, and doesn't address the actual goal (closing the AWS resume/JD gap) — deliberately cut, not an oversight.
- **This AWS deployment is not the long-term demo.** Render remains the permanent, always-on version. The AWS deployment was kept within free-tier limits and is intended to be torn down (EC2/RDS stopped, Elastic IP released) once documentation and screenshots were captured.

### Still open at time of writing
- SSH security group rule was temporarily widened to `0.0.0.0/0` to unblock EC2 Instance Connect — deliberately left open until the end of the exercise, to be scoped back to "My IP" (with Instance Connect re-verified afterward) before considering the deployment fully closed out.
- Eventual teardown of EC2/RDS/Elastic IP once this documentation was captured, to avoid running unattended against the AWS free-tier credit balance.

## 10. Async rework (September 2026)

Section 7 flagged "thread-pool concurrency, not true async" as a known limitation, and Section 8 listed moving to true async as the first "what I'd do at scale" item. This section closes that gap.

### Decision
Converted the full request path — DB layer, retrieval, LLM call, and routes — to genuine `async`/`await`, rather than relying on FastAPI's automatic thread-pool handling of sync `def` routes. Chose this over a partial conversion (e.g., only the DB layer) because a naive partial conversion — marking routes `async def` without offloading the CPU-bound calls (`fastembed`, PDF parsing, text splitting) — reintroduces the exact event-loop-blocking defect already fixed once in this project's history (Section 2, problem #3), just with a different library. Verified this experimentally before writing any conversion code: a synthetic concurrent-request test showed a CPU-bound call blocking the event loop for 814ms under naive `async def`, versus 20-24ms under both the original thread-pool approach and the target `asyncio.to_thread` approach.

### What changed
| Component | Before | After |
|---|---|---|
| DB engine/session | `sqlalchemy.create_engine` (sync) | `create_async_engine` + `asyncpg` driver, `AsyncSession` |
| Queries | `Session.query(...)` | SQLAlchemy 2.0 `select()` + `await session.execute()` |
| LLM client | `groq.Groq` | `groq.AsyncGroq`, `await client.chat.completions.create(...)`, `async for` over the stream |
| PDF parsing | `PyPDFLoader.load()` | `await loader.aload()` — confirmed from `langchain-core` source that this already offloads via `run_in_executor`, no manual wrapping needed |
| Text splitting / embedding | inline sync calls | wrapped in `asyncio.to_thread()` — both are CPU-bound with no async-native equivalent, so they must be explicitly offloaded even inside an `async def` route |
| Routes (`app.py`) | `def` | `async def`, file write offloaded via `asyncio.to_thread` |
| Migrations (`alembic/env.py`) | sync engine via `engine_from_config` | **unchanged** — it never touched the app's engine object, so it was already correctly isolated from this conversion |

### Neon-specific translation required
Neon's connection string uses libpq-only query parameters (`sslmode`, `channel_binding`) that `asyncpg` doesn't accept directly, and its `-pooler` hostname means PgBouncer in transaction-pooling mode, which is incompatible with asyncpg's default server-side prepared-statement caching. Solved with a small URL-rewriting function (`_to_asyncpg_url`) plus `statement_cache_size=0` in `connect_args` when a pooler hostname is detected — verified against the actual production connection string shape, not a generic placeholder.

### Real bugs found and fixed (verified, not assumed)
1. **`query_logs` table missing during test setup.** A test-harness gap (an earlier bootstrap script didn't register the `QueryLog` model), not an application bug — creating tables from the real `models.py` module resolved it.
2. **`TestClient` incompatible with an async SQLAlchemy engine** (`RuntimeError: ... attached to a different loop`) — `fastapi.testclient.TestClient` runs the ASGI app on its own background event loop, and `asyncpg` connections can't cross event loops. Replaced with `httpx.AsyncClient` + `ASGITransport` for all HTTP-level tests, so app and test share one loop — the documented, correct fix for this known incompatibility, not a workaround.
3. **Neon migration never applied in production** — `alembic current` against the live database came back empty; the schema had simply never been migrated onto Neon. Running `alembic upgrade head` resolved it. Not an async-conversion bug, but only surfaced once the app was actually run against real infrastructure.

### Verification
- Local test suite (7 tests) ported to `pytest-asyncio`, run against a real Postgres+pgvector instance (Docker), all passing
- CI (`ci.yml`) required **no changes** — confirmed its Postgres service, env vars, and Python version are all already compatible with the async stack
- Full manual run against the real Neon database and real Groq API key: upload → ingest → retrieve → streamed answer, confirmed working end-to-end
- Additionally connected Claude Desktop itself as an MCP client (not just the Inspector debugging tool) — configured via its `claude_desktop_config.json`, pointed at the project's actual venv interpreter. One real bug surfaced during this setup, worth noting since it's a genuinely non-obvious Windows-specific gotcha: Claude Desktop is installed as a packaged/sandboxed app on this machine, so its real config lives under `%LOCALAPPDATA%\Packages\Claude_<id>\LocalCache\Roaming\Claude\`, not the conventional `%APPDATA%\Claude\` path — editing the wrong file produced no error and no effect, which was more confusing than a clean failure would have been. Once pointed at the correct file, a natural-language request in a real Claude Desktop chat correctly triggered `search_documents` and returned accurate, document-grounded content from the real ingested PDF.

### Residual, honest limitation
`asyncio.to_thread` is still used for `fastembed` inference and PDF parsing/splitting — these have no true async-native implementation in their respective libraries. This isn't a gap in the conversion; it's the correct pattern for offloading CPU-bound work under `asyncio`, and is worth being able to explain as such rather than presenting the whole pipeline as "100% async" without qualification.

## 11. MCP tool integration (September 2026)

### Decision
Wrapped the existing pgvector retrieval service as a single MCP tool — `search_documents(query, top_k=5, doc_id=None)` — in a standalone `mcp_server.py` at the repo root, importing `src.retrieve` directly rather than duplicating any retrieval logic. No existing FastAPI route was touched; the MCP tool and the web app are two independent entry points into the same service layer.

Used the current `mcp` SDK (`MCPServer`, v2.x) rather than pinning `mcp<2` to keep the more commonly-referenced `FastMCP` class name from v1. The actual usage pattern — decorator-based tool registration, `.run(transport="stdio")` — is identical between versions; only internal naming changed. Chose to build on the actively-maintained version rather than a deprecated one for name-familiarity.

### Retrieval enrichment for the MCP response
The tool's stated return type (`list[Chunk]`) needed more than the retrieval function originally returned. Extended `retrieve_relevant_chunks` with:
- an optional `doc_id` parameter, added as a `WHERE` clause on the existing query
- a join against `Document` to include `filename` per chunk
- the actual cosine-distance value, previously used only for `ORDER BY` and never returned, now selected explicitly and converted to a 0-1 `similarity` score (`1 - distance`)

This is backward-compatible: `generate.py`'s existing call site (`await retrieve_relevant_chunks(question)`) is unaffected, since `k` and `doc_id` both keep their previous defaults and it only reads the `content` field.

### Real bugs found and fixed (verified, not assumed)
1. **Test-vector design bug, not an app bug.** An early verification test used constant-valued vectors (e.g., `[0.9]*384` vs `[0.01]*384`) to simulate a "near" and "far" match. Both returned identical (maximal) similarity — because any two vectors where every component is the same constant are scalar multiples of each other, i.e. they point in the exact same direction, so cosine distance between them is always zero regardless of magnitude. Fixed by using vectors with genuinely different component patterns (varying by index, not just scale). Worth remembering as a general gotcha when hand-constructing test embeddings, not specific to this project.
2. **Three SDK API-surface mismatches**, all found by testing against the real `MCPServer` object instead of assuming its shape: an internal attribute referenced by an old name (`_mcp_server`, actually `_lowlevel_server` in this version); `list_tools()` being genuinely `async` despite a type hint that read as synchronous; and Python-side fields using snake_case (`is_error`, `input_schema`, `structured_content`) while the wire protocol they serialize to and from uses camelCase (`isError`, `inputSchema`) — none were application bugs, all were caught before they could reach real usage.

### Verification
- In-sandbox: registered the tool, checked its generated schema exposes exactly `query`/`top_k`/`doc_id`, and called it via `MCPServer`'s own public `call_tool()` against a local pgvector database seeded with two documents — confirmed correct similarity ordering *and* that the `doc_id` filter genuinely excludes the other document's closer-matching chunk (not just present in the schema, actually enforced in the query)
- Real end-to-end: connected the official MCP Inspector (`npx @modelcontextprotocol/inspector python mcp_server.py`) to the actual server over real stdio transport, against the real, already-populated Neon database — `search_documents` returned real, correctly-ranked chunks and filenames from an actually-ingested PDF (`ICICNS2026_PaperID539_AR_CITIZEN.pdf`) for a genuinely relevant query, then confirmed the `doc_id` filter correctly scoped results in production and that an unfiltered call correctly retrieved across multiple distinct documents already in the database

## 12. AWS Bedrock as an alternate LLM provider (September 2026)

### Decision
Added Bedrock alongside Groq — not a replacement — with the caller choosing per-request via `stream_answer(question, provider=...)`. Implemented both integration generations explicitly, since speaking to that tradeoff was the point of this step:
- `invoke_model_with_response_stream` — the older, per-model-family API
- `converse_stream` — the newer, unified API, same shape regardless of model family

### The async bridge problem
`boto3` has no async API at all, including its streaming response object. Naively wrapping the whole stream in `asyncio.to_thread(list, stream)` would technically work but defeats streaming entirely — it drains every chunk in the background thread before returning anything, so nothing reaches the user until the full answer has already arrived. Built and verified a proper bridge (`src/async_bridge.py`): a background thread iterates the blocking stream and pushes each item onto an `asyncio.Queue` via `loop.call_soon_threadsafe` as it arrives; the async generator pulls and yields incrementally. Verified experimentally, not assumed: a synthetic blocking stream with real per-chunk delays showed chunks arriving at the correct intervals (not bunched at the end), and concurrent trivial async tasks completed at their normal latency while the bridge was actively running in the background — confirming the event loop stays free.

### Model churn — three real model changes forced by real-world constraints, not code bugs
1. **Anthropic Claude 3 Haiku (original choice)** — blocked before any code ran: Bedrock's Anthropic use-case approval form returned "account not authorized," reproducible even from the root account. An account-level restriction, not an IAM permissions issue.
2. **Amazon Titan Text Lite (first switch)** — no vendor gate, worked initially, but later invocation failed with `ResourceNotFoundException: This model version has reached the end of its life` — Titan Text has been superseded by the Nova lineup.
3. **OpenAI gpt-oss-20b (second switch, attempting a real GPT model)** — both APIs failed identically with `ValidationException: Operation not allowed`. Root-caused via AWS documentation and multiple corroborating community reports (not guessed): gpt-oss is a third-party, Marketplace-billed model requiring an account-level Marketplace subscription/entitlement that this account doesn't currently satisfy — the same category of restriction as the Anthropic block, not a code or credentials issue.
4. **Amazon Nova Micro (final choice)** — AWS's own first-party successor to Titan, no vendor gate. Notable finding: Nova's `invoke_model` request/response shape is nearly identical to `converse_stream`'s (both use `messages`/`contentBlockDelta`) — Nova's native Messages API was designed to mirror Converse from the start. This is model-family-specific, not a general property of `invoke_model`: Titan and Claude both use structurally different, incompatible bodies for the same operation.

### Blocked: account-level Bedrock restriction (open at time of writing)
Even after switching to Nova Micro — a first-party AWS model with no vendor gate — every invocation still fails with `ValidationException: Operation not allowed`. Isolated this down to a bare, synchronous, non-streaming `boto3.client('bedrock-runtime').converse()` call, completely outside the app, async, and streaming code, with credentials confirmed loaded correctly. It still fails identically. This rules out the application code, model choice, and IAM permissions entirely — it's an AWS account-level Bedrock-enablement restriction, matching a documented pattern (multiple AWS re:Post threads report the identical symptom — every model, including AWS-native ones, rejected under full `AdministratorAccess`) whose only known resolution is an AWS Support case (filed under Account and Billing, no paid plan required).

### Verification status — stated precisely, not rounded up
- **Fully verified:** the async bridge (real incremental delivery, real non-blocking behavior under concurrent load), both event-parsing paths against realistic event shapes matching AWS's documented schemas for Titan, Nova, and gpt-oss, the full provider-dispatch flow (retrieval → provider call → DB logging) with real Postgres writes for all three providers, and the `/stream` route's provider parameter (default, explicit selection, and rejection of invalid values) — all via real HTTP requests against the real ASGI app.
- **Not yet verified:** an actual live call reaching Bedrock and returning a real model response. Blocked by the account-level restriction above, not by anything in the code. Once the AWS Support case resolves, the fastest confirmation is the same isolated `boto3.converse()` script used to diagnose this — if that succeeds, the full app almost certainly works immediately, since every layer above it is already tested.