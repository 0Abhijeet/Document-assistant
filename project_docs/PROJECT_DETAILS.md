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

## 13. Langfuse tracing (September 2026)

### Decision
Langfuse Cloud (free tier) over self-hosting — zero extra infrastructure, and it's what "Langfuse experience" means in most job postings (the SaaS product, not necessarily self-hosting it). Region: Japan (`jp.cloud.langfuse.com`), chosen for rough geographic proximity to Neon's Singapore hosting — a low-stakes choice, since span export happens after the response has already streamed to the user (fire-and-forget background batching), not in the request's critical path the way Neon's region actually is.

Manual instrumentation over the `@observe` decorator, for two concrete reasons rather than general preference:
1. `stream_answer` is an async generator with no single return value — `@observe`'s core behavior ("call the function, capture whatever it returns") doesn't map onto a generator, which yields values incrementally rather than returning one. Confirmed via Langfuse's own GitHub issues, not assumed: issue #7226 documents that the decorator's generator-output-concatenation logic only checks `inspect.isgenerator()` internally, which returns `False` for async generators — meaning the intended fix for this exact situation is currently broken for exactly this case.
2. Manual placement inside `generate.py` — rather than decorating `retrieve_relevant_chunks` in `retrieve.py` directly — keeps tracing scoped to the `/stream` HTTP path only. `retrieve_relevant_chunks` is also called directly by the MCP tool (`mcp_server.py`, a separate process). Decorating the function itself would have silently instrumented that path too, an unintended coupling between two features built in separate steps.

### A real correctness question, verified rather than assumed
`stream_answer` opens its tracing spans in a `with` block that spans multiple `yield` points — the consuming `StreamingResponse` resumes the generator repeatedly as chunks are written to the socket, not all at once. Whether OpenTelemetry's `contextvars`-based span context correctly survives across those yield/resume cycles (versus silently losing the parent-child relationship, or worse, mixing spans from concurrent requests) was a genuine open question, not something to assume worked. Verified with a real `InMemorySpanExporter` injected in place of the network exporter: built a synthetic async generator with real `asyncio.sleep()` suspensions between yields (forcing genuine, not instant, resumption) and a separate consumer loop with interleaved work between pulls (simulating what a real ASGI server does). Result: context survived correctly — all spans landed in one shared trace with correct parent-child nesting, confirmed by inspecting the exporter's captured spans directly (trace IDs, parent span IDs), not by absence of exceptions.

### Real production finding
First trace after a cold `uvicorn` restart showed `retrieve_relevant_chunks` taking 11.19s of a 12.06s total request. Traced to fastembed's model load and/or Neon's serverless compute wake-up — both real, expected one-time costs, not a query performance bug. Confirmed via subsequent traces settling to 1.06s–1.58s total once warm. A concrete example of the tracing setup catching something worth investigating on the very first real use, rather than being purely decorative for a resume line.

### Verification status
- **Fully verified:** correct trace/span/generation structure and nesting (via injected in-memory exporter, sandbox), correct output content landing on every span, DB logging (`QueryLog`) continuing to work unchanged alongside tracing, and — the one thing that mattered most to get right — context survival across real async suspension.
- **Verified in production:** real traces confirmed in the live Langfuse dashboard for the Groq provider, across multiple real questions against the real ingested document, showing correct `stream_answer` → `retrieve_relevant_chunks` + `{provider}-generation` nesting with real content.
- **Not yet verified:** Bedrock provider traces specifically — blocked by the same AWS account-level restriction documented in §12, not a Langfuse or tracing-code issue. The tracing code path is identical across all three providers (same `start_as_current_observation` wrapper around whichever provider function is dispatched), so this is expected to work immediately once that account clears, with no additional changes required.

## 14. Empty/near-empty retrieval guard (September 2026)

### Decision
Before this change, `retrieve_relevant_chunks` had no similarity floor at all — it was an unfiltered top-k query, so it always returned exactly `k` chunks regardless of how irrelevant they were to the question, and `generate.py` passed whatever came back straight into the prompt with zero inspection. A question about a completely different topic than the ingested document still got `k` chunks of unrelated text silently stuffed into the LLM call, inviting a confidently-wrong answer with no signal to the user that retrieval had actually failed.

Fixed at the `generate.py` layer, not inside `retrieve.py`: `stream_answer` now filters `context_chunks` down to `relevant_chunks` using a new `SIMILARITY_THRESHOLD` (env-overridable, default `0.35`, explicitly *not* benchmarked against a labeled relevance set — same honest caveat already applied to the ivfflat `lists=100` default in §7). If `relevant_chunks` is empty — whether because retrieval returned nothing at all (empty result set, e.g. no documents ingested yet) or because every candidate fell below the threshold (near-empty/irrelevant context) — the function short-circuits before the provider call entirely: it yields a fixed `NO_CONTEXT_ANSWER` message, skips the LLM round-trip, and still logs the exchange to `query_logs` and closes out the Langfuse root span, so both the query-history table and tracing stay consistent whether or not a real generation happened.

Deliberately left `retrieve_relevant_chunks` itself untouched: it's also the function backing the MCP `search_documents` tool (§11), which is an explicit search surface where a caller may legitimately want to see low-similarity results rather than have them silently dropped. Filtering only in `generate.py` keeps that tool's behavior and return shape exactly as documented, and keeps the "what counts as good enough to answer from" policy scoped to the one place that actually generates an answer.

### Real bugs found and fixed (verified, not assumed)
None in the application code itself — this was a missing-behavior gap (no threshold existed to be wrong), not a defect in existing logic. One environment issue surfaced while verifying: running the test suite hit `asyncpg.exceptions.InvalidPasswordError: password authentication failed for user "rag"` against `localhost:5432` — the exact port-conflict class of bug already logged in §4 item 6 (a local Postgres install already owns 5432, so `DATABASE_URL`'s default `localhost:5432/ragdb_test` silently connects to the wrong server instead of the project's own container). The project's `docker-compose.yml` already avoids this by publishing the test container on `15432:5432`; the fix was operational, not code — start Docker Desktop, confirm the existing `pgvector/pgvector:pg16` container on `15432`, and point `DATABASE_URL` at `localhost:15432/ragdb` for the run, rather than relying on conftest's bare `setdefault` default. Caught by the actual connection error, not assumed.

### Verification
Two new tests added to `tests/test_rag_pipeline.py`, both passing against the real Dockerized pgvector instance alongside the existing 7 (9/9 passing):
- `test_stream_answer_short_circuits_on_empty_result_set` — no documents ingested at all, asserts `stream_answer` yields exactly `NO_CONTEXT_ANSWER` and logs it to `query_logs`, with the LLM provider function monkeypatched to raise if called at all (an explicit assertion, not an accidental pass from the fixture's fake Groq key)
- `test_stream_answer_short_circuits_when_all_chunks_below_threshold` — a real document is ingested and retrieval genuinely returns chunks, but `SIMILARITY_THRESHOLD` is monkeypatched to `1.1` (above the maximum possible cosine similarity of 1.0) to deterministically force every chunk below the bar without depending on the fake test-embedding model's exact vector math; same short-circuit and logging assertions
## 15. OpenAI direct as a fourth LLM provider (September 2026)

### Decision
Added `provider=openai` via the native async client (`AsyncOpenAI`, openai 3.16.2) in its own module, `src/openai_provider.py` (not `src/openai.py`, which would shadow the SDK). It reuses `generate._build_prompt` (imported inside the function to avoid a circular import), builds the client lazily so a missing `OPENAI_API_KEY` only fails when this provider is selected, and reads `OPENAI_MODEL` (default `gpt-4o-mini`) and optional `OPENAI_BASE_URL`. Registered in `generate._PROVIDERS` and in the allowlist in `app.py::stream_endpoint`.

### Real bugs found and fixed
None in code. Live check note: the first two `/stream` questions (one generic, one keyword-style) fell below `SIMILARITY_THRESHOLD` and short-circuited to `NO_CONTEXT_ANSWER`; only a question that quoted the document title cleared 0.35.

### Verification status
- **Verified:** 3 new tests (dispatch + `query_logs` row, delta parsing that skips empty/None/no-choices chunks, route accepts `openai` and rejects unknown with 400); suite 12/12. Live run of the real app against a local fake OpenAI-compatible SSE server (via `OPENAI_BASE_URL`): deltas arrived incrementally (~0.5 s apart), a `query_logs` row was written with the full answer, unknown provider returned 400.
- **Not verified:** any call to the real OpenAI service. Reason: pricing concerns (OpenAI direct needs a separate paid account; the owner chose not to buy one, so there is no key). The real endpoint's auth-error path with a dummy key was not exercised. Langfuse traces for this provider unverified.

### 15.1 Claude Code skill and plugin for adding providers (September 2026)

#### Decision
Wrote a project skill, `add-llm-provider` (`.claude/skills/add-llm-provider/SKILL.md`), from the real provider abstraction (`_PROVIDERS` dict in `src/generate.py`, the allowlist in `src/app.py::stream_endpoint`, `src/bedrock.py`, `src/async_bridge.py`), then packaged it as a plugin, `rag-provider-tools` (`rag-provider-tools/`, marketplace `rag-project-tools`). OpenAI direct was the test provider: its streaming shape matches Groq's and the SDK accepts a `base_url` override, which allows a real end-to-end check without a key. The task brief mentioned "Azure implementations already built"; there are none (`git grep -i azure` empty, no history). The abstraction has exactly Groq, `bedrock_invoke`, `bedrock_converse`; the skill was written against that.

#### Real bugs found and fixed (verified, not assumed)
1. **Loose skill not discoverable in the session that created it.** Invoking `add-llm-provider` via the Skill tool right after creating it returned `Unknown skill`. `.claude/skills/` did not exist when the session started, so it was not being watched. Not verified: whether adding a skill to an already-existing skills dir hot-reloads. Workaround: a fresh headless session (`claude -p "/add-llm-provider ..."`) discovers it at startup. The `claude` binary is not on PATH here; it lives at `%USERPROFILE%\.vscode\extensions\anthropic.claude-code-<ver>\resources\native-binary\claude.exe`.
2. **Headless run 1 could not run tests or servers.** Under `--permission-mode acceptEdits` every form of setting `DATABASE_URL` inline (bash `VAR=x cmd`, `env VAR=x cmd`, PowerShell `$env:`) was blocked, so the run stopped after writing code and 3 tests, before any test run, live check or docs. Fix for run 2: pre-set the env vars in the child process environment and allow-list command patterns. Run 1's remainder (tests, live check, docs) was finished by hand.
3. **My own cleanup deleted a tracked file.** `rm data/docs/*_sample.pdf` (to remove the stray PDF the test suite leaves) also deleted the tracked `6c84beec..._sample.pdf`. Caught by `git status` (` D`), restored with `git checkout -- <file>`. CLAUDE.md already warned that `data/docs/` holds tracked PDFs; the skill's original "delete strays" wording did not. Don't reintroduce: skill now says delete only `git ls-files --others --exclude-standard data/docs` output.
4. **SKILL.md edit mangled by my own scripted edit.** A `\n` in a Python replacement string became a real line break inside step 9, and two sentence fixes read awkwardly. Caught from the file-changed notice, fixed and re-read.
5. **Skill gaps found by the two runs**, all fixed in SKILL.md: (a) a top-level `from src.generate import _build_prompt` in a provider module is circular; import it inside the function; (b) a module named after the SDK (`src/openai.py`) would shadow the package, so use `src/openai_provider.py`; (c) generic live-check questions fall below `SIMILARITY_THRESHOLD` (0.35) and return `NO_CONTEXT_ANSWER` without calling the provider, and each failed attempt writes a `query_logs` row (run 2 needed a question quoting the PDF title); (d) the suite's `clean_db` drops every table in `DATABASE_URL`'s database, so live checks need a separate/throwaway DB or must run first; (e) cleanup of the stray test PDF is needed after the suite, not just after a manual upload; (f) docs step now lists the CLAUDE.md file-map row and test count; (g) a hardcoded "currently registered" provider list in the skill went stale immediately, replaced with "read `_PROVIDERS`".
6. **CLAUDE.md said "no torch" but the local venv has torch 2.14.0 and transformers 5.15.1.** They are leftovers (not in `requirements.txt`, unused by the app). CLAUDE.md now says so instead of implying they are absent. Still: do not add them to `requirements.txt` (Render OOM, §4).
7. **`claude plugin validate` warning:** marketplace.json without a description passes with a warning; added `metadata.description`, then it passes clean.

#### Plugin install and namespacing (checked, not assumed)
- Layout: `rag-provider-tools/.claude-plugin/{plugin.json,marketplace.json}`, `skills/add-llm-provider/SKILL.md`, `README.md`. `plugin.json` `repository` is a plain URL string. `marketplace.json` uses `"source": "./"` (the marketplace and plugin are the same directory). No `commands/` directory (optional; not needed).
- Install: `claude plugin marketplace add ./rag-provider-tools` then `claude plugin install rag-provider-tools@rag-project-tools`. Default scope is **user**: this wrote the marketplace and plugin into the user-level Claude settings, a global side effect. Undo with `claude plugin uninstall rag-provider-tools@rag-project-tools` and `claude plugin marketplace remove rag-project-tools`.
- Namespacing confirmed from a fresh session's init message: the plugin skill is `rag-provider-tools:add-llm-provider`, distinct from the loose `add-llm-provider`; both appear together in `slash_commands` when both exist. For the plugin re-run the loose skill was moved out of `.claude/skills/` so only the plugin could serve it.
- With a directory-source marketplace and `source: "./"`, the installed plugin's `path` is the live source directory, not a cached copy, so it tracks edits on disk. Bumping the version to 0.1.1 and running `plugin marketplace update` + `plugin update` reported "updated from 0.1.0 to 0.1.1 ... Restart to apply"; a fresh session then loaded 0.1.1.
- Drift risk: the skill exists twice (`.claude/skills/add-llm-provider/SKILL.md` and `rag-provider-tools/skills/add-llm-provider/SKILL.md`), currently byte-identical; symlinks are unreliable on Windows, so this is manual.

#### Verification
- **Run 1 (loose skill, fresh headless session):** wrote `src/openai_provider.py`, registered it, updated the allowlist, added `openai` to `requirements.txt`. After the manual finish: 12/12 tests; real app via `POST /stream` against a local fake OpenAI-compatible SSE server (`OPENAI_BASE_URL`) on a throwaway DB migrated with `alembic upgrade head`: incremental deltas, `query_logs` row, unknown provider returns 400, the server saw `Bearer` dummy key, `gpt-4o-mini`, `stream: true`. A direct call to api.openai.com with a dummy key returned the expected 401 `AuthenticationError`.
- **Run 2 (installed plugin, loose skill moved aside, same task from a clean baseline):** produced equivalent code, 3 new tests, docs including this section's parent; live check against the fake server (deltas ~0.5 s apart, `query_logs` row written, 400 on unknown); 12/12 tests, re-run independently afterwards. The two runs differed in small ways (run 1 `os.environ.get("OPENAI_API_KEY")`, run 2 `os.environ["OPENAI_API_KEY"]`); both are lazy, so a missing key only fails when `provider=openai` is selected.
- **Not verified:** any call that succeeded against the real OpenAI service (unverified due to pricing concerns; no paid account or key). The dummy-key 401 check was done in run 1 only; run 2 skipped it as ambiguous. Langfuse traces for the OpenAI provider. Whether a skill added to an existing `.claude/skills/` hot-reloads. `openai` 3.16.2 lists `httpx2` among its requirements (confirmed with `pip show`); not investigated, but it matters for unpinned installs.

## 16. Azure AI Foundry as an alternate LLM provider (September 2026)

### Decision (each choice confirmed with the project owner before any code)
Added `provider=azure` alongside the existing values, chosen per request by the caller (no env toggle, no auto-fallback), same pattern as Bedrock. Registered next to `groq`, `bedrock_invoke`, `bedrock_converse` and `openai`, so it is the fifth registered value, not the third.
- **Model: Azure OpenAI gpt-4.1-mini (Global Standard deployment, deployment name `gpt-4.1-mini`).** First-party, Azure-billed, small, so verification spends little of the $200 / 30-day credit. The original pick, gpt-4o-mini, was changed after the Foundry catalog page showed it as Deprecated (banner: April 14, 2027; lifecycle field: Deprecated); the owner chose gpt-4.1-mini instead. Its Direct-from-Azure badge, non-deprecated status and pricing still need confirming on its own Details page. Partner models (Claude, Llama, Mistral via the Foundry catalog) were explicitly rejected for the first live call because they can be billed through the Marketplace rather than against free credits.
- **SDK: `openai` package pointed at Azure's v1 route (`AsyncOpenAI`, `base_url=<resource>/openai/v1/`), no framework wrapper.** Same "raw SDK" position as boto3 over LangChain for Bedrock. Natively async (no `to_thread`, no `async_bridge`), already installed, chunk shape matches Groq's. First version used `AsyncAzureOpenAI` with the legacy `/openai/deployments/<name>/...?api-version=` route and a guessed default `2024-10-21`; replaced (owner's decision, after the portal's own snippet for the deployment showed the v1 route with a plain `OpenAI` client and no api-version) to remove the unverifiable api-version guess. Trade-off: Azure is now structurally close to the `openai` provider; `azure-ai-inference` (model-agnostic) was the other alternative and was not chosen. Chat completions (not the Responses API the portal snippet uses) keeps the streaming shape identical to Groq/OpenAI; that chat completions works on the v1 route for this deployment is an assumption until the live call. Contrast with Bedrock: boto3 is blocking, so it needs the thread bridge; this one does not.
- **Auth: API key + endpoint from env vars** (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, plus `AZURE_OPENAI_DEPLOYMENT`, default `gpt-4.1-mini`). Entra ID (`DefaultAzureCredential`) was the alternative; rejected because it needs role assignments and more setup inside a 30-day window. The client is built lazily, so the app still imports and other providers still work with no Azure variables set.
- **Azure specifics:** `model=` in the request is the *deployment name*, not the base model name. Azure's stream can include chunks whose `choices` list is empty (content-filter results); the provider skips them (a `chunk.choices[0]` on those would raise `IndexError`). `AZURE_OPENAI_ENDPOINT` may be the resource root (`...services.ai.azure.com` or `...openai.azure.com`) or a URL already ending `/openai/v1`; the code normalizes it.
- **Resource mismatch caught from screenshots:** the first credentials screenshot showed resource `...-4841-resource`, the gpt-4.1-mini deployment page showed `...-4286-resource`. The key and endpoint must come from the same resource as the deployment.
- Module: `src/azure_foundry.py`, registered in `src/generate.py::_PROVIDERS` and the `/stream` allowlist in `src/app.py`. Routes, MCP tool and other providers unchanged.

### Credit / billing verification (a real cost risk; status: owner confirmed the deployment is covered by the free $200 credit; budget alert / low quota not separately confirmed)
This is a cost gate, not a technical detail. Nothing in this repo can check it: there is no `az` CLI and no Azure variables on this machine, and the credit balance and model billing path are only visible in the Azure portal. Required before the first live call, by the account owner:
1. Confirm the credit is active and see the balance (Cost Management + Billing, credits/subscription view) and that the subscription is the one holding the deployment.
2. Confirm the gpt-4.1-mini deployment is a native Azure OpenAI deployment billed by Microsoft, not a partner/Marketplace offer. Check the model's pricing/billing information in the Foundry model catalog or deployment details.
3. Confirm Azure OpenAI/Foundry models are actually eligible to draw from the credit on this account (not assumed here).
4. Set a budget alert and keep the deployment's tokens-per-minute quota low, so a mistake cannot burn the credit.
Only after all four are confirmed: one short `POST /stream` with `provider=azure`. The 30-day window starts at signup, so this is time-boxed.

### Real bugs found and fixed (verified, not assumed)
None in application logic. Process notes: my scripted edit of `src/generate.py` failed twice (CRLF line endings on disk broke a multi-line match, then a wrong docstring pattern); both times the assert fired before anything was written, so no partial edit landed. The first Azure test failed on `NameError: _fake_openai_client` because the committed OpenAI tests use inline helpers, not shared ones; the Azure tests now carry their own helpers.

### Verification status
- **Verified:** 16/16 tests pass (4 new: deployment name passed as `model` and empty-`choices` chunks skipped, lazy client needs key and endpoint and builds the `/openai/v1/` base URL, base-URL normalization, `/stream` dispatch + `query_logs` row). Real app via `POST /stream` against a local fake server: the server saw `/openai/v1/chat/completions`, `Authorization: Bearer <key>`, `model: gpt-4.1-mini`, `stream: true`; output was incremental and a `query_logs` row was written.
- **Confirmed live (2026-09-19):** after fixing `.env` to use the key and endpoint of the resource that holds the deployment, one real `POST /stream` with `provider=azure` through the real app returned a correct answer grounded in the uploaded document (first byte 0.21 s, total 2.5 s) and wrote a `query_logs` row. **Not verified:** incremental token arrival on the real service (single short answer; incremental delivery only shown against the fake server), long answers, Langfuse traces for this provider.

### Live call attempt 1 (reached Azure, rejected: 401)
Billing was confirmed by the owner ("under the free $200") before the call; the budget alert and low tokens-per-minute quota were not separately confirmed. One real `POST /stream` with `provider=azure` was made through the real app (throwaway DB) using the owner's `.env`, whose endpoint was `https://<account>-4841-resource.openai.azure.com/openai/v1` and deployment `gpt-4.1-mini`. Azure answered `401: Access denied due to invalid subscription key or wrong API endpoint`. This proves the request reached the real service, but no completion was returned. Most likely cause (not yet confirmed): the deployment page screenshot showed resource `...-4286-resource` while `.env` points at `...-4841-resource`, so key and endpoint are not from the resource that holds the deployment. Side effects seen, consistent with existing behavior for every provider: the provider error surfaced after `StreamingResponse` had started, so the client received HTTP 200 with an empty body, and no `query_logs` row was written (logging only follows a successful generation). Status: blocked on the owner copying the key and endpoint from the same resource/page as the gpt-4.1-mini deployment; the 30-day credit clock is running.

### Live call attempt 2 (success)
After the owner replaced the key and endpoint with those of resource `...-4286-resource` (the one holding the `gpt-4.1-mini` deployment), the same call succeeded: HTTP 200, a correct one-line answer, `query_logs` row written. Confirms the attempt-1 diagnosis: the 401 was a key/endpoint-from-the-wrong-resource mismatch, not a code problem. Root cause of the earlier failure was configuration, so nothing in `src/azure_foundry.py` changed between the two attempts.

## 17. Second OOM crash on upload: fastembed's default batch size (September 2026)

### Symptom
Render's event log showed `Ran out of memory (used over 512MB)` on every document upload, after §11's torch→fastembed swap had already fixed the *first* OOM crash loop. This is a distinct bug: it happens inside fastembed itself, not because torch got reintroduced (`requirements.txt` was checked — no torch, no sentence-transformers).

### Diagnosis (measured, not assumed)
`ingest.py::_FastEmbedWrapper.embed_documents()` called `TextEmbedding.embed(texts)` with no `batch_size`, which defaults to 256 — every chunk of an uploaded PDF gets padded to the longest sequence in the batch and run through the ONNX model in one forward pass. Memory scales with `batch_size * seq_len^2` (attention), so peak memory grows with document size, not just with the model's own footprint.
Reproduced locally, real venv, real 387KB sample PDF (71 chunks), measured with `psutil` (RSS):
- Model loaded, before embedding: ~223MB
- After `embed_documents()` at the default `batch_size=256` (i.e. all 71 chunks in one batch): ~669MB
- After `embed_documents()` at `batch_size=8`: ~283MB
Ruled out first, both measured and found to make no difference: onnxruntime thread count (`threads=1` vs default), and the CPU memory arena (`enable_cpu_mem_arena=False` + `arena_extend_strategy=kSameAsRequested`). The batch size of the single `.embed()` call is the actual lever.

### Fix
`embed_documents()` now passes an explicit `batch_size` (env-overridable `EMBED_BATCH_SIZE`, default 8) into fastembed's `.embed()`. `embed_query()` is untouched — it always embeds exactly one string, so batch size doesn't apply.

### Real bugs found and fixed (verified, not assumed)
Confirmed batch_size has no effect on the embedding values themselves: embedding the same 20 texts at `batch_size=256` and `batch_size=8` produced bit-identical vectors (max abs diff `0.0`). So the fix only bounds peak memory; retrieval quality is unaffected.

### Verification status
- **Verified:** the memory measurement above, and the batch-size-invariance of embedding output, both against the real fastembed model in this project's venv.
- **Not verified:** the real pytest suite against this change (Docker Desktop was not running at fix time). No live upload against the hosted Render instance has been done since the fix — the Render OOM is not yet confirmed gone in production, only strongly diagnosed and fixed locally.

## 18. Golden-set eval harness + a real ivfflat retrieval bug it surfaced (September 2026)

### Decision
Built a 38-question golden set (`Golden_questions.json`, hand-verified against real chunk UUIDs pulled from the live Neon `chunks` table for the two uploaded AR-CITIZEN paper versions) and a eval harness under `eval/` (new, dev-only — its one dependency, `deepeval`, lives in `requirements-eval.txt`, never in `requirements.txt`, so it never reaches the Docker image or Render). Chose **DeepEval over RAGAS**: RAGAS's LLM-judge integration is built around LangChain's `BaseChatModel`, which would force-add a LangChain dependency tree onto this project purely for eval plumbing, directly against this repo's own already-logged "raw provider SDKs, no LangChain wrapper" decision. DeepEval's `DeepEvalBaseLLM` is a plain ABC — `eval/azure_judge_llm.py` wraps the *already-verified-live* Azure client from `src/azure_foundry.py` directly, no new client code, no LangChain. Retrieval metrics (precision@3/recall@3/MRR against real `expected_chunk_ids`) are plain Python (`eval/retrieval_metrics.py`) — neither RAGAS's nor DeepEval's built-in context metrics do exact chunk-ID overlap against known ground truth, they do LLM-judged relevance, which is a different thing.

**Judge model, deliberately separate from the production model being evaluated:** Azure `gpt-4.1-mini` judges; Groq `openai/gpt-oss-20b` (via the real `src.generate.stream_answer(provider="groq")`) generates. Avoids a model grading its own answers.

**Judge prompt** (`eval/judge_metric.py`, module-level constants `JUDGE_SYSTEM_PROMPT` / `JUDGE_USER_TEMPLATE`, printable/inspectable verbatim, not a library-internal template): scores faithfulness (are the answer's claims supported by the *actual* retrieved, threshold-filtered context — not the golden chunk IDs) and relevance (does the answer address the question) in one combined call, returning strict JSON.

**Langfuse wiring:** not a standalone script. Uses the SDK's own `dataset.run_experiment(task=..., evaluators=[...])` (Langfuse 4.15.2) — creates the `golden-set-v1` dataset + items once, then every run is a first-class Langfuse dataset run with per-item scores (`precision_at_3`, `recall_at_3`, `mrr_at_3`, `faithfulness`, `relevance`) attached to the real trace.

### Real bug found and fixed (verified, not assumed): `ivfflat.probes` default silently truncates retrieval
Smoke-testing one golden question before the paid run, `retrieve_relevant_chunks(query, k=3)` returned **1** chunk, not 3. Traced through every layer (ORM query construct, asyncpg, raw psycopg2, `EXPLAIN`) before concluding the cause: pgvector's own `ivfflat.probes` default is **1**, and nothing in this codebase ever set it. The `chunks` table currently has 236 rows against the migration's `lists = 100` — ~2.4 rows per partition, severely over-partitioned for this size — so scanning only 1 of 100 partitions (`probes=1`) frequently misses the true nearest neighbors entirely.

Measured directly, forcing `enable_indexscan=off` (sequential scan, bypasses the index) vs. varying `probes` on the exact failing query/vector:
| `ivfflat.probes` | rows returned (LIMIT 10) |
|---|---|
| 1 (the actual production default) | 1 |
| 2 | 1 |
| 5 | 10 |
| 10 | 10 |

Checked across **all 38 real golden questions**, not just the one anecdote: at `probes=1`, **5/38 (13%) got fewer than the requested k=3 rows back, and 4 of those got zero rows** — meaning `generate.py`'s `SIMILARITY_THRESHOLD` guard would fire `NO_CONTEXT_ANSWER` for those regardless of whether the real answer existed in the DB. At `probes=10`: 0/38 truncated.

**Fix:** `src/database.py` now sets `ivfflat.probes` (env-overridable `IVFFLAT_PROBES`, default 10) on every new physical connection via the `"connect"` event on `engine.sync_engine`, using SQLAlchemy's asyncpg adapter's `dbapi_connection.run_async(...)` — the correct hook for running a real post-connect `SET` on an async driver. Tried `connect_args={"server_settings": {...}}` first (asyncpg's normal per-connection-setting mechanism); it does **not** work for extension-defined GUCs — raises `asyncpg.exceptions.UndefinedObjectError: unrecognized configuration parameter "ivfflat.probes"`, because custom GUCs aren't recognized in the startup packet before the extension loads. Verified the fix against the real production code path (`retrieve_relevant_chunks`) directly, not just the raw SQL repro.

Also additive, non-breaking: `retrieve.py`'s returned chunk dicts now include `"id"` (the chunk's own UUID) — needed for the golden set's `expected_chunk_ids` to mean anything; existing consumers (app.py's prompt builder, the MCP tool, `tests/test_rag_pipeline.py`) only ever read specific keys or check `"content" in results[0]`, so this doesn't break anything (checked, not assumed).

`lists = 100` itself remains oversized for the current row count and is not fixed here — that needs a new migration (drop/rebuild the index with a smaller `lists`), out of scope for this pass; `IVFFLAT_PROBES=10` is a value chosen from the measurement above, not benchmarked against a larger dataset, same "starting value" caveat as `SIMILARITY_THRESHOLD`.

### First real eval run — actual numbers, not tuned or cherry-picked
Run against the *fixed* system (probes bug fixed first, per explicit instruction, before spending eval tokens on numbers that would mostly measure an index bug rather than RAG quality). First attempt at `max_concurrency=3` completed only 26/38 items — 12 failed on Groq free-tier rate limits (8000 TPM, `openai/gpt-oss-20b`). Rerun at `max_concurrency=1` (sequential) completed 38/38, 0 failures. **Reported numbers below are from the complete 38/38 sequential run**, not the partial one.

**Aggregate scores (38 questions, 3 excluded from precision/recall/MRR by design — no ground-truth chunk):**
| Metric | Score |
|---|---|
| faithfulness (mean) | 0.945 |
| relevance (mean) | 1.000 |
| precision@3 (mean, n=35) | 0.143 |
| recall@3 (mean, n=35) | 0.386 |
| MRR@3 (mean, n=35) | 0.295 |

Langfuse dataset run: https://jp.cloud.langfuse.com/project/cmtyj56ul02b5ad0imca6cfld/datasets/cmujmrb1w00c8ad0efyya9ugf/runs/031210ef-56e0-4ccf-a5ef-7d65314a9ac1

**Finding 1 — precision@3/recall@3 understate real answer quality here, and it's traceable to a specific cause, not just "bad retrieval":** relevance is 1.000 and faithfulness is 0.945 even though recall@3 is only 0.386 (many questions retrieved zero of their single hand-picked "gold" chunk in the top 3, yet still got a fully faithful, relevant answer — e.g. q01 "What is AR-CITIZEN?": 0 precision/recall, 1.0 faithfulness/relevance, because a *different* chunk in the same document (or the near-duplicate other uploaded paper) restates the same fact). Root cause: this golden set's `expected_chunk_ids` name one specific chunk per fact, but 500-char/50-overlap chunking plus two near-duplicate uploaded document versions (see §-implied earlier discussion of `AR_CITIZEN_Paper.pdf` vs `ICICNS2026_PaperID539_AR_CITIZEN.pdf`) mean several chunks legitimately contain the same fact. Exact chunk-ID recall is a real, conservative, correctly-measured number — it is not a proxy for "was the retrieved context adequate."

**Finding 2 — a real cross-document identity-contamination bug, confirmed live (q21):** asked specifically about the ICICNS2026 submission's author title for Sonia Jenifer Rayen (correct answer: no title given), retrieval returned two chunks from the *other* uploaded document (`AR_CITIZEN_Paper.pdf`, which does give her "Dr.") alongside one ICICNS2026 chunk, and the model answered "Dr." — wrong for the document actually asked about. Faithfulness scored 0.0 (the judge correctly flagged the claim as present in *a* retrieved chunk but not resolving the actual question asked). This is the exact failure mode predicted when the golden set was built (`/stream` has no `doc_id` scoping, so k=3 draws from the whole store) — now measured, not just predicted.

**Finding 3 — a real pattern: when retrieval misses the right chunk, the production model sometimes fabricates plausible specifics instead of staying conservative (q06, q09, q16 — faithfulness 0.5, 0.5, 0.9):** in each case the correct chunk was outside the top-3 (0 recall), and rather than answering only from what was actually retrieved, the model added specific-sounding claims not present in the retrieved text (e.g. q16: invented a "before-and-after... measured by looking at how many duplicate tickets were correctly merged" methodology narrative around otherwise-correct percentages that were in context; q06/q09: invented specific mechanisms — "paint patterns, mounting hardware," "lets analysts... trace back to the underlying cause" — not stated in what was retrieved). This is a real, non-cherry-picked faithfulness risk correlated with retrieval misses, distinct from Finding 1 (which is a golden-set measurement artifact, not a model behavior problem).

### Verification status
- **Verified:** the `ivfflat.probes` bug and its fix, directly against the real production code path and across all 38 real questions (not a single anecdote). The full 38/38 real eval run, judge prompt shown verbatim, Langfuse dataset run live and linked above.
- **Not verified:** the real pytest suite against the `database.py`/`retrieve.py` changes in this section (Docker Desktop was not running at fix time — same gap as §17). No decision has been made yet on the `lists=100` re-migration; that's flagged, not fixed.

### Real bug the fix itself introduced, caught by CI (not local pytest)
Pushing the fix broke GitHub Actions CI immediately (job failed 3s after "Install dependencies" — consistent with failing on the very first DB interaction, not a real test). Root cause, traced from the code (GitHub's log API returned 403 without a token, no `gh` CLI available in this environment, so this was traced by reading the actual fixture code rather than the raw log text): `tests/conftest.py`'s `clean_db` fixture opens its *first* connection via this same `engine` specifically to run `CREATE EXTENSION IF NOT EXISTS vector` — but the new `"connect"` event fires *before* that SQL runs, on a brand-new CI database (the CI workflow runs no migration, per its own documented behavior) where the extension doesn't exist yet. `SET ivfflat.probes` on that connection raised `asyncpg.exceptions.UndefinedObjectError`, which propagated out of the `"connect"` event and killed the very first connection outright.

**Fix:** wrapped the `SET` in a try/except that silently skips it on any failure (`src/database.py::_try_set_ivfflat_probes`) — the extension gets created moments later by the fixture itself, and every subsequent pooled connection succeeds normally. Verified two things directly, not assumed: (1) a fake connection object that raises on `execute()` doesn't propagate past this wrapper, and (2) a real connection against the live Neon DB (extension already present there) still correctly reports `ivfflat.probes = 10` afterward — the fix doesn't silently defeat itself on the happy path.

**Not verified:** whether this actually fixes the CI run, since that needs a real push and a real Actions run to confirm — diagnosed and fixed from reading the code, not from the actual failing log text.
