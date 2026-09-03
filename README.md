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
## Tech stack
| Layer | Technology |
|---|---|
| API | FastAPI (async) |
| Vector storage | PostgreSQL + pgvector (via Neon; also deployed on AWS RDS — see below) |
| Embeddings | fastembed (BAAI/bge-small-en-v1.5, ONNX runtime — no torch dependency) |
| LLM | Groq (Llama-based, async client, streamed responses) |
| ORM / migrations | SQLAlchemy 2.0 (async) + asyncpg; Alembic (sync, unaffected) |
| Testing | pytest, pytest-asyncio, run against a real Postgres instance |
| CI/CD | GitHub Actions |
| Deployment | Docker → Render (production); Docker → AWS EC2 + RDS (infrastructure exercise) |
| Agent tooling | Model Context Protocol (MCP) — stdio transport, official `mcp` SDK |
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
## Known limitations
Said out loud on purpose — demonstrating I understand the tradeoffs matters more than pretending they don't exist:
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

Same app deployed separately to AWS EC2 + RDS to build hands-on cloud infrastructure experience (VPC networking, IAM, security groups, cross-provider DB migration). Full write-up, architecture, and debugging log: [`docs/PROJECT_DETAILS.md`]