---
name: add-llm-provider
description: Use when adding a new LLM provider option (e.g. OpenAI, Gemini, Azure OpenAI) to this RAG app's model-serving abstraction, i.e. a new value for the `provider` parameter of `stream_answer()` / the `/stream` route. Covers the provider function contract, registration in generate.py and app.py, env vars, tests, and docs.
---

# Add an LLM provider

Providers are plain async generators registered in a dict. There is no base class. Adding one touches a small, fixed set of places; the failure mode is forgetting one of them (the provider allowlist exists twice).

The registered providers are whatever `_PROVIDERS` in `src/generate.py` says (`grep -n "_PROVIDERS" -A6 src/generate.py`); do not trust any list written in docs.

## How the abstraction works

- `src/generate.py::stream_answer(question, provider="groq")` retrieves chunks, filters by `SIMILARITY_THRESHOLD`, then calls `_PROVIDERS[provider](question, relevant_chunks)` and iterates the result with `async for`, wrapping it in a Langfuse `generation` span and accumulating the answer for `query_logs`.
- `src/app.py::stream_endpoint` validates `provider` against a hardcoded set literal before calling `stream_answer`. `stream_answer` validates again against `_PROVIDERS`. **Both must list the new name.**
- Providers never retrieve, log, or trace. `stream_answer` owns all three identically for every provider. Do not add any of them to a provider function.
- The no-context short-circuit happens before dispatch, so a provider is only ever called with a non-empty `context_chunks` list of dicts with (at least) a `content` key.

## The contract a provider function must satisfy

```python
async def stream_answer_<name>(question: str, context_chunks: list[dict]):
    ...  # async generator: `yield` non-empty str deltas only
```

1. **Async generator**, signature `(question, context_chunks)`. Callers iterate it with `async for`; it is never awaited. Register the function object, not a call.
2. **Yield only non-empty `str` text deltas.** Skip empty/None deltas (see `_stream_groq`: `if delta: yield delta`). No dicts, no SSE framing, no final "done" marker. The route streams these straight to `text/plain`.
3. **Use only `context_chunks[i]["content"]`** from the chunks.
4. **Build the prompt the same way as the others.** The prompt text is duplicated: `_build_prompt` in `src/generate.py` (Groq) and `_build_messages` in `src/bedrock.py`. New providers should reuse `generate._build_prompt`, imported *inside the provider function* (a top-level `from src.generate import ...` is circular, because `generate.py` imports the provider module at load). If you need a different message shape, copy the same wording instead. If you change the wording, change every copy.
5. **Never block the event loop.** Prefer the provider's native async SDK client and `await`/`async for` it. If the SDK is sync-only: run the call that opens the request with `asyncio.to_thread`, then iterate the returned blocking stream with `src/async_bridge.py::sync_iter_to_async`. Never `await asyncio.to_thread(list, stream)`, which drains the whole stream before yielding anything. (Build-log bug: blocking `async def` measured 814 ms vs 20-24 ms with `to_thread`.)
6. **Do not require credentials at import time.** `generate.py` currently does `AsyncGroq(api_key=os.environ["GROQ_API_KEY"])` at import; do not copy that for a new provider or the whole app fails to import when only some keys are set. Use a lazy module-level singleton like `get_bedrock_client()` in `src/bedrock.py`, so a missing key fails only when that provider is selected.
7. **Model ID must be overridable by env var with a default** (pattern: `BEDROCK_MODEL_ID = os.environ.get(..., "<default>")`). Hardcoded model IDs go stale (Groq and Bedrock Titan both had deprecations); say in a comment where the default came from only if non-obvious.
8. **Error handling, as it actually is today:** provider functions have no try/except. An exception mid-stream propagates out of `stream_answer` and the `StreamingResponse`; `_log_query` runs only after a successful generation, so failed calls are not logged. Do not silently swallow errors or yield error text as if it were an answer. If you want friendlier errors, that is a cross-provider change to `stream_answer`, not something to do in one provider; ask first.
9. **Keep it in its own module** when it needs its own SDK (`src/bedrock.py` precedent), and do not name the module after the SDK: `src/openai.py` would shadow the `openai` package, so use `src/openai_provider.py`. Import lazily-constructed clients only. Do not add tracing (`@observe` on async generators is broken upstream, langfuse#7226) or DB access.

## Steps

1. **Read first**: `src/generate.py`, `src/bedrock.py`, `src/async_bridge.py`, and the `/stream` route in `src/app.py`. Confirm the registered list and the contract above still hold; if they drifted, trust the code.
2. **Choose the SDK style**: native async client (preferred) vs sync SDK (needs `to_thread` + `sync_iter_to_async`). Note the choice in the module docstring in one line.
3. **Add the dependency** to `requirements.txt` (it is unpinned; append the bare package name), `pip install` it into `venv/`, and record the installed version.
4. **Write the provider function** in `src/<provider>.py` per the contract. Name it `stream_answer_<name>` / follow the existing naming.
5. **Register it** in `src/generate.py`: add the import and a `_PROVIDERS["<name>"]` entry, and update the `provider:` line in the `stream_answer` docstring.
6. **Update the allowlist in `src/app.py::stream_endpoint`** (the set literal). Grep for the previous provider names (`grep -rn "bedrock_converse" src tests`) to find any other spot listing them.
7. **Env vars**: document the key/model/base-url variables in `CLAUDE.md` (the "App" section lists optional vars). Do not write secrets into any file. `.env` is local-only; never print or commit its values.
8. **Tests** in `tests/test_rag_pipeline.py` (run with `python -m pytest`, real Postgres, see CLAUDE.md for the `DATABASE_URL` on port 15432):
   - Provider dispatch: `monkeypatch.setitem(generate._PROVIDERS, "<name>", fake_async_gen)`, ingest the sample PDF (the `sample_pdf` fixture), iterate `stream_answer(q, provider="<name>")` with `async for`, assert the concatenated deltas and that a `query_logs` row was written. The existing short-circuit tests show the pattern.
   - Delta parsing: feed the provider function a fake SDK stream (skip empty deltas, yield text only) and assert the output. Use `@pytest.mark.asyncio` (strict mode). Never use `TestClient`; use `httpx.AsyncClient` with `ASGITransport(app=app)`.
   - Route: unknown provider still returns 400; the new name is accepted (with the generator monkeypatched, no network).
   - Test embeddings must vary by index (constant vectors have cosine distance 0).
   - `test_upload_endpoint_accepts_pdf` leaves a stray `data/docs/*_sample.pdf`, and a manual `/upload` leaves another file there. `data/docs/` also holds TRACKED sample PDFs, some named `*_sample.pdf`, so never `rm data/docs/*_sample.pdf`. Remove only untracked strays: `git ls-files --others --exclude-standard data/docs`, delete exactly those, then confirm `git status` shows no ` D` lines.
9. **Live verification**: make one real call through `POST /stream` with `provider=<name>` against the running app, using a question that quotes a distinctive phrase (e.g. the title) from the uploaded PDF. Generic questions can score below `SIMILARITY_THRESHOLD` (0.35) and return `NO_CONTEXT_ANSWER` without ever calling the provider; that is not a provider bug, and you must not lower the threshold in code to get around it. Each attempt writes a `query_logs` row. If the API key or account access is unavailable, do NOT claim it works. Verify the wiring with a local fake server if the SDK supports a base-URL override (worked for OpenAI: a ~15-line stdlib `http.server` handler that replies `text/event-stream` with `data: {chat.completion.chunk JSON}` + a blank line as SSE events and `data: [DONE]`, plus a throwaway DB created with `CREATE DATABASE` in the compose db on port 15432 and `alembic upgrade head`; drop it afterwards). Optionally (ask the user if outbound calls are in doubt) make one call to the real endpoint with a dummy key to confirm you get the provider's auth error, which proves you reached the real service. State clearly that the real service is untested (Bedrock precedent: `CLAUDE.md` "Not implemented / not verified live").
10. **Docs**: update `CLAUDE.md` (a file-map row for the new module, tech-stack version line, the test count wherever it appears (`grep -n "tests" CLAUDE.md`), provider list in the "Running things" section, and the "Not implemented / not verified live" section if the live call was blocked), and add a section to `project_docs/PROJECT_DETAILS.md` in the existing "Decision / Real bugs found and fixed / Verification status" format. Do not edit `data/docs/PROJECT_DETAILS.md` (stale duplicate). README is partly stale; only touch it if you are also fixing what you changed.
11. **Run the full suite** (it drops every table in `DATABASE_URL`'s database, so do any live check against a different or throwaway DB first, or before the suite if re-migrating), delete the untracked stray PDF it leaves in `data/docs/` (same rule as above), and confirm it passes before reporting done. Only commit if the user asked.

## Do not

- Do not import `AsyncGroq`/the Groq client for anything but the Groq provider.
- Do not add provider-specific branches inside `stream_answer`; the dispatch dict is the seam.
- Do not touch `retrieve.py` (it also backs the MCP tool) or `mcp_server.py`.
- Do not run the tests against a Neon/RDS/dev `DATABASE_URL`; `clean_db` drops all tables.
