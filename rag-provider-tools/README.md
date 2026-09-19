# rag-provider-tools

Claude Code plugin with one skill, `add-llm-provider`, for the rag-project repo (FastAPI + pgvector RAG app). It walks through adding a new value for the `provider` parameter of `stream_answer()` / `POST /stream`: the async-generator provider contract, registration in `src/generate.py` and the allowlist in `src/app.py`, env vars, tests, and docs.

The skill contains repo-specific paths (`src/generate.py`, `src/bedrock.py`, `tests/test_rag_pipeline.py`, `CLAUDE.md`), so it is only useful inside this repo.

## Install (local)

```
claude plugin marketplace add ./rag-provider-tools
claude plugin install rag-provider-tools@rag-project-tools
```

Restart Claude Code (or run `/reload-plugins`) so the skill is picked up. Invoke it as `/rag-provider-tools:add-llm-provider <provider and details>`; Claude also triggers it automatically when you ask to add an LLM provider.

## Layout

```
.claude-plugin/plugin.json        plugin manifest
.claude-plugin/marketplace.json   single-plugin marketplace (source "./")
skills/add-llm-provider/SKILL.md  the skill
```

`.claude/skills/add-llm-provider/SKILL.md` in the repo root is a copy of the same file. Keep them identical when editing.
