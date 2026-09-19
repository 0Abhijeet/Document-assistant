"""OpenAI direct provider. SDK style: native async client (AsyncOpenAI), no threads needed."""
import os

from openai import AsyncOpenAI

_openai_client = None


def get_openai_client():
    """Lazy singleton -- a missing OPENAI_API_KEY fails only when this
    provider is selected, not at app import."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
    return _openai_client


OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


async def stream_answer_openai(question: str, context_chunks: list[dict]):
    """Async generator of text deltas only. No retrieval, DB logging, or
    tracing here -- generate.stream_answer() owns those."""
    # Imported here: generate.py imports this module at load (circular otherwise).
    from src.generate import _build_prompt

    stream = await get_openai_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": _build_prompt(question, context_chunks)}],
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
