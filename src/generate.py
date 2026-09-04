import os
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq

from src.retrieve import retrieve_relevant_chunks
from src.database import AsyncSessionLocal
from src.models import QueryLog
from src.bedrock import stream_answer_bedrock_invoke_model, stream_answer_bedrock_converse

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"


def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n".join(c["content"] for c in context_chunks)
    return f"""You are a helpful assistant. Use the following context to answer the question.
If the context doesn't contain the answer, say so — don't make one up.

Context:
{context_text}

Question: {question}
Answer:"""


async def _stream_groq(question: str, context_chunks: list[dict]):
    """Same interface as the Bedrock functions in src/bedrock.py: an async
    generator of text deltas only. No retrieval, no DB logging here --
    stream_answer() below owns both, identically regardless of provider."""
    prompt = _build_prompt(question, context_chunks)

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


_PROVIDERS = {
    "groq": _stream_groq,
    "bedrock_invoke": stream_answer_bedrock_invoke_model,
    "bedrock_converse": stream_answer_bedrock_converse,
}


async def stream_answer(question: str, provider: str = "groq"):
    """
    Async generator -- `async def` + `yield` inside. Call it without
    `await` (same as the old sync generator); FastAPI's StreamingResponse
    accepts async generators natively.

    provider: "groq" (default), "bedrock_invoke", or "bedrock_converse".
    Retrieval and DB logging happen here, once, regardless of which
    provider is selected -- each provider function only knows how to talk
    to its own API and yields text deltas.
    """
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r}. Must be one of {list(_PROVIDERS)}")

    context_chunks = await retrieve_relevant_chunks(question)
    provider_stream = _PROVIDERS[provider](question, context_chunks)

    full_answer = ""
    async for delta in provider_stream:
        full_answer += delta
        yield delta

    await _log_query(question, full_answer)


async def _log_query(question: str, answer: str):
    async with AsyncSessionLocal() as db:
        try:
            db.add(QueryLog(question=question, answer=answer))
            await db.commit()
        except Exception:
            db.rollback()