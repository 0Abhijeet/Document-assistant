import os
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq

from src.retrieve import retrieve_relevant_chunks
from src.database import AsyncSessionLocal
from src.models import QueryLog
from src.bedrock import stream_answer_bedrock_invoke_model, stream_answer_bedrock_converse
from src.tracing import get_langfuse_client

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"

# bge-small-en-v1.5 cosine similarity for genuinely unrelated text still lands
# well above 0, so 0 is not a usable "no match" floor -- an unfiltered top-k
# will always return *something*, regardless of relevance. 0.35 is a starting
# cutoff (not benchmarked against a labeled relevance set -- same honest
# caveat as the ivfflat `lists=100` default elsewhere in this project),
# overridable per-deployment without a code change.
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.35"))

NO_CONTEXT_ANSWER = (
    "I couldn't find anything in the uploaded documents relevant to that question. "
    "Try rephrasing, or upload a document that covers this topic."
)


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
    generator of text deltas only. No retrieval, no DB logging, no tracing
    here -- stream_answer() below owns all three, identically regardless
    of provider."""
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

    Tracing: instrumented manually (not via @observe) for two reasons --
    this function is an async generator with no single return value, which
    @observe's automatic output-capture doesn't handle correctly for async
    generators specifically (a known, currently-open upstream bug); and
    manual instrumentation here, rather than inside retrieve.py, keeps the
    MCP tool's calls to retrieve_relevant_chunks (a separate process,
    mcp_server.py) completely untouched by Langfuse -- tracing is scoped
    to this HTTP-facing path only, not the shared service layer.

    Verified experimentally (not assumed): OTel's contextvar-based span
    context correctly survives across this generator's yield/resume
    cycles even under real async suspension between yields -- retrieval
    and generation spans land correctly nested under one shared trace,
    not scattered into separate traces.
    """
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r}. Must be one of {list(_PROVIDERS)}")

    langfuse = get_langfuse_client()

    with langfuse.start_as_current_observation(
        as_type="span", name="stream_answer", input={"question": question, "provider": provider}
    ) as root_span:

        with langfuse.start_as_current_observation(
            as_type="retriever", name="retrieve_relevant_chunks", input={"question": question}
        ) as retrieval_span:
            context_chunks = await retrieve_relevant_chunks(question)
            relevant_chunks = [c for c in context_chunks if c["similarity"] >= SIMILARITY_THRESHOLD]
            retrieval_span.update(output={
                "chunk_count": len(context_chunks),
                "relevant_chunk_count": len(relevant_chunks),
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "chunks": context_chunks,
            })

        if not relevant_chunks:
            # Zero chunks cleared the bar -- either nothing was retrieved at
            # all, or every candidate was too dissimilar to be real context.
            # Short-circuits before the provider call: no LLM round-trip on
            # context the model would otherwise have to either hallucinate
            # around or (best case) just be told is irrelevant anyway.
            full_answer = NO_CONTEXT_ANSWER
            root_span.update(output=full_answer)
            yield full_answer
        else:
            provider_stream = _PROVIDERS[provider](question, relevant_chunks)

            full_answer = ""
            with langfuse.start_as_current_observation(
                as_type="generation", name=f"{provider}-generation", input=question, model=provider,
            ) as generation:
                async for delta in provider_stream:
                    full_answer += delta
                    yield delta
                generation.update(output=full_answer)

            root_span.update(output=full_answer)

    await _log_query(question, full_answer)


async def _log_query(question: str, answer: str):
    async with AsyncSessionLocal() as db:
        try:
            db.add(QueryLog(question=question, answer=answer))
            await db.commit()
        except Exception:
            await db.rollback()