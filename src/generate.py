import os
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq

from src.retrieve import retrieve_relevant_chunks
from src.database import AsyncSessionLocal
from src.models import QueryLog

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"


async def stream_answer(question: str):
    """Async generator -- `async def` + `yield` inside. Call it without
    `await` (same as the old sync generator); FastAPI's StreamingResponse
    accepts async generators natively."""
    context_chunks = await retrieve_relevant_chunks(question)
    context_text = "\n\n".join(c["content"] for c in context_chunks)

    prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the context doesn't contain the answer, say so — don't make one up.

Context:
{context_text}

Question: {question}
Answer:"""

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_answer = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_answer += delta
            yield delta

    await _log_query(question, full_answer)


async def _log_query(question: str, answer: str):
    async with AsyncSessionLocal() as db:
        try:
            db.add(QueryLog(question=question, answer=answer))
            await db.commit()
        except Exception:
            # A logging failure should never break the user-facing answer that already streamed.
            await db.rollback()