import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

from src.retrieve import retrieve_relevant_chunks
from src.database import SessionLocal
from src.models import QueryLog

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"


def stream_answer(question: str):
    context_chunks = retrieve_relevant_chunks(question)
    context_text = "\n\n".join(c["content"] for c in context_chunks)



    prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the context doesn't contain the answer, say so — don't make one up.

Context:
{context_text}

Question: {question}
Answer:"""

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_answer += delta
            yield delta

    _log_query(question, full_answer)


def _log_query(question: str, answer: str):
    db = SessionLocal()
    try:
        db.add(QueryLog(question=question, answer=answer))
        db.commit()
    except Exception:
        # A logging failure should never break the user-facing answer that already streamed.
        db.rollback()
    finally:
        db.close()