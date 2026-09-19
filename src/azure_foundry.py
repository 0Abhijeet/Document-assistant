"""Azure AI Foundry provider via Azure's OpenAI-compatible v1 route. SDK style: native async client (openai.AsyncOpenAI pointed at <resource>/openai/v1/), no threads needed."""
import os

from openai import AsyncOpenAI

_azure_client = None


def _v1_base_url(endpoint: str) -> str:
    """Accepts the resource root (https://<resource>.services.ai.azure.com or
    ...openai.azure.com) or a URL already ending in /openai/v1."""
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/openai/v1"):
        endpoint += "/openai/v1"
    return endpoint + "/"


def get_azure_client():
    """Lazy singleton -- missing Azure env vars fail only when this provider
    is selected, not at app import."""
    global _azure_client
    if _azure_client is None:
        _azure_client = AsyncOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url=_v1_base_url(os.environ["AZURE_OPENAI_ENDPOINT"]),
        )
    return _azure_client


# On Azure, `model=` in the request is the *deployment name* chosen in
# Foundry, not the base model name.
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")


async def stream_answer_azure(question: str, context_chunks: list[dict]):
    """Async generator of text deltas only. No retrieval, DB logging, or
    tracing here -- generate.stream_answer() owns those."""
    # Imported here: generate.py imports this module at load (circular otherwise).
    from src.generate import _build_prompt

    stream = await get_azure_client().chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": _build_prompt(question, context_chunks)}],
        stream=True,
    )
    async for chunk in stream:
        # Azure can emit chunks with empty `choices` (content-filter results).
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
