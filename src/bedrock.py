import os
import json
import asyncio
import boto3

from src.async_bridge import sync_iter_to_async

_bedrock_client = None


def get_bedrock_client():
    """Lazy singleton -- boto3 client construction does a bit of config
    resolution work, not worth repeating per call."""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _bedrock_client


# Amazon Nova Micro -- AWS's own first-party model, no vendor Marketplace
# subscription gate (unlike Anthropic and OpenAI gpt-oss models on Bedrock,
# both of which hit an account-level Marketplace/use-case-approval wall on
# this account). Titan Text Lite (the original choice) has since reached
# end-of-life; Nova is Titan's official successor lineup. Cheapest
# text-only model in the Nova family. Override via env var without
# touching code.
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")


def _build_messages(question: str, context_chunks: list[dict]) -> list[dict]:
    context_text = "\n\n".join(c["content"] for c in context_chunks)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the context doesn't contain the answer, say so — don't make one up.

Context:
{context_text}

Question: {question}
Answer:"""
    return [{"role": "user", "content": [{"text": prompt}]}]


async def stream_answer_bedrock_invoke_model(question: str, context_chunks: list[dict]):
    """
    Approach 1: invoke_model_with_response_stream -- the older, per-model-
    family API. Notably, for Nova specifically, the request body
    ({"messages": [...], "inferenceConfig": {...}}) and the streamed event
    shape (contentBlockDelta) are nearly identical to converse_stream's --
    Nova's native "Messages API" was designed to mirror Converse from the
    start. That's NOT true for every model family (Titan and Claude both
    use structurally different invoke_model bodies), so this convergence
    is Nova-specific, not a general property of the invoke_model API.

    Yields text deltas only -- no retrieval, no DB logging. generate.py
    owns both, identically regardless of which provider is active.
    """
    messages = _build_messages(question, context_chunks)
    client = get_bedrock_client()

    body = json.dumps({
        "messages": messages,
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.7, "topP": 0.9},
    })

    response = await asyncio.to_thread(
        client.invoke_model_with_response_stream,
        modelId=BEDROCK_MODEL_ID,
        body=body,
    )

    async for event in sync_iter_to_async(response["body"]):
        chunk = json.loads(event["chunk"]["bytes"])
        if "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"].get("text", "")
            if delta:
                yield delta


async def stream_answer_bedrock_converse(question: str, context_chunks: list[dict]):
    """
    Approach 2: converse_stream -- the unified API. Same request/response
    shape regardless of model family; swapping BEDROCK_MODEL_ID to a
    Claude or Llama model needs zero changes to this function.
    """
    messages = _build_messages(question, context_chunks)
    client = get_bedrock_client()

    response = await asyncio.to_thread(
        client.converse_stream,
        modelId=BEDROCK_MODEL_ID,
        messages=messages,
        inferenceConfig={"maxTokens": 1024},
    )

    async for event in sync_iter_to_async(response["stream"]):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"].get("text", "")
            if delta:
                yield delta