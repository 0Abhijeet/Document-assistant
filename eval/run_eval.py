"""Golden-set eval runner. Wires retrieval metrics (precision@k/recall@k/MRR,
plain Python) and an LLM-as-judge (faithfulness + relevance, DeepEval custom
metric, Azure gpt-4.1-mini) into a Langfuse dataset + experiment run, using
the real, live pipeline (src.generate.stream_answer, provider="groq" --
the app's default production path).

Usage (from repo root, with DATABASE_URL/GROQ_API_KEY/Azure+Langfuse env vars
set -- this hits the real Neon DB and real APIs):
    python -m eval.run_eval
"""
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

from src.retrieve import retrieve_relevant_chunks
from src.generate import stream_answer, SIMILARITY_THRESHOLD
from src.tracing import get_langfuse_client

from eval.retrieval_metrics import score_retrieval
from eval.judge_metric import FaithfulnessRelevanceMetric
from eval.azure_judge_llm import AzureJudgeLLM
from deepeval.test_case import LLMTestCase

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "Golden_questions.json"
DATASET_NAME = "golden-set-v1"
K = 3  # matches retrieve_relevant_chunks's production default in generate.py

_judge_model = AzureJudgeLLM()
_judge_metric = FaithfulnessRelevanceMetric(model=_judge_model)


def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def ensure_dataset(langfuse: Langfuse, questions: list[dict]) -> None:
    """Idempotent: creates the dataset and items once; safe to re-run --
    create_dataset_item with an explicit `id` upserts rather than duplicating."""
    try:
        langfuse.create_dataset(
            name=DATASET_NAME,
            description="AR-CITIZEN RAG golden set: 38 questions over the two "
                         "uploaded AR-CITIZEN paper versions, hand-verified "
                         "against real chunk IDs in the Neon DB.",
        )
    except Exception as e:
        # Already exists on a re-run -- fine, not fatal.
        print(f"[dataset] create_dataset: {e!r} (likely already exists, continuing)")

    for q in questions:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=q["id"],
            input=q["query"],
            expected_output=q["expected_answer"],
            metadata={
                "expected_chunk_ids": q["expected_chunk_ids"],
                "source_document": q["source_document"],
                "difficulty": q["difficulty"],
                "notes": q.get("notes", ""),
            },
        )


async def task(*, item, **kwargs):
    """Runs the real production pipeline for one golden question:
    1) raw top-K retrieval (for retrieval metrics, against expected_chunk_ids)
    2) the real stream_answer(provider="groq") call (production default)
    Returns everything the evaluators need."""
    query = item.input

    context_chunks = await retrieve_relevant_chunks(query, k=K)
    retrieved_ids = [c["id"] for c in context_chunks]
    relevant_chunks = [c for c in context_chunks if c["similarity"] >= SIMILARITY_THRESHOLD]

    full_answer = ""
    async for delta in stream_answer(query, provider="groq"):
        full_answer += delta

    return {
        "answer": full_answer,
        "retrieved_ids": retrieved_ids,
        "relevant_context": [c["content"] for c in relevant_chunks],
        "relevant_chunk_count": len(relevant_chunks),
    }


def retrieval_evaluator(*, input, output, expected_output, metadata, **kwargs):
    expected_ids = (metadata or {}).get("expected_chunk_ids") or []
    if not expected_ids:
        return {
            "name": "retrieval_no_ground_truth",
            "value": True,
            "comment": "No expected_chunk_ids for this question (trap/no-context question) -- "
                       "excluded from precision/recall/MRR aggregates by design.",
        }

    result = score_retrieval(output["retrieved_ids"], expected_ids, k=K)
    return [
        {"name": "precision_at_3", "value": result.precision_at_k},
        {"name": "recall_at_3", "value": result.recall_at_k},
        {"name": "mrr_at_3", "value": result.reciprocal_rank},
    ]


async def judge_evaluator(*, input, output, expected_output, metadata, **kwargs):
    test_case = LLMTestCase(
        input=input,
        actual_output=output["answer"],
        retrieval_context=output["relevant_context"],
    )
    await _judge_metric.a_measure(test_case)
    breakdown = _judge_metric.score_breakdown
    return [
        {
            "name": "faithfulness",
            "value": breakdown["faithfulness_score"],
            "comment": f"unsupported_claims={breakdown['unsupported_claims']!r}"
                       if breakdown["unsupported_claims"] else "fully supported",
        },
        {
            "name": "relevance",
            "value": breakdown["relevance_score"],
            "comment": breakdown["relevance_reasoning"],
        },
    ]


async def main():
    langfuse = get_langfuse_client()
    questions = load_golden_set()
    print(f"Loaded {len(questions)} golden questions from {GOLDEN_SET_PATH}")

    ensure_dataset(langfuse, questions)
    langfuse.flush()

    dataset = langfuse.get_dataset(DATASET_NAME)
    print(f"Dataset '{DATASET_NAME}' has {len(dataset.items)} items on Langfuse.")

    result = dataset.run_experiment(
        name="groq-v1 real run",
        description="First real run: retrieval (precision@3/recall@3/mrr@3, plain "
                     "Python against real chunk IDs) + LLM-judge (faithfulness/"
                     "relevance, Azure gpt-4.1-mini judging Groq gpt-oss-20b "
                     "production answers).",
        task=task,
        evaluators=[retrieval_evaluator, judge_evaluator],
        # Sequential: Groq's free tier caps at 8000 TPM. A first attempt at
        # max_concurrency=3 hit 429s on 12/38 items (bursts of concurrent
        # ~700-1100 token requests blew the per-minute budget). Sequential
        # keeps each request's cost isolated in time; slower, but this is a
        # one-off diagnostic run, not a load test.
        max_concurrency=1,
    )

    print(result.format(include_item_results=True))
    if result.dataset_run_url:
        print(f"\nLangfuse dataset run: {result.dataset_run_url}")

    out_path = Path(__file__).resolve().parent / "last_run_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_run_id": result.dataset_run_id,
                "dataset_run_url": result.dataset_run_url,
                "items": [
                    {
                        "id": ir.item.id if hasattr(ir.item, "id") else None,
                        "input": ir.item.input if hasattr(ir.item, "input") else ir.item.get("input"),
                        "output": ir.output,
                        "evaluations": [
                            {"name": e.name, "value": e.value, "comment": e.comment}
                            for e in ir.evaluations
                        ],
                    }
                    for ir in result.item_results
                ],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nFull results written to {out_path}")

    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
