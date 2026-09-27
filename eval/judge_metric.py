"""Custom DeepEval metric: faithfulness + relevance in one judge call, against
the *actual* retrieved/threshold-filtered context the live system used -- not
the golden set's expected_chunk_ids. Prompt text below is exactly what is sent
to the judge model; nothing else is added by DeepEval internals for this
metric (no auto-generated criteria, no template substitution beyond the three
{question}/{context}/{answer} slots)."""
import json
import re

from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase

# ---- Exact judge prompt (shown verbatim in the eval writeup) ----

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a retrieval-augmented generation (RAG) system. You will be given a user's question, the exact context chunks that were retrieved and given to the answering model, and the answer the model produced. Score the answer on two dimensions. Do not use any outside knowledge -- you are not judging factual correctness against the real world, only whether the answer is properly grounded in and responsive to what was given.

FAITHFULNESS: For every factual claim in the answer, decide whether it is directly supported by the retrieved context. An answer that adds information not present in the context (even if true in general) is unfaithful. An answer that correctly says the context does not contain the information is faithful.
Score faithfulness as a single number from 0.0 to 1.0: 1.0 = every claim is fully supported by the retrieved context; 0.0 = the answer is entirely unsupported or fabricated. List each unsupported claim you find, verbatim, in "unsupported_claims".

RELEVANCE: Judge whether the answer actually addresses the user's question, regardless of grounding. An answer can be faithful but off-topic (e.g. correctly quoting the context but not answering what was asked).
Score relevance as a single number from 0.0 to 1.0: 1.0 = directly and completely answers the question; 0.0 = does not address the question at all.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{
  "faithfulness_score": <float 0.0-1.0>,
  "unsupported_claims": [<string>, ...],
  "relevance_score": <float 0.0-1.0>,
  "relevance_reasoning": "<one sentence>"
}"""

JUDGE_USER_TEMPLATE = """QUESTION:
{question}

RETRIEVED CONTEXT (this is ALL the information the answering model had access to):
{context}

ANSWER TO EVALUATE:
{answer}

Score this answer now, following the rules in the system prompt exactly."""


def render_judge_prompt(question: str, context: str, answer: str) -> str:
    return JUDGE_SYSTEM_PROMPT + "\n\n" + JUDGE_USER_TEMPLATE.format(
        question=question, context=context, answer=answer
    )


def _parse_judge_json(raw: str) -> dict:
    # Judge is instructed to return only JSON; strip markdown fences defensively
    # in case the model wraps it anyway (observed behavior across providers).
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Judge response was not JSON: {raw!r}")
    return json.loads(match.group(0))


class FaithfulnessRelevanceMetric(BaseMetric):
    """One judge call scores both faithfulness and relevance. `score` (the
    BaseMetric-required field) is set to faithfulness_score; the full
    breakdown (both scores + unsupported claims + relevance reasoning) is in
    `score_breakdown`, which is what the eval runner actually reads."""

    def __init__(self, model, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
        self.async_mode = True

    @property
    def __name__(self):
        return "FaithfulnessRelevance"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raise NotImplementedError("Sync path unused -- eval harness is async throughout.")

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        context = "\n\n".join(test_case.retrieval_context or [])
        prompt = render_judge_prompt(
            question=test_case.input,
            context=context if context else "(no context was retrieved)",
            answer=test_case.actual_output,
        )
        raw = await self.model.a_generate(prompt)
        parsed = _parse_judge_json(raw)

        self.score_breakdown = {
            "faithfulness_score": float(parsed["faithfulness_score"]),
            "unsupported_claims": parsed.get("unsupported_claims", []),
            "relevance_score": float(parsed["relevance_score"]),
            "relevance_reasoning": parsed.get("relevance_reasoning", ""),
        }
        self.score = self.score_breakdown["faithfulness_score"]
        self.success = self.is_successful()
        self.reason = self.score_breakdown["relevance_reasoning"]
        return self.score

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold
