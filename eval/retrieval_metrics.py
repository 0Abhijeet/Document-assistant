"""Precision@k, recall@k, MRR against the golden set's expected_chunk_ids.
Plain Python -- no eval framework involved. Deliberately NOT delegated to
RAGAS/DeepEval's context_precision/context_recall, which score *LLM-judged
relevance* rather than exact chunk-ID overlap against known ground truth;
that's a different (and for this golden set, wrong) thing to measure."""
from dataclasses import dataclass


@dataclass
class RetrievalScore:
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    retrieved_ids: list[str]
    hit_ids: list[str]


def score_retrieval(retrieved_ids: list[str], expected_ids: list[str], k: int) -> RetrievalScore:
    """`retrieved_ids` must already be ranked (top-1 first)."""
    if not expected_ids:
        raise ValueError("score_retrieval requires a non-empty expected_ids list; "
                          "questions with no ground-truth chunk are scored separately.")

    top_k = retrieved_ids[:k]
    expected_set = set(expected_ids)
    hits = [cid for cid in top_k if cid in expected_set]

    precision_at_k = len(hits) / k
    recall_at_k = len(set(hits)) / len(expected_set)

    reciprocal_rank = 0.0
    for rank, cid in enumerate(top_k, start=1):
        if cid in expected_set:
            reciprocal_rank = 1.0 / rank
            break

    return RetrievalScore(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        retrieved_ids=top_k,
        hit_ids=hits,
    )
