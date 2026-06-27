"""
eval/metrics.py — NDCG@k, MAP, P@k implementations.

These match the hackathon scoring formula:
  Composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10

Graded relevance: 0–5 (tier 0 = honeypot/irrelevant, tier 5 = perfect fit).
"Relevant" = tier ≥ 3 (used for MAP and P@k binary decisions).
"""

import numpy as np
from typing import Optional


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Standard DCG@k with log2 discounting."""
    rel = np.array(relevances[:k], dtype=float)
    if len(rel) == 0:
        return 0.0
    positions = np.arange(1, len(rel) + 1)
    gains = (2.0 ** rel - 1.0) / np.log2(positions + 1)
    return float(gains.sum())


def ndcg_at_k(
    ranked_relevances: list[float],
    k: int,
    ideal_relevances: Optional[list[float]] = None,
) -> float:
    """
    NDCG@k.

    Parameters
    ----------
    ranked_relevances : relevance scores in rank order (rank 1 first)
    k                 : cutoff
    ideal_relevances  : sorted-descending ideal relevances; if None, derived
                        from ranked_relevances
    """
    if ideal_relevances is None:
        ideal_relevances = sorted(ranked_relevances, reverse=True)

    dcg   = dcg_at_k(ranked_relevances, k)
    idcg  = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked_relevances: list[float], threshold: float = 3.0) -> float:
    """
    MAP (Average Precision) for a single query.
    "Relevant" = relevance >= threshold.
    """
    hits = 0
    precision_sum = 0.0
    for i, rel in enumerate(ranked_relevances, 1):
        if rel >= threshold:
            hits += 1
            precision_sum += hits / i
    n_relevant = sum(1 for r in ranked_relevances if r >= threshold)
    return precision_sum / n_relevant if n_relevant > 0 else 0.0


def precision_at_k(ranked_relevances: list[float], k: int, threshold: float = 3.0) -> float:
    """P@k: fraction of top-k candidates that are relevant (relevance >= threshold)."""
    top_k = ranked_relevances[:k]
    return sum(1 for r in top_k if r >= threshold) / k if k > 0 else 0.0


def composite_score(
    ranked_relevances: list[float],
    ideal_relevances: Optional[list[float]] = None,
) -> dict:
    """
    Compute the hackathon composite score plus individual components.

    Composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
    """
    ndcg10 = ndcg_at_k(ranked_relevances, 10,  ideal_relevances)
    ndcg50 = ndcg_at_k(ranked_relevances, 50,  ideal_relevances)
    ap     = average_precision(ranked_relevances)
    p10    = precision_at_k(ranked_relevances, 10)

    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * ap + 0.05 * p10

    return {
        "composite": composite,
        "ndcg@10":   ndcg10,
        "ndcg@50":   ndcg50,
        "MAP":       ap,
        "P@10":      p10,
    }


if __name__ == "__main__":
    # Sanity check: perfect ranking should score 1.0 on all metrics
    perfect = [5, 5, 4, 4, 3, 3, 3, 2, 2, 1] * 10
    result = composite_score(perfect)
    print("Perfect ranking (graded):", result)

    # Baseline (random relevances)
    import random
    random.seed(42)
    random_rels = [random.choice([0, 0, 0, 1, 2, 3, 4, 5]) for _ in range(100)]
    result2 = composite_score(random_rels)
    print("Random ranking:", result2)

    # Keyword-stuffer failure (top-10 are all tier-0 honeypots)
    bad = [0] * 10 + [5] * 10 + [0] * 80
    ideal = sorted(bad, reverse=True)
    result3 = composite_score(bad, ideal)
    print("Keyword-stuffer failure (top 10 = tier 0):", result3)
