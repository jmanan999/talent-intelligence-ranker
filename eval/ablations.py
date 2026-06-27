"""
eval/ablations.py — Component ablations and top-10 stability analysis.

Tests:
  1. Dense-only ranking vs. sparse-only vs. hybrid vs. hybrid+signals vs. full
  2. Top-10 stability: shuffle weights ±15%, check how many top-10 candidates change
  3. Honeypot rate in top-100 for each ablation variant

Run: python eval/ablations.py --artifacts ./artifacts --candidates ../candidates.jsonl
"""

import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.metrics import composite_score
from eval.pseudo_labels import compute_pseudo_labels
from src.consistency import compute_consistency_scores
from src.signals import compute_availability
from src.score import compute_base_score, compute_fit_score, sort_for_submission


def run_ablation(
    variant_name: str,
    df: pd.DataFrame,
    feat: pd.DataFrame,
    consistency: pd.Series,
    availability: pd.Series,
    pseudo_labels: pd.Series,
    weights: dict,
) -> dict:
    """Run one ablation variant and return metrics."""
    fit = compute_fit_score(df, feat, consistency, availability, weights)
    sorted_df = sort_for_submission(fit)
    top100_ids = sorted_df["candidate_id"].head(100).tolist()

    # Build ranked relevances from pseudo labels
    ranked_rels = [int(pseudo_labels.get(cid, 0)) for cid in top100_ids]
    all_rels = list(pseudo_labels.values)
    ideal_rels = sorted(all_rels, reverse=True)[:100]

    metrics = composite_score(ranked_rels, ideal_rels)

    # Honeypot rate (consistency < 0.20 in top-100)
    hp_in_top100 = sum(
        1 for cid in top100_ids
        if consistency.get(cid, 1.0) < 0.20
    )
    kw_stuffers_in_top100 = df.loc[top100_ids, "keyword_stuffer"].sum()

    return {
        "variant": variant_name,
        "composite": round(metrics["composite"], 4),
        "ndcg@10":  round(metrics["ndcg@10"], 4),
        "ndcg@50":  round(metrics["ndcg@50"], 4),
        "MAP":      round(metrics["MAP"], 4),
        "P@10":     round(metrics["P@10"], 4),
        "honeypots_in_top100": hp_in_top100,
        "kw_stuffers_in_top100": int(kw_stuffers_in_top100),
        "top10_ids": top100_ids[:10],
    }


def stability_analysis(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    consistency: pd.Series,
    availability: pd.Series,
    base_weights: dict,
    n_trials: int = 20,
    perturb: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Perturb weights ±perturb% on n_trials random seeds.
    Count how many top-10 candidates stay in top-10 across perturbations.
    """
    rng = np.random.default_rng(seed)

    # Baseline top-10
    fit0 = compute_fit_score(df, feat, consistency, availability, base_weights)
    base_top10 = set(sort_for_submission(fit0)["candidate_id"].head(10))

    stable_counts = {cid: 0 for cid in base_top10}

    for _ in range(n_trials):
        perturbed = {}
        for k, v in base_weights.items():
            delta = rng.uniform(-perturb, perturb)
            perturbed[k] = max(0.01, v * (1 + delta))
        # Renormalize
        total = sum(perturbed.values())
        perturbed = {k: v / total for k, v in perturbed.items()}

        fit_p = compute_fit_score(df, feat, consistency, availability, perturbed)
        top10_p = set(sort_for_submission(fit_p)["candidate_id"].head(10))

        for cid in base_top10:
            if cid in top10_p:
                stable_counts[cid] += 1

    stability_rate = np.mean(list(stable_counts.values())) / n_trials
    return {
        "n_trials": n_trials,
        "perturb_pct": perturb * 100,
        "avg_stability_rate": round(stability_rate, 3),
        "stable_counts": stable_counts,
        "base_top10": list(base_top10),
    }


def run_all_ablations(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    consistency: pd.Series,
    availability: pd.Series,
    pseudo_labels: pd.Series,
    base_weights: dict,
) -> list[dict]:
    """Run 5 ablation variants."""
    results = []

    # 1. Dense only
    dense_only = {k: 0.0 for k in base_weights}
    dense_only["role_fit_dense"] = 1.0
    # Availability = uniform (no discount)
    avail_uniform = pd.Series(
        np.ones(len(df), dtype=np.float32), index=df.index, name="availability"
    )
    cons_ones = pd.Series(np.ones(len(df), dtype=np.float32), index=df.index, name="consistency")
    results.append(run_ablation(
        "dense_only", df, feat, cons_ones, avail_uniform, pseudo_labels, dense_only
    ))

    # 2. Hybrid (dense + sparse), no signals, no consistency
    hybrid_w = {k: 0.0 for k in base_weights}
    hybrid_w["role_fit_dense"]  = 0.50
    hybrid_w["role_fit_sparse"] = 0.50
    results.append(run_ablation(
        "hybrid_no_signals", df, feat, cons_ones, avail_uniform, pseudo_labels, hybrid_w
    ))

    # 3. Hybrid + signals (no consistency guard)
    results.append(run_ablation(
        "hybrid_+signals", df, feat, cons_ones, availability, pseudo_labels, hybrid_w
    ))

    # 4. Full composite (no LTR)
    results.append(run_ablation(
        "full_composite", df, feat, consistency, availability, pseudo_labels, base_weights
    ))

    # 5. Full composite — what if we didn't penalize keyword stuffers?
    df_no_kw_pen = df.copy()
    df_no_kw_pen["keyword_stuffer"] = False
    results.append(run_ablation(
        "no_kw_stuffer_penalty", df_no_kw_pen, feat, consistency, availability, pseudo_labels, base_weights
    ))

    return results


def print_report(results: list[dict], stability: dict) -> None:
    print(f"\n{'='*90}")
    print(f"ABLATION RESULTS")
    print(f"{'='*90}")
    print(f"{'Variant':<30} {'Composite':>9} {'NDCG@10':>8} {'NDCG@50':>8} {'MAP':>6} {'P@10':>6} {'HP':>4} {'KW':>4}")
    print("-" * 90)
    for r in results:
        print(
            f"  {r['variant']:<28} {r['composite']:>9.4f} {r['ndcg@10']:>8.4f}"
            f" {r['ndcg@50']:>8.4f} {r['MAP']:>6.4f} {r['P@10']:>6.4f}"
            f" {r['honeypots_in_top100']:>4d} {r['kw_stuffers_in_top100']:>4d}"
        )

    print(f"\n{'='*90}")
    print(f"TOP-10 STABILITY ANALYSIS (±{stability['perturb_pct']:.0f}% weight perturbation, {stability['n_trials']} trials)")
    print(f"  Avg stability rate: {stability['avg_stability_rate']:.1%}  (fraction of top-10 that stay top-10)")
    print(f"  Base top-10: {stability['base_top10']}")
    print(f"  Per-candidate stability counts: {stability['stable_counts']}")


if __name__ == "__main__":
    import argparse
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts",  default="./artifacts")
    ap.add_argument("--candidates", default="../candidates.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    art = Path(args.artifacts)

    from src.parse import load_candidates
    from src.features import compute_features

    print("Loading data ...")
    df, blobs, raw = load_candidates(args.candidates, args.limit)
    cons = compute_consistency_scores(df, raw)
    avail = compute_availability(df)
    pseudo_labels = compute_pseudo_labels(df, cons)

    print("Loading artifacts ...")
    embeddings        = np.load(art / "embeddings.npy")
    anchor_embeddings = np.load(art / "anchor_embeddings.npy")
    ids               = list(np.load(art / "ids.npy", allow_pickle=True))
    with open(art / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(art / "bm25_ids.json") as f:
        bm25_ids = json.load(f)

    # Align embeddings to df order
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    emb_idx = [id_to_idx[cid] for cid in df.index]
    aligned_emb = embeddings[emb_idx]

    feat = compute_features(df, aligned_emb, anchor_embeddings, bm25, bm25_ids)

    with open(REPO / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    base_weights = cfg["weights"]

    print("Running ablations ...")
    ablation_results = run_all_ablations(df, feat, cons, avail, pseudo_labels, base_weights)

    print("Running stability analysis ...")
    stability = stability_analysis(df, feat, cons, avail, base_weights)

    print_report(ablation_results, stability)
