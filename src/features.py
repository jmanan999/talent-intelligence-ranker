"""
features.py — Vectorized feature engineering (Stage B).

All operations run in numpy/pandas over the full 100K pool.
No per-candidate Python loops in the hot path.

Features produced:
  role_fit_dense           — cosine similarity to JD anchor embeddings
  role_fit_sparse          — BM25 score against JD query + phrase patterns
  evidence_eval            — NDCG/MRR/MAP/A-B language in career text
  product_ratio            — fraction of career at product-AI companies
  yoe_band                 — smooth Gaussian bump centered at 5–8 yrs
  tenure_stability         — penalize many <18-mo stints; reward 3+ yr stints
  recent_handon            — current role writes code (not pure architect)
  location_fit             — location preference score
  skill_assessment         — verified assessment scores on relevant skills
  notice_score             — JD prefers sub-30-day notice (explicitly stated)
  github_activity          — open-source activity (JD nice-to-have)
  engagement_score         — platform engagement composite (all 23 signals)
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ── JD query for BM25 ─────────────────────────────────────────────────────────
# These terms come from the rubric must-haves. We score all 100K docs against
# this query in one vectorized BM25 call.
JD_QUERY_TOKENS = (
    "ranking retrieval recommendation search semantic vector embedding "
    "ndcg mrr learning rank personalization recsys "
    "pinecone weaviate faiss milvus opensearch elasticsearch "
    "sentence transformer python pytorch deployed production "
    "a/b test evaluation offline online"
).split()


# ── Location scoring ──────────────────────────────────────────────────────────
PREFERRED_LOCS = {"Noida", "Pune"}
WELCOME_LOCS = {
    "Hyderabad", "Mumbai", "Delhi", "Gurgaon", "Gurugram",
    "Bangalore", "Bengaluru", "NCR",
}


def _location_score(location: str, country: str, willing_to_relocate: bool) -> float:
    """Return a [0,1] location preference score."""
    if country == "India":
        for pl in PREFERRED_LOCS:
            if pl.lower() in location.lower():
                return 1.0
        for wl in WELCOME_LOCS:
            if wl.lower() in location.lower():
                return 0.95
        return 0.87   # other India
    else:
        # International — no visa sponsorship
        base = 0.40
        return min(base + 0.20, 0.60) if willing_to_relocate else base


def _yoe_band_score(yoe: float) -> float:
    """
    Smooth bell curve peaked at 5–8 yrs.
    - yoe < 2: heavy penalty
    - yoe 2–4: gentle ramp up
    - yoe 4–9: peak zone (1.0)
    - yoe 9–13: gentle decline
    - yoe > 13: moderate penalty
    """
    if yoe >= 4 and yoe <= 9:
        return 1.0
    elif yoe >= 2 and yoe < 4:
        return 0.60 + 0.40 * ((yoe - 2) / 2)   # ramp 2→4
    elif yoe > 9 and yoe <= 13:
        return 1.0 - 0.30 * ((yoe - 9) / 4)     # decline 9→13
    elif yoe > 13:
        return 0.70 - 0.20 * min((yoe - 13) / 5, 1.0)
    else:
        return max(0.20, 0.40 * (yoe / 2))       # yoe < 2


def _tenure_stability_score(short_stints: int, long_stints: int, n_roles: int) -> float:
    """
    Penalize title-chasing (many <18-mo stints), reward ≥30-mo stints.
    """
    if n_roles == 0:
        return 0.50
    short_ratio = short_stints / n_roles
    score = 1.0
    score -= 0.30 * short_ratio        # lose up to 30 pts for all short stints
    score += 0.10 * min(long_stints, 3) / 3   # gain up to 10 pts for long stints
    return max(0.30, min(1.0, score))


def compute_dense_scores(
    embeddings: np.ndarray,          # (N, D) candidate embeddings, L2-normalized
    anchor_embeddings: np.ndarray,   # (A, D) JD anchor embeddings, L2-normalized
) -> np.ndarray:
    """
    role_fit_dense: for each candidate, max cosine similarity across all JD anchors.
    Since embeddings are L2-normalized, cosine = dot product.
    Shape: (N,)
    """
    # (N, A) similarity matrix; take max across anchors
    sims = embeddings @ anchor_embeddings.T          # (N, A)
    return sims.max(axis=1).astype(np.float32)       # (N,)


def compute_sparse_scores(bm25, query_tokens: list[str]) -> np.ndarray:
    """
    role_fit_sparse: BM25 scores for all candidates against the JD query.
    rank_bm25 returns a plain Python list; convert to float32 numpy.
    """
    scores = bm25.get_scores(query_tokens)
    arr = np.array(scores, dtype=np.float32)
    return arr


def normalize_to_01(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + eps)


def compute_features(
    df: pd.DataFrame,
    embeddings: np.ndarray,          # (N, D), ordered by df.index
    anchor_embeddings: np.ndarray,   # (A, D)
    bm25,                            # BM25Okapi instance
    bm25_ids: list[str],             # candidate IDs in BM25 corpus order
) -> pd.DataFrame:
    """
    Compute all engineered features and return as a DataFrame indexed by candidate_id.

    Parameters
    ----------
    df             : parsed feature table (from parse.py)
    embeddings     : L2-normalized candidate embeddings
    anchor_embeddings : L2-normalized JD anchor embeddings
    bm25           : fitted BM25Okapi
    bm25_ids       : list of candidate_ids in BM25 corpus order

    Returns
    -------
    feat: pd.DataFrame (same index as df) with all numeric features
    """
    N = len(df)

    # ── Dense role fit ─────────────────────────────────────────────────────────
    print("  Computing dense role-fit scores ...")
    dense_raw = compute_dense_scores(embeddings, anchor_embeddings)
    dense_scores = normalize_to_01(dense_raw)

    # ── Sparse / BM25 role fit ─────────────────────────────────────────────────
    print("  Computing BM25 sparse scores ...")
    sparse_raw = compute_sparse_scores(bm25, JD_QUERY_TOKENS)
    # BM25 corpus may be in a different order; reindex to match df
    bm25_series = pd.Series(sparse_raw, index=bm25_ids)
    sparse_aligned = bm25_series.reindex(df.index).fillna(0).values
    sparse_scores = normalize_to_01(sparse_aligned.astype(np.float32))

    # ── Evidence-eval score (NDCG/MRR/A-B language) ───────────────────────────
    # Already a boolean in df; convert and add partial credit for embedding mentions
    evidence_eval = (
        df["eval_in_desc"].astype(float) * 1.0
        + df["embedding_in_desc"].astype(float) * 0.40
        + df["vector_db_in_desc"].astype(float) * 0.35
    ).clip(0, 1).values.astype(np.float32)

    # ── YOE band ──────────────────────────────────────────────────────────────
    yoe_band = np.array([_yoe_band_score(y) for y in df["yoe"]], dtype=np.float32)

    # ── Tenure stability ──────────────────────────────────────────────────────
    tenure_stability = np.array([
        _tenure_stability_score(ss, ls, nr)
        for ss, ls, nr in zip(df["short_stints"], df["long_stints"], df["n_roles"])
    ], dtype=np.float32)

    # ── Location fit ──────────────────────────────────────────────────────────
    location_fit = np.array([
        _location_score(loc, ctr, rel)
        for loc, ctr, rel in zip(df["location"], df["country"], df["willing_to_relocate"])
    ], dtype=np.float32)

    # ── Product ratio (already in df, clip to [0,1]) ──────────────────────────
    product_ratio = df["product_ratio"].clip(0, 1).values.astype(np.float32)

    # ── Recent hands-on coding ────────────────────────────────────────────────
    recent_handon = df["recent_handon"].astype(float).values.astype(np.float32)

    # ── Skill assessment composite (verified scores, not self-reported) ────────
    # Values in df: -1.0 = no assessments taken, [0,1] = weighted avg of relevant scores
    # For candidates without assessments, use 0.45 as neutral (slight downward nudge)
    raw_sa = df["skill_assessment_composite"].values.astype(np.float32)
    skill_assessment = np.where(raw_sa < 0, 0.45, raw_sa.clip(0, 1)).astype(np.float32)

    # ── Notice period score (JD: loves sub-30, can buy out 30, 30+ bar gets higher) ─
    notice_score = df["notice_score"].values.astype(np.float32)

    # ── GitHub activity (nice-to-have per JD: open-source contributions) ──────
    github_activity = df["github_norm"].values.astype(np.float32)

    # ── Platform engagement (job-seeking signal composite from all 23 signals) ─
    engagement = df["engagement_score"].values.astype(np.float32)

    # ── Build feature DataFrame ───────────────────────────────────────────────
    feat = pd.DataFrame({
        "role_fit_dense":    dense_scores,
        "role_fit_sparse":   sparse_scores,
        "evidence_eval":     evidence_eval,
        "product_ratio":     product_ratio,
        "yoe_band":          yoe_band,
        "tenure_stability":  tenure_stability,
        "recent_handon":     recent_handon,
        "location_fit":      location_fit,
        "skill_assessment":  skill_assessment,
        "notice_score":      notice_score,
        "github_activity":   github_activity,
        "engagement_score":  engagement,
    }, index=df.index)

    return feat


if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO))

    from src.parse import load_candidates

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="../candidates.jsonl")
    ap.add_argument("--artifacts",  default="./artifacts")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    art = Path(args.artifacts)
    print("Loading candidates ...")
    df, blobs, raw = load_candidates(args.candidates, args.limit)

    # Load precomputed artifacts
    embeddings       = np.load(art / "embeddings.npy")[:len(df)]
    anchor_embeddings = np.load(art / "anchor_embeddings.npy")
    ids              = list(np.load(art / "ids.npy", allow_pickle=True))[:len(df)]
    with open(art / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(art / "bm25_ids.json") as f:
        bm25_ids = json.load(f)[:len(df)]

    feat = compute_features(df, embeddings, anchor_embeddings, bm25, bm25_ids)
    print(f"\nFeature table shape: {feat.shape}")
    print(feat.describe().round(3))
