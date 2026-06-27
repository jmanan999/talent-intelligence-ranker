"""
ltr.py — Weak-supervision LightGBM LambdaMART calibration (Stage D).

Since we have no ground-truth labels, we generate high-precision pseudo-labels
from the rubric and train a LambdaMART ranker to learn feature weights.

The LTR layer AUGMENTS the hand-crafted scorer — it doesn't replace the
transparent features. The same multiplicative penalties (keyword_stuffer,
honeypot, etc.) still apply after LTR scoring.

Why LambdaMART?
- Directly optimizes NDCG (the hackathon's primary metric)
- CPU-fast (LightGBM's gradient-boosting is vectorized C++)
- Interpretable: feature importances confirm our manual weight choices
- Robust to pseudo-label noise via soft graded relevance (0–5)
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


FEATURE_COLS = [
    "role_fit_dense",
    "role_fit_sparse",
    "evidence_eval",
    "product_ratio",
    "yoe_band",
    "tenure_stability",
    "recent_handon",
    "location_fit",
    # New features (v2: all 23 signals, skill assessments, notice period)
    "skill_assessment",
    "notice_score",
    "github_activity",
    "engagement_score",
    # Extra behavioral signals from df
    "response_rate",
    "open_to_work_f",
    "days_active_norm",
    "github_norm",
    "saved_30d_norm",
    "notice_score_raw",
    "offer_accept_norm",
]


def build_ltr_features(
    feat: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine engineered features (feat) with selected behavioral signals (df)
    into the LTR input matrix.
    """
    extra = pd.DataFrame({
        "response_rate":    df["response_rate"].clip(0, 1),
        "open_to_work_f":   df["open_to_work"].astype(float),
        "days_active_norm": 1.0 / (1.0 + df["days_since_active"] / 30.0),
        "github_norm":      df["github_score"].clip(0, 100) / 100.0,
        "saved_30d_norm":   df["saved_30d"].clip(0, 20) / 20.0,
        # New v2 signals
        "notice_score_raw": df["notice_score"].clip(0, 1),
        "offer_accept_norm": df["offer_acceptance_rate"].clip(0, 1),
    }, index=df.index)

    return pd.concat([feat, extra], axis=1)


def train_ltr(
    ltr_features: pd.DataFrame,
    pseudo_labels: pd.Series,
    cfg: dict,
    seed: int = 42,
) -> object:
    """
    Train LightGBM LambdaMART ranker on pseudo-labels.

    Uses a single group (all 100K as one query) for training since we have
    one JD. For grouped CV this would split by candidate cohort.

    Returns the fitted LGBMRanker.
    """
    import lightgbm as lgb

    ltr_cfg = cfg.get("ltr", {})
    n_estimators     = ltr_cfg.get("n_estimators", 200)
    num_leaves       = ltr_cfg.get("num_leaves", 31)
    learning_rate    = ltr_cfg.get("learning_rate", 0.05)
    min_child_samples = ltr_cfg.get("min_child_samples", 10)

    # Filter to candidates with confident labels
    pos_thresh = ltr_cfg.get("pseudo_label_positive_threshold", 0.60)
    neg_thresh = ltr_cfg.get("pseudo_label_negative_threshold", 0.15)

    # Align
    X = ltr_features.reindex(pseudo_labels.index).fillna(0.0)
    y = pseudo_labels.values

    print(f"  LTR features shape: {X.shape}")
    print(f"  Label distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

    # LightGBM LambdaMART max group size = 10,000.
    # Sample a balanced training subset: keep all high-tier + random low-tier.
    # Inference is still run on all 100K.
    rng = np.random.default_rng(seed)
    idx_series = pd.Series(range(len(y)), index=X.index)

    high_tier  = idx_series[y >= 2].values
    mid_tier   = idx_series[y == 1].values
    low_tier   = idx_series[y == 0].values
    budget     = min(9800, 10000) - len(high_tier)
    n_mid      = min(len(mid_tier), budget // 2)
    n_low      = min(len(low_tier), budget - n_mid)
    sampled    = np.concatenate([
        high_tier,
        rng.choice(mid_tier, n_mid, replace=False),
        rng.choice(low_tier, n_low, replace=False),
    ])
    sampled.sort()
    X_train = X.iloc[sampled]
    y_train = y[sampled]
    groups   = [len(y_train)]
    print(f"  Training subset: {len(y_train)} candidates "
          f"(tier≥2: {len(high_tier)}, tier-1: {n_mid}, tier-0: {n_low})")

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10, 50],
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    ranker.fit(
        X_train.values,
        y_train,
        group=groups,
    )

    # Print feature importances
    importances = pd.Series(
        ranker.feature_importances_,
        index=X.columns,
        name="importance"
    ).sort_values(ascending=False)
    print(f"\n  Feature importances:\n{importances.to_string()}")

    return ranker, X.columns.tolist()


def ltr_predict(
    ranker,
    ltr_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """
    Generate LTR ranking scores for all candidates.
    Returns pd.Series(index=candidate_id, values=float).
    """
    X = ltr_features[feature_cols].fillna(0.0)
    scores = ranker.predict(X.values)
    return pd.Series(scores.astype(np.float32), index=ltr_features.index, name="ltr_score")


def save_ltr_model(ranker, feature_cols: list[str], path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump({"ranker": ranker, "feature_cols": feature_cols}, f, protocol=4)
    print(f"  LTR model saved → {path}")


def load_ltr_model(path: Path) -> tuple:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["ranker"], obj["feature_cols"]


if __name__ == "__main__":
    import argparse
    from src.parse import load_candidates
    from src.consistency import compute_consistency_scores
    from src.signals import compute_availability
    from src.features import compute_features
    from eval.pseudo_labels import compute_pseudo_labels, pseudo_label_report

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="../candidates.jsonl")
    ap.add_argument("--artifacts",  default="./artifacts")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    art = Path(args.artifacts)

    with open(REPO / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Loading candidates ...")
    df, blobs, raw = load_candidates(args.candidates, args.limit)
    cons   = compute_consistency_scores(df, raw)
    avail  = compute_availability(df)
    pseudo = compute_pseudo_labels(df, cons)
    pseudo_label_report(pseudo)

    print("Loading artifacts ...")
    embeddings        = np.load(art / "embeddings.npy")
    anchor_embeddings = np.load(art / "anchor_embeddings.npy")
    ids               = list(np.load(art / "ids.npy", allow_pickle=True))
    with open(art / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(art / "bm25_ids.json") as f:
        bm25_ids = json.load(f)

    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    emb_idx = [id_to_idx[cid] for cid in df.index]
    aligned_emb = embeddings[emb_idx]

    feat = compute_features(df, aligned_emb, anchor_embeddings, bm25, bm25_ids)

    print("\nBuilding LTR features ...")
    ltr_feat = build_ltr_features(feat, df)

    print("Training LambdaMART ...")
    ranker, feature_cols = train_ltr(ltr_feat, pseudo, cfg)

    ltr_scores = ltr_predict(ranker, ltr_feat, feature_cols)
    top20 = ltr_scores.nlargest(20)
    print(f"\nTop-20 by LTR score:")
    for cid, s in top20.items():
        row = df.loc[cid]
        print(f"  {cid}  {row['current_title']:<35}  YOE={row['yoe']:.1f}  score={s:.4f}")

    save_ltr_model(ranker, feature_cols, art / "ltr_model.pkl")
