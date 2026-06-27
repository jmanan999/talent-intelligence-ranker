"""
score.py — Composite fit scorer (Stage C).

Combines:
  base score    = weighted sum of 8 engineered features
  × availability multiplier  [0.60, 1.00]
  × consistency multiplier   [0.05, 1.00]  (honeypot guard)
  × multiplicative penalties (keyword_stuffer, all_consulting, etc.)

The final score is normalized to (0, 1], strictly non-increasing by rank,
with tie-breaks by candidate_id ascending (per validator spec).

Design principle: use multiplicative penalties not hard filters so the
ranking stays smooth and defensible. The only hard rule is the top-100 output
format (enforced by validate_submission.py).
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_weights(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["weights"]


# ── Penalty functions (multiplicative) ────────────────────────────────────────

def _penalty_keyword_stuffer(df: pd.DataFrame) -> np.ndarray:
    """
    Non-technical title + AI skills only in skills[], absent from career desc.
    This is the SINGLE MOST IMPORTANT negative — the sample_submission trap.
    """
    # keyword_stuffer flag already computed in parse.py
    penalty = np.where(df["keyword_stuffer"].values, 0.10, 1.0)
    return penalty.astype(np.float32)


def _penalty_all_consulting(df: pd.DataFrame) -> np.ndarray:
    """Entire career at services firms (TCS/Infosys/Wipro/…)."""
    penalty = np.where(df["all_consulting"].values, 0.35, 1.0)
    return penalty.astype(np.float32)


def _penalty_cv_speech_primary(df: pd.DataFrame) -> np.ndarray:
    """CV / speech / robotics only, no NLP/IR transfer."""
    penalty = np.where(df["is_cv_speech_primary"].values, 0.50, 1.0)
    return penalty.astype(np.float32)


def _penalty_llm_wrapper_only(df: pd.DataFrame) -> np.ndarray:
    """Recent LangChain wrapper only, no pre-LLM ML production history."""
    penalty = np.where(df["llm_wrapper_only"].values, 0.55, 1.0)
    return penalty.astype(np.float32)


def _penalty_non_tech_non_stuffer(df: pd.DataFrame) -> np.ndarray:
    """
    Non-technical title with limited retrieval evidence but not caught by
    keyword_stuffer (e.g., 0-3 AI skills). Still a negative signal.
    """
    is_non_tech_no_retrieval = (
        df["is_non_tech_title"].values
        & ~df["retrieval_in_desc"].values
        & ~df["keyword_stuffer"].values
    )
    penalty = np.where(is_non_tech_no_retrieval, 0.40, 1.0)
    return penalty.astype(np.float32)


def _penalty_international_no_relocate(df: pd.DataFrame) -> np.ndarray:
    """International candidate without willing_to_relocate flag."""
    is_international = (df["country"] != "India").values
    no_relocate = ~df["willing_to_relocate"].values
    penalty = np.where(is_international & no_relocate, 0.50, 1.0)
    return penalty.astype(np.float32)


def compute_base_score(
    feat: pd.DataFrame,
    weights: dict,
) -> np.ndarray:
    """
    Weighted sum of 12 features → base score in [0,1].

    feat columns: role_fit_dense, role_fit_sparse, evidence_eval,
                  product_ratio, yoe_band, tenure_stability,
                  recent_handon, location_fit,
                  skill_assessment, notice_score, github_activity,
                  engagement_score
    """
    w = weights
    base = (
        w["role_fit_dense"]    * feat["role_fit_dense"].values
        + w["role_fit_sparse"] * feat["role_fit_sparse"].values
        + w["evidence_eval"]   * feat["evidence_eval"].values
        + w["product_ratio"]   * feat["product_ratio"].values
        + w["yoe_band"]        * feat["yoe_band"].values
        + w["tenure_stability"] * feat["tenure_stability"].values
        + w["recent_handon"]   * feat["recent_handon"].values
        + w["location_fit"]    * feat["location_fit"].values
        + w.get("skill_assessment", 0) * feat["skill_assessment"].values
        + w.get("notice_score", 0)     * feat["notice_score"].values
        + w.get("github_activity", 0)  * feat["github_activity"].values
        + w.get("engagement_score", 0) * feat["engagement_score"].values
    )
    # Normalize to [0,1] by the theoretical max (sum of weights)
    total_w = sum(w.values())
    base = base / total_w
    return base.astype(np.float32)


def compute_fit_score(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    consistency: pd.Series,
    availability: pd.Series,
    weights: dict,
) -> pd.Series:
    """
    Full composite fit score:
      fit = base × availability × consistency × penalty_1 × penalty_2 × ...

    All arrays must be aligned with df.index.

    Returns pd.Series (index=candidate_id, dtype=float32) in (0,1].
    """
    # ── Base score ────────────────────────────────────────────────────────────
    base = compute_base_score(feat, weights)

    # ── Availability & consistency ────────────────────────────────────────────
    avail_arr = availability.reindex(df.index).values.astype(np.float32)
    cons_arr  = consistency.reindex(df.index).values.astype(np.float32)

    # ── Multiplicative penalties ──────────────────────────────────────────────
    pen_kw     = _penalty_keyword_stuffer(df)
    pen_cons   = _penalty_all_consulting(df)
    pen_cv     = _penalty_cv_speech_primary(df)
    pen_llm    = _penalty_llm_wrapper_only(df)
    pen_non    = _penalty_non_tech_non_stuffer(df)
    pen_intl   = _penalty_international_no_relocate(df)

    fit = (
        base
        * avail_arr
        * cons_arr
        * pen_kw
        * pen_cons
        * pen_cv
        * pen_llm
        * pen_non
        * pen_intl
    )

    # Clip to (0, 1] — a score of exactly 0 would be invisible in a ranking
    fit = np.clip(fit, 1e-6, 1.0)

    return pd.Series(fit, index=df.index, name="fit_score", dtype=np.float32)


def compute_fit_score_with_llm(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    consistency: pd.Series,
    availability: pd.Series,
    weights: dict,
    llm_scores: "pd.DataFrame | None",
    llm_weight: float = 0.40,
) -> pd.Series:
    """
    Full composite fit score blending feature-based score with Claude LLM scores.

    When LLM scores are available for a candidate, the final score is:
      fit = (llm_weight * llm_overall + (1-llm_weight) * feature_score)
            × availability × consistency × penalties

    Candidates without LLM scores (outside the top-5000 precomputed set)
    fall back to feature-only scoring.

    llm_weight=0.40 means LLM contributes 40% and features 60%.
    This is intentionally conservative: the feature score catches clear
    anti-patterns even when the LLM is generous.
    """
    base = compute_base_score(feat, weights)

    avail_arr = availability.reindex(df.index).values.astype(np.float32)
    cons_arr  = consistency.reindex(df.index).values.astype(np.float32)

    pen_kw     = _penalty_keyword_stuffer(df)
    pen_cons   = _penalty_all_consulting(df)
    pen_cv     = _penalty_cv_speech_primary(df)
    pen_llm    = _penalty_llm_wrapper_only(df)
    pen_non    = _penalty_non_tech_non_stuffer(df)
    pen_intl   = _penalty_international_no_relocate(df)

    # Blend LLM scores where available
    if llm_scores is not None and len(llm_scores) > 0:
        llm_overall = llm_scores["llm_overall"].reindex(df.index)
        has_llm = ~llm_overall.isna()
        llm_arr = llm_overall.fillna(0.0).values.astype(np.float32)
        has_llm_arr = has_llm.values

        # Normalize base to [0,1] first, then blend
        base_norm = base.copy()

        blended = np.where(
            has_llm_arr,
            llm_weight * llm_arr + (1.0 - llm_weight) * base_norm,
            base_norm,
        ).astype(np.float32)
    else:
        blended = base

    fit = (
        blended
        * avail_arr
        * cons_arr
        * pen_kw
        * pen_cons
        * pen_cv
        * pen_llm
        * pen_non
        * pen_intl
    )

    fit = np.clip(fit, 1e-6, 1.0)
    return pd.Series(fit, index=df.index, name="fit_score", dtype=np.float32)


def compute_fit_score_from_ltr(
    ltr_scores: np.ndarray,
    df: pd.DataFrame,
) -> pd.Series:
    """
    Alternative: use LTR-calibrated scores as the fit signal.
    Still applies multiplicative penalties for keyword stuffers and honeypots
    so the LTR can't accidentally elevate obvious non-fits.
    """
    ltr_norm = (ltr_scores - ltr_scores.min()) / (ltr_scores.max() - ltr_scores.min() + 1e-9)

    pen_kw  = _penalty_keyword_stuffer(df)
    pen_con = _penalty_all_consulting(df)
    pen_non = _penalty_non_tech_non_stuffer(df)

    fit = ltr_norm * pen_kw * pen_con * pen_non
    fit = np.clip(fit, 1e-6, 1.0)
    return pd.Series(fit, index=df.index, name="fit_score", dtype=np.float32)


def sort_for_submission(fit: pd.Series) -> pd.DataFrame:
    """
    Sort candidates for submission:
      1. Descending by score
      2. Ascending by candidate_id (tie-break — per validator spec)

    Returns DataFrame with columns [candidate_id, score] sorted correctly.
    """
    df_out = fit.reset_index()
    df_out.columns = ["candidate_id", "score"]

    # Stable sort: primary desc score, secondary asc candidate_id
    df_out = df_out.sort_values(
        by=["score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return df_out


if __name__ == "__main__":
    print("score.py: use rank.py to run the full pipeline end-to-end.")
