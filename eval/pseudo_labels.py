"""
eval/pseudo_labels.py — Rubric-based weak supervision labels for validation.

Since we have no ground truth, we generate high-precision pseudo-labels
based on the rubric (§2.3 "what a genuine strong candidate looks like").
These are used for:
  1. Pseudo-label validation of our ranking (approximate NDCG before submission)
  2. Positive/negative anchors for LightGBM LambdaMART calibration

Label tiers (0–5):
  5 = India + product-AI + explicit retrieval/ranking + 5–9 yrs + active + consistent
  4 = India + product-AI + retrieval + 4–11 yrs (slight gap vs ideal band)
  3 = India + some retrieval + some product experience + reasonable YOE
  2 = Adjacent: tech role, some ML, but no retrieval evidence
  1 = Edge: non-tech title but retrieval in descriptions (rare), or int'l with relocate
  0 = Honeypot / keyword-stuffer / all-consulting+no-retrieval / international+no-relocate

These labels are deliberately conservative: we'd rather label 3 a "2" than risk
labeling a non-fit as "5". High-tier labels require MULTIPLE positive signals.
"""

import numpy as np
import pandas as pd
from pathlib import Path


PRODUCT_AI_INDICATORS = {
    "sarvam", "cred", "razorpay", "paytm", "zoho", "mad street den",
    "yellow", "verloop", "unacademy", "nykaa", "freshworks",
    "swiggy", "zomato", "flipkart", "meesho", "ola", "rapido",
    "phonepe", "groww", "zepto", "blinkit", "myntra", "lenskart",
    "slice", "browserstack", "cleartax", "chargebee", "postman",
    "leadsquared", "darwinbox", "hasura", "setu", "signzy",
    "artivatic", "observe", "sprinklr", "keka", "haptik", "juspay",
    "sharechat", "inmobi", "dream11", "smallcase", "jupiter",
    "khatabook", "dukaan", "udaan", "moglix", "zetwerk", "delhivery",
    "porter", "byjus", "upgrad", "vedantu", "toppr", "doubtnut",
    "scaler", "healthifyme", "practo", "cure.fit", "oyo", "cars24",
    "nobroker", "naukri", "instahyre", "iimjobs", "hirist", "cutshort",
    "apna", "redrob", "pubmatic", "moengage", "clevertap",
}


def _is_product_ai(row: pd.Series) -> bool:
    """Check if current company is a product-AI company (loose match)."""
    company = (row.get("current_company", "") or "").lower()
    return any(pa in company for pa in PRODUCT_AI_INDICATORS)


def _is_product_heavy_career(row: pd.Series) -> bool:
    """product_ratio ≥ 0.5 and not all_consulting."""
    return row.get("product_ratio", 0) >= 0.5 and not row.get("all_consulting", False)


def _yoe_ideal(yoe: float) -> bool:
    """YOE in the preferred 4–9 year range."""
    return 4.0 <= yoe <= 9.0


def _yoe_ok(yoe: float) -> bool:
    """YOE in the acceptable 2–12 year range."""
    return 2.0 <= yoe <= 12.0


def _active(row: pd.Series) -> bool:
    """Active recently (days_since_active < 90) and decent response rate."""
    return (
        row.get("days_since_active", 9999) < 90
        and row.get("response_rate", 0) >= 0.30
    )


def _responsive(row: pd.Series) -> bool:
    """Above-average responsiveness (response_rate ≥ 0.5)."""
    return row.get("response_rate", 0) >= 0.50


def _consistent(row: pd.Series, consistency: pd.Series) -> bool:
    """No internal numeric contradictions (consistency ≥ 0.90)."""
    return consistency.get(row.name, 1.0) >= 0.90


def assign_pseudo_label(
    row: pd.Series,
    consistency: pd.Series,
) -> int:
    """
    Assign a pseudo-relevance label (0–5) for one candidate.
    High precision: multiple required signals for high tiers.
    """
    cid = row.name
    cons_ok = consistency.get(cid, 1.0) >= 0.90

    # ── Tier 0: hard negatives ─────────────────────────────────────────────
    if not cons_ok:
        return 0   # honeypot / internal contradiction

    if row.get("keyword_stuffer", False):
        return 0   # non-tech title + keyword stuffing

    if row.get("is_non_tech_title", False) and not row.get("retrieval_in_desc", False):
        return 0   # pure non-tech, no retrieval signal at all

    if row.get("country", "India") != "India" and not row.get("willing_to_relocate", False):
        return 0   # international, no relocate willingness

    # ── Tier 5: ideal fit ─────────────────────────────────────────────────
    retrieval = row.get("retrieval_in_desc", False)
    eval_rig  = row.get("eval_in_desc", False)
    embedding = row.get("embedding_in_desc", False)
    vecdb     = row.get("vector_db_in_desc", False)

    india = row.get("country", "") == "India"
    yoe   = float(row.get("yoe", 0) or 0)
    product_ok = _is_product_heavy_career(row)

    if (
        india and retrieval and eval_rig and product_ok
        and _yoe_ideal(yoe) and _active(row) and _responsive(row) and cons_ok
    ):
        return 5

    # ── Tier 4 ────────────────────────────────────────────────────────────
    if (
        india and retrieval and product_ok
        and _yoe_ok(yoe) and _active(row) and cons_ok
    ):
        return 4

    # ── Tier 3 ────────────────────────────────────────────────────────────
    if (
        india and retrieval and _yoe_ok(yoe) and cons_ok
    ):
        return 3

    # ── Tier 2: adjacent tech ─────────────────────────────────────────────
    if (
        not row.get("is_non_tech_title", False)
        and (embedding or vecdb)
        and _yoe_ok(yoe) and cons_ok
    ):
        return 2

    # ── Tier 1: edge cases ────────────────────────────────────────────────
    if row.get("willing_to_relocate", False) and retrieval and _yoe_ok(yoe):
        return 1

    return 1 if not row.get("is_non_tech_title", False) else 0


def compute_pseudo_labels(
    df: pd.DataFrame,
    consistency: pd.Series,
) -> pd.Series:
    """
    Compute pseudo-labels for all candidates.

    Returns pd.Series (index=candidate_id, dtype=int) with values 0–5.
    """
    labels = {}
    for cid, row in df.iterrows():
        labels[cid] = assign_pseudo_label(row, consistency)
    return pd.Series(labels, name="pseudo_label", dtype=int)


def pseudo_label_report(labels: pd.Series) -> None:
    """Print tier distribution."""
    counts = labels.value_counts().sort_index()
    total = len(labels)
    print("\nPseudo-label distribution:")
    for tier, cnt in counts.items():
        bar = "█" * min(40, cnt // 100)
        print(f"  Tier {tier}: {cnt:>7,} ({cnt/total*100:5.1f}%)  {bar}")
    print(f"  Total:   {total:>7,}")
    print(f"  Tier≥3 (relevant): {(labels >= 3).sum():,} ({(labels >= 3).mean()*100:.1f}%)")
    print(f"  Tier 5 (ideal):    {(labels == 5).sum():,}")


if __name__ == "__main__":
    import argparse, sys
    REPO = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO))

    from src.parse import load_candidates
    from src.consistency import compute_consistency_scores

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="../candidates.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df, _, raw = load_candidates(args.candidates, args.limit)
    cons = compute_consistency_scores(df, raw)
    labels = compute_pseudo_labels(df, cons)
    pseudo_label_report(labels)

    print(f"\nTop 20 tier-5 candidates:")
    tier5 = df[labels == 5]
    for cid in list(tier5.index[:20]):
        row = df.loc[cid]
        print(f"  {cid}  {row['current_title']:<35}  YOE={row['yoe']:.1f}"
              f"  product_ratio={row['product_ratio']:.2f}"
              f"  response={row['response_rate']:.2f}")
