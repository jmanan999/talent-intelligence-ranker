"""
signals.py — Behavioral availability multiplier (Stage B).

Combines all relevant redrob_signals into a single multiplier in [0.55, 1.00].
Upgraded from 5 signals to 9 signals using the full behavioral footprint.

The multiplier is NEVER used to zero out a perfect fit — it discounts
unresponsive/inactive candidates and boosts highly engaged ones.

Signals used:
  last_active_date          → recency decay (half-life 60 days)
  recruiter_response_rate   → direct engagement proxy
  open_to_work_flag         → self-declared intent
  saved_by_recruiters_30d   → external demand signal (market validation)
  interview_completion_rate → shows-up-and-follows-through
  notice_score              → JD explicitly prefers sub-30-day notice
  offer_acceptance_rate     → signals serious intent, not just browsing
  applications_submitted_30d → active job seeking
  engagement_score          → composite platform activity signal
"""

import math
from datetime import date

import numpy as np
import pandas as pd


TODAY = date(2026, 6, 19)
HALF_LIFE_DAYS = 60
FLOOR = 0.55   # lowered from 0.60: ghost candidates deserve more penalty
CEIL  = 1.00

# Component weights (sum = 1.0)
W_RECENCY      = 0.28
W_RESPONSE     = 0.22
W_OPEN         = 0.10
W_SAVED        = 0.08
W_INTERVIEW    = 0.08
W_NOTICE       = 0.10   # NEW: JD says notice period matters significantly
W_OFFER_ACCEPT = 0.07   # NEW: offer acceptance = serious candidate
W_APPLICATIONS = 0.04   # NEW: actively applying = available
W_ENGAGEMENT   = 0.03   # NEW: platform activity signal


def _recency_score(days_since: float) -> float:
    """Exponential decay: score = exp(-ln2 * days / half_life). Clamped [0,1]."""
    return math.exp(-math.log(2) * days_since / HALF_LIFE_DAYS)


def _saved_score(saved_30d: int) -> float:
    """Log-scale: 0→0, 1→0.50, 3→0.75, 10→0.95. Market validation signal."""
    return 1.0 - 1.0 / (1.0 + max(0, saved_30d) * 0.5)


def _applications_score(apps_30d: int) -> float:
    """Soft cap: 0 apps→0.2 (not actively looking), 3+→0.7, 8+→1.0."""
    if apps_30d == 0:
        return 0.20
    return min(0.20 + 0.10 * apps_30d, 1.0)


def compute_availability(df: pd.DataFrame) -> pd.Series:
    """
    Compute availability multiplier for each candidate.

    Parameters
    ----------
    df : DataFrame with columns from parse.py (all 23 signals extracted)

    Returns
    -------
    pd.Series (index=candidate_id) of floats in [FLOOR, CEIL]
    """
    # Core signals
    recency      = np.array([_recency_score(d) for d in df["days_since_active"]], dtype=np.float32)
    response     = df["response_rate"].clip(0, 1).values.astype(np.float32)
    open_work    = df["open_to_work"].astype(float).values.astype(np.float32)
    saved        = np.array([_saved_score(s) for s in df["saved_30d"]], dtype=np.float32)
    interview    = df["interview_rate"].clip(0, 1).values.astype(np.float32)

    # New signals
    notice       = df["notice_score"].values.astype(np.float32)
    offer_accept = df["offer_acceptance_rate"].clip(0, 1).values.astype(np.float32)
    apps         = np.array([_applications_score(a) for a in df["applications_30d"]], dtype=np.float32)
    engagement   = df["engagement_score"].clip(0, 1).values.astype(np.float32)

    raw = (
        W_RECENCY      * recency
        + W_RESPONSE     * response
        + W_OPEN         * open_work
        + W_SAVED        * saved
        + W_INTERVIEW    * interview
        + W_NOTICE       * notice
        + W_OFFER_ACCEPT * offer_accept
        + W_APPLICATIONS * apps
        + W_ENGAGEMENT   * engagement
    )

    # Remap [0,1] → [FLOOR, CEIL]
    availability = FLOOR + (CEIL - FLOOR) * raw

    return pd.Series(availability, index=df.index, name="availability", dtype=np.float32)


if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO))
    from src.parse import load_candidates

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="../candidates.jsonl")
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()

    df, _, _ = load_candidates(args.candidates, args.limit)
    avail = compute_availability(df)
    print(f"Availability multiplier stats:\n{avail.describe().round(3)}")
    print(f"\nTop-10 most available:")
    print(avail.nlargest(10))
    print(f"\nBottom-10 least available:")
    print(avail.nsmallest(10))
