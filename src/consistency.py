"""
consistency.py — Internal-consistency scorer and honeypot detector.

Honeypots (~80 in the full 100K pool) are disguised as perfect ML candidates
but contain numerical contradictions that no legitimate profile would have.
We detect them structurally, never by ID.

Consistency score: 1.0 = fully consistent, ~0.05 = honeypot, values in (0,1].
The score is used as a MULTIPLICATIVE penalty in score.py.

Detection axes (§2.4 from the brief):
  A. expert-proficiency skills with duration_months = 0 (≥2 → strong signal)
  B. a skill's duration_months > total career months by a large margin
  C. sum-of-tenures implies career far longer than years_of_experience
  D. (C reversed) years_of_experience wildly exceeds tenure sum (ghost career)
"""

import re
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


# ── Thresholds ─────────────────────────────────────────────────────────────────
# A: expert-with-zero-duration threshold
EXPERT_ZERO_MIN = 2       # ≥2 expert-zero skills → flag A

# B: skill duration to career months ratio
SKILL_CAREER_RATIO = 1.60  # skill_duration > career_months * 1.60 → flag B

# C: sum-of-tenures vs. yoe
YOE_HIGH_RATIO   = 2.40   # sum_tenures > yoe_months * 2.40 → flag C (career longer than stated)
YOE_LOW_RATIO    = 0.20   # sum_tenures < yoe_months * 0.20 (and yoe > 24 mo) → flag D (ghost career)

# Penalty multipliers per flag (multiplicative — multiple flags stack)
PENALTY_A = 0.25   # expert-zero contradiction
PENALTY_B = 0.20   # skill exceeds career
PENALTY_C = 0.30   # career exceeds stated YOE significantly
PENALTY_D = 0.50   # ghost career (YOE >> tenures) — softer; could be gaps

FLOOR     = 0.05   # never fully zero out


def consistency_score_single(c: dict) -> tuple[float, list[str]]:
    """
    Compute consistency multiplier [FLOOR, 1.0] for one candidate JSON record.

    Returns (score, list_of_flags_triggered).
    """
    p = c.get("profile", {})
    yoe = float(p.get("years_of_experience", 0) or 0)
    career = c.get("career_history", []) or []
    skills = c.get("skills", []) or []

    total_career_months = sum(r.get("duration_months", 0) or 0 for r in career)
    yoe_months = yoe * 12.0

    flags = []
    score = 1.0

    # ── Flag A: expert-proficiency with zero duration on multiple skills ───────
    expert_zeros = [
        s.get("name", "?")
        for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    ]
    if len(expert_zeros) >= EXPERT_ZERO_MIN:
        flags.append(f"expert_zero({len(expert_zeros)} skills: {expert_zeros[:3]})")
        score *= PENALTY_A

    # ── Flag B: single skill duration massively exceeds total career ──────────
    if total_career_months > 6:   # skip edge-case of brand-new candidates
        for s in skills:
            sk_dur = s.get("duration_months") or 0
            if sk_dur > total_career_months * SKILL_CAREER_RATIO:
                flags.append(
                    f"skill_exceeds_career({s.get('name','?')} "
                    f"{sk_dur}mo > career {total_career_months}mo)"
                )
                score *= PENALTY_B
                break  # one flag is enough; avoid stacking identical penalties

    # ── Flag C: sum-of-tenures implies career far longer than YOE ─────────────
    if yoe_months > 12 and total_career_months > yoe_months * YOE_HIGH_RATIO:
        ratio = total_career_months / max(yoe_months, 1)
        flags.append(
            f"yoe_mismatch_high(career={total_career_months:.0f}mo "
            f"vs yoe={yoe_months:.0f}mo ratio={ratio:.1f})"
        )
        score *= PENALTY_C

    # ── Flag D: years_of_experience wildly exceeds actual tenure sum ──────────
    # (ghost career: claims 10 yrs but has only 6 months of jobs listed)
    if yoe_months > 24 and total_career_months > 0 and total_career_months < yoe_months * YOE_LOW_RATIO:
        ratio = total_career_months / max(yoe_months, 1)
        flags.append(
            f"ghost_career(career={total_career_months:.0f}mo "
            f"vs yoe={yoe_months:.0f}mo ratio={ratio:.2f})"
        )
        score *= PENALTY_D

    score = max(FLOOR, score)
    return score, flags


def compute_consistency_scores(
    df: pd.DataFrame,
    raw: dict,
) -> pd.Series:
    """
    Vectorized over the full candidate pool.

    Parameters
    ----------
    df   : DataFrame with index=candidate_id and columns built by parse.py
    raw  : dict[candidate_id -> full JSON record]

    Returns
    -------
    pd.Series (index=candidate_id) of consistency multipliers in [FLOOR, 1.0]
    """
    scores = {}
    for cid, row in df.iterrows():
        c = raw[cid]
        s, _ = consistency_score_single(c)
        scores[cid] = s
    return pd.Series(scores, name="consistency")


def flag_report(raw: dict, top_n: int = 20) -> list[dict]:
    """
    Produce a human-readable report of the top_n most contradicted candidates.
    Useful for verifying the detector catches the right people.
    """
    results = []
    for cid, c in raw.items():
        score, flags = consistency_score_single(c)
        if flags:
            p = c["profile"]
            results.append({
                "candidate_id": cid,
                "score": round(score, 4),
                "title": p.get("current_title", ""),
                "yoe": p.get("years_of_experience", 0),
                "flags": flags,
            })
    results.sort(key=lambda x: x["score"])
    return results[:top_n]


if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.parse import load_candidates

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="../candidates.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("Loading candidates...")
    df, blobs, raw = load_candidates(args.candidates, args.limit)
    print(f"Loaded {len(df):,} candidates")

    print("Computing consistency scores...")
    cons = compute_consistency_scores(df, raw)

    n_flagged = (cons < 0.99).sum()
    n_honeypot = (cons < 0.15).sum()
    print(f"\nCandidates with any flag (score < 0.99):  {n_flagged:,}")
    print(f"Near-honeypot (score < 0.15):              {n_honeypot:,}")
    print(f"Distribution:\n{cons.describe()}")

    print(f"\n--- Top 25 most contradicted candidates ---")
    report = flag_report(raw)
    for r in report[:25]:
        print(f"  {r['candidate_id']}  {r['title']:<35}  YOE={r['yoe']:.1f}  "
              f"score={r['score']:.3f}  flags={r['flags']}")
