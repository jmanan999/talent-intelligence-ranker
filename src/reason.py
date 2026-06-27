"""
reason.py — Grounded per-candidate reasoning generator (Stage E).

Generates a 1–2 sentence, specific, honest reasoning string for each of
the top-100 candidates. The reasoning is assembled from the candidate's
ACTUAL fields — never hallucinated.

Rules (per submission_spec §6):
  - Reference real specifics: YOE, title, company, concrete retrieval/ranking
    evidence from career history, key behavioral signal
  - Acknowledge concerns where they exist
  - Tone must match rank (rank-5 confident; rank-90 reads "adjacent / filler")
  - Varied — built from candidate-specific driving features
  - If Claude LLM scoring note is available, use it as the lead signal

The final reasoning is a single-line string with no internal newlines.
We wrap in quotes in the CSV writer, so internal commas are fine.
"""

import re
from datetime import date
from typing import Optional

import pandas as pd

TODAY = date(2026, 6, 19)

RETRIEVAL_SNIPPET_RE = re.compile(
    r"(built|designed|implemented|shipped|developed|owned|led|created|"
    r"deployed|architected|optimized|scaled)\s+[^.]{0,80}"
    r"(?:ranking|retrieval|recommendation|recsys|search|embedding|vector|"
    r"rerank|ltr|learn.*rank|semantic|personali[sz]|ndcg|mrr)[^.]{0,60}",
    re.I,
)

EVAL_SNIPPET_RE = re.compile(
    r"(?:ndcg|mrr|\bmap\b|a/b test|a-b test|ab test|offline.*online|"
    r"precision@|recall@|relevance|graded)[^.]{0,80}",
    re.I,
)

SCALE_SNIPPET_RE = re.compile(
    r"(\d+[\.,]?\d*\s*[MBK]?\+?\s*(?:candidates|users|documents|requests|items|"
    r"queries|impressions|corpus|records|rows|vectors)[^.]{0,60})",
    re.I,
)


def _extract_best_snippet(career: list) -> Optional[str]:
    """Pull the most compelling retrieval/ranking snippet from career descriptions."""
    # Score each role by tenure so we pull from longer stints
    for role in sorted(career, key=lambda r: r.get("duration_months", 0), reverse=True):
        desc = role.get("description", "") or ""
        m = RETRIEVAL_SNIPPET_RE.search(desc)
        if m:
            snippet = m.group(0).strip()
            snippet = re.sub(r"\s+", " ", snippet)
            if len(snippet) > 120:
                snippet = snippet[:120].rsplit(" ", 1)[0]
            return snippet
    return None


def _extract_eval_snippet(career: list) -> Optional[str]:
    """Find evaluation-rigour language in career descriptions."""
    for role in career:
        desc = role.get("description", "") or ""
        m = EVAL_SNIPPET_RE.search(desc)
        if m:
            snippet = m.group(0).strip()
            return re.sub(r"\s+", " ", snippet)[:80]
    return None


def _extract_scale_signal(career: list) -> Optional[str]:
    """Find scale indicator like '30m+ candidates', '500GB daily'."""
    for role in career:
        desc = role.get("description", "") or ""
        m = SCALE_SNIPPET_RE.search(desc)
        if m:
            return m.group(0).strip()[:60]
    return None


def _location_phrase(row: pd.Series) -> str:
    loc = row.get("location", "") or ""
    country = row.get("country", "") or ""
    if country == "India":
        city = loc.split(",")[0].strip()
        if row.get("willing_to_relocate", False):
            return f"{city}-based, open to relocate"
        return f"{city}-based"
    else:
        if row.get("willing_to_relocate", False):
            return f"International ({country}), willing to relocate"
        return f"International ({country}), no relocate flag"


def _availability_phrase(row: pd.Series) -> str:
    """Concise availability summary using behavioral signals."""
    parts = []
    rr = row.get("response_rate", 0)
    if rr >= 0.80:
        parts.append(f"{rr:.0%} recruiter response")
    elif rr >= 0.50:
        parts.append(f"{rr:.0%} response rate")
    else:
        parts.append(f"low response ({rr:.0%})")

    days = row.get("days_since_active", 9999)
    if days < 7:
        parts.append("active this week")
    elif days < 30:
        parts.append("active this month")
    elif days < 90:
        parts.append("active recently")
    else:
        parts.append(f"last active {days}d ago")

    if row.get("open_to_work", False):
        parts.append("open to work")

    notice = row.get("notice_days", 30)
    if notice <= 15:
        parts.append(f"{notice}d notice")
    elif notice <= 30:
        parts.append(f"{notice}d notice")
    elif notice >= 90:
        parts.append(f"{notice}-day notice")

    # Skill assessment bonus mention
    sa = row.get("skill_assessment_composite", -1.0)
    if sa > 0.70:
        parts.append("strong verified skill scores")

    return ", ".join(parts)


def _concern_phrase(row: pd.Series, rank: int, cons_score: float) -> Optional[str]:
    """Surface the biggest concern honestly, if one exists."""
    concerns = []

    if cons_score < 0.50:
        concerns.append("profile numerical inconsistency (possible honeypot)")
    if row.get("keyword_stuffer", False):
        concerns.append("AI skills only in skills list, absent from career descriptions")
    if row.get("all_consulting", False):
        concerns.append("entire career at services firms — no product company experience")
    if row.get("is_cv_speech_primary", False):
        concerns.append("CV/speech-primary background; limited NLP/IR exposure")
    if row.get("llm_wrapper_only", False):
        concerns.append("recent LLM-API wrapping only; no pre-LLM ML production history")
    if row.get("country", "India") != "India" and not row.get("willing_to_relocate", False):
        concerns.append("international; no relocation flag — no visa sponsorship per JD")

    notice = row.get("notice_days", 30)
    if notice >= 90:
        concerns.append(f"{notice}-day notice period — JD says bar gets higher")
    elif notice >= 60:
        concerns.append(f"{notice}-day notice period")

    days = row.get("days_since_active", 9999)
    if days > 180:
        concerns.append(f"inactive {days}d — may not be actively looking")

    rr = row.get("response_rate", 0.5)
    if rr < 0.20:
        concerns.append(f"very low recruiter response rate ({rr:.0%})")

    if not concerns:
        return None
    return concerns[0]


def generate_reasoning(
    cid: str,
    rank: int,
    fit_score: float,
    row: pd.Series,
    raw: dict,
    cons_score: float,
    feat_row: Optional[pd.Series] = None,
    llm_note: Optional[str] = None,
) -> str:
    """
    Generate a grounded 1–2 sentence reasoning for one candidate.

    Parameters
    ----------
    cid        : candidate_id
    rank       : rank in submission (1-indexed)
    fit_score  : composite fit score
    row        : parsed feature row (from df)
    raw        : full JSON record for this candidate
    cons_score : consistency multiplier
    feat_row   : optional feature row for dense/sparse scores
    llm_note   : optional one-sentence note from Claude Haiku LLM scorer
    """
    p = raw["profile"]
    career = raw.get("career_history", []) or []

    title   = p.get("current_title", "?")
    company = p.get("current_company", "?")
    yoe     = float(p.get("years_of_experience", 0) or 0)

    loc_phrase  = _location_phrase(row)
    concern     = _concern_phrase(row, rank, cons_score)
    avail_str   = _availability_phrase(row)
    snippet     = _extract_best_snippet(career)
    eval_snip   = _extract_eval_snippet(career)
    scale_snip  = _extract_scale_signal(career)

    # ── Build sentence 1: identity + core signal ─────────────────────────────
    if rank <= 10:
        # Top 10: be very specific — what exactly makes them the best
        if llm_note and len(llm_note) > 15:
            s1 = f"{title} ({yoe:.1f} yrs) at {company}: {llm_note.rstrip('.')}."
        elif snippet:
            s1 = f"{title} ({yoe:.1f} yrs) at {company}: {snippet.lower()}."
        else:
            signals = []
            if row.get("retrieval_in_desc"): signals.append("retrieval/ranking in career history")
            if row.get("eval_in_desc"):      signals.append("NDCG/A-B eval rigor")
            if row.get("embedding_in_desc"): signals.append("embedding/semantic search work")
            if row.get("vector_db_in_desc"): signals.append("vector DB production experience")
            sig_str = "; ".join(signals) if signals else "strong ML background"
            s1 = f"{title} ({yoe:.1f} yrs) at {company}: {sig_str}."

    elif rank <= 30:
        # Moderately confident
        if snippet:
            s1 = f"{title} ({yoe:.1f} yrs) at {company}: {snippet.lower()}."
        elif llm_note and len(llm_note) > 15:
            s1 = f"{title} ({yoe:.1f} yrs) at {company}: {llm_note.rstrip('.')}."
        else:
            signals = []
            if row.get("retrieval_in_desc"):  signals.append("retrieval work in descriptions")
            if row.get("eval_in_desc"):       signals.append("evaluation rigor")
            pr = row.get("product_ratio", 0)
            if pr >= 0.6: signals.append(f"{pr:.0%} product-AI career")
            sig_str = "; ".join(signals) if signals else "ML-adjacent background"
            s1 = f"{title} ({yoe:.1f} yrs) at {company} — {sig_str}."

    elif rank <= 60:
        # Mid-tier: honest but encouraging
        signals = []
        if row.get("retrieval_in_desc"):  signals.append("retrieval/ranking experience")
        if row.get("embedding_in_desc"):  signals.append("embedding experience")
        pr = row.get("product_ratio", 0)
        if pr >= 0.4: signals.append(f"some product-company experience")
        if llm_note and len(llm_note) > 15:
            s1 = f"{title} ({yoe:.1f} yrs) at {company}: {llm_note.rstrip('.')}."
        elif signals:
            sig_str = "; ".join(signals[:2])
            s1 = f"{title} ({yoe:.1f} yrs) at {company} — {sig_str}."
        else:
            s1 = f"{title} ({yoe:.1f} yrs) at {company} — adjacent ML background."

    else:
        # Bottom tier (61–100): honest about why they're marginal
        if concern and concern != _concern_phrase(row, rank, cons_score):
            pass  # will show in s2
        category = "adjacent tech"
        if row.get("is_non_tech_title"):    category = "non-tech background"
        if row.get("is_cv_speech_primary"): category = "CV/speech-primary profile"
        if row.get("all_consulting"):       category = "services-firm career"
        if row.get("keyword_stuffer"):      category = "keyword inflation pattern"
        s1 = f"{title} ({yoe:.1f} yrs) at {company} — {category}; below median on retrieval evidence."

    # ── Build sentence 2: location + availability + concern ──────────────────
    if concern:
        s2 = f"{loc_phrase}; {avail_str}; concern: {concern}."
    else:
        s2 = f"{loc_phrase}; {avail_str}."

    # Combine and clean
    reasoning = f"{s1} {s2}"
    reasoning = re.sub(r"\s+", " ", reasoning).strip()

    # Truncate at word boundary to 300 chars
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    return reasoning


def generate_all_reasonings(
    top100_df: pd.DataFrame,
    raw: dict,
    consistency: pd.Series,
    feat: Optional[pd.DataFrame] = None,
    llm_scores: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Generate reasoning for all top-100 candidates.

    Parameters
    ----------
    top100_df   : DataFrame (100 rows) with index=candidate_id
    raw         : full JSON records dict
    consistency : consistency scores
    feat        : optional feature rows
    llm_scores  : optional LLM score DataFrame (index=candidate_id, col='note')
    """
    reasonings = {}
    for rank, (cid, row) in enumerate(top100_df.iterrows(), 1):
        cons_score = float(consistency.get(cid, 1.0))
        feat_row   = feat.loc[cid] if feat is not None and cid in feat.index else None
        fit_score  = float(row.get("fit_score", 0.5))

        # Pull LLM note if available
        llm_note = None
        if llm_scores is not None and cid in llm_scores.index:
            note = llm_scores.at[cid, "note"] if "note" in llm_scores.columns else None
            if note and str(note) not in ("nan", "None", "parse_error", "") and not str(note).startswith("api_error"):
                llm_note = str(note)

        reasonings[cid] = generate_reasoning(
            cid, rank, fit_score, row, raw[cid], cons_score, feat_row, llm_note
        )
    return pd.Series(reasonings, name="reasoning")


if __name__ == "__main__":
    import argparse, sys
    REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))
    from src.parse import load_candidates
    from src.consistency import compute_consistency_scores

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="../candidates.jsonl")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    df, _, raw = load_candidates(args.candidates, args.limit)
    cons = compute_consistency_scores(df, raw)

    print("Sample reasonings:")
    for i, (cid, row) in enumerate(df.head(10).iterrows()):
        r = generate_reasoning(cid, i+1, 0.5, row, raw[cid], float(cons.get(cid, 1.0)))
        print(f"\n  [{i+1}] {cid}: {r}")
