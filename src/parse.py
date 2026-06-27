"""
parse.py — Load candidates.jsonl into a flat Pandas DataFrame + candidate text blobs.

Output:
  df: one row per candidate with all structured fields extracted
  text_blobs: dict[candidate_id -> str]  — concatenated searchable text
  raw: dict[candidate_id -> dict]        — original JSON for feature.py / reason.py

Key design choice: we extract EVERY field here so all downstream modules work from
a single pass over the JSONL. The JSONL is 100K × ~3 KB ≈ 487 MB; one pass
takes ~60 s and the result fits easily in RAM as a ~500 MB DataFrame + index.
"""

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

TODAY = date(2026, 6, 19)   # matches config.yaml today

SERVICES_FIRMS_RE = re.compile(
    r"\b(tcs|tata consultancy|infosys|wipro|accenture|cognizant|capgemini|"
    r"tech mahindra|hcl technologies|hcltech|mindtree|ltimindtree|lti|mphasis|"
    r"l&t infotech|hexaware|niit technologies|syntel|persistent systems|birlasoft|"
    r"igate|mastech|zensar|cyient|sonata software|coforge|mphasis|sasken|"
    r"geometric|cybage)\b",
    re.I,
)

NON_TECH_TITLES_RE = re.compile(
    r"\b(hr manager|human resource|accountant|content writer|graphic design(?:er)?|"
    r"business analyst|sales executive|sales manager|marketing manager|"
    r"civil engineer|mechanical engineer|administrative|recruiter|"
    r"customer support|operations manager|project manager|"
    r"supply chain|procurement|logistics|financial analyst|"
    r"chartered accountant|ca (?:intern|fresher)|"
    r"legal|compliance officer|teacher|professor|nurse|doctor)\b",
    re.I,
)

TECH_TITLES_RE = re.compile(
    r"\b(ml engineer|machine learning engineer|data scientist|nlp engineer|"
    r"search engineer|recsys engineer|recommendation engine|"
    r"ai engineer|applied scientist|research scientist|"
    r"software engineer|software developer|backend engineer|"
    r"data engineer|platform engineer|full.?stack|"
    r"deep learning|llm engineer|ir engineer|information retrieval|"
    r"ranking engineer|relevance engineer)\b",
    re.I,
)

CV_SPEECH_RE = re.compile(
    r"\b(computer vision|object detection|yolo|faster.?rcnn|"
    r"speech recognition|asr|tts|text.to.speech|"
    r"robotics|ros\b|lidar|3d detection|pose estimation|"
    r"image segmentation|image classification)\b",
    re.I,
)

RETRIEVAL_RE = re.compile(
    r"\b(ranking|ranker|rerank|retrieval|recommendation system|recsys|"
    r"collaborative filtering|matrix factori|two.tower|"
    r"semantic search|vector search|hybrid search|dense retrieval|"
    r"learning.to.rank|\bltr\b|lambdamart|ranknet|"
    r"personali[sz]|candidate generation|"
    r"information retrieval|search relevance|feed ranking)\b",
    re.I,
)

EVAL_RE = re.compile(
    r"\b(ndcg|mrr|mean reciprocal|mean average precision|\bmap\b|"
    r"recall@|precision@|a/b test|ab test|a-b test|"
    r"offline.*online|online.*offline|holdout|"
    r"click.through|ctr|relevance label|graded relevance|"
    r"interleaving|counterfactual)\b",
    re.I,
)

EMBEDDING_RE = re.compile(
    r"\b(sentence.transformer|openai embedding|text.embedding|"
    r"bge|e5.embed|bi.encoder|dense embed|neural embed|"
    r"semantic embed|vector embed|embed)\b",
    re.I,
)

VECTOR_DB_RE = re.compile(
    r"\b(pinecone|weaviate|qdrant|milvus|faiss|annoy|"
    r"opensearch|elasticsearch|vector database|vector store|"
    r"vector index|approximate nearest|ann\b|hnsw)\b",
    re.I,
)

LLM_WRAPPER_RE = re.compile(
    r"\b(langchain|llamaindex|llama.index|openai api|chatgpt api|"
    r"gpt.4|gpt.3|claude api|llm wrapper|prompt engineer)\b",
    re.I,
)

RESEARCH_RE = re.compile(
    r"\b(research paper|published|arxiv|phd|ph\.d|"
    r"university lab|academic research|iit|iim|"
    r"postdoc|dissertation|thesis)\b",
    re.I,
)


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _days_ago(d: Optional[date]) -> float:
    """Days between date and TODAY; None → 9999 (treat as very stale)."""
    if d is None:
        return 9999.0
    return max(0.0, (TODAY - d).days)


def _career_total_months(career: list) -> int:
    return sum(r.get("duration_months", 0) or 0 for r in career)


def _career_text(career: list) -> str:
    """All career description text concatenated, descriptions weighted 3× over titles."""
    parts = []
    for r in career:
        title = r.get("title", "") or ""
        company = r.get("company", "") or ""
        desc = r.get("description", "") or ""
        parts.append(f"{title} {company} {desc} {desc} {desc}")   # desc weighted 3×
    return " ".join(parts)


def build_text_blob(c: dict) -> str:
    """
    Searchable text for BM25 indexing (longer, descriptions weighted 3×).
    Career descriptions dominate; skills list is last to avoid keyword trap.
    Used by BM25; NOT used for dense embeddings (use build_embed_text instead).
    """
    p = c["profile"]
    headline = p.get("headline", "") or ""
    summary = p.get("summary", "") or ""
    title = p.get("current_title", "") or ""

    career = c.get("career_history", [])
    desc_text = _career_text(career)   # descriptions repeated 3×

    skills = " ".join(s.get("name", "") for s in c.get("skills", []))
    certs = " ".join(
        f"{cert.get('name','')} {cert.get('issuer','')}"
        for cert in c.get("certifications", [])
    )

    # Career descriptions dominate; skills are last (the trap to avoid)
    return f"{title} {headline} {summary} {desc_text} {certs} {skills}"


def build_embed_text(c: dict) -> str:
    """
    Short, focused embedding text for sentence-transformer (≤150 tokens).
    We want the embedding to capture the candidate's domain, not their keywords.
    Shorter = faster CPU inference; truncation is wasteful for 512-token blobs.

    Structure:
      {title}. {headline truncated}.
      Career: {first 100 chars of each role description, up to 3 roles}.
    """
    p = c["profile"]
    title   = p.get("current_title", "") or ""
    headline = (p.get("headline", "") or "")[:120]

    career = c.get("career_history", []) or []
    # Take the 3 most recent roles; truncate each description to 150 chars
    sorted_career = sorted(career, key=lambda r: _parse_date(r.get("start_date")) or _parse_date("2000-01-01"), reverse=True)
    snippets = []
    for role in sorted_career[:3]:
        desc = (role.get("description", "") or "")[:150]
        role_title = (role.get("title", "") or "")
        snippets.append(f"{role_title}: {desc}")

    career_str = " | ".join(snippets)

    return f"{title}. {headline}. {career_str}"


def extract_row(c: dict) -> dict:
    """Extract all structured features from a single candidate JSON into a flat dict."""
    p = c["profile"]
    sig = c.get("redrob_signals", {}) or {}
    career = c.get("career_history", []) or []
    skills = c.get("skills", []) or []

    cid = c["candidate_id"]
    yoe = float(p.get("years_of_experience", 0) or 0)
    country = p.get("country", "") or ""
    location = p.get("location", "") or ""
    current_title = p.get("current_title", "") or ""
    current_company = p.get("current_company", "") or ""
    current_industry = p.get("current_industry", "") or ""

    # ── Career summary ────────────────────────────────────────────────────────
    total_career_months = _career_total_months(career)
    career_text = _career_text(career)
    desc_text = " ".join(r.get("description", "") or "" for r in career)

    # Per-role stats
    tenures = [r.get("duration_months", 0) or 0 for r in career]
    short_stints = sum(1 for t in tenures if 0 < t < 18)
    long_stints = sum(1 for t in tenures if t >= 30)
    n_roles = len(career)
    max_tenure = max(tenures) if tenures else 0

    # Company classification
    companies_lower = [
        (r.get("company", "") or "").lower() for r in career
    ]
    industries = [(r.get("industry", "") or "").lower() for r in career]
    months_by_role = list(zip(tenures, companies_lower, industries))

    is_services = [
        bool(SERVICES_FIRMS_RE.search(co)) for co in companies_lower
    ]
    all_consulting = all(is_services) and n_roles > 0

    # product_ratio: months at non-services companies / total career months
    product_months = sum(
        t for t, co, ind in months_by_role
        if not SERVICES_FIRMS_RE.search(co)
    )
    product_ratio = (product_months / total_career_months) if total_career_months > 0 else 0.0

    # ── Text signal flags (from career descriptions, NOT skills list) ─────────
    retrieval_in_desc = bool(RETRIEVAL_RE.search(desc_text))
    eval_in_desc = bool(EVAL_RE.search(desc_text))
    embedding_in_desc = bool(EMBEDDING_RE.search(desc_text))
    vector_db_in_desc = bool(VECTOR_DB_RE.search(desc_text))
    cv_speech_in_desc = bool(CV_SPEECH_RE.search(desc_text))
    research_signals = bool(RESEARCH_RE.search(desc_text))
    llm_wrapper_in_desc = bool(LLM_WRAPPER_RE.search(desc_text))

    # ── LLM wrapper check (only if langchain in recent role, no older retrieval) ──
    # Sort career by start date descending
    def role_start(r):
        return _parse_date(r.get("start_date")) or date(2000, 1, 1)

    sorted_career = sorted(career, key=role_start, reverse=True)
    recent_role = sorted_career[0] if sorted_career else {}
    recent_desc = recent_role.get("description", "") or ""
    recent_title = (recent_role.get("title", "") or "").lower()
    recent_company = (recent_role.get("company", "") or "").lower()
    recent_is_services = bool(SERVICES_FIRMS_RE.search(recent_company))

    older_desc = " ".join(r.get("description", "") or "" for r in sorted_career[1:])
    llm_wrapper_only = (
        bool(LLM_WRAPPER_RE.search(recent_desc))
        and not bool(RETRIEVAL_RE.search(older_desc))
        and not bool(RETRIEVAL_RE.search(recent_desc))
    )

    # ── Title classification ──────────────────────────────────────────────────
    is_non_tech_title = bool(NON_TECH_TITLES_RE.search(current_title)) and not bool(
        TECH_TITLES_RE.search(current_title)
    )
    is_cv_speech_primary = (
        bool(CV_SPEECH_RE.search(desc_text))
        and not bool(RETRIEVAL_RE.search(desc_text))
    )

    # AI skills listed vs. mentioned in descriptions
    ai_skill_names = {
        "rag", "pinecone", "faiss", "embeddings", "vector search",
        "semantic search", "sentence transformers", "llms", "fine-tuning llms",
        "recommendation systems", "information retrieval", "langchain",
        "weaviate", "qdrant", "milvus", "openai", "hugging face",
        "transformers", "bert", "gpt", "nlp", "deep learning",
        "machine learning", "pytorch", "tensorflow", "scikit-learn",
        "xgboost", "lightgbm", "elasticsearch", "opensearch", "bm25",
    }
    candidate_ai_skills = sum(
        1 for s in skills if s.get("name", "").lower() in ai_skill_names
    )

    # keyword_stuffer: non-tech title + AI skills in skills[] but NOT in descriptions
    keyword_stuffer = (
        is_non_tech_title
        and candidate_ai_skills >= 4
        and not retrieval_in_desc
    )

    # ── Recent hands-on code (current role) ──────────────────────────────────
    code_signals_re = re.compile(
        r"\b(built|implemented|developed|wrote|engineer|cod|deploy|"
        r"python|pytorch|tensorflow|api|microservice|pipeline|model|"
        r"inference|training|fine.tun)\b",
        re.I,
    )
    recent_handon = bool(code_signals_re.search(recent_desc))

    # ── Skills structure (for consistency checks) ─────────────────────────────
    skill_durations = [s.get("duration_months", 0) or 0 for s in skills]
    expert_zero_count = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    max_skill_duration = max(skill_durations) if skill_durations else 0

    # ── Behavioral signals (all 23) ───────────────────────────────────────────
    last_active = _parse_date(sig.get("last_active_date"))
    days_since_active = _days_ago(last_active)
    open_to_work = bool(sig.get("open_to_work_flag", False))
    response_rate = float(sig.get("recruiter_response_rate", 0.5) or 0.5)
    avg_response_h = float(sig.get("avg_response_time_hours", 48) or 48)
    notice_days = int(sig.get("notice_period_days", 30) or 30)
    willing_to_relocate = bool(sig.get("willing_to_relocate", False))
    github_score = float(sig.get("github_activity_score", -1) or -1)
    saved_30d = int(sig.get("saved_by_recruiters_30d", 0) or 0)
    interview_rate = float(sig.get("interview_completion_rate", 0.5) or 0.5)
    profile_completeness = float(sig.get("profile_completeness_score", 50) or 50)
    # New: previously extracted but unused signals
    profile_views_30d = int(sig.get("profile_views_received_30d", 0) or 0)
    applications_30d = int(sig.get("applications_submitted_30d", 0) or 0)
    search_appearances_30d = int(sig.get("search_appearance_30d", 0) or 0)
    connection_count = int(sig.get("connection_count", 0) or 0)
    endorsements_received = int(sig.get("endorsements_received", 0) or 0)
    offer_acceptance_rate = float(sig.get("offer_acceptance_rate", 0.5) or 0.5)
    linkedin_connected = bool(sig.get("linkedin_connected", False))
    verified_email = bool(sig.get("verified_email", True))
    verified_phone = bool(sig.get("verified_phone", True))
    preferred_work_mode = str(sig.get("preferred_work_mode", "hybrid") or "hybrid")
    salary_min = float((sig.get("expected_salary_range_inr_lpa") or {}).get("min", 0) or 0)
    salary_max = float((sig.get("expected_salary_range_inr_lpa") or {}).get("max", 0) or 0)

    # ── Skill assessment composite (verified scores, not self-reported) ────────
    # For this JD: retrieval/NLP/ML/recommendation/embedding skills matter most
    RELEVANT_ASSESSMENT_SKILLS = {
        "nlp": 1.0,
        "recommendation systems": 1.0,
        "information retrieval": 1.0,
        "faiss": 0.9,
        "vector search": 0.9,
        "semantic search": 0.9,
        "fine-tuning llms": 0.8,
        "transformers": 0.8,
        "deep learning": 0.7,
        "machine learning": 0.7,
        "python": 0.8,
        "pytorch": 0.8,
        "tensorflow": 0.6,
        "elasticsearch": 0.7,
        "pinecone": 0.9,
        "qdrant": 0.8,
        "weaviate": 0.8,
        "milvus": 0.8,
        "xgboost": 0.5,
        "lightgbm": 0.5,
    }
    raw_assessments = sig.get("skill_assessment_scores", {}) or {}
    relevant_scores = []
    for skill_name, score_val in raw_assessments.items():
        normalized_key = skill_name.lower()
        for rel_skill, weight in RELEVANT_ASSESSMENT_SKILLS.items():
            if rel_skill in normalized_key or normalized_key in rel_skill:
                relevant_scores.append(float(score_val or 0) / 100.0 * weight)
                break
    skill_assessment_composite = float(sum(relevant_scores) / max(len(relevant_scores), 1)) if relevant_scores else -1.0
    has_skill_assessments = len(raw_assessments) > 0

    # ── Notice period scoring (JD explicitly prefers sub-30 days) ────────────
    if notice_days <= 0:
        notice_score = 1.0
    elif notice_days <= 15:
        notice_score = 0.97
    elif notice_days <= 30:
        notice_score = 0.90
    elif notice_days <= 60:
        notice_score = 0.72
    elif notice_days <= 90:
        notice_score = 0.52
    else:
        notice_score = 0.33   # 90+ day notice → JD says "bar gets higher"

    # ── GitHub activity normalized (JD nice-to-have) ─────────────────────────
    github_norm = min(github_score / 100.0, 1.0) if github_score >= 0 else 0.30

    # ── Platform engagement score (active job seeker signals) ─────────────────
    engagement_score = (
        0.25 * min(profile_views_30d / 30.0, 1.0)
        + 0.20 * min(applications_30d / 8.0, 1.0)
        + 0.15 * min(search_appearances_30d / 400.0, 1.0)
        + 0.15 * offer_acceptance_rate
        + 0.10 * profile_completeness / 100.0
        + 0.10 * min(connection_count / 500.0, 1.0)
        + 0.05 * (1.0 if linkedin_connected else 0.3)
    )

    return {
        "candidate_id": cid,
        # Profile basics
        "yoe": yoe,
        "country": country,
        "location": location,
        "current_title": current_title,
        "current_company": current_company,
        "current_industry": current_industry,
        # Career structure
        "n_roles": n_roles,
        "total_career_months": total_career_months,
        "product_ratio": product_ratio,
        "product_months": product_months,
        "short_stints": short_stints,
        "long_stints": long_stints,
        "max_tenure": max_tenure,
        "all_consulting": all_consulting,
        "recent_is_services": recent_is_services,
        # Text signals (from career descriptions)
        "retrieval_in_desc": retrieval_in_desc,
        "eval_in_desc": eval_in_desc,
        "embedding_in_desc": embedding_in_desc,
        "vector_db_in_desc": vector_db_in_desc,
        "cv_speech_in_desc": cv_speech_in_desc,
        "research_signals": research_signals,
        "llm_wrapper_in_desc": llm_wrapper_in_desc,
        "llm_wrapper_only": llm_wrapper_only,
        # Classification flags
        "is_non_tech_title": is_non_tech_title,
        "is_cv_speech_primary": is_cv_speech_primary,
        "keyword_stuffer": keyword_stuffer,
        "candidate_ai_skills": candidate_ai_skills,
        "recent_handon": recent_handon,
        # Skills stats
        "expert_zero_count": expert_zero_count,
        "max_skill_duration": max_skill_duration,
        # Behavioral signals — core
        "days_since_active": days_since_active,
        "open_to_work": open_to_work,
        "response_rate": response_rate,
        "avg_response_hours": avg_response_h,
        "notice_days": notice_days,
        "willing_to_relocate": willing_to_relocate,
        "github_score": github_score,
        "saved_30d": saved_30d,
        "interview_rate": interview_rate,
        "profile_completeness": profile_completeness,
        # Behavioral signals — new (all 23 used)
        "profile_views_30d": profile_views_30d,
        "applications_30d": applications_30d,
        "search_appearances_30d": search_appearances_30d,
        "connection_count": connection_count,
        "endorsements_received": endorsements_received,
        "offer_acceptance_rate": offer_acceptance_rate,
        "linkedin_connected": linkedin_connected,
        "verified_email": verified_email,
        "verified_phone": verified_phone,
        "preferred_work_mode": preferred_work_mode,
        "salary_min": salary_min,
        "salary_max": salary_max,
        # Derived composite signals
        "skill_assessment_composite": skill_assessment_composite,
        "has_skill_assessments": has_skill_assessments,
        "notice_score": notice_score,
        "github_norm": github_norm,
        "engagement_score": engagement_score,
    }


def load_candidates(
    path: str, limit: int = 0
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Parse candidates.jsonl into (df, text_blobs, raw_records).

    Returns:
      df:          pd.DataFrame, one row per candidate, all structured fields
      text_blobs:  dict[candidate_id -> blob_str]  — for embedding / BM25
      raw:         dict[candidate_id -> dict]       — full JSON for reason.py
    """
    rows = []
    text_blobs = {}
    raw = {}

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = c["candidate_id"]
            rows.append(extract_row(c))
            text_blobs[cid] = build_text_blob(c)
            raw[cid] = c

    df = pd.DataFrame(rows)
    df.set_index("candidate_id", inplace=True)
    return df, text_blobs, raw


# ── Sanity printer ─────────────────────────────────────────────────────────────

def _print_candidate_summary(cid: str, df: pd.DataFrame, raw: dict) -> None:
    row = df.loc[cid]
    p = raw[cid]["profile"]
    print(
        f"\n{cid}  {p['current_title']} @ {p['current_company']}"
        f"  YOE={row['yoe']:.1f}  {row['country']}"
    )
    print(
        f"  retrieval={row['retrieval_in_desc']}  eval={row['eval_in_desc']}"
        f"  embedding={row['embedding_in_desc']}  vecdb={row['vector_db_in_desc']}"
    )
    print(
        f"  product_ratio={row['product_ratio']:.2f}  all_consulting={row['all_consulting']}"
        f"  kw_stuffer={row['keyword_stuffer']}  response_rate={row['response_rate']:.2f}"
    )
    print(
        f"  consistency: expert_zero={row['expert_zero_count']}"
        f"  max_skill_mo={row['max_skill_duration']}  career_mo={row['total_career_months']}"
    )


if __name__ == "__main__":
    import argparse, random

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="../candidates.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("Loading candidates...")
    df, blobs, raw = load_candidates(args.candidates, args.limit)
    print(f"Loaded {len(df):,} candidates")
    print(f"DataFrame shape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")

    # Print 5 random
    print("\n\n=== 5 RANDOM CANDIDATES ===")
    for cid in random.sample(list(df.index), min(5, len(df))):
        _print_candidate_summary(cid, df, raw)

    # Print 5 strong (retrieval in desc, India)
    strong = df[df["retrieval_in_desc"] & (df["country"] == "India")]
    print(f"\n\n=== 5 STRONG CANDIDATES (retrieval in desc, India) ===")
    print(f"Pool size: {len(strong):,}")
    for cid in list(strong.index[:5]):
        _print_candidate_summary(cid, df, raw)

    # Print 1 honeypot candidate
    honeypot = df[
        (df["expert_zero_count"] >= 2) |
        ((df["max_skill_duration"] > df["total_career_months"] * 1.5) & (df["total_career_months"] > 0))
    ]
    print(f"\n\n=== 1 HONEYPOT CANDIDATE (internal contradiction) ===")
    print(f"Pool size (loose): {len(honeypot):,}")
    if len(honeypot) > 0:
        cid = list(honeypot.index)[0]
        _print_candidate_summary(cid, df, raw)
