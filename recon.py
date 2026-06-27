#!/usr/bin/env python3
"""
recon.py — Verify the §2 data-composition claims before baking them in.

Run: python recon.py --candidates ../candidates.jsonl
Prints a short report; takes ~60 s on the full 100K file.
"""

import argparse
import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

TODAY = date(2026, 6, 19)

# ── Keyword sets ──────────────────────────────────────────────────────────────

AI_SKILLS = {
    "rag", "pinecone", "faiss", "embeddings", "vector search", "semantic search",
    "sentence transformers", "llms", "fine-tuning llms", "recommendation systems",
    "information retrieval", "langchain", "weaviate", "qdrant", "milvus",
    "openai", "hugging face", "transformers", "bert", "gpt", "llm", "nlp",
    "deep learning", "machine learning", "pytorch", "tensorflow", "scikit-learn",
    "xgboost", "lightgbm", "a/b testing", "mlflow", "weights & biases",
    "elasticsearch", "opensearch", "bm25", "ranking", "recsys",
    "vector database", "vector db", "hybrid search",
}

PRODUCT_AI_COMPANIES = {
    "sarvam ai", "cred", "razorpay", "paytm", "zoho", "mad street den",
    "yellow.ai", "verloop.io", "unacademy", "nykaa", "freshworks",
    "swiggy", "zomato", "flipkart", "meesho", "ola", "rapido",
    "phonepe", "groww", "zepto", "blinkit", "myntra", "lenskart",
    "slice", "browserstack", "cleartax", "chargebee", "postman",
    "leadsquared", "darwinbox", "hasura", "setu", "signzy",
    "artivatic", "observe.ai", "sprinklr", "keka", "darwinbox",
    "haptik", "niki.ai", "juspay", "nium", "open financial",
}

SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "lti", "mphasis",
    "l&t infotech", "hexaware", "mphasis", "niit technologies",
    "syntel", "persistent systems", "birlasoft",
}

RETRIEVAL_KEYWORDS = [
    r"\branking\b", r"\bretrieval\b", r"\brecommendation\b", r"\bsearch\b",
    r"\bembedding", r"\bvector\b", r"\bndcg\b", r"\bmrr\b", r"\bmap\b",
    r"\blearning.to.rank\b", r"\bltr\b", r"\bpersonali[sz]", r"\breranki",
    r"\bsemantic search\b", r"\bhybrid search\b", r"\brag\b",
]
RETRIEVAL_RE = re.compile("|".join(RETRIEVAL_KEYWORDS), re.I)

HONEYPOT_KEYS = [
    "expert_zero_duration",   # proficiency=expert, duration_months=0
    "skill_exceeds_career",   # a skill duration > total career months
    "tenure_exceeds_career",  # a job tenure > years_of_experience * 12 + 24
    "yoe_mismatch",           # sum of tenures > yoe * 12 * 2 or < yoe * 12 * 0.3
]


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def career_months(career):
    return sum(r.get("duration_months", 0) for r in career)


def is_honeypot(c):
    flags = []
    yoe = c["profile"].get("years_of_experience", 0) or 0
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    total_career_mo = career_months(career)
    yoe_months = yoe * 12

    # expert with 0 months on multiple skills
    expert_zeros = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zeros >= 2:
        flags.append("expert_zero_duration")

    # skill duration exceeds total career
    for s in skills:
        if s.get("duration_months", 0) > total_career_mo + 6:
            flags.append("skill_exceeds_career")
            break

    # yoe wildly smaller than sum of tenures
    if total_career_mo > yoe_months * 2.5 and yoe_months > 0:
        flags.append("yoe_mismatch_high")

    # yoe wildly larger than tenures (ghost career)
    if total_career_mo < yoe_months * 0.25 and yoe_months > 24:
        flags.append("yoe_mismatch_low")

    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="../candidates.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    args = ap.parse_args()

    path = Path(args.candidates)
    assert path.exists(), f"File not found: {path}"

    total = 0
    country_counter = Counter()
    location_counter = Counter()
    title_counter = Counter()
    skill_names = Counter()
    ai_skill_counts = Counter()  # per-candidate count of AI skills
    all_consulting_count = 0
    keyword_stuffers = 0   # non-technical + 6+ AI skills
    strong_candidates = 0  # retrieval evidence in descriptions
    honeypot_count = 0
    honeypot_examples = []
    strong_examples = []

    TECH_TITLE_RE = re.compile(
        r"\b(ml|machine learning|data scien|nlp|search|recsys|rec sys|"
        r"recommendation|ai engineer|software engineer|software developer|"
        r"backend|data engineer|platform engineer|research scientist|"
        r"applied scientist|deep learning|llm|ir engineer|information retrieval)\b",
        re.I,
    )

    NON_TECH_RE = re.compile(
        r"\b(hr |human resource|accountant|content writer|graphic design|"
        r"business analyst|sales|marketing|civil engineer|mechanical engineer|"
        r"administrative|recruiter|teacher|professor|nurse|doctor|lawyer|"
        r"project manager)\b",
        re.I,
    )

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1

            p = c["profile"]
            country_counter[p.get("country", "Unknown")] += 1
            location_counter[p.get("location", "Unknown")] += 1
            title_counter[p.get("current_title", "Unknown")] += 1

            # Skills
            candidate_skills = {s["name"].lower() for s in c.get("skills", [])}
            for sn in candidate_skills:
                skill_names[sn] += 1

            ai_count = sum(1 for s in candidate_skills if s in AI_SKILLS)
            ai_skill_counts[ai_count] += 1

            # Keyword-stuffer: non-technical title + 6+ AI skills
            ctitle = p.get("current_title", "")
            is_non_tech = bool(NON_TECH_RE.search(ctitle)) and not bool(TECH_TITLE_RE.search(ctitle))
            if is_non_tech and ai_count >= 6:
                keyword_stuffers += 1

            # All-consulting career
            career = c.get("career_history", [])
            if career:
                companies_lower = {r.get("company", "").lower() for r in career}
                all_srv = all(
                    any(sf in co for sf in SERVICES_FIRMS)
                    for co in companies_lower
                )
                if all_srv:
                    all_consulting_count += 1

            # Strong candidates: retrieval/ranking evidence in career descriptions
            desc_text = " ".join(r.get("description", "") for r in career)
            has_retrieval = bool(RETRIEVAL_RE.search(desc_text))
            india_or_relocate = (
                p.get("country", "") == "India"
                or c.get("redrob_signals", {}).get("willing_to_relocate", False)
            )
            if has_retrieval and india_or_relocate:
                strong_candidates += 1
                if len(strong_examples) < 3:
                    strong_examples.append({
                        "id": c["candidate_id"],
                        "title": ctitle,
                        "company": p.get("current_company"),
                        "yoe": p.get("years_of_experience"),
                        "country": p.get("country"),
                        "desc_snippet": desc_text[:200],
                    })

            # Honeypot detection
            hp_flags = is_honeypot(c)
            if hp_flags:
                honeypot_count += 1
                if len(honeypot_examples) < 3:
                    honeypot_examples.append({
                        "id": c["candidate_id"],
                        "title": ctitle,
                        "yoe": p.get("years_of_experience"),
                        "career_months": career_months(career),
                        "flags": hp_flags,
                    })

            if total % 10000 == 0:
                print(f"  ...processed {total:,}")

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"REDROB RECON REPORT")
    print(f"{'='*60}")
    print(f"\nTotal candidates:      {total:,}")
    print(f"\n--- Country breakdown (top 10) ---")
    for country, cnt in country_counter.most_common(10):
        pct = cnt / total * 100
        print(f"  {country:<30} {cnt:>7,}  ({pct:5.1f}%)")

    print(f"\n--- Top 20 locations ---")
    for loc, cnt in location_counter.most_common(20):
        print(f"  {loc:<30} {cnt:>7,}")

    print(f"\n--- Top 20 job titles ---")
    for title, cnt in title_counter.most_common(20):
        print(f"  {title:<40} {cnt:>6,}")

    print(f"\n--- Unique skill names: {len(skill_names):,} ---")
    print(f"  Top 20 skills:")
    for sk, cnt in skill_names.most_common(20):
        print(f"    {sk:<35} {cnt:>6,}")

    print(f"\n--- AI skill count distribution (per candidate) ---")
    for k in sorted(ai_skill_counts.keys()):
        if k >= 3:
            print(f"  AI skills >= {k}: {sum(v for kk,v in ai_skill_counts.items() if kk>=k):,}")
    print(f"  Keyword stuffers (non-tech + 6+ AI skills): {keyword_stuffers:,}")

    print(f"\n--- Service-firm candidates (all-consulting career): {all_consulting_count:,} ---")
    print(f"\n--- Strong candidates (retrieval in desc + India/relocate): {strong_candidates:,} ---")
    print(f"\n--- Honeypot candidates (internal contradiction): {honeypot_count:,} ---")

    print(f"\n--- STRONG EXAMPLES (first 3) ---")
    for ex in strong_examples:
        print(f"  {ex['id']}  {ex['title']} @ {ex['company']}  YOE={ex['yoe']}  {ex['country']}")
        print(f"    desc: {ex['desc_snippet'][:120]}...")

    print(f"\n--- HONEYPOT EXAMPLES (first 3) ---")
    for ex in honeypot_examples:
        print(f"  {ex['id']}  {ex['title']}  YOE={ex['yoe']}  career_mo={ex['career_months']}  flags={ex['flags']}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
