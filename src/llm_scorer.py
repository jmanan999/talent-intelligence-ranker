"""
llm_scorer.py — Semantic pre-scoring for top candidates (precompute phase only).

Supports Anthropic native API and any OpenAI-compatible API (OpenRouter, Groq, DeepSeek).
Results saved to artifacts/llm_scores.parquet; loaded by rank.py at ranking time.
Zero API calls during the 5-minute ranking step.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DIM_WEIGHTS = {
    "retrieval_depth":    0.35,
    "production_quality": 0.30,
    "fit_trajectory":     0.20,
    "antipattern_score":  0.15,
}

SCORE_PROMPT_TEMPLATE = """\
You are a technical hiring filter for a Senior AI Engineer role at Redrob AI (Series A startup, Pune/Noida India).

HARD REQUIREMENTS:
- Production embeddings-based retrieval (sentence-transformers, BGE, Pinecone, FAISS, Weaviate, Qdrant, Elasticsearch)
- Evaluation frameworks for ranking (NDCG, MRR, MAP, A/B tests, offline-to-online)
- 5-9 years, 4-5 in applied ML/AI at PRODUCT companies (not services firms, not research)
- Strong Python

DISQUALIFIERS:
1. Entire career at services firms (TCS/Infosys/Wipro/Accenture/Capgemini) with no product experience
2. AI experience is only recent LangChain wrapping, no pre-LLM ML production history
3. Primary domain is CV/speech/robotics without NLP/IR exposure
4. Pure academia, no production deployment
5. Title-chasing: many 1-1.5 year stints

CANDIDATE:
Title: {title} | {yoe:.1f} years
Company: {company} ({industry})
Location: {location}, {country}

Career (most recent first):
{career_text}

IMPORTANT RULES — failure to follow these disqualifies your response:
1. Score ONLY what is explicitly stated in the career text above. Do NOT infer, assume, or guess skills not mentioned.
2. If the career text is vague or templated, score conservatively (5 or below).
3. The "note" must reference a specific detail from the career text — do not make up company names, projects, or numbers.
4. Scores must be integers 0-10. Do not use decimals or ranges.

Rate 0-10 (integers only):
- retrieval_depth: production retrieval/ranking/recsys evidence EXPLICITLY described in career text (0=none/vague, 10=specific systems with tech details)
- production_quality: real deployment at scale EXPLICITLY mentioned with metrics/numbers (0=no evidence, 10=multiple prod systems with explicit metrics)
- fit_trajectory: career path toward AI at product companies based on ACTUAL role history (0=wrong direction, 10=ideal trajectory)
- antipattern_score: absence of disqualifiers based on ACTUAL career (0=clear disqualifiers, 10=none present)

Reply with ONLY valid JSON, no other text:
{{"retrieval_depth": X, "production_quality": X, "fit_trajectory": X, "antipattern_score": X, "note": "one sentence citing a specific detail from their career text"}}"""


def _build_career_text(raw: dict, max_roles: int = 3, max_chars: int = 280) -> str:
    from datetime import datetime, date
    career = raw.get("career_history", []) or []

    def role_date(r):
        try:
            return datetime.strptime((r.get("start_date") or "")[:10], "%Y-%m-%d").date()
        except Exception:
            return date(2000, 1, 1)

    parts = []
    for r in sorted(career, key=role_date, reverse=True)[:max_roles]:
        months = r.get("duration_months", 0) or 0
        duration = f"{months//12}y{months%12}m" if months else "?"
        desc = (r.get("description", "") or "")[:max_chars]
        parts.append(f"[{r.get('title','')} @ {r.get('company','')} — {duration}]: {desc}")
    return "\n".join(parts)


def _build_prompt(cid: str, raw: dict, df_row: pd.Series) -> str:
    return SCORE_PROMPT_TEMPLATE.format(
        title=df_row.get("current_title", "Unknown"),
        yoe=float(df_row.get("yoe", 0)),
        company=df_row.get("current_company", "Unknown"),
        industry=df_row.get("current_industry", "Unknown"),
        location=df_row.get("location", "Unknown"),
        country=df_row.get("country", "Unknown"),
        career_text=_build_career_text(raw),
    )


def _parse_llm_response(text: str, cid: str) -> Optional[dict]:
    text = text.strip()
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            obj = json.loads(text[start:end])
            dims = ["retrieval_depth", "production_quality", "fit_trajectory", "antipattern_score"]
            for d in dims:
                if d not in obj:
                    return None
                obj[d] = max(0, min(10, int(obj[d])))
            obj["llm_overall"] = round(sum(DIM_WEIGHTS[d] * obj[d] for d in dims) / 10.0, 4)
            obj["candidate_id"] = cid
            return obj
    except Exception:
        pass
    return None


# ── OpenAI-compatible async scorer (OpenRouter, Groq, DeepSeek, etc.) ─────────

async def _score_one(client, cid: str, raw: dict, df_row: pd.Series,
                     semaphore: asyncio.Semaphore, model: str) -> dict:
    async with semaphore:
        prompt = _build_prompt(cid, raw, df_row)
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=160,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                text = resp.choices[0].message.content or ""
                parsed = _parse_llm_response(text, cid)
                if parsed:
                    return parsed
                return {"candidate_id": cid, "llm_overall": 0.5, "note": "parse_error"}
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    wait = 4 ** attempt          # 1s, 4s, 16s back-off
                    await asyncio.sleep(wait)
                else:
                    return {"candidate_id": cid, "llm_overall": 0.5, "note": err[:60]}
        return {"candidate_id": cid, "llm_overall": 0.5, "note": "rate_limit_exhausted"}


async def _run_async(candidate_ids, raw_records, df, api_key, base_url, model,
                     max_concurrent, extra_headers) -> pd.DataFrame:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url=base_url,
                         default_headers=extra_headers or {})
    semaphore = asyncio.Semaphore(max_concurrent)

    ids_to_score = [c for c in candidate_ids if c in raw_records and c in df.index]
    tasks = [_score_one(client, c, raw_records[c], df.loc[c], semaphore, model)
             for c in ids_to_score]

    t0 = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    ok = sum(1 for r in results if r.get("llm_overall", 0.5) != 0.5)
    print(f"  Done: {len(results)} scored in {elapsed:.1f}s "
          f"({len(results)/max(elapsed,1):.1f}/s) | {ok} valid responses")
    return pd.DataFrame([r for r in results if r])


# ── Public API ─────────────────────────────────────────────────────────────────

PROVIDERS = {
    # key_prefix  → (base_url, default_model)
    "sk-ant":  ("https://api.anthropic.com/v1",     "claude-haiku-4-5-20251001"),
    "sk-or-":  ("https://openrouter.ai/api/v1",     "deepseek/deepseek-chat-v3-0324:free"),
    "gsk_":    ("https://api.groq.com/openai/v1",   "llama-3.3-70b-versatile"),
    "sk-":     ("https://api.deepseek.com",          "deepseek-chat"),   # DeepSeek
}


def resolve_provider(api_key: Optional[str] = None, model: Optional[str] = None,
                     base_url: Optional[str] = None):
    """Return (api_key, base_url, model, extra_headers)."""
    key = api_key or ""
    # Try env vars if no explicit key
    for env in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY"):
        if not key:
            key = os.environ.get(env, "")

    if not key:
        raise ValueError("No API key. Set OPENROUTER_API_KEY / GROQ_API_KEY / DEEPSEEK_API_KEY / ANTHROPIC_API_KEY")

    # Infer provider from key prefix
    if base_url is None:
        for prefix, (url, _) in PROVIDERS.items():
            if key.startswith(prefix):
                base_url = url
                break
        if base_url is None:
            base_url = "https://openrouter.ai/api/v1"   # default fallback

    if model is None:
        for prefix, (_, mdl) in PROVIDERS.items():
            if key.startswith(prefix):
                model = mdl
                break
        if model is None:
            model = "deepseek/deepseek-chat-v3-0324:free"

    extra_headers = {}
    if "openrouter" in base_url:
        extra_headers = {
            "HTTP-Referer": "https://github.com/redrob-ranker",
            "X-Title": "Redrob Candidate Ranker",
        }

    return key, base_url, model, extra_headers


def run_llm_scoring(
    candidate_ids: list,
    raw_records: dict,
    df: pd.DataFrame,
    output_path: Path,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    max_concurrent: int = 15,
) -> pd.DataFrame:
    """Score top candidates and save to parquet. Called during precompute only."""
    key, b_url, mdl, headers = resolve_provider(api_key, model, base_url)

    is_free = mdl.endswith(":free")
    if is_free and max_concurrent > 12:
        max_concurrent = 12   # free models have stricter rate limits

    print(f"  Provider: {b_url.split('/')[2]}")
    print(f"  Model:    {mdl}  ({'FREE' if is_free else 'paid'})")
    print(f"  Scoring:  {len(candidate_ids)} candidates (concurrent={max_concurrent})")
    if is_free:
        est = len(candidate_ids) / max_concurrent * 3
        print(f"  Est. time: {est/60:.0f}–{est*2/60:.0f} min (free-tier rate limits)")

    scores_df = asyncio.run(_run_async(
        candidate_ids, raw_records, df, key, b_url, mdl, max_concurrent, headers
    ))

    for col in ["retrieval_depth", "production_quality", "fit_trajectory",
                "antipattern_score", "llm_overall"]:
        if col not in scores_df.columns:
            scores_df[col] = 5.0 if col != "llm_overall" else 0.5

    output_path = Path(output_path)
    scores_df.to_parquet(output_path, index=False)
    print(f"  Saved → {output_path}")
    return scores_df


def load_llm_scores(path: Path) -> Optional[pd.DataFrame]:
    """Load pre-computed LLM scores. Only returns VALID scores (not rate-exhausted defaults).

    Critical: candidates with llm_overall==0.5 (rate_exhausted/parse_error) are excluded
    so they fall back to feature-only scoring rather than being penalised by a fake 0.5.
    """
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "candidate_id" not in df.columns or "llm_overall" not in df.columns:
        return None
    df = df.set_index("candidate_id")
    # Only keep genuinely scored candidates — exclude rate_exhausted (0.5 default)
    valid = df[df["llm_overall"] != 0.5]
    print(f"  Loaded LLM scores: {len(valid):,} valid (of {len(df):,} total) from {path}")
    return valid if len(valid) > 0 else None
