# Redrob Candidate Ranker

> **"Understand the career, not the keywords."**

Production-grade candidate ranking system for the Redrob Intelligent Candidate Discovery hackathon.
Ranks the top 100 of 100,000 candidates for a Senior AI Engineer role at a Series A startup —
by reading career history, not keyword lists.

## The Problem

The sample baseline ranks candidates by skill keyword density.
This is the trap. Our system reads `career_history[].description` — career narratives, not skill lists —
and uses Claude Haiku to semantically understand the QUALITY of evidence, not just its presence.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Precompute (one-time, ~20–30 min on Apple MPS / ~60 min on CPU)

```bash
python src/precompute.py --candidates ./candidates.jsonl
```

Downloads `BAAI/bge-small-en-v1.5` (~130 MB), embeds all 100K candidates,
builds BM25 index over career descriptions, saves feature table to `artifacts/`.

### 3. (Optional) LLM semantic scoring

```bash
# ANTHROPIC_API_KEY or OPENROUTER_API_KEY or GROQ_API_KEY
GROQ_API_KEY=... python run_llm_scoring.py --top-n 500
```

**This step is optional — `artifacts/llm_scores.parquet` is already committed.**
`rank.py` requires zero API keys. Clone → precompute embeddings → rank. Done.

If you want to regenerate LLM scores: uses Groq/OpenRouter (free tier available),
runs in precompute phase only. Results saved to `artifacts/llm_scores.parquet`.

### 4. Rank (≤5 minutes, CPU only, no network)

```bash
python src/rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### 5. Validate

```bash
python validate_submission.py ./submission.csv
# Expected: "Submission is valid."
```

---

## Architecture

```
candidates.jsonl
    │
    ▼ [once, offline — can use network]
precompute.py ──────────────────────────→ artifacts/
    ├── embeddings.npy         (100K × 384, float32, L2-normalized)
    ├── anchor_embeddings.npy  (7 JD facets × 384)
    ├── ids.npy
    ├── bm25_index.pkl
    ├── bm25_ids.json
    ├── feature_table.parquet
    └── llm_scores.parquet     ← Claude Haiku scores for top 5000
    │
    ▼ [≤5 min, CPU only, no network — reproducible in Docker]
rank.py ──→ submission.csv
    1. load artifacts (sub-second)
    2. parse candidates.jsonl  (~78s)
    3. consistency.py → honeypot multiplier (all 23 signals)
    4. signals.py    → availability multiplier [0.55–1.00] (9 signals)
    5. features.py   → 12 vectorized features
    6. score.py      → blend(LLM scores, feature score) × 6 penalties
    7. reason.py     → grounded per-candidate reasoning (LLM note aware)
    8. write + validate
```

## Key Design Decisions

### Why descriptions ≫ skills list
The `skills[]` array is trivially gameable — candidates add AI keywords to look relevant.
Career descriptions require actual experience. We weight descriptions 3× and place the
skill list at the very end of the BM25 corpus text.

### Honeypot detection
~80 candidates are disguised as perfect ML experts but have internal numerical
contradictions (e.g., claims 6 years of experience but tenure totals 18 months;
skills used for longer than the candidate's entire career). Detected by 4 structural
checks — never hardcoded IDs.

### Multiplicative penalties (not hard filters)
Each negative signal (keyword stuffer: 0.10×, all-consulting: 0.35×, international
without relocate: 0.50×) is applied multiplicatively. This keeps the ranking smooth
and defensible — no brittle cutoffs.

### Availability multiplier
A perfect-on-paper candidate who hasn't logged in for 6 months and replies to 5%
of recruiter messages is not actually hirable. We discount them to ~0.62× without
zeroing out their fit signal.

---

## Repository Structure

```
redrob-ranker/
├── README.md
├── requirements.txt
├── submission_metadata.yaml
├── rubric.yaml          ← distilled JD rubric (must-haves, penalties, weights)
├── config.yaml          ← weights, paths, thresholds
├── recon.py             ← data exploration / §2 verification
├── src/
│   ├── parse.py         ← JSONL → DataFrame + text blobs
│   ├── precompute.py    ← embeddings + BM25 + anchors → artifacts/
│   ├── consistency.py   ← honeypot / internal-contradiction detector
│   ├── features.py      ← vectorized feature engineering
│   ├── signals.py       ← behavioral availability multiplier
│   ├── score.py         ← composite scorer + penalties
│   ├── ltr.py           ← LightGBM LambdaMART calibration
│   ├── reason.py        ← grounded reasoning generator
│   └── rank.py          ← ENTRYPOINT
├── eval/
│   ├── metrics.py       ← NDCG@k, MAP, P@k
│   ├── pseudo_labels.py ← rubric-based weak supervision labels
│   └── ablations.py     ← component ablations + stability analysis
├── tests/
│   └── test_submission.py
├── sandbox/
│   ├── app.py           ← Streamlit demo
│   └── Dockerfile
├── artifacts/           ← (gitignored) precomputed data
└── deck/
    └── approach.md      ← slide deck source (Marp) → deck.pdf
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Core unit tests (no artifacts needed):
- Metrics: DCG/NDCG/MAP/P@k correctness
- Consistency: honeypot detector catches known contradictions
- Availability: multiplier stays in [0.60, 1.00]

Integration tests (require submission.csv and candidates.jsonl):
- Validator passes
- No keyword stuffers in top-10
- Honeypot rate < 10% in top-100
- Scores monotonically non-increasing

## Sandbox Demo

```bash
streamlit run sandbox/app.py
```

Or via Docker:
```bash
docker build -t redrob-ranker .
docker run -p 8501:8501 redrob-ranker
```

Upload ≤100 candidates (JSON array) → ranked CSV output. Uses the same pipeline
as `rank.py` with zero network access during ranking.

## Compute Constraints

| Constraint | Requirement | Actual |
|------------|-------------|--------|
| Wall-clock (rank step) | ≤5 min | ~90s |
| RAM | ≤16 GB | ~4 GB |
| Device | CPU only | CPU only ✓ |
| Network (rank step) | None | None ✓ |
| Disk | ≤5 GB | ~200 MB |

## Evaluation

### Scoring formula
```
Composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
```

### Pseudo-label ablations

| Variant | NDCG@10 | Honeypots | KW-Stuffers |
|---------|---------|-----------|-------------|
| Dense only | 0.41 | 2 | 12 |
| Hybrid (no signals) | 0.58 | 1 | 8 |
| Full composite | 0.84 | 0 | 0 |

---

*Built for the Redrob Intelligent Candidate Discovery & Ranking hackathon.*
*AI tools: Claude (architecture + code review); Groq/OpenRouter LLMs for semantic pre-scoring of top candidates (precompute phase only — zero LLM calls during the 5-minute ranking step).*
