#!/usr/bin/env python3
"""
run_llm_scoring.py — Standalone Claude Haiku semantic scoring script.

Runs AFTER precompute.py has generated embeddings + BM25.
Picks the top-N candidates by initial feature score and scores them
with Claude Haiku to produce artifacts/llm_scores.parquet.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python run_llm_scoring.py
    python run_llm_scoring.py --top-n 5000 --candidates path/to/candidates.jsonl

Requirements:
    pip install anthropic
    ANTHROPIC_API_KEY must be set in environment.

This script runs BEFORE rank.py. rank.py will automatically detect and
use llm_scores.parquet if it exists in the artifacts directory.

Runtime: ~3-5 minutes for 5000 candidates (async, 25 concurrent).
Cost: ~$1.00 with Claude Haiku at $0.80/1M input + $4/1M output tokens.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.parse import load_candidates
from src.features import compute_features
from src.consistency import compute_consistency_scores
from src.signals import compute_availability
from src.score import compute_base_score
from src.llm_scorer import run_llm_scoring


CONFIG_PATH = REPO_ROOT / "config.yaml"


def main():
    ap = argparse.ArgumentParser(description="Run LLM scoring on top candidates.")
    ap.add_argument("--candidates", default=None, help="Path to candidates.jsonl")
    ap.add_argument("--artifacts", default=None, help="Artifacts directory")
    ap.add_argument("--top-n", type=int, default=5000,
                    help="Number of top candidates to score (default: 5000)")
    ap.add_argument("--max-concurrent", type=int, default=25,
                    help="Max concurrent API calls (default: 25)")
    ap.add_argument("--api-key", default=None,
                    help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if llm_scores.parquet already exists")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit candidates for testing (0=all)")
    args = ap.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    candidates_path = args.candidates or cfg["paths"]["candidates"]
    art_dir = Path(args.artifacts or cfg["paths"]["artifacts"])
    output_path = art_dir / "llm_scores.parquet"

    if output_path.exists() and not args.force:
        print(f"llm_scores.parquet already exists ({output_path})")
        print("Use --force to re-run. Current scores:")
        df_existing = pd.read_parquet(output_path)
        print(f"  {len(df_existing)} candidates scored")
        if "llm_overall" in df_existing.columns:
            print(f"  llm_overall: mean={df_existing['llm_overall'].mean():.3f}, "
                  f"std={df_existing['llm_overall'].std():.3f}")
        return

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable or use --api-key")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"LLM Scoring — {args.top_n} candidates with Claude Haiku")
    print(f"  Candidates: {candidates_path}")
    print(f"  Artifacts:  {art_dir}")
    print(f"  Output:     {output_path}")
    print(f"{'='*60}\n")

    # ── Load precomputed artifacts ────────────────────────────────────────────
    print("[1/4] Loading precomputed artifacts ...")
    embeddings        = np.load(art_dir / "embeddings.npy")
    anchor_embeddings = np.load(art_dir / "anchor_embeddings.npy")
    ids               = list(np.load(art_dir / "ids.npy", allow_pickle=True))
    with open(art_dir / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(art_dir / "bm25_ids.json") as f:
        bm25_ids = json.load(f)

    # ── Parse candidates ──────────────────────────────────────────────────────
    print(f"[2/4] Parsing candidates from {candidates_path} ...")
    t0 = time.time()
    df, _, raw = load_candidates(candidates_path, limit=args.limit)
    print(f"  Parsed {len(df):,} candidates in {time.time()-t0:.1f}s")

    # ── Compute fast feature score to pick top-N ──────────────────────────────
    print(f"[3/4] Computing fast feature scores to select top-{args.top_n} ...")
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    aligned   = embeddings[[id_to_idx[cid] for cid in df.index if cid in id_to_idx]]

    feat = compute_features(df, aligned, anchor_embeddings, bm25, bm25_ids)
    fast_scores = compute_base_score(feat, cfg["weights"])
    fast_series = pd.Series(fast_scores, index=df.index).sort_values(ascending=False)
    top_ids = list(fast_series.head(args.top_n).index)

    print(f"  Selected top-{len(top_ids)} by fast feature score")
    print(f"  Score range: {fast_series.iloc[0]:.4f} → {fast_series.iloc[len(top_ids)-1]:.4f}")

    # ── Run LLM scoring ───────────────────────────────────────────────────────
    print(f"\n[4/4] Running Claude Haiku scoring on {len(top_ids)} candidates ...")
    t0 = time.time()
    scores_df = run_llm_scoring(
        candidate_ids=top_ids,
        raw_records=raw,
        df=df,
        output_path=output_path,
        api_key=api_key,
        max_concurrent=args.max_concurrent,
    )

    print(f"\n✓ LLM scoring complete in {time.time()-t0:.1f}s")
    print(f"  Scored: {len(scores_df):,} candidates")
    if "llm_overall" in scores_df.columns:
        print(f"  llm_overall stats: {scores_df['llm_overall'].describe().round(3).to_dict()}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
