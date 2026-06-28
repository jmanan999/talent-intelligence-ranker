"""
rank.py — ENTRYPOINT: candidates.jsonl → submission.csv (≤5 min, CPU, no net).

Usage:
  python src/rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints enforced:
  - ≤5 min wall-clock
  - ≤16 GB RAM
  - CPU only (no GPU, no MPS)
  - No network access during ranking (embeddings loaded from disk)
  - Emits exactly 100 rows, ranks 1–100, monotonic non-increasing score

Pipeline:
  1. Load precomputed artifacts from artifacts/ (sub-second)
  2. Parse candidates.jsonl into DataFrame (≈60 s)
  3. Align embeddings to DataFrame order (vectorized)
  4. Compute consistency + availability (vectorized)
  5. Compute features (vectorized)
  6. Score: composite + penalties (vectorized)
  7. Optional: LTR calibration (LightGBM predict, sub-second)
  8. Generate reasoning for top-100
  9. Write submission.csv
  10. Run validate_submission.py (assert passes)
"""

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Force CPU — constraint compliance and reproducibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.parse import load_candidates, build_embed_text
from src.consistency import compute_consistency_scores
from src.signals import compute_availability
from src.features import compute_features
from src.score import compute_fit_score, compute_fit_score_with_llm, sort_for_submission, load_weights
from src.reason import generate_all_reasonings
from src.llm_scorer import load_llm_scores

# Optional LTR (silently skipped if model not found)
try:
    from src.ltr import load_ltr_model, build_ltr_features, ltr_predict
    LTR_AVAILABLE = True
except ImportError:
    LTR_AVAILABLE = False


def load_artifacts(artifacts_dir: Path):
    """Load all precomputed artifacts from disk."""
    t = time.time()
    embeddings        = np.load(artifacts_dir / "embeddings.npy")
    anchor_embeddings = np.load(artifacts_dir / "anchor_embeddings.npy")
    ids               = list(np.load(artifacts_dir / "ids.npy", allow_pickle=True))

    with open(artifacts_dir / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(artifacts_dir / "bm25_ids.json") as f:
        bm25_ids = json.load(f)

    print(f"  Artifacts loaded in {time.time()-t:.1f}s")
    return embeddings, anchor_embeddings, ids, bm25, bm25_ids


def align_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    ids: list[str],
) -> np.ndarray:
    """
    Reorder the (N, D) embeddings matrix to match df.index order.
    Candidates not in the index get a zero embedding (→ zero dense score).
    """
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    D = embeddings.shape[1]
    result = np.zeros((len(df), D), dtype=np.float32)
    for i, cid in enumerate(df.index):
        if cid in id_to_idx:
            result[i] = embeddings[id_to_idx[cid]]
    return result


def write_submission_csv(
    top100_df: pd.DataFrame,   # columns: candidate_id, rank, score, reasoning
    out_path: Path,
) -> None:
    """Write submission CSV with exact header and format required by the validator."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for _, row in top100_df.iterrows():
            # Score to 6 decimal places, reasoning cleaned of internal newlines
            reasoning = str(row["reasoning"]).replace("\n", " ").replace("\r", " ").strip()
            writer.writerow([
                row["candidate_id"],
                int(row["rank"]),
                f"{row['score']:.6f}",
                reasoning,
            ])


def enforce_monotonic_scores(sorted_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure scores are strictly non-increasing by rank.
    On exact ties the validator requires candidate_id ascending — our sort
    already handles that. We just clamp any float rounding violations.
    """
    scores = sorted_df["score"].values.copy()
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            scores[i] = scores[i - 1] - 1e-9
    sorted_df = sorted_df.copy()
    sorted_df["score"] = scores
    return sorted_df


def main():
    wall_start = time.time()

    ap = argparse.ArgumentParser(
        description="Rank candidates and emit submission.csv"
    )
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    ap.add_argument("--out", default="./submission.csv", help="Output CSV path")
    ap.add_argument("--artifacts", default=None, help="Artifacts directory")
    ap.add_argument("--config",    default=None, help="Config YAML path")
    ap.add_argument("--no-ltr",    action="store_true", help="Skip LTR calibration")
    ap.add_argument("--no-validate", action="store_true", help="Skip validate_submission.py")
    ap.add_argument("--limit",     type=int, default=0, help="Limit candidates (debug)")
    args = ap.parse_args()

    cfg_path = Path(args.config or REPO_ROOT / "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    art_dir = Path(args.artifacts or cfg["paths"]["artifacts"])
    out_path = Path(args.out)

    print(f"\n{'='*60}")
    print(f"REDROB RANKER — rank.py")
    print(f"  Candidates:  {args.candidates}")
    print(f"  Artifacts:   {art_dir}")
    print(f"  Output:      {out_path}")
    print(f"{'='*60}\n")

    # ── Step 1: Load artifacts (fast) ─────────────────────────────────────────
    print("[1/7] Loading precomputed artifacts ...")
    embeddings, anchor_embeddings, ids, bm25, bm25_ids = load_artifacts(art_dir)

    # ── Step 2: Parse candidates.jsonl ────────────────────────────────────────
    print(f"[2/7] Parsing candidates from {args.candidates} ...")
    t = time.time()
    df, blobs, raw = load_candidates(args.candidates, args.limit)
    print(f"  Parsed {len(df):,} candidates in {time.time()-t:.1f}s")

    # Validate all IDs exist in artifacts
    missing = set(df.index) - set(ids)
    if missing:
        print(f"  WARNING: {len(missing)} candidates not in embedding index — "
              f"they'll get zero dense score")

    # ── Step 3: Align embeddings ──────────────────────────────────────────────
    print("[3/7] Aligning embeddings ...")
    t = time.time()
    aligned_emb = align_embeddings(df, embeddings, ids)
    print(f"  Aligned {aligned_emb.shape} embeddings in {time.time()-t:.1f}s")

    # ── Step 4: Consistency + availability ────────────────────────────────────
    print("[4/7] Computing consistency and availability ...")
    t = time.time()
    consistency  = compute_consistency_scores(df, raw)
    availability = compute_availability(df)
    n_hp = (consistency < 0.20).sum()
    print(f"  Consistency: {n_hp} near-honeypots detected  [{time.time()-t:.1f}s]")

    # ── Step 5: Feature engineering ───────────────────────────────────────────
    print("[5/7] Computing features ...")
    t = time.time()
    feat = compute_features(df, aligned_emb, anchor_embeddings, bm25, bm25_ids)
    print(f"  Features computed: {feat.shape}  [{time.time()-t:.1f}s]")

    # ── Step 5.5: Load LLM scores if available ────────────────────────────────
    llm_scores_path = art_dir / "llm_scores.parquet"
    llm_scores = load_llm_scores(llm_scores_path)
    if llm_scores is not None:
        n_llm = (~llm_scores["llm_overall"].isna()).sum()
        print(f"  LLM scores: {n_llm:,} candidates have Claude Haiku semantic scores")
    else:
        print("  LLM scores: not found (run precompute.py --llm-score to generate)")

    # ── Step 6: Composite score ───────────────────────────────────────────────
    print("[6/7] Scoring candidates ...")
    t = time.time()
    weights = load_weights(cfg_path)
    llm_weight = cfg.get("llm", {}).get("blend_weight", 0.40)

    use_ltr = (
        LTR_AVAILABLE
        and not args.no_ltr
        and cfg.get("ltr", {}).get("enabled", False)
        and (art_dir / "ltr_model.pkl").exists()
    )

    if use_ltr:
        from src.ltr import load_ltr_model, build_ltr_features, ltr_predict, FEATURE_COLS
        from src.score import (
            _penalty_keyword_stuffer, _penalty_all_consulting,
            _penalty_non_tech_non_stuffer, _penalty_cv_speech_primary,
            _penalty_llm_wrapper_only, _penalty_international_no_relocate,
        )
        ranker, feature_cols = load_ltr_model(art_dir / "ltr_model.pkl")
        ltr_feat   = build_ltr_features(feat, df)
        ltr_scores_raw = ltr_predict(ranker, ltr_feat, feature_cols)
        ltr_norm = (ltr_scores_raw.values - ltr_scores_raw.min()) / (ltr_scores_raw.max() - ltr_scores_raw.min() + 1e-9)

        # Blend LTR with LLM scores when available (LTR 60% + LLM 40%)
        if llm_scores is not None:
            print(f"  Using LTR + LLM blend (LTR {1-llm_weight:.0%} + LLM {llm_weight:.0%}) ...")
            llm_overall = llm_scores["llm_overall"].reindex(df.index)
            has_llm = ~llm_overall.isna()
            llm_arr = llm_overall.fillna(0.0).values.astype(np.float32)
            base = np.where(has_llm.values,
                            (1 - llm_weight) * ltr_norm + llm_weight * llm_arr,
                            ltr_norm).astype(np.float32)
        else:
            print("  Using LTR calibrated scores ...")
            base = ltr_norm.astype(np.float32)

        pen_kw   = _penalty_keyword_stuffer(df)
        pen_con  = _penalty_all_consulting(df)
        pen_non  = _penalty_non_tech_non_stuffer(df)
        pen_cv   = _penalty_cv_speech_primary(df)
        pen_llm  = _penalty_llm_wrapper_only(df)
        pen_intl = _penalty_international_no_relocate(df)
        cons_arr  = consistency.reindex(df.index).values
        avail_arr = availability.reindex(df.index).values
        fit_vals = base * pen_kw * pen_con * pen_non * pen_cv * pen_llm * pen_intl * cons_arr * avail_arr
        fit_vals = np.clip(fit_vals, 1e-6, 1.0)
        fit = pd.Series(fit_vals.astype(np.float32), index=df.index, name="fit_score")
    elif llm_scores is not None:
        print(f"  Using LLM-blended score (llm_weight={llm_weight:.0%}) ...")
        fit = compute_fit_score_with_llm(
            df, feat, consistency, availability, weights,
            llm_scores=llm_scores,
            llm_weight=llm_weight,
        )
    else:
        print("  Using feature-only composite score ...")
        fit = compute_fit_score(df, feat, consistency, availability, weights)

    print(f"  Scored {len(fit):,} candidates  [{time.time()-t:.1f}s]")

    # ── Step 7: Select top-100, generate reasonings, write CSV ───────────────
    print("[7/7] Selecting top-100 and generating reasonings ...")
    t = time.time()

    sorted_candidates = sort_for_submission(fit)
    top100 = sorted_candidates.head(100).copy()
    top100["rank"] = range(1, 101)
    top100 = enforce_monotonic_scores(top100)

    # Stats on top-100
    top100_ids = top100["candidate_id"].tolist()
    top100_df_rows = df.loc[top100_ids]
    n_india   = (top100_df_rows["country"] == "India").sum()
    n_kw_stuf = top100_df_rows["keyword_stuffer"].sum()
    n_hp_top  = (consistency.reindex(top100_ids) < 0.20).sum()
    n_retrieval = top100_df_rows["retrieval_in_desc"].sum()

    print(f"\n  ── Top-100 composition ──")
    print(f"    India:          {n_india}/100")
    print(f"    Keyword stuffers: {n_kw_stuf}/100  (target: 0)")
    print(f"    Honeypots (<.20): {n_hp_top}/100  (target: <10)")
    print(f"    With retrieval:   {n_retrieval}/100")
    print(f"\n  ── Top-20 candidates ──")
    for rank, row in top100.head(20).iterrows():
        cid = row["candidate_id"]
        dr  = df.loc[cid]
        print(
            f"    #{row['rank']:>3}  {cid}  "
            f"{dr['current_title'][:30]:<30}  YOE={dr['yoe']:.1f}  "
            f"score={row['score']:.4f}  retrieval={dr['retrieval_in_desc']}"
        )

    # Generate reasonings (pass LLM notes when available for richer text)
    top100_with_fit = top100_df_rows.copy()
    top100_with_fit["fit_score"] = fit.reindex(top100_ids).values
    reasonings = generate_all_reasonings(top100_with_fit, raw, consistency, feat, llm_scores)

    # Build final output
    submission = top100[["candidate_id", "rank", "score"]].copy()
    submission["reasoning"] = submission["candidate_id"].map(reasonings)

    write_submission_csv(submission, out_path)
    print(f"\n  Written {len(submission)} rows → {out_path}  [{time.time()-t:.1f}s]")

    # ── Final timing ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - wall_start
    print(f"\n  Total wall-clock time: {total_elapsed:.1f}s")
    if total_elapsed > 300:
        print(f"  ⚠ WARNING: {total_elapsed:.0f}s exceeds 5-minute budget!")
    else:
        print(f"  ✓ Within 5-minute budget ({300 - total_elapsed:.0f}s remaining)")

    # ── Validate ──────────────────────────────────────────────────────────────
    if not args.no_validate:
        validator = REPO_ROOT / "validate_submission.py"
        if validator.exists():
            result = subprocess.run(
                [sys.executable, str(validator), str(out_path)],
                capture_output=True, text=True
            )
            print(f"\n  Validator output: {result.stdout.strip()}")
            if result.returncode != 0:
                print(f"  ⚠ VALIDATION FAILED:\n{result.stderr}")
                sys.exit(1)
            else:
                print("  ✓ Submission is valid.")
        else:
            print(f"  (validate_submission.py not found at {validator}, skipping)")

    print(f"\n{'='*60}\n")
    return out_path


if __name__ == "__main__":
    main()
