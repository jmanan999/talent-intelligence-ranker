"""
precompute.py — Offline precomputation stage (may be slow; run once).

Produces in artifacts/:
  embeddings.npy        float32 array (N, D), normalized, row-order matches ids.npy
  ids.npy               str array of candidate_ids (same order as embeddings)
  anchor_embeddings.npy float32 array (n_anchors, D), normalized
  anchor_labels.json    list of anchor names/descriptions
  bm25_index.pkl        serialized BM25Okapi index
  bm25_ids.json         list of candidate_ids matching BM25 corpus rows
  feature_table.parquet flat feature DataFrame (from parse.py)

The rank step loads these artifacts from disk and does only vectorized math.
No network access is needed during ranking.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Add repo root to path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.parse import load_candidates, build_embed_text

RUBRIC_PATH = REPO_ROOT / "rubric.yaml"
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def load_rubric() -> dict:
    with open(RUBRIC_PATH) as f:
        return yaml.safe_load(f)


def embed_texts(texts: list[str], model_name: str, batch_size: int, cache_dir: str) -> np.ndarray:
    """
    Embed a list of texts with a sentence-transformer model.
    Returns float32 array of shape (N, D), L2-normalized.
    Uses local cache; no network needed after first download.

    Device selection:
      - Tries MPS (Apple GPU) first with batch_size=32 (memory-safe)
      - Falls back to CPU if MPS OOMs or is unavailable
    """
    import os
    import torch
    from sentence_transformers import SentenceTransformer

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Try MPS first (Apple M-series GPU: much faster than CPU for this)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        # MPS optimal batch_size=64 on M4 with short texts (~113 tokens)
        # gives ~137 texts/sec → 12 min for 100K candidates
        batch_size = min(batch_size, 64)
        print(f"  Using MPS (Apple GPU) device with batch_size={batch_size}")
    else:
        device = "cpu"
        # CPU: use smaller batches; ~7 texts/sec with batch_size=512 is too slow
        batch_size = min(batch_size, 64)
        print(f"  Using CPU with batch_size={batch_size}")

    print(f"  Loading model '{model_name}' from cache {cache_dir} (device={device}) ...")
    model = SentenceTransformer(model_name, cache_folder=cache_dir, device=device)

    print(f"  Embedding {len(texts):,} texts in batches of {batch_size} ...")
    t0 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # L2 normalize so cosine = dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s — shape {embeddings.shape}")
    return embeddings


def build_bm25(texts: list[str]) -> object:
    """
    Build BM25Okapi index over tokenized texts.
    Simple whitespace tokenization is sufficient here; the descriptions
    are English prose and BM25 is robust to minor tokenization variation.
    """
    from rank_bm25 import BM25Okapi

    print(f"  Building BM25 index over {len(texts):,} documents ...")
    t0 = time.time()
    corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(corpus)
    dt = time.time() - t0
    print(f"  BM25 built in {dt:.1f}s")
    return bm25


def build_anchor_sentences(rubric: dict) -> tuple[list[str], list[str]]:
    """
    Build the JD anchor sentences from rubric.yaml.
    Returns (labels, sentences).
    """
    anchors = rubric.get("jd_anchors", {})
    labels   = list(anchors.keys())
    sentences = list(anchors.values())
    return labels, sentences


def main():
    ap = argparse.ArgumentParser(
        description="Precompute embeddings, BM25 index, feature table, and LLM scores."
    )
    ap.add_argument("--candidates", default=None, help="Path to candidates.jsonl")
    ap.add_argument("--artifacts", default=None, help="Output directory for artifacts")
    ap.add_argument("--limit", type=int, default=0, help="Limit candidates (0=all)")
    ap.add_argument("--skip-embeddings", action="store_true",
                    help="Skip embedding step (reuse existing embeddings.npy)")
    ap.add_argument("--skip-bm25", action="store_true",
                    help="Skip BM25 step (reuse existing bm25_index.pkl)")
    ap.add_argument("--llm-score", action="store_true",
                    help="Run Claude Haiku LLM scoring on top candidates (requires ANTHROPIC_API_KEY)")
    ap.add_argument("--llm-top-n", type=int, default=5000,
                    help="Score top-N candidates by fast embedding score (default: 5000)")
    ap.add_argument("--skip-llm", action="store_true",
                    help="Skip LLM scoring even if --llm-score is set")
    args = ap.parse_args()

    cfg = load_config()
    rubric = load_rubric()

    candidates_path = args.candidates or cfg["paths"]["candidates"]
    artifacts_dir = Path(args.artifacts or cfg["paths"]["artifacts"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    emb_cfg = cfg["embedding"]
    model_name = emb_cfg["model"]
    batch_size = emb_cfg["batch_size"]
    cache_dir  = str(REPO_ROOT / emb_cfg["cache_dir"])

    # ── Step 1: Parse candidates ───────────────────────────────────────────────
    print(f"\n[1/4] Parsing candidates from {candidates_path} ...")
    t0 = time.time()
    df, text_blobs, raw = load_candidates(candidates_path, limit=args.limit)
    print(f"  Loaded {len(df):,} candidates in {time.time()-t0:.1f}s")

    # Save feature table (used by rank.py to skip re-parsing)
    parquet_path = artifacts_dir / "feature_table.parquet"
    df.to_parquet(parquet_path)
    print(f"  Feature table saved → {parquet_path} ({parquet_path.stat().st_size/1e6:.1f} MB)")

    # Keep ordered list of IDs
    ordered_ids = list(df.index)

    # ── Step 2: Candidate embeddings ───────────────────────────────────────────
    emb_path   = artifacts_dir / "embeddings.npy"
    ids_path   = artifacts_dir / "ids.npy"

    if args.skip_embeddings and emb_path.exists():
        print(f"\n[2/4] Skipping embeddings (reusing {emb_path})")
    else:
        print(f"\n[2/4] Embedding {len(ordered_ids):,} candidates ...")
        # Use short embed text (~150 tokens) for fast CPU/MPS embedding.
        # BM25 uses the full blob (build_text_blob) separately.
        embed_texts_list = [build_embed_text(raw[cid]) for cid in ordered_ids]
        embeddings = embed_texts(embed_texts_list, model_name, batch_size, cache_dir)
        np.save(emb_path, embeddings)
        np.save(ids_path, np.array(ordered_ids))
        print(f"  Saved embeddings → {emb_path} ({emb_path.stat().st_size/1e6:.1f} MB)")

    # ── Step 3: Anchor embeddings (JD facets) ──────────────────────────────────
    anchor_emb_path   = artifacts_dir / "anchor_embeddings.npy"
    anchor_label_path = artifacts_dir / "anchor_labels.json"

    print(f"\n[3/4] Embedding JD anchor sentences ...")
    anchor_labels, anchor_sentences = build_anchor_sentences(rubric)
    anchor_embs = embed_texts(anchor_sentences, model_name, batch_size, cache_dir)
    np.save(anchor_emb_path, anchor_embs)
    with open(anchor_label_path, "w") as f:
        json.dump(anchor_labels, f)
    print(f"  Saved {len(anchor_labels)} anchors → {anchor_emb_path}")

    # ── Step 4: BM25 index ────────────────────────────────────────────────────
    bm25_path  = artifacts_dir / "bm25_index.pkl"
    bm25ids_path = artifacts_dir / "bm25_ids.json"

    if args.skip_bm25 and bm25_path.exists():
        print(f"\n[4/4] Skipping BM25 (reusing {bm25_path})")
    else:
        print(f"\n[4/4] Building BM25 index ...")
        texts_for_bm25 = [text_blobs[cid] for cid in ordered_ids]
        bm25 = build_bm25(texts_for_bm25)
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f, protocol=4)
        with open(bm25ids_path, "w") as f:
            json.dump(ordered_ids, f)
        print(f"  Saved BM25 index → {bm25_path} ({bm25_path.stat().st_size/1e6:.1f} MB)")

    # ── Step 5: LLM scoring (optional, requires ANTHROPIC_API_KEY) ───────────
    llm_scores_path = artifacts_dir / "llm_scores.parquet"
    run_llm = args.llm_score and not args.skip_llm

    if run_llm:
        print(f"\n[5/5] LLM semantic scoring (Claude Haiku) on top-{args.llm_top_n} candidates ...")
        try:
            from src.llm_scorer import run_llm_scoring
            from src.consistency import compute_consistency_scores
            from src.signals import compute_availability
            from src.features import compute_features

            import numpy as _np_llm
            _embs      = _np_llm.load(emb_path)
            _anchors   = _np_llm.load(anchor_emb_path)
            _ids_saved = list(_np_llm.load(ids_path, allow_pickle=True))

            with open(bm25_path, "rb") as _f:
                _bm25_idx = pickle.load(_f)
            with open(bm25ids_path) as _f:
                _bm25_ids = json.load(_f)

            # Reorder embeddings to match df
            _id_to_idx = {cid: i for i, cid in enumerate(_ids_saved)}
            _emb_idx   = [_id_to_idx[cid] for cid in ordered_ids if cid in _id_to_idx]
            _aligned   = _embs[[_id_to_idx[cid] for cid in ordered_ids]]

            _consistency  = compute_consistency_scores(df, raw)
            _availability = compute_availability(df)
            _feat         = compute_features(df, _aligned, _anchors, _bm25_idx, _bm25_ids)

            # Fast composite for pre-ranking to pick top-N
            from src.score import compute_base_score
            import yaml as _yaml_llm
            with open(CONFIG_PATH) as _f:
                _cfg2 = _yaml_llm.safe_load(_f)
            _fast_score = compute_base_score(_feat, _cfg2["weights"])
            _fast_series = pd.Series(_fast_score, index=df.index).sort_values(ascending=False)
            _top_ids = list(_fast_series.head(args.llm_top_n).index)

            print(f"  Scoring top-{len(_top_ids)} candidates by initial feature score ...")
            run_llm_scoring(
                candidate_ids=_top_ids,
                raw_records=raw,
                df=df,
                output_path=llm_scores_path,
                use_async=True,
                max_concurrent=25,
            )
        except ImportError as e:
            print(f"  WARNING: LLM scoring skipped — {e}")
            print("  Install: pip install anthropic")
        except Exception as e:
            print(f"  WARNING: LLM scoring failed — {e}")
    elif llm_scores_path.exists():
        print(f"\n[5/5] LLM scores already exist at {llm_scores_path} (use --llm-score to refresh)")
    else:
        print(f"\n[5/5] LLM scoring skipped (use --llm-score flag + ANTHROPIC_API_KEY to enable)")

    print(f"\n✓ Precomputation complete. Artifacts in {artifacts_dir}/")
    print(f"  feature_table.parquet  |  embeddings.npy  |  anchor_embeddings.npy")
    print(f"  bm25_index.pkl         |  ids.npy         |  anchor_labels.json")
    if llm_scores_path.exists():
        print(f"  llm_scores.parquet     ← Claude Haiku semantic scores for top candidates")


if __name__ == "__main__":
    main()
