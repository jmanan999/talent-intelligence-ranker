"""
sandbox/app.py — Streamlit demo: upload ≤100 candidates → ranked CSV.

Usage:
  streamlit run sandbox/app.py

Or via Docker:
  docker build -t redrob-ranker ./
  docker run -p 8501:8501 redrob-ranker

The sandbox runs the full rank pipeline on small samples end-to-end,
demonstrating the system without network access during ranking.
"""

import io
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ART_DIR = REPO_ROOT / "artifacts"
SAMPLE_PATH = REPO_ROOT.parent / "sample_candidates.json"


def check_artifacts() -> bool:
    """Check if precomputed artifacts are present."""
    required = [
        "embeddings.npy", "anchor_embeddings.npy", "ids.npy",
        "bm25_index.pkl", "bm25_ids.json",
    ]
    return all((ART_DIR / f).exists() for f in required)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    embeddings        = np.load(ART_DIR / "embeddings.npy")
    anchor_embeddings = np.load(ART_DIR / "anchor_embeddings.npy")
    ids               = list(np.load(ART_DIR / "ids.npy", allow_pickle=True))
    with open(ART_DIR / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(ART_DIR / "bm25_ids.json") as f:
        bm25_ids = json.load(f)
    return embeddings, anchor_embeddings, ids, bm25, bm25_ids


def run_ranking(candidates_data: list[dict]) -> pd.DataFrame:
    """Run the full pipeline on a small candidate list."""
    from src.parse import load_candidates, extract_row, build_text_blob, build_embed_text
    from src.consistency import compute_consistency_scores
    from src.signals import compute_availability
    from src.features import compute_features, normalize_to_01, compute_dense_scores, compute_sparse_scores, JD_QUERY_TOKENS
    from src.score import compute_fit_score, sort_for_submission, load_weights
    from src.reason import generate_all_reasonings

    with open(REPO_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Build minimal structures from candidate list
    rows = []
    text_blobs = {}
    raw = {}
    for c in candidates_data:
        cid = c["candidate_id"]
        rows.append(extract_row(c))
        text_blobs[cid] = build_text_blob(c)
        raw[cid] = c

    df = pd.DataFrame(rows).set_index("candidate_id")

    # Load artifacts
    embeddings, anchor_embeddings, ids, bm25, bm25_ids = load_artifacts()

    # For sandbox candidates not in the full index, embed them on-the-fly
    known_ids = set(ids)
    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    in_index = [cid for cid in df.index if cid in known_ids]
    not_in_index = [cid for cid in df.index if cid not in known_ids]

    if not_in_index:
        # Embed unknown candidates using sentence-transformers
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        from sentence_transformers import SentenceTransformer
        model_name = cfg["embedding"]["model"]
        cache_dir  = str(REPO_ROOT / cfg["embedding"]["cache_dir"])
        model = SentenceTransformer(model_name, cache_folder=cache_dir, device="cpu")
        new_texts = [build_embed_text(c) for c in candidates_data if c["candidate_id"] in set(not_in_index)]
        new_embs = model.encode(new_texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        # Extend index
        all_ids    = list(df.index)
        all_embs   = np.zeros((len(df), embeddings.shape[1]), dtype=np.float32)
        for i, cid in enumerate(all_ids):
            if cid in id_to_idx:
                all_embs[i] = embeddings[id_to_idx[cid]]
            else:
                j = not_in_index.index(cid)
                all_embs[i] = new_embs[j]
        aligned_emb = all_embs
    else:
        emb_idx = [id_to_idx[cid] for cid in df.index]
        aligned_emb = embeddings[emb_idx]

    # Features
    consistency  = compute_consistency_scores(df, raw)
    availability = compute_availability(df)

    # Build mini BM25 for the sandbox (fast on ≤100 docs)
    from rank_bm25 import BM25Okapi
    sandbox_ids = list(df.index)
    corpus = [text_blobs[cid].lower().split() for cid in sandbox_ids]
    mini_bm25 = BM25Okapi(corpus)
    feat = compute_features(df, aligned_emb, anchor_embeddings, mini_bm25, sandbox_ids)

    weights = load_weights(REPO_ROOT / "config.yaml")
    fit = compute_fit_score(df, feat, consistency, availability, weights)
    sorted_df = sort_for_submission(fit)

    top_n = min(100, len(df))
    top_df = sorted_df.head(top_n).copy()
    top_df["rank"] = range(1, top_n + 1)

    top_ids = top_df["candidate_id"].tolist()
    top_df_rows = df.loc[top_ids].copy()
    top_df_rows["fit_score"] = fit.reindex(top_ids).values
    reasonings = generate_all_reasonings(top_df_rows, raw, consistency, feat)

    result = top_df[["candidate_id", "rank", "score"]].copy()
    result["reasoning"] = result["candidate_id"].map(reasonings)
    return result


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Redrob Ranker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Talent Intelligence Ranker")
st.markdown("""
**Hybrid semantic + structured candidate ranking for a Senior AI Engineer role.**

Upload a JSON file (array of candidate records, ≤100) or use the sample.
The system ranks them by fit — reading career history, not keyword lists.
""")

if not check_artifacts():
    st.error(
        "⚠ Precomputed artifacts not found in `artifacts/`. "
        "Run `python src/precompute.py --candidates candidates.jsonl` first."
    )
    st.stop()

with st.sidebar:
    st.header("About the system")
    st.markdown("""
**Architecture:**
1. Dense: BGE-small embeddings vs. JD anchors
2. Sparse: BM25 on career descriptions
3. Structured: YOE band, product-ratio, tenure
4. Penalties: keyword-stuffer, all-consulting, honeypot
5. Availability: recency × response rate × open-to-work

**The trap we avoided:**
The sample baseline ranks an HR Manager #1 because she has 9 AI skills listed.
Our system reads career descriptions, not skill lists.

**Compute constraints:**
- CPU only
- No network during ranking
- ≤5 min for 100K candidates
""")

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload candidates JSON (array of candidate objects)",
        type=["json", "jsonl"],
    )

with col2:
    use_sample = st.button("Use sample_candidates.json", type="primary")

candidates_data = None

if use_sample and SAMPLE_PATH.exists():
    with open(SAMPLE_PATH) as f:
        candidates_data = json.load(f)
    if isinstance(candidates_data, dict):
        candidates_data = [candidates_data]
    st.success(f"Loaded {len(candidates_data)} candidates from sample file")

elif uploaded:
    raw_bytes = uploaded.read().decode("utf-8")
    try:
        candidates_data = json.loads(raw_bytes)
        if isinstance(candidates_data, dict):
            candidates_data = [candidates_data]
    except json.JSONDecodeError:
        # Try JSONL
        candidates_data = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    st.success(f"Loaded {len(candidates_data)} candidates")

if candidates_data:
    if len(candidates_data) > 100:
        st.warning(f"Truncating to 100 candidates (sandbox limit).")
        candidates_data = candidates_data[:100]

    with st.spinner(f"Ranking {len(candidates_data)} candidates ..."):
        t0 = time.time()
        try:
            result_df = run_ranking(candidates_data)
            elapsed = time.time() - t0
            st.success(f"✓ Ranked {len(result_df)} candidates in {elapsed:.1f}s")
        except Exception as e:
            st.error(f"Ranking failed: {e}")
            st.exception(e)
            st.stop()

    # ── Results display ───────────────────────────────────────────────────────
    st.subheader(f"Top {len(result_df)} Candidates")

    display_df = result_df.copy()
    display_df["score"] = display_df["score"].round(4)
    st.dataframe(display_df, use_container_width=True, height=400)

    # Download button
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ranked CSV",
        data=csv_bytes,
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )

    # Show top-5 with reasoning
    st.subheader("Top 5 — Detailed Reasoning")
    for _, row in result_df.head(5).iterrows():
        st.markdown(f"**#{row['rank']}** `{row['candidate_id']}` — score {row['score']:.4f}")
        st.caption(row["reasoning"])
        st.divider()
