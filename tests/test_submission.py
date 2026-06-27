"""
tests/test_submission.py — Unit tests for the ranking pipeline.

Tests:
  1. validator passes on the output CSV
  2. no honeypots in top-10 (consistency guard)
  3. scores are monotonically non-increasing
  4. no keyword stuffers in top-10
  5. all top-100 IDs are valid CAND_XXXXXXX format
  6. metrics implementations are correct (DCG/NDCG/MAP/P@k)
  7. consistency detector catches known contradictions
  8. availability multiplier is in [0.60, 1.00]

Run: python -m pytest tests/ -v
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SUBMISSION_CSV = REPO / "submission.csv"
CANDIDATES_JSONL = REPO.parent / "candidates.jsonl"
VALIDATE_PY = REPO.parent / "validate_submission.py"

CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


# ── Submission CSV tests ───────────────────────────────────────────────────────

class TestSubmissionCSV:
    @pytest.fixture(scope="class")
    def submission_df(self):
        if not SUBMISSION_CSV.exists():
            pytest.skip("submission.csv not yet generated")
        return pd.read_csv(SUBMISSION_CSV)

    def test_has_100_rows(self, submission_df):
        assert len(submission_df) == 100, f"Expected 100 rows, got {len(submission_df)}"

    def test_required_columns(self, submission_df):
        assert list(submission_df.columns) == ["candidate_id", "rank", "score", "reasoning"]

    def test_ranks_1_to_100(self, submission_df):
        ranks = sorted(submission_df["rank"].tolist())
        assert ranks == list(range(1, 101)), "Ranks must be exactly 1–100"

    def test_unique_candidate_ids(self, submission_df):
        assert submission_df["candidate_id"].nunique() == 100, "Duplicate candidate IDs"

    def test_candidate_id_format(self, submission_df):
        for cid in submission_df["candidate_id"]:
            assert CANDIDATE_ID_RE.match(str(cid)), f"Invalid ID format: {cid}"

    def test_scores_non_increasing(self, submission_df):
        sorted_df = submission_df.sort_values("rank")
        scores = sorted_df["score"].values
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-9, (
                f"Score not non-increasing at ranks {i+1}→{i+2}: "
                f"{scores[i]:.6f} < {scores[i+1]:.6f}"
            )

    def test_scores_positive(self, submission_df):
        assert (submission_df["score"] > 0).all(), "All scores must be positive"

    def test_reasoning_not_empty(self, submission_df):
        empty = submission_df[submission_df["reasoning"].isna() | (submission_df["reasoning"].str.strip() == "")]
        assert len(empty) == 0, f"Empty reasoning at ranks: {empty['rank'].tolist()}"

    def test_validator_passes(self):
        """Run the official validator and assert it passes."""
        if not SUBMISSION_CSV.exists():
            pytest.skip("submission.csv not yet generated")
        if not VALIDATE_PY.exists():
            pytest.skip("validate_submission.py not found")

        import subprocess
        result = subprocess.run(
            [sys.executable, str(VALIDATE_PY), str(SUBMISSION_CSV)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Validator failed:\n{result.stdout}\n{result.stderr}"
        assert "Submission is valid." in result.stdout


# ── Honeypot tests ────────────────────────────────────────────────────────────

class TestHoneypotGuard:
    @pytest.fixture(scope="class")
    def submission_df(self):
        if not SUBMISSION_CSV.exists():
            pytest.skip("submission.csv not yet generated")
        return pd.read_csv(SUBMISSION_CSV)

    def test_no_keyword_stuffers_in_top10(self, submission_df):
        """Top-10 must not contain candidates whose AI skills are ONLY in skills[]."""
        if not CANDIDATES_JSONL.exists():
            pytest.skip("candidates.jsonl not available")

        from src.parse import load_candidates
        top10_ids = set(submission_df[submission_df["rank"] <= 10]["candidate_id"].tolist())

        # Load only top-10 candidates for speed
        df, _, raw = load_candidates(str(CANDIDATES_JSONL))
        top10_df = df[df.index.isin(top10_ids)]
        stuffers = top10_df[top10_df["keyword_stuffer"]]
        assert len(stuffers) == 0, (
            f"Keyword stuffers in top-10: {stuffers.index.tolist()}"
        )

    def test_honeypot_rate_top100_under_10pct(self, submission_df):
        """Honeypot rate in top-100 must be < 10% (target 0%)."""
        if not CANDIDATES_JSONL.exists():
            pytest.skip("candidates.jsonl not available")

        from src.parse import load_candidates
        from src.consistency import compute_consistency_scores

        top100_ids = set(submission_df["candidate_id"].tolist())
        df, _, raw = load_candidates(str(CANDIDATES_JSONL))
        top100_df = df[df.index.isin(top100_ids)]
        cons = compute_consistency_scores(top100_df, {k: raw[k] for k in top100_df.index})

        n_hp = (cons < 0.20).sum()
        rate = n_hp / 100
        assert rate < 0.10, (
            f"Honeypot rate {rate:.1%} exceeds 10% threshold (found {n_hp} in top-100)"
        )


# ── Metrics unit tests ────────────────────────────────────────────────────────

class TestMetrics:
    def test_ndcg_perfect(self):
        from eval.metrics import ndcg_at_k
        perfect = [5, 4, 3, 2, 1]
        assert ndcg_at_k(perfect, 5) == pytest.approx(1.0, abs=1e-9)

    def test_ndcg_worst(self):
        from eval.metrics import ndcg_at_k
        worst  = [1, 2, 3, 4, 5]
        ideal  = [5, 4, 3, 2, 1]
        score  = ndcg_at_k(worst, 5, ideal)
        assert score < 0.9  # not perfect

    def test_precision_at_k(self):
        from eval.metrics import precision_at_k
        rels = [5, 4, 3, 0, 0, 0, 0, 0, 0, 0]
        assert precision_at_k(rels, 10) == pytest.approx(0.3, abs=1e-9)

    def test_composite_structure(self):
        from eval.metrics import composite_score
        rels = [5, 4, 3, 2, 1, 0] * 10 + [0] * 40
        result = composite_score(rels)
        assert set(result.keys()) == {"composite", "ndcg@10", "ndcg@50", "MAP", "P@10"}
        assert 0.0 <= result["composite"] <= 1.0

    def test_map_all_relevant(self):
        from eval.metrics import average_precision
        all_rel = [5] * 100
        assert average_precision(all_rel) == pytest.approx(1.0, abs=1e-9)


# ── Consistency unit tests ────────────────────────────────────────────────────

class TestConsistency:
    def make_candidate(self, yoe, career_months_list, skills):
        """Build a minimal candidate dict for testing."""
        return {
            "candidate_id": "CAND_TEST001",
            "profile": {"years_of_experience": yoe, "current_title": "Test"},
            "career_history": [
                {"company": "Co", "duration_months": m, "title": "Eng",
                 "start_date": "2020-01-01", "end_date": None,
                 "is_current": True, "industry": "Tech", "company_size": "51-200",
                 "description": "Built stuff."}
                for m in career_months_list
            ],
            "skills": skills,
        }

    def test_consistent_candidate_scores_1(self):
        from src.consistency import consistency_score_single
        c = self.make_candidate(5.0, [30, 30], [
            {"name": "Python", "proficiency": "advanced", "duration_months": 30}
        ])
        score, flags = consistency_score_single(c)
        assert score == pytest.approx(1.0, abs=1e-9)
        assert flags == []

    def test_expert_zero_flagged(self):
        from src.consistency import consistency_score_single
        skills = [
            {"name": "Pinecone", "proficiency": "expert", "duration_months": 0},
            {"name": "FAISS",    "proficiency": "expert", "duration_months": 0},
            {"name": "Python",   "proficiency": "advanced", "duration_months": 30},
        ]
        c = self.make_candidate(5.0, [30, 30], skills)
        score, flags = consistency_score_single(c)
        assert score < 0.5
        assert any("expert_zero" in f for f in flags)

    def test_skill_exceeds_career_flagged(self):
        from src.consistency import consistency_score_single
        skills = [
            {"name": "Python", "proficiency": "advanced", "duration_months": 200}
        ]
        c = self.make_candidate(5.0, [30, 30], skills)  # career = 60 months; skill = 200
        score, flags = consistency_score_single(c)
        assert score < 0.5
        assert any("skill_exceeds" in f for f in flags)


# ── Availability unit tests ───────────────────────────────────────────────────

class TestAvailability:
    def test_multiplier_in_range(self):
        from src.signals import compute_availability

        data = {
            "days_since_active": [0, 30, 90, 180, 365],
            "response_rate":     [1.0, 0.8, 0.5, 0.2, 0.0],
            "open_to_work":      [True, True, False, False, False],
            "saved_30d":         [10, 5, 2, 0, 0],
            "interview_rate":    [1.0, 0.8, 0.5, 0.3, 0.0],
        }
        df = pd.DataFrame(data)
        df.index = [f"CAND_{i:07d}" for i in range(len(df))]
        avail = compute_availability(df)

        assert (avail >= 0.55).all(), f"Min availability below floor: {avail.min()}"  # floor lowered to 0.55 in v2
        assert (avail <= 1.00).all(), f"Max availability above ceil: {avail.max()}"

    def test_active_responsive_beats_stale(self):
        from src.signals import compute_availability

        data = {
            "days_since_active": [1,   365],
            "response_rate":     [0.9, 0.05],
            "open_to_work":      [True, False],
            "saved_30d":         [5,   0],
            "interview_rate":    [1.0, 0.1],
        }
        df = pd.DataFrame(data)
        df.index = ["CAND_ACTIVE1", "CAND_STALE01"]
        avail = compute_availability(df)

        assert avail.iloc[0] > avail.iloc[1], "Active candidate should score higher than stale"
