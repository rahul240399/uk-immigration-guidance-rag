"""Tests for near_duplicates.py (T2c-2)."""
import numpy as np
import pytest

from code.evalset.near_duplicates import (
    COSINE_THRESHOLD,
    OUT_COLUMNS,
    TOP_K,
    compute_cosine_matrix,
    find_pairs,
)


# ── helpers ──────────────────────────────────────────────────────────

def _q(qid, route="student", tier="T1", gold="sec-a:00001",
       question="What is the Student requirement?"):
    return {
        "question_id": qid, "route": route, "tier": tier,
        "link_type": "", "question": question,
        "reference_answer": "answer", "gold_paragraph_ids": gold,
        "source": "prompt-v2", "generator_model": "", "embedding_model": "",
        "validated_by_student": "Y", "notes": "",
    }


def _normalised_vec(dim=768, seed=0):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── cosine matrix ────────────────────────────────────────────────────

class TestCosineMatrix:
    def test_identity_diagonal(self):
        emb = np.stack([_normalised_vec(seed=i) for i in range(5)])
        mat = compute_cosine_matrix(emb)
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)

    def test_symmetric(self):
        emb = np.stack([_normalised_vec(seed=i) for i in range(5)])
        mat = compute_cosine_matrix(emb)
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)

    def test_identical_vectors(self):
        v = _normalised_vec(seed=42)
        emb = np.stack([v, v, _normalised_vec(seed=99)])
        mat = compute_cosine_matrix(emb)
        assert abs(mat[0, 1] - 1.0) < 1e-5


# ── pair-set union logic ─────────────────────────────────────────────

class TestPairSetUnion:
    def _make_questions_and_matrix(self, n, high_pairs=None,
                                   same_gold_pairs=None):
        """Build n questions with controlled cosine values.

        high_pairs: list of (i, j, cosine) to force high similarity
        same_gold_pairs: list of (i, j) that share gold ids
        """
        questions = [_q(f"Q{k:04d}", gold=f"g{k}") for k in range(n)]
        if same_gold_pairs:
            for i, j in same_gold_pairs:
                questions[j]["gold_paragraph_ids"] = \
                    questions[i]["gold_paragraph_ids"]

        # Start with low cosine everywhere
        mat = np.full((n, n), 0.3, dtype=np.float32)
        np.fill_diagonal(mat, 1.0)
        if high_pairs:
            for i, j, c in high_pairs:
                mat[i, j] = c
                mat[j, i] = c
        return questions, mat

    def test_threshold_set(self):
        qs, mat = self._make_questions_and_matrix(
            5, high_pairs=[(0, 1, 0.95), (2, 3, 0.91)])
        rows, report = find_pairs(qs, mat)
        assert report["threshold_count"] == 2

    def test_top20_always_present(self):
        """Even with low cosines, top 20 pairs are included."""
        qs, mat = self._make_questions_and_matrix(25)
        rows, report = find_pairs(qs, mat)
        assert report["top20_count"] == TOP_K
        assert report["union_count"] >= TOP_K

    def test_same_gold_set(self):
        qs, mat = self._make_questions_and_matrix(
            5, same_gold_pairs=[(0, 1), (2, 3)])
        rows, report = find_pairs(qs, mat)
        assert report["same_gold_count"] == 2
        same_rows = [r for r in rows if "same_gold" in r["reason_set"]]
        assert len(same_rows) == 2
        assert all(r["same_gold"] == "Y" for r in same_rows)

    def test_union_deduplicates(self):
        """A pair in both threshold and same_gold appears once."""
        qs, mat = self._make_questions_and_matrix(
            5, high_pairs=[(0, 1, 0.95)], same_gold_pairs=[(0, 1)])
        rows, report = find_pairs(qs, mat)
        pair_01 = [r for r in rows
                   if r["qid_a"] == "Q0000" and r["qid_b"] == "Q0001"]
        assert len(pair_01) == 1
        assert "threshold" in pair_01[0]["reason_set"]
        assert "same_gold" in pair_01[0]["reason_set"]

    def test_reason_set_joined(self):
        """reason_set uses + as separator."""
        qs, mat = self._make_questions_and_matrix(
            5, high_pairs=[(0, 1, 0.95)], same_gold_pairs=[(0, 1)])
        rows, _ = find_pairs(qs, mat)
        pair_01 = [r for r in rows
                   if r["qid_a"] == "Q0000" and r["qid_b"] == "Q0001"][0]
        parts = pair_01["reason_set"].split("+")
        assert "threshold" in parts
        assert "same_gold" in parts

    def test_small_n_fewer_than_20(self):
        """With fewer than 20 possible pairs, top20 = all pairs."""
        qs, mat = self._make_questions_and_matrix(4)  # 6 pairs
        rows, report = find_pairs(qs, mat)
        assert report["top20_count"] == 6  # C(4,2) = 6
        assert report["union_count"] == 6


# ── output columns ───────────────────────────────────────────────────

class TestOutputColumns:
    def test_columns_match_spec(self):
        qs, mat = self._make_simple()
        rows, _ = find_pairs(qs, mat)
        if rows:
            assert list(rows[0].keys()) == OUT_COLUMNS

    def test_verdict_and_reworded_empty(self):
        qs, mat = self._make_simple()
        rows, _ = find_pairs(qs, mat)
        for r in rows:
            assert r["verdict"] == ""
            assert r["reworded_question_b"] == ""
            assert r["note"] == ""

    def test_pair_id_sequential(self):
        qs, mat = self._make_simple()
        rows, _ = find_pairs(qs, mat)
        for i, r in enumerate(rows, 1):
            assert r["pair_id"] == f"P{i:04d}"

    @staticmethod
    def _make_simple():
        questions = [_q(f"Q{k:04d}", gold=f"g{k}") for k in range(25)]
        mat = np.full((25, 25), 0.3, dtype=np.float32)
        np.fill_diagonal(mat, 1.0)
        mat[0, 1] = mat[1, 0] = 0.95
        return questions, mat


# ── top-20 rule specifics ────────────────────────────────────────────

class TestTop20Rule:
    def test_top20_picks_highest(self):
        """The 20 highest-cosine pairs should be in the top20 set."""
        n = 30
        questions = [_q(f"Q{k:04d}", gold=f"g{k}") for k in range(n)]
        rng = np.random.RandomState(42)
        mat = np.full((n, n), 0.1, dtype=np.float32)
        np.fill_diagonal(mat, 1.0)
        # Set 25 upper-triangle pairs to descending high values
        pairs_ij = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs_ij.append((i, j))
        rng.shuffle(pairs_ij)
        for rank, (i, j) in enumerate(pairs_ij[:25]):
            val = 0.80 - rank * 0.01  # 0.80, 0.79, ...
            mat[i, j] = val
            mat[j, i] = val

        rows, report = find_pairs(questions, mat)
        assert report["top20_count"] == 20

        # The top-20 rows should have the 20 highest cosines
        top20_rows = [r for r in rows if "top20" in r["reason_set"]]
        cosines = sorted([float(r["cosine"]) for r in top20_rows], reverse=True)
        assert len(cosines) == 20
        assert cosines[0] >= cosines[-1]
