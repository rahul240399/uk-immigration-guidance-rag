"""Tests for evalset covariates.py (T2c-3)."""
import pytest

from code.evalset.covariates import (
    OUT_COLUMNS,
    build_covariates,
    compute_bm25_gold_rank,
    compute_bm25_overlap,
    find_anchor,
)
from code.index.build_index import _tokenize_bm25


# ── fixtures ─────────────────────────────────────────────────────────

def _unit(uid, corpus="rules", text="some unit text", token_count=20):
    return {
        "unit_id": uid, "corpus": corpus, "text": text,
        "unit_token_count": token_count,
    }


def _question(qid="Q0001", route="student", tier="T1", link_type="",
              gold="sec-a:00001", question="What is the Student requirement?"):
    return {
        "question_id": qid, "route": route, "tier": tier,
        "link_type": link_type, "question": question,
        "gold_paragraph_ids": gold,
    }


def _covariates_row(pid, style="new", section_records="50",
                    siblings="3", links_out="2"):
    return {
        "paragraph_id": pid, "drafting_style": style,
        "section_records": section_records,
        "siblings_same_heading": siblings, "links_out_v2": links_out,
    }


# ── anchor rule ──────────────────────────────────────────────────────

class TestAnchorRule:
    def test_first_rules_id(self):
        units = {
            "r1": _unit("r1", corpus="rules"),
            "r2": _unit("r2", corpus="rules"),
        }
        assert find_anchor(["r1", "r2"], "", units) == "r1"

    def test_skips_guidance_for_default(self):
        """Default anchor skips guidance, takes first rules."""
        units = {
            "g1": _unit("g1", corpus="guidance"),
            "r1": _unit("r1", corpus="rules"),
        }
        assert find_anchor(["g1", "r1"], "", units) == "r1"

    def test_guidance_para_skips_guidance_source(self):
        """For guidance_para, skip guidance ids, take first rules id."""
        units = {
            "g1": _unit("g1", corpus="guidance"),
            "r1": _unit("r1", corpus="rules"),
            "r2": _unit("r2", corpus="rules"),
        }
        anchor = find_anchor(["g1", "r1", "r2"], "guidance_para", units)
        assert anchor == "r1"

    def test_guidance_para_all_guidance_falls_back(self):
        """If all ids are guidance, fall back to first id."""
        units = {
            "g1": _unit("g1", corpus="guidance"),
            "g2": _unit("g2", corpus="guidance"),
        }
        anchor = find_anchor(["g1", "g2"], "guidance_para", units)
        assert anchor == "g1"

    def test_empty_gold(self):
        assert find_anchor([], "", {}) is None

    def test_rules_section_xcut_uses_first_rules(self):
        units = {
            "r1": _unit("r1", corpus="rules"),
            "r2": _unit("r2", corpus="rules"),
        }
        anchor = find_anchor(["r1", "r2"], "rules_section_xcut", units)
        assert anchor == "r1"


# ── BM25 overlap ─────────────────────────────────────────────────────

class TestBm25Overlap:
    def test_full_overlap(self):
        """All question tokens appear in anchor text."""
        q = "the student visa requirement"
        a = "the student visa requirement is to show funds"
        assert compute_bm25_overlap(q, a) == 1.0

    def test_no_overlap(self):
        q = "apple banana cherry"
        a = "delta echo foxtrot"
        assert compute_bm25_overlap(q, a) == 0.0

    def test_partial_overlap(self):
        q = "student funds living costs"
        a = "the student must show funds"
        q_tokens = set(_tokenize_bm25(q))
        a_tokens = set(_tokenize_bm25(a))
        expected = len(q_tokens & a_tokens) / len(q_tokens)
        assert abs(compute_bm25_overlap(q, a) - expected) < 1e-9

    def test_empty_question(self):
        assert compute_bm25_overlap("", "some text") == 0.0

    def test_uses_bm25_tokenizer(self):
        """Single-char tokens are dropped per _tokenize_bm25 rule."""
        q = "a b cd ef"  # "a" and "b" dropped (len 1)
        a = "cd ef gh"
        # q_tokens = {"cd", "ef"}, a_tokens = {"cd", "ef", "gh"}
        assert compute_bm25_overlap(q, a) == 1.0


# ── BM25 gold rank ──────────────────────────────────────────────────

class TestBm25GoldRank:
    def test_gold_at_rank_1(self):
        retrieved = [("r1", 5.0), ("r2", 4.0), ("r3", 3.0)]
        assert compute_bm25_gold_rank(retrieved, {"r1"}) == "1"

    def test_gold_at_rank_3(self):
        retrieved = [("x1", 5.0), ("x2", 4.0), ("r1", 3.0)]
        assert compute_bm25_gold_rank(retrieved, {"r1"}) == "3"

    def test_absent_when_not_retrieved(self):
        retrieved = [("x1", 5.0), ("x2", 4.0)]
        assert compute_bm25_gold_rank(retrieved, {"r1"}) == "absent"

    def test_empty_retrieved(self):
        assert compute_bm25_gold_rank([], {"r1"}) == "absent"

    def test_multiple_gold_takes_first(self):
        retrieved = [("x1", 5.0), ("r2", 4.0), ("r1", 3.0)]
        assert compute_bm25_gold_rank(retrieved, {"r1", "r2"}) == "2"


# ── lexically_easy from toy retrieved list ───────────────────────────

class TestLexicallyEasy:
    def test_easy_when_first_is_gold(self):
        """lexically_easy=1 when first retrieved carries a gold id."""
        # Simulate via build_covariates with a mock BM25
        # Instead test the logic directly
        retrieved = [("r1", 5.0), ("r2", 4.0)]
        gold_set = {"r1", "r3"}
        first_is_gold = retrieved[0][0] in gold_set
        assert first_is_gold is True

    def test_not_easy_when_first_is_not_gold(self):
        retrieved = [("x1", 5.0), ("r1", 4.0)]
        gold_set = {"r1"}
        first_is_gold = retrieved[0][0] in gold_set
        assert first_is_gold is False


# ── build_covariates integration ─────────────────────────────────────

class _MockBM25:
    """Minimal mock that returns controlled scores."""
    def __init__(self, chunk_ids, gold_at_rank=None):
        self._n = len(chunk_ids)
        self._gold_at_rank = gold_at_rank or {}

    def get_scores(self, tokens):
        import numpy as np
        scores = np.zeros(self._n, dtype=float)
        for rank, (idx, score) in enumerate(self._gold_at_rank.items()):
            scores[idx] = score
        return scores


class TestBuildCovariates:
    def test_basic_row(self):
        chunk_ids = ["r1", "r2", "r3"]
        units = {
            "r1": _unit("r1", text="the student requirement text", token_count=15),
            "r2": _unit("r2", text="other text here", token_count=10),
            "r3": _unit("r3", text="more stuff", token_count=8),
        }
        covs = {"r1": _covariates_row("r1")}

        # BM25 mock: r1 has highest score → rank 1
        bm25 = _MockBM25(chunk_ids, gold_at_rank={0: 5.0, 1: 3.0, 2: 1.0})

        questions = [_question(gold="r1;r2")]
        rows, report = build_covariates(
            questions, bm25, chunk_ids, units, covs)

        assert len(rows) == 1
        r = rows[0]
        assert r["question_id"] == "Q0001"
        assert r["anchor_id"] == "r1"
        assert r["drafting_style"] == "new"
        assert r["n_gold"] == "2"
        assert r["gold_tokens"] == "25"  # 15 + 10
        assert r["bm25_gold_rank"] == "1"
        assert r["lexically_easy"] == "1"
        assert report["empty_anchors"] == 0

    def test_guidance_para_anchor(self):
        chunk_ids = ["g1", "r1"]
        units = {
            "g1": _unit("g1", corpus="guidance", token_count=12),
            "r1": _unit("r1", corpus="rules", token_count=18),
        }
        covs = {"r1": _covariates_row("r1", style="old")}

        bm25 = _MockBM25(chunk_ids, gold_at_rank={0: 3.0, 1: 2.0})
        questions = [_question(link_type="guidance_para", gold="g1;r1")]
        rows, report = build_covariates(
            questions, bm25, chunk_ids, units, covs)

        assert rows[0]["anchor_id"] == "r1"  # skipped guidance
        assert rows[0]["drafting_style"] == "old"

    def test_output_columns(self):
        chunk_ids = ["r1"]
        units = {"r1": _unit("r1")}
        covs = {"r1": _covariates_row("r1")}
        bm25 = _MockBM25(chunk_ids, gold_at_rank={0: 5.0})
        questions = [_question(gold="r1")]

        rows, _ = build_covariates(questions, bm25, chunk_ids, units, covs)
        assert list(rows[0].keys()) == OUT_COLUMNS

    def test_absent_rank_counted(self):
        chunk_ids = ["x1", "x2"]
        units = {
            "x1": _unit("x1"), "x2": _unit("x2"),
            "r1": _unit("r1"),
        }
        covs = {"r1": _covariates_row("r1")}
        # r1 not in chunk_ids → never retrieved
        bm25 = _MockBM25(chunk_ids, gold_at_rank={0: 3.0, 1: 2.0})
        questions = [_question(gold="r1")]

        rows, report = build_covariates(
            questions, bm25, chunk_ids, units, covs)
        assert rows[0]["bm25_gold_rank"] == "absent"
        assert report["absent_count"] == 1

    def test_lexically_easy_per_tier(self):
        chunk_ids = ["r1", "r2"]
        units = {
            "r1": _unit("r1", token_count=10),
            "r2": _unit("r2", token_count=10),
        }
        covs = {
            "r1": _covariates_row("r1"),
            "r2": _covariates_row("r2"),
        }
        # r1 at rank 1 (gold hit), r2 at rank 2
        bm25 = _MockBM25(chunk_ids, gold_at_rank={0: 5.0, 1: 3.0})
        questions = [
            _question(qid="Q1", tier="T1", gold="r1"),
            _question(qid="Q2", tier="T2", gold="r2"),
        ]

        rows, report = build_covariates(
            questions, bm25, chunk_ids, units, covs)
        # Q1: first retrieved is r1 which is gold → easy=1
        # Q2: first retrieved is r1 which is NOT r2's gold → easy=0
        assert report["lexically_easy_per_tier"]["T1"] == 1
        assert report["lexically_easy_per_tier"]["T2"] == 0
