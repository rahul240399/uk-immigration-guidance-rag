"""Tests for freeze_evalset.py (T2c-4)."""
import pytest

from code.evalset.freeze_evalset import (
    VALID_VERDICTS,
    apply_verdicts,
    floor_check,
)
from code.evalset.merge_questions import COLUMNS


# ── fixtures ─────────────────────────────────────────────────────────

def _row(qid, route="student", tier="T1", link_type="", source="prompt-v2"):
    return {c: "" for c in COLUMNS} | {
        "question_id": qid, "route": route, "tier": tier,
        "link_type": link_type, "question": f"Question {qid}",
        "reference_answer": "Answer text here.",
        "gold_paragraph_ids": f"sec:0000{qid[-1]}",
        "source": source, "validated_by_student": "Y",
    }


def _nd(pair_id, qid_a, qid_b, verdict, reworded=""):
    return {
        "pair_id": pair_id, "qid_a": qid_a, "qid_b": qid_b,
        "cosine": "0.92", "same_gold": "N",
        "reason_set": "threshold", "route_a": "student",
        "route_b": "student", "tier_a": "T1", "tier_b": "T1",
        "question_a": "Q A", "question_b": "Q B",
        "verdict": verdict, "reworded_question_b": reworded, "note": "",
    }


# ── verdict application ─────────────────────────────────────────────

class TestApplyVerdicts:
    def test_distinct_keeps_both(self):
        rows = [_row("Q0001"), _row("Q0002")]
        nds = [_nd("P0001", "Q0001", "Q0002", "distinct")]
        kept, removed = apply_verdicts(rows, nds)
        assert len(kept) == 2
        assert len(removed) == 0

    def test_duplicate_drops_later_qid(self):
        rows = [_row("Q0001"), _row("Q0005"), _row("Q0013")]
        nds = [_nd("P0001", "Q0005", "Q0013", "duplicate")]
        kept, removed = apply_verdicts(rows, nds)
        assert len(kept) == 2
        ids = [r["question_id"] for r in kept]
        assert "Q0013" not in ids  # later id dropped
        assert "Q0005" in ids
        assert len(removed) == 1
        assert removed[0]["qid_dropped"] == "Q0013"

    def test_duplicate_drops_later_regardless_of_order_in_pair(self):
        """The pair might list qid_b < qid_a; still drop the later."""
        rows = [_row("Q0001"), _row("Q0010"), _row("Q0003")]
        nds = [_nd("P0001", "Q0010", "Q0003", "duplicate")]
        kept, removed = apply_verdicts(rows, nds)
        ids = [r["question_id"] for r in kept]
        assert "Q0010" not in ids  # Q0010 > Q0003
        assert "Q0003" in ids

    def test_reworded_replaces_text(self):
        rows = [_row("Q0001"), _row("Q0002")]
        nds = [_nd("P0001", "Q0001", "Q0002", "reworded",
                    reworded="New question text here")]
        kept, removed = apply_verdicts(rows, nds)
        assert len(kept) == 2
        q2 = [r for r in kept if r["question_id"] == "Q0002"][0]
        assert q2["question"] == "New question text here"

    def test_reworded_empty_text_raises(self):
        rows = [_row("Q0001"), _row("Q0002")]
        nds = [_nd("P0001", "Q0001", "Q0002", "reworded", reworded="")]
        with pytest.raises(ValueError, match="reworded_question_b is empty"):
            apply_verdicts(rows, nds)

    def test_empty_verdict_stops(self):
        rows = [_row("Q0001"), _row("Q0002")]
        nds = [_nd("P0001", "Q0001", "Q0002", "")]
        with pytest.raises(ValueError, match="Empty verdict"):
            apply_verdicts(rows, nds)

    def test_invalid_verdict_stops(self):
        rows = [_row("Q0001"), _row("Q0002")]
        nds = [_nd("P0001", "Q0001", "Q0002", "maybe")]
        with pytest.raises(ValueError, match="Invalid verdict"):
            apply_verdicts(rows, nds)

    def test_multiple_duplicates(self):
        rows = [_row("Q0001"), _row("Q0005"), _row("Q0010"), _row("Q0013")]
        nds = [
            _nd("P0001", "Q0005", "Q0013", "duplicate"),
            _nd("P0002", "Q0001", "Q0010", "duplicate"),
        ]
        kept, removed = apply_verdicts(rows, nds)
        ids = {r["question_id"] for r in kept}
        assert "Q0013" not in ids
        assert "Q0010" not in ids
        assert len(removed) == 2

    def test_no_neardup_rows(self):
        rows = [_row("Q0001"), _row("Q0002")]
        kept, removed = apply_verdicts(rows, [])
        assert len(kept) == 2
        assert len(removed) == 0


# ── floor check ──────────────────────────────────────────────────────

class TestFloorCheck:
    def test_all_above_floor(self):
        rows = [_row(f"Q{i:04d}", route="student", tier="T1")
                for i in range(12)]
        floor = {"student": {"T1": 10}}
        shortfalls = floor_check(rows, floor)
        assert shortfalls == []

    def test_below_floor(self):
        rows = [_row(f"Q{i:04d}", route="student", tier="T2")
                for i in range(5)]
        floor = {"student": {"T2": 7}}
        shortfalls = floor_check(rows, floor)
        assert len(shortfalls) == 1
        assert shortfalls[0]["route"] == "student"
        assert shortfalls[0]["actual"] == 5
        assert shortfalls[0]["floor"] == 7
        assert shortfalls[0]["shortfall"] == 2

    def test_t3_synthetic_counts_hand_and_synthetic(self):
        """T3_synthetic floor key maps to tier T3 in the data."""
        rows = [
            _row("Q0001", route="family", tier="T3", source="prompt-v2"),
            _row("Q0002", route="family", tier="T3", source="hand"),
            _row("Q0003", route="family", tier="T3", source="hand"),
        ]
        floor = {"family": {"T3_synthetic": 4}}
        shortfalls = floor_check(rows, floor)
        # 3 T3 rows vs floor 4
        assert len(shortfalls) == 1
        assert shortfalls[0]["actual"] == 3
        assert shortfalls[0]["floor"] == 4

    def test_zero_floor_skipped(self):
        rows = []
        floor = {"graduate": {"T3_synthetic": 0}}
        shortfalls = floor_check(rows, floor)
        assert shortfalls == []

    def test_missing_route_counted(self):
        """A route with 0 rows still fails if floor > 0."""
        rows = [_row("Q0001", route="student", tier="T1")]
        floor = {"visitor": {"T1": 10}}
        shortfalls = floor_check(rows, floor)
        assert len(shortfalls) == 1
        assert shortfalls[0]["actual"] == 0
