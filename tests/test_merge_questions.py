"""Tests for merge_questions.py (T2c-1)."""
import csv
import json
from pathlib import Path

import pytest

from code.evalset.merge_questions import (
    COLUMNS,
    HAND_LINK_TYPES,
    HAND_TIERS,
    VALID_ROUTES,
    _strata_counts,
    _validate_gold_ids,
    _validate_hand,
    _validate_route_names,
    merge,
)


# ── fixtures ─────────────────────────────────────────────────────────

def _syn_row(qid="Q0001", route="student", tier="T1", link_type="",
             validated="Y", gold="sec-a:00001"):
    return {c: "" for c in COLUMNS} | {
        "question_id": qid, "route": route, "tier": tier,
        "link_type": link_type,
        "question": "What is the Student visa requirement for funds?",
        "reference_answer": "The student must show at least 1000 pounds.",
        "gold_paragraph_ids": gold,
        "source": "prompt-v2", "generator_model": "gemma4:12b",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "validated_by_student": validated, "notes": "",
    }


def _hand_row(qid="Q0203", route="skilled_worker", tier="T3",
              link_type="rules_section_xcut", gold="sec-b:00002"):
    return {c: "" for c in COLUMNS} | {
        "question_id": qid, "route": route, "tier": tier,
        "link_type": link_type,
        "question": "What does the Skilled Worker route require?",
        "reference_answer": "The applicant must meet several requirements.",
        "gold_paragraph_ids": gold,
        "source": "hand", "generator_model": "",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "validated_by_student": "Y", "notes": "",
    }


def _corpus_ids(*extras):
    base = {"sec-a:00001", "sec-b:00002", "sec-c:00003"}
    return base | set(extras)


# ── merge assertions ─────────────────────────────────────────────────

class TestMergeBasic:
    def test_keeps_validated_Y(self):
        syn = [_syn_row(qid="Q0001"), _syn_row(qid="Q0002")]
        hand = [_hand_row()]
        merged, report = merge(syn, hand, _corpus_ids())
        assert report["synthetic_Y"] == 2
        assert report["merged_rows"] == 3

    def test_filters_validated_N(self):
        syn = [_syn_row(qid="Q0001", validated="Y"),
               _syn_row(qid="Q0002", validated="N")]
        hand = [_hand_row()]
        merged, report = merge(syn, hand, _corpus_ids())
        assert report["synthetic_Y"] == 1
        assert report["merged_rows"] == 2
        assert "student_T1" in report["synthetic_N_per_stratum"]

    def test_reports_N_per_stratum(self):
        syn = [
            _syn_row(qid="Q0001", validated="N", route="student", tier="T1"),
            _syn_row(qid="Q0002", validated="N", route="family", tier="T2"),
            _syn_row(qid="Q0003", validated="Y", route="visitor", tier="T1"),
        ]
        hand = [_hand_row()]
        merged, report = merge(syn, hand, _corpus_ids())
        assert report["synthetic_N_per_stratum"]["student_T1"] == 1
        assert report["synthetic_N_per_stratum"]["family_T2"] == 1
        assert report["synthetic_Y"] == 1

    def test_ordering_synthetic_by_qid_then_hand(self):
        syn = [_syn_row(qid="Q0003"), _syn_row(qid="Q0001"),
               _syn_row(qid="Q0002")]
        hand = [_hand_row(qid="Q0205"), _hand_row(qid="Q0203")]
        merged, report = merge(syn, hand, _corpus_ids())
        ids = [r["question_id"] for r in merged]
        # synthetic sorted, then hand in original order
        assert ids == ["Q0001", "Q0002", "Q0003", "Q0205", "Q0203"]

    def test_ids_unique(self):
        syn = [_syn_row(qid="Q0001")]
        hand = [_hand_row(qid="Q0001")]  # duplicate
        with pytest.raises(ValueError, match="Duplicate question_ids"):
            merge(syn, hand, _corpus_ids())

    def test_columns_d53(self):
        """Extra or missing columns raise ValueError."""
        bad = _syn_row()
        bad["extra_col"] = "oops"
        with pytest.raises(ValueError, match="extra="):
            merge([bad], [_hand_row()], _corpus_ids())


# ── hand rejection cases ─────────────────────────────────────────────

class TestHandRejection:
    def test_rejects_wrong_source(self):
        row = _hand_row()
        row["source"] = "prompt-v2"
        problems = _validate_hand([row])
        assert any("source=" in p for p in problems)

    def test_rejects_wrong_tier(self):
        row = _hand_row()
        row["tier"] = "T1"
        problems = _validate_hand([row])
        assert any("tier=" in p for p in problems)

    def test_rejects_wrong_link_type(self):
        row = _hand_row()
        row["link_type"] = "guidance_para"
        problems = _validate_hand([row])
        assert any("link_type=" in p for p in problems)

    def test_rejects_unvalidated(self):
        row = _hand_row()
        row["validated_by_student"] = "N"
        problems = _validate_hand([row])
        assert any("validated_by_student=" in p for p in problems)

    def test_accepts_valid_hand(self):
        assert _validate_hand([_hand_row()]) == []

    def test_merge_raises_on_bad_hand(self):
        bad_hand = _hand_row()
        bad_hand["source"] = "llm"
        with pytest.raises(ValueError, match="Hand file rejected"):
            merge([_syn_row()], [bad_hand], _corpus_ids())


# ── gold id validation ───────────────────────────────────────────────

class TestGoldValidation:
    def test_valid_gold_ids(self):
        rows = [_syn_row(gold="sec-a:00001;sec-b:00002")]
        assert _validate_gold_ids(rows, _corpus_ids()) == []

    def test_missing_gold_id(self):
        rows = [_syn_row(gold="sec-a:00001;nonexistent:99999")]
        problems = _validate_gold_ids(rows, _corpus_ids())
        assert len(problems) == 1
        assert "nonexistent:99999" in problems[0]

    def test_empty_gold_ids(self):
        rows = [_syn_row(gold="")]
        problems = _validate_gold_ids(rows, _corpus_ids())
        assert any("empty" in p for p in problems)

    def test_merge_raises_on_bad_gold(self):
        syn = [_syn_row(gold="missing:99999")]
        with pytest.raises(ValueError, match="Gold-id validation failed"):
            merge(syn, [_hand_row()], _corpus_ids())


# ── route name validation (\b boundaries) ────────────────────────────

class TestRouteValidation:
    def test_valid_routes(self):
        rows = [_syn_row(route=r) for r in VALID_ROUTES]
        assert _validate_route_names(rows) == []

    def test_invalid_route(self):
        rows = [_syn_row(route="part_9")]
        problems = _validate_route_names(rows)
        assert len(problems) == 1

    def test_route_whole_word_match(self):
        """Route names must match whole words, not substrings."""
        # All valid routes should pass
        for route in VALID_ROUTES:
            assert _validate_route_names([_syn_row(route=route)]) == []
        # A route that is a substring of a valid route should fail
        assert len(_validate_route_names([_syn_row(route="skill")])) == 1


# ── strata counts ────────────────────────────────────────────────────

class TestStrataCounts:
    def test_counts(self):
        rows = [
            _syn_row(route="student", tier="T1", link_type=""),
            _syn_row(route="student", tier="T1", link_type=""),
            _hand_row(route="family", tier="T3",
                      link_type="rules_section_xcut"),
        ]
        counts = _strata_counts(rows)
        assert counts["student_T1"] == 2
        assert counts["family_T3_rules_section_xcut"] == 1

    def test_link_type_in_key(self):
        rows = [_syn_row(tier="T3", link_type="guidance_para")]
        counts = _strata_counts(rows)
        assert "student_T3_guidance_para" in counts


# ── integration: write and read back ─────────────────────────────────

class TestRoundTrip:
    def test_csv_round_trip(self, tmp_path):
        syn = [_syn_row(qid="Q0001"), _syn_row(qid="Q0002")]
        hand = [_hand_row(qid="Q0203")]
        merged, _ = merge(syn, hand, _corpus_ids())

        out = tmp_path / "merged.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(merged)

        with out.open(encoding="utf-8", newline="") as f:
            read_back = list(csv.DictReader(f))

        assert len(read_back) == 3
        assert [r["question_id"] for r in read_back] == ["Q0001", "Q0002", "Q0203"]
        # verify D53 columns exactly
        assert list(read_back[0].keys()) == COLUMNS
