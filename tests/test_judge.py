"""Tests for run_judge.py (T6b-2)."""
import json
from pathlib import Path

import pytest

from code.eval.run_judge import (
    _extract_json,
    compute_faithfulness,
    handle_not_found,
    parse_accuracy,
    parse_faithfulness,
    verify,
)


# ── parse_faithfulness (Call A) ──────────────────────────────────────

class TestParseFaithfulness:
    def test_well_formed(self):
        raw = json.dumps({
            "statements": [
                {"statement": "claim A", "verdict": "supported"},
                {"statement": "claim B", "verdict": "unsupported"},
                {"statement": "claim C", "verdict": "supported"},
            ]
        })
        result = parse_faithfulness(raw)
        assert result["statements"] == ["claim A", "claim B", "claim C"]
        assert result["verdicts"] == ["supported", "unsupported", "supported"]
        assert abs(result["faithfulness"] - 2 / 3) < 1e-5
        assert result["flag"] is None

    def test_all_supported(self):
        raw = json.dumps({
            "statements": [
                {"statement": "a", "verdict": "supported"},
                {"statement": "b", "verdict": "supported"},
            ]
        })
        result = parse_faithfulness(raw)
        assert result["faithfulness"] == 1.0

    def test_empty_statements(self):
        """Empty statements list → faithfulness 1.0 (vacuously faithful)."""
        raw = json.dumps({"statements": []})
        result = parse_faithfulness(raw)
        assert result["statements"] == []
        assert result["verdicts"] == []
        assert result["faithfulness"] == 1.0
        assert result["flag"] is None

    def test_fenced_json(self):
        raw = '```json\n{"statements": [{"statement": "x", "verdict": "supported"}]}\n```'
        result = parse_faithfulness(raw)
        assert result["faithfulness"] == 1.0
        assert result["flag"] is None

    def test_malformed_unparseable(self):
        result = parse_faithfulness("no json at all")
        assert result["faithfulness"] is None
        assert result["flag"] == "unparseable_json"

    def test_malformed_no_response(self):
        result = parse_faithfulness(None)
        assert result["faithfulness"] is None
        assert result["flag"] == "no_response"

    def test_malformed_missing_statements(self):
        raw = json.dumps({"score": 1})
        result = parse_faithfulness(raw)
        assert result["faithfulness"] is None
        assert result["flag"] == "missing_statements_list"

    def test_malformed_invalid_verdict(self):
        raw = json.dumps({
            "statements": [{"statement": "x", "verdict": "maybe"}]
        })
        result = parse_faithfulness(raw)
        assert result["faithfulness"] is None
        assert "invalid_verdict" in result["flag"]

    def test_malformed_statement_not_dict(self):
        raw = json.dumps({"statements": ["just a string"]})
        result = parse_faithfulness(raw)
        assert result["faithfulness"] is None
        assert result["flag"] == "statement_not_dict"

    def test_length_mismatch_impossible(self):
        """statements and verdicts always have same length by construction."""
        raw = json.dumps({
            "statements": [
                {"statement": "a", "verdict": "supported"},
                {"statement": "b", "verdict": "unsupported"},
            ]
        })
        result = parse_faithfulness(raw)
        assert len(result["statements"]) == len(result["verdicts"])


# ── parse_accuracy (Call B) ──────────────────────────────────────────

class TestParseAccuracy:
    def test_well_formed(self):
        raw = json.dumps({"score": 1, "reason": "Fully correct."})
        result = parse_accuracy(raw)
        assert result["accuracy"] == 1.0
        assert result["reason"] == "Fully correct."
        assert result["flag"] is None

    def test_half_score(self):
        raw = json.dumps({"score": 0.5, "reason": "Partial."})
        result = parse_accuracy(raw)
        assert result["accuracy"] == 0.5

    def test_zero_score(self):
        raw = json.dumps({"score": 0, "reason": "Wrong."})
        result = parse_accuracy(raw)
        assert result["accuracy"] == 0.0

    def test_malformed_no_response(self):
        result = parse_accuracy(None)
        assert result["accuracy"] is None
        assert result["flag"] == "no_response"

    def test_malformed_unparseable(self):
        result = parse_accuracy("{broken")
        assert result["accuracy"] is None
        assert result["flag"] == "unparseable_json"

    def test_malformed_missing_score(self):
        raw = json.dumps({"reason": "no score key"})
        result = parse_accuracy(raw)
        assert result["accuracy"] is None
        assert result["flag"] == "missing_score"

    def test_malformed_score_outside_set(self):
        raw = json.dumps({"score": 0.7, "reason": "x"})
        result = parse_accuracy(raw)
        assert result["accuracy"] is None
        assert "score_outside_set" in result["flag"]

    def test_malformed_score_not_numeric(self):
        raw = json.dumps({"score": "high", "reason": "x"})
        result = parse_accuracy(raw)
        assert result["accuracy"] is None
        assert "score_not_numeric" in result["flag"]


# ── Not-found handling ───────────────────────────────────────────────

class TestNotFound:
    def test_gold_absent_from_passages(self):
        """'Not found' + gold absent → faithfulness 1.0."""
        result = handle_not_found(
            gold_ids={"g1", "g2"}, passage_pids={"x1", "x2"})
        assert result["statements"] == []
        assert result["verdicts"] == []
        assert result["faithfulness"] == 1.0
        assert result["not_found_case"] is True

    def test_gold_present_in_passages(self):
        """'Not found' + gold present → faithfulness 0.0."""
        result = handle_not_found(
            gold_ids={"g1"}, passage_pids={"g1", "x2"})
        assert result["faithfulness"] == 0.0
        assert result["not_found_case"] is True

    def test_empty_gold(self):
        result = handle_not_found(gold_ids=set(), passage_pids={"x1"})
        assert result["faithfulness"] == 1.0  # no gold to miss


# ── compute_faithfulness (pure) ──────────────────────────────────────

class TestComputeFaithfulness:
    def test_all_supported(self):
        assert compute_faithfulness(["a", "b"], ["supported", "supported"]) == 1.0

    def test_none_supported(self):
        assert compute_faithfulness(["a"], ["unsupported"]) == 0.0

    def test_mixed(self):
        f = compute_faithfulness(["a", "b", "c"],
                                 ["supported", "supported", "unsupported"])
        assert abs(f - 2 / 3) < 1e-5

    def test_empty(self):
        assert compute_faithfulness([], []) == 1.0


# ── _extract_json ────────────────────────────────────────────────────

class TestExtractJson:
    def test_fenced(self):
        text = 'Some text\n```json\n{"key": "val"}\n```\nmore'
        assert _extract_json(text) == {"key": "val"}

    def test_bare(self):
        assert _extract_json('blah {"score": 1} blah') == {"score": 1}

    def test_none_input(self):
        assert _extract_json(None) is None

    def test_no_json(self):
        assert _extract_json("no json here at all") is None

    def test_malformed_json(self):
        assert _extract_json("{malformed") is None


# ── verify mode ──────────────────────────────────────────────────────

class TestVerify:
    def test_verify_passes_on_consistent(self, tmp_path):
        """Verify passes when stored faithfulness matches recomputed."""
        judgements = [
            {"question_id": "Q0001",
             "statements": ["a", "b"],
             "verdicts": ["supported", "unsupported"],
             "faithfulness": 0.5,  # 1/2
             "flag": None},
            {"question_id": "Q0002",
             "statements": ["x"],
             "verdicts": ["supported"],
             "faithfulness": 1.0,
             "flag": None},
        ]
        jpath = tmp_path / "judgements.jsonl"
        with jpath.open("w") as f:
            for j in judgements:
                f.write(json.dumps(j) + "\n")

        assert verify(tmp_path, Path("dummy.csv")) is True

    def test_verify_fails_on_mismatch(self, tmp_path):
        """Verify fails when stored value doesn't match recomputed."""
        judgements = [
            {"question_id": "Q0001",
             "statements": ["a", "b"],
             "verdicts": ["supported", "unsupported"],
             "faithfulness": 0.99,  # wrong — should be 0.5
             "flag": None},
        ]
        jpath = tmp_path / "judgements.jsonl"
        with jpath.open("w") as f:
            for j in judgements:
                f.write(json.dumps(j) + "\n")

        assert verify(tmp_path, Path("dummy.csv")) is False

    def test_verify_skips_null_rows(self, tmp_path):
        """Null (malformed) rows are skipped, not failed."""
        judgements = [
            {"question_id": "Q0001",
             "statements": None, "verdicts": None,
             "faithfulness": None,
             "flag": "unparseable_json"},
            {"question_id": "Q0002",
             "statements": ["a"], "verdicts": ["supported"],
             "faithfulness": 1.0, "flag": None},
        ]
        jpath = tmp_path / "judgements.jsonl"
        with jpath.open("w") as f:
            for j in judgements:
                f.write(json.dumps(j) + "\n")

        assert verify(tmp_path, Path("dummy.csv")) is True

    def test_verify_skips_not_found_case(self, tmp_path):
        """Not-found cases are skipped (faithfulness set by rule, not compute)."""
        judgements = [
            {"question_id": "Q0001",
             "statements": [], "verdicts": [],
             "faithfulness": 0.0,
             "not_found_case": True, "flag": None},
            {"question_id": "Q0002",
             "statements": ["a"], "verdicts": ["supported"],
             "faithfulness": 1.0, "flag": None},
        ]
        jpath = tmp_path / "judgements.jsonl"
        with jpath.open("w") as f:
            for j in judgements:
                f.write(json.dumps(j) + "\n")

        assert verify(tmp_path, Path("dummy.csv")) is True

    def test_verify_missing_file(self, tmp_path):
        """Verify returns False if judgements.jsonl missing."""
        assert verify(tmp_path, Path("dummy.csv")) is False
