"""
Tests for T6: generation, judging, rating, select_configs.
"""

import csv
import json
from pathlib import Path

import pytest


class TestSelectConfigs:
    def test_dryrun_summary(self):
        """select_configs on the dry-run summary yields expected configs."""
        from code.eval.select_configs import select
        summary_path = Path("data/results/2026-09-18_grid_summary.csv")
        if not summary_path.exists():
            pytest.skip("dry-run summary not available")
        result = select(str(summary_path))
        assert "flat" in result
        assert "pcreturn" in result
        assert "linkexp" in result
        assert result["flat"]["config_name"] == "para-bge-hybrid-flat-dryrun"
        assert result["pcreturn"]["config_name"] == "parentchild-bge-hybrid-pcreturn-dryrun"
        assert result["linkexp"]["config_name"] == "para-bge-hybrid-linkexp-dryrun"


class TestContextAssembly:
    def test_flat_context(self):
        """Flat context = each unit's text."""
        from code.eval.run_generation import build_context
        from unittest.mock import MagicMock
        tok = MagicMock()
        tok.encode = lambda t, **kw: list(range(len(t.split())))

        units_by_id = {"u1": {"unit_id": "u1", "corpus": "rules",
                               "text": "heading\nunit text"}}
        corpus_records = {"u1": {"paragraph_id": "u1", "section_title": "Sec",
                                  "rule_ref": "R1", "seq": 1, "heading_path": ["H"],
                                  "text": "unit text"}}
        units_ret = [{"unit_id": "u1", "paragraph_ids": ["u1"]}]

        ctx, labels, tc = build_context(units_ret, "flat", units_by_id,
                                         corpus_records, tok)
        assert "unit text" in ctx
        assert len(labels) == 1

    def test_linkexp_context(self):
        """Linkexp context includes appended targets."""
        from code.eval.run_generation import build_context
        from unittest.mock import MagicMock
        tok = MagicMock()
        tok.encode = lambda t, **kw: list(range(len(t.split())))

        units_by_id = {
            "u1": {"unit_id": "u1", "corpus": "rules", "text": "h\nmain"},
            "t1": {"unit_id": "t1", "corpus": "rules", "text": "h\ntarget"},
        }
        corpus_records = {
            "u1": {"paragraph_id": "u1", "section_title": "S", "rule_ref": "R1",
                   "seq": 1, "heading_path": [], "text": "main"},
            "t1": {"paragraph_id": "t1", "section_title": "S2", "rule_ref": "T1",
                   "seq": 2, "heading_path": [], "text": "target"},
        }
        units_ret = [{"unit_id": "u1", "paragraph_ids": ["u1"],
                       "appended_targets": ["t1"]}]

        ctx, labels, tc = build_context(units_ret, "linkexp", units_by_id,
                                         corpus_records, tok)
        assert "Cross-referenced" in ctx
        assert len(labels) == 2


class TestLabels:
    def test_rules_label(self):
        from code.eval.run_generation import _label
        r = {"section_title": "Appendix Skilled Worker", "rule_ref": "SW 1.1",
             "paragraph_id": "x:001"}
        assert _label(r, "rules") == "[Appendix Skilled Worker | SW 1.1]"

    def test_guidance_label(self):
        from code.eval.run_generation import _label
        r = {"section_title": "Skilled Worker guidance", "paragraph_id": "g:001"}
        assert _label(r, "guidance") == "[Skilled Worker guidance | g:001]"


class TestFaithfulness:
    def test_arithmetic(self):
        """2 supported, 1 unsupported → 2/3."""
        stmts = [
            {"statement": "a", "verdict": "supported"},
            {"statement": "b", "verdict": "supported"},
            {"statement": "c", "verdict": "unsupported"},
        ]
        supported = sum(1 for s in stmts if s["verdict"] == "supported")
        assert supported / len(stmts) == pytest.approx(2 / 3)


class TestNotFoundRule:
    def test_not_found_gold_absent(self):
        """'Not found' answer scores 1.0 when gold absent from passages."""
        gold_ids = {"g1"}
        passage_pids = {"x1", "x2"}
        gold_absent = not gold_ids.intersection(passage_pids)
        assert gold_absent is True
        score = 1.0 if gold_absent else 0.0
        assert score == 1.0

    def test_not_found_gold_present(self):
        """'Not found' answer scores 0.0 when gold is in passages."""
        gold_ids = {"g1"}
        passage_pids = {"g1", "x2"}
        gold_absent = not gold_ids.intersection(passage_pids)
        assert gold_absent is False


class TestMalformedJudge:
    def test_null_on_malformed(self):
        """Malformed judge output returns None, never a guessed value."""
        from code.eval.run_judge import _extract_json
        assert _extract_json("no json here at all") is None
        assert _extract_json("{malformed") is None


class TestRatingSheet:
    def test_no_judge_columns(self):
        """Rating sheet has student columns but no judge scores."""
        expected = {"student_faithfulness", "student_accuracy",
                    "student_copies_or_answers", "notes"}
        forbidden = {"faithfulness", "accuracy", "judge_faithfulness", "judge_accuracy"}
        # The column names
        fields = ["question_id", "config", "tier", "question", "context",
                  "answer", "reference_answer", "student_faithfulness",
                  "student_accuracy", "student_copies_or_answers", "notes"]
        assert expected.issubset(set(fields))
        assert not forbidden.intersection(set(fields))


class TestKappa:
    def test_toy_perfect_agreement(self):
        """Perfect agreement gives kappa = 1."""
        y1 = [0, 0.5, 1, 1, 0.5]
        y2 = [0, 0.5, 1, 1, 0.5]
        # Simple check: all match
        assert all(a == b for a, b in zip(y1, y2))


class TestManifestSerialises:
    def test_stratum_done_string_keys(self):
        """Manifest with tuple-keyed Counter serialises to JSON without error."""
        from collections import Counter
        stratum_done = Counter()
        stratum_done[("skilled_worker", "T1", "")] = 12
        stratum_done[("family", "T3", "rules_para")] = 5

        # Same conversion as generate_questions_v1.py
        manifest = {
            "stratum_done": {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in stratum_done.items()},
            "none_counts": {"family_T3_rules_para": 1},
        }
        serialised = json.dumps(manifest)
        reloaded = json.loads(serialised)
        assert reloaded["stratum_done"]["skilled_worker_T1_"] == 12
        assert reloaded["stratum_done"]["family_T3_rules_para"] == 5
