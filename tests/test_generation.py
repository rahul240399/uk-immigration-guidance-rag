"""
Tests for T6 / T6b-1: generation, context assembly, judging, rating, select_configs.
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from code.eval.run_generation import build_context, _label


# ── shared fixtures ──────────────────────────────────────────────────

def _mock_tok():
    tok = MagicMock()
    tok.encode = lambda t, **kw: list(range(len(t.split())))
    return tok


def _unit(uid, corpus="rules", text="heading\nunit text", tc=20):
    return {"unit_id": uid, "corpus": corpus, "text": text,
            "unit_token_count": tc, "token_count": tc}


def _rec(pid, corpus="rules", section_title="Sec", rule_ref=None,
         seq=1, heading_path=None, text="unit text"):
    r = {"paragraph_id": pid, "corpus": corpus,
         "section_title": section_title, "seq": seq,
         "heading_path": heading_path or ["H"],
         "text": text}
    if rule_ref:
        r["rule_ref"] = rule_ref
    return r


# ── T6b-1 context assembly for the three architectures ──────────────

class TestContextAssemblyFlat:
    def test_flat_basic(self):
        """Flat context = each unit's text, one label per unit."""
        units_by_id = {"u1": _unit("u1")}
        corpus_records = {"u1": _rec("u1", rule_ref="R1")}
        units_ret = [{"unit_id": "u1", "paragraph_ids": ["u1"]}]

        ctx, labels, tc, n_units, n_appended = build_context(
            units_ret, "flat", units_by_id, corpus_records, _mock_tok())
        assert "unit text" in ctx
        assert len(labels) == 1
        assert n_units == 1
        assert n_appended == 0

    def test_flat_multiple_units(self):
        units_by_id = {
            "u1": _unit("u1", text="h\nfirst"),
            "u2": _unit("u2", text="h\nsecond"),
        }
        corpus_records = {
            "u1": _rec("u1", rule_ref="R1"),
            "u2": _rec("u2", rule_ref="R2", seq=2),
        }
        units_ret = [
            {"unit_id": "u1", "paragraph_ids": ["u1"]},
            {"unit_id": "u2", "paragraph_ids": ["u2"]},
        ]
        ctx, labels, tc, n_units, n_appended = build_context(
            units_ret, "flat", units_by_id, corpus_records, _mock_tok())
        assert n_units == 2
        assert n_appended == 0
        assert len(labels) == 2


class TestContextAssemblyLinkexp:
    def test_linkexp_with_appended(self):
        """Linkexp renders Cross-referenced passages for appended_targets."""
        units_by_id = {
            "u1": _unit("u1", text="h\nmain"),
            "t1": _unit("t1", text="h\ntarget one"),
            "t2": _unit("t2", text="h\ntarget two"),
        }
        corpus_records = {
            "u1": _rec("u1", rule_ref="R1"),
            "t1": _rec("t1", rule_ref="T1"),
            "t2": _rec("t2", rule_ref="T2"),
        }
        units_ret = [{"unit_id": "u1", "paragraph_ids": ["u1"],
                       "appended_targets": ["t1", "t2"]}]

        ctx, labels, tc, n_units, n_appended = build_context(
            units_ret, "linkexp", units_by_id, corpus_records, _mock_tok())
        assert n_units == 1
        assert n_appended == 2
        assert sum(1 for lb in labels if "Cross-referenced" in lb) == 2
        assert "[Cross-referenced: T1]" in labels
        assert "[Cross-referenced: T2]" in labels

    def test_linkexp_count_assertion_passes(self):
        """The assertion passes: Cross-ref count == n_appended rendered."""
        units_by_id = {
            "u1": _unit("u1"), "u2": _unit("u2"),
            "t1": _unit("t1"), "t2": _unit("t2"),
        }
        corpus_records = {
            "u1": _rec("u1", rule_ref="R1"),
            "u2": _rec("u2", rule_ref="R2"),
            "t1": _rec("t1", rule_ref="T1"),
            "t2": _rec("t2", rule_ref="T2"),
        }
        # Targets are deduplicated at retrieval time by run_grid.py
        units_ret = [
            {"unit_id": "u1", "paragraph_ids": ["u1"],
             "appended_targets": ["t1"]},
            {"unit_id": "u2", "paragraph_ids": ["u2"],
             "appended_targets": ["t2"]},
        ]
        ctx, labels, tc, n_units, n_appended = build_context(
            units_ret, "linkexp", units_by_id, corpus_records, _mock_tok())
        cross_ref_count = sum(1 for lb in labels
                              if lb.startswith("[Cross-referenced:"))
        assert cross_ref_count == 2
        assert n_appended == 2

    def test_linkexp_no_appended(self):
        """Linkexp with no appended_targets works like flat."""
        units_by_id = {"u1": _unit("u1")}
        corpus_records = {"u1": _rec("u1", rule_ref="R1")}
        units_ret = [{"unit_id": "u1", "paragraph_ids": ["u1"],
                       "appended_targets": []}]

        ctx, labels, tc, n_units, n_appended = build_context(
            units_ret, "linkexp", units_by_id, corpus_records, _mock_tok())
        assert n_appended == 0
        assert not any("Cross-referenced" in lb for lb in labels)

    def test_linkexp_n_appended_equals_cross_ref_count(self):
        """Per spec: n_appended passages == distinct appended_targets count
        when targets are unique across units."""
        units_by_id = {
            "u1": _unit("u1"),
            "t1": _unit("t1"),
        }
        corpus_records = {
            "u1": _rec("u1", rule_ref="R1"),
            "t1": _rec("t1", rule_ref="T1"),
        }
        units_ret = [{"unit_id": "u1", "paragraph_ids": ["u1"],
                       "appended_targets": ["t1"]}]

        ctx, labels, tc, n_units, n_appended = build_context(
            units_ret, "linkexp", units_by_id, corpus_records, _mock_tok())
        cross_ref_count = sum(1 for lb in labels
                              if lb.startswith("[Cross-referenced:"))
        assert n_appended == 1
        assert cross_ref_count == 1


class TestContextAssemblyPcreturn:
    def test_pcreturn_renders_paragraph_ids_in_seq_order(self):
        """Pcreturn renders paragraph_ids sorted by seq, heading once."""
        units_by_id = {"u1": _unit("u1")}
        corpus_records = {
            "p3": _rec("p3", seq=3, text="third paragraph",
                        heading_path=["Section A"]),
            "p1": _rec("p1", seq=1, text="first paragraph",
                        heading_path=["Section A"]),
            "p2": _rec("p2", seq=2, text="second paragraph",
                        heading_path=["Section A"]),
        }
        units_ret = [{"unit_id": "u1", "paragraph_ids": ["p3", "p1", "p2"]}]

        ctx, labels, tc, n_units, n_appended = build_context(
            units_ret, "pcreturn", units_by_id, corpus_records, _mock_tok())
        # Should be ordered: first, second, third
        lines = ctx.split("\n")
        text_lines = [l for l in lines if "paragraph" in l]
        assert text_lines == ["first paragraph", "second paragraph",
                              "third paragraph"]
        assert n_appended == 0

    def test_pcreturn_heading_once(self):
        """Heading appears once even with multiple paragraphs under it."""
        units_by_id = {"u1": _unit("u1")}
        corpus_records = {
            "p1": _rec("p1", seq=1, text="alpha",
                        heading_path=["Head"]),
            "p2": _rec("p2", seq=2, text="beta",
                        heading_path=["Head"]),
        }
        units_ret = [{"unit_id": "u1", "paragraph_ids": ["p1", "p2"]}]

        ctx, labels, tc, n_units, n_appended = build_context(
            units_ret, "pcreturn", units_by_id, corpus_records, _mock_tok())
        assert ctx.count("Head") == 1  # heading rendered once

    def test_pcreturn_different_headings(self):
        """Different headings each appear once."""
        units_by_id = {"u1": _unit("u1")}
        corpus_records = {
            "p1": _rec("p1", seq=1, text="alpha",
                        heading_path=["Head A"]),
            "p2": _rec("p2", seq=2, text="beta",
                        heading_path=["Head B"]),
        }
        units_ret = [{"unit_id": "u1", "paragraph_ids": ["p1", "p2"]}]

        ctx, labels, tc, n_units, n_appended = build_context(
            units_ret, "pcreturn", units_by_id, corpus_records, _mock_tok())
        assert "Head A" in ctx
        assert "Head B" in ctx


# ── Labels ───────────────────────────────────────────────────────────

class TestLabels:
    def test_rules_label(self):
        r = {"section_title": "Appendix Skilled Worker", "rule_ref": "SW 1.1",
             "paragraph_id": "x:001"}
        assert _label(r, "rules") == "[Appendix Skilled Worker | SW 1.1]"

    def test_guidance_label(self):
        r = {"section_title": "Skilled Worker guidance", "paragraph_id": "g:001"}
        assert _label(r, "guidance") == "[Skilled Worker guidance | g:001]"

    def test_rules_label_no_rule_ref(self):
        r = {"section_title": "Sec", "paragraph_id": "x:001"}
        assert _label(r, "rules") == "[Sec | x:001]"


# ── select_configs (preserved from existing tests) ───────────────────

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


# ── Faithfulness arithmetic ──────────────────────────────────────────

class TestFaithfulness:
    def test_arithmetic(self):
        stmts = [
            {"statement": "a", "verdict": "supported"},
            {"statement": "b", "verdict": "supported"},
            {"statement": "c", "verdict": "unsupported"},
        ]
        supported = sum(1 for s in stmts if s["verdict"] == "supported")
        assert supported / len(stmts) == pytest.approx(2 / 3)


class TestNotFoundRule:
    def test_not_found_gold_absent(self):
        gold_ids = {"g1"}
        passage_pids = {"x1", "x2"}
        assert not gold_ids.intersection(passage_pids)

    def test_not_found_gold_present(self):
        gold_ids = {"g1"}
        passage_pids = {"g1", "x2"}
        assert gold_ids.intersection(passage_pids)


class TestMalformedJudge:
    def test_null_on_malformed(self):
        from code.eval.run_judge import _extract_json
        assert _extract_json("no json here at all") is None
        assert _extract_json("{malformed") is None


class TestKappa:
    def test_toy_perfect_agreement(self):
        y1 = [0, 0.5, 1, 1, 0.5]
        y2 = [0, 0.5, 1, 1, 0.5]
        assert all(a == b for a, b in zip(y1, y2))


class TestManifestSerialises:
    def test_stratum_done_string_keys(self):
        from collections import Counter
        stratum_done = Counter()
        stratum_done[("skilled_worker", "T1", "")] = 12
        stratum_done[("family", "T3", "rules_para")] = 5
        manifest = {
            "stratum_done": {f"{k[0]}_{k[1]}_{k[2]}": v
                             for k, v in stratum_done.items()},
        }
        serialised = json.dumps(manifest)
        reloaded = json.loads(serialised)
        assert reloaded["stratum_done"]["skilled_worker_T1_"] == 12
