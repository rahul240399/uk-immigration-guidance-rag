"""Tests for export_rating.py (T6b-3)."""
import csv
import json

import pytest

from code.eval.export_rating import (
    FORBIDDEN_SHEET_COLUMNS,
    KEY_COLUMNS,
    SHEET_COLUMNS,
    build_sheet_and_key,
    draw_question_ids,
)


# ── fixtures ─────────────────────────────────────────────────────────

def _questions(n_t1=25, n_t2=15, n_t3=20):
    """Build a question list with the given tier counts."""
    qs = []
    idx = 1
    for tier, n in [("T1", n_t1), ("T2", n_t2), ("T3", n_t3)]:
        for _ in range(n):
            qs.append({
                "question_id": f"Q{idx:04d}",
                "tier": tier,
                "question": f"Question {idx}?",
                "reference_answer": f"Answer {idx}.",
                "route": "student",
                "gold_paragraph_ids": f"sec:{idx:05d}",
            })
            idx += 1
    return qs


def _answers(question_ids, config_name):
    """Build answers dict for given question ids."""
    return {
        qid: {
            "question_id": qid,
            "answer": f"Answer for {qid} under {config_name}.",
            "labels": [f"[Label for {qid}]"],
            "context_tokens": 100,
            "n_units": 5,
            "n_appended": 0,
        }
        for qid in question_ids
    }


RATED_BY_TIER = {"T1": 20, "T2": 13, "T3": 17}
SEED = 20260918


# ── draw_question_ids ────────────────────────────────────────────────

class TestDrawQuestionIds:
    def test_correct_counts(self):
        qs = _questions()
        drawn = draw_question_ids(qs, RATED_BY_TIER, SEED)
        assert len(drawn) == 50
        # Check per tier
        by_tier = {}
        qmap = {q["question_id"]: q for q in qs}
        for qid in drawn:
            t = qmap[qid]["tier"]
            by_tier[t] = by_tier.get(t, 0) + 1
        assert by_tier == {"T1": 20, "T2": 13, "T3": 17}

    def test_deterministic(self):
        qs = _questions()
        draw1 = draw_question_ids(qs, RATED_BY_TIER, SEED)
        draw2 = draw_question_ids(qs, RATED_BY_TIER, SEED)
        assert draw1 == draw2

    def test_different_seed(self):
        qs = _questions()
        draw1 = draw_question_ids(qs, RATED_BY_TIER, SEED)
        draw2 = draw_question_ids(qs, RATED_BY_TIER, 99999)
        # Very unlikely to be identical
        assert draw1 != draw2

    def test_insufficient_pool_raises(self):
        qs = _questions(n_t1=5)  # only 5 T1 but need 20
        with pytest.raises(ValueError, match="Tier T1"):
            draw_question_ids(qs, RATED_BY_TIER, SEED)

    def test_unique_ids(self):
        qs = _questions()
        drawn = draw_question_ids(qs, RATED_BY_TIER, SEED)
        assert len(set(drawn)) == len(drawn)


# ── build_sheet_and_key ──────────────────────────────────────────────

class TestBuildSheetAndKey:
    def _setup(self):
        qs = _questions()
        qmap = {q["question_id"]: q for q in qs}
        drawn = draw_question_ids(qs, RATED_BY_TIER, SEED)
        configs = {
            "flat": {"config_name": "para-bge-hybrid-flat"},
            "linkexp": {"config_name": "para-bge-hybrid-linkexp"},
            "pcreturn": {"config_name": "parentchild-bge-hybrid-pcreturn"},
        }
        all_ids = [q["question_id"] for q in qs]
        answers_by_config = {
            "para-bge-hybrid-flat": _answers(all_ids, "flat"),
            "para-bge-hybrid-linkexp": _answers(all_ids, "linkexp"),
            "parentchild-bge-hybrid-pcreturn": _answers(all_ids, "pcreturn"),
        }
        gen_runs = {
            "para-bge-hybrid-flat": "run_flat",
            "para-bge-hybrid-linkexp": "run_linkexp",
            "parentchild-bge-hybrid-pcreturn": "run_pcreturn",
        }
        judge_runs = {
            "para-bge-hybrid-flat": "judge_flat",
            "para-bge-hybrid-linkexp": "judge_linkexp",
            "parentchild-bge-hybrid-pcreturn": "judge_pcreturn",
        }
        return drawn, configs, qmap, answers_by_config, gen_runs, judge_runs

    def test_150_rows(self):
        drawn, configs, qmap, abc, gr, jr = self._setup()
        sheet, key = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)
        assert len(sheet) == 150  # 50 ids × 3 configs
        assert len(key) == 150

    def test_50_ids_times_3(self):
        drawn, configs, qmap, abc, gr, jr = self._setup()
        sheet, key = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)
        # Each drawn id should appear exactly 3 times in the key
        id_counts = {}
        for k in key:
            id_counts[k["question_id"]] = id_counts.get(k["question_id"], 0) + 1
        assert all(v == 3 for v in id_counts.values())
        assert len(id_counts) == 50

    def test_sheet_has_no_forbidden_columns(self):
        drawn, configs, qmap, abc, gr, jr = self._setup()
        sheet, key = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)
        for row in sheet:
            assert not FORBIDDEN_SHEET_COLUMNS.intersection(row.keys()), \
                f"Found forbidden columns: {FORBIDDEN_SHEET_COLUMNS & row.keys()}"

    def test_sheet_columns_match_spec(self):
        drawn, configs, qmap, abc, gr, jr = self._setup()
        sheet, key = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)
        assert list(sheet[0].keys()) == SHEET_COLUMNS

    def test_key_columns_match_spec(self):
        drawn, configs, qmap, abc, gr, jr = self._setup()
        sheet, key = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)
        assert list(key[0].keys()) == KEY_COLUMNS

    def test_sheet_row_sequential(self):
        drawn, configs, qmap, abc, gr, jr = self._setup()
        sheet, key = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)
        assert [r["sheet_row"] for r in sheet] == list(range(1, 151))
        assert [r["sheet_row"] for r in key] == list(range(1, 151))

    def test_key_round_trip(self):
        """Key file maps sheet_row → (question_id, config); round-trippable."""
        drawn, configs, qmap, abc, gr, jr = self._setup()
        sheet, key = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)
        key_map = {k["sheet_row"]: k for k in key}
        for s in sheet:
            k = key_map[s["sheet_row"]]
            # The key should have a question_id and config
            assert k["question_id"]
            assert k["config"]
            # The sheet question should match the question for that id
            q = qmap[k["question_id"]]
            assert s["question"] == q["question"]

    def test_empty_hand_columns(self):
        drawn, configs, qmap, abc, gr, jr = self._setup()
        sheet, _ = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)
        for r in sheet:
            assert r["faithfulness_hand"] == ""
            assert r["accuracy_hand"] == ""
            assert r["copies_or_answers"] == ""
            assert r["notes"] == ""

    def test_shuffled_order(self):
        """Rows are shuffled, not in question_id × config order."""
        drawn, configs, qmap, abc, gr, jr = self._setup()
        _, key = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)
        # Check that the sequence of (qid, config) is not sorted
        pairs = [(k["question_id"], k["config"]) for k in key]
        assert pairs != sorted(pairs), "Rows should be shuffled"

    def test_deterministic_shuffle(self):
        drawn, configs, qmap, abc, gr, jr = self._setup()
        _, key1 = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)
        _, key2 = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)
        pairs1 = [(k["question_id"], k["config"]) for k in key1]
        pairs2 = [(k["question_id"], k["config"]) for k in key2]
        assert pairs1 == pairs2


# ── CSV round-trip ───────────────────────────────────────────────────

class TestCSVRoundTrip:
    def test_write_read_sheet(self, tmp_path):
        drawn, configs, qmap, abc, gr, jr = TestBuildSheetAndKey()._setup()
        sheet, key = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)

        sheet_path = tmp_path / "sheet.csv"
        with sheet_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
            w.writeheader()
            w.writerows(sheet)

        with sheet_path.open(encoding="utf-8", newline="") as f:
            read_back = list(csv.DictReader(f))

        assert len(read_back) == 150
        assert list(read_back[0].keys()) == SHEET_COLUMNS

    def test_write_read_key(self, tmp_path):
        drawn, configs, qmap, abc, gr, jr = TestBuildSheetAndKey()._setup()
        sheet, key = build_sheet_and_key(
            drawn, configs, qmap, abc, gr, jr, SEED)

        key_path = tmp_path / "key.csv"
        with key_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=KEY_COLUMNS)
            w.writeheader()
            w.writerows(key)

        with key_path.open(encoding="utf-8", newline="") as f:
            read_back = list(csv.DictReader(f))

        assert len(read_back) == 150
        assert list(read_back[0].keys()) == KEY_COLUMNS
