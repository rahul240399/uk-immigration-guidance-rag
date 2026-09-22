"""Tests for tools/rate_sheet.py: parse and write-back on a 3-row fixture."""
import csv
from pathlib import Path

import pytest

# Import from tools — add to path if needed
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.rate_sheet import (
    COLUMNS,
    apply_rating,
    is_rated,
    load_sheet,
    rated_count,
    save_sheet,
)


# ── fixture ──────────────────────────────────────────────────────────

def _write_fixture(tmp_path: Path) -> Path:
    """Write a 3-row sheet: row 1 rated, rows 2-3 unrated."""
    p = tmp_path / "sheet.csv"
    rows = [
        {"sheet_row": "1", "question": "Q1?", "context": "ctx1",
         "answer": "A1", "reference_answer": "R1",
         "faithfulness_hand": "N", "accuracy_hand": "1",
         "copies_or_answers": "answer", "notes": "ok"},
        {"sheet_row": "2", "question": "Q2?", "context": "ctx2",
         "answer": "A2", "reference_answer": "R2",
         "faithfulness_hand": "", "accuracy_hand": "",
         "copies_or_answers": "", "notes": ""},
        {"sheet_row": "3", "question": "Q3?", "context": "ctx3",
         "answer": "A3", "reference_answer": "R3",
         "faithfulness_hand": "", "accuracy_hand": "",
         "copies_or_answers": "", "notes": ""},
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return p


# ── load_sheet ───────────────────────────────────────────────────────

class TestLoadSheet:
    def test_loads_3_rows(self, tmp_path):
        p = _write_fixture(tmp_path)
        rows = load_sheet(p)
        assert len(rows) == 3

    def test_columns_match(self, tmp_path):
        p = _write_fixture(tmp_path)
        rows = load_sheet(p)
        assert list(rows[0].keys()) == COLUMNS


# ── is_rated ─────────────────────────────────────────────────────────

class TestIsRated:
    def test_rated(self):
        r = {"faithfulness_hand": "Y", "accuracy_hand": "1",
             "copies_or_answers": "copy"}
        assert is_rated(r) is True

    def test_unrated(self):
        r = {"faithfulness_hand": "", "accuracy_hand": "",
             "copies_or_answers": ""}
        assert is_rated(r) is False

    def test_partial(self):
        r = {"faithfulness_hand": "Y", "accuracy_hand": "",
             "copies_or_answers": ""}
        assert is_rated(r) is False


# ── rated_count ──────────────────────────────────────────────────────

class TestRatedCount:
    def test_fixture(self, tmp_path):
        rows = load_sheet(_write_fixture(tmp_path))
        assert rated_count(rows) == 1


# ── apply_rating ─────────────────────────────────────────────────────

class TestApplyRating:
    def test_fills_fields(self):
        row = {"sheet_row": "2", "question": "Q?", "context": "c",
               "answer": "A", "reference_answer": "R",
               "faithfulness_hand": "", "accuracy_hand": "",
               "copies_or_answers": "", "notes": ""}
        rated = apply_rating(row, "Y", "0.5", "copy", "some note")
        assert rated["faithfulness_hand"] == "Y"
        assert rated["accuracy_hand"] == "0.5"
        assert rated["copies_or_answers"] == "copy"
        assert rated["notes"] == "some note"

    def test_does_not_mutate_original(self):
        row = {"faithfulness_hand": "", "accuracy_hand": "",
               "copies_or_answers": "", "notes": ""}
        apply_rating(row, "N", "1", "answer", "")
        assert row["faithfulness_hand"] == ""


# ── save_sheet + round trip ──────────────────────────────────────────

class TestSaveSheet:
    def test_write_back(self, tmp_path):
        p = _write_fixture(tmp_path)
        rows = load_sheet(p)

        # Rate row 2
        rows[1] = apply_rating(rows[1], "Y", "0.5", "copy", "test note")
        save_sheet(p, rows)

        # Reload
        reloaded = load_sheet(p)
        assert len(reloaded) == 3
        assert reloaded[1]["faithfulness_hand"] == "Y"
        assert reloaded[1]["accuracy_hand"] == "0.5"
        assert reloaded[1]["copies_or_answers"] == "copy"
        assert reloaded[1]["notes"] == "test note"
        # Row 1 untouched
        assert reloaded[0]["faithfulness_hand"] == "N"
        # Row 3 still unrated
        assert reloaded[2]["faithfulness_hand"] == ""

    def test_atomic_no_tmp_left(self, tmp_path):
        p = _write_fixture(tmp_path)
        rows = load_sheet(p)
        save_sheet(p, rows)
        assert not (p.with_suffix(".tmp")).exists()

    def test_resume_after_partial(self, tmp_path):
        """Rate row 2, save, reload, rate row 3, verify all 3 rated."""
        p = _write_fixture(tmp_path)
        rows = load_sheet(p)

        rows[1] = apply_rating(rows[1], "N", "1", "answer", "")
        save_sheet(p, rows)

        rows2 = load_sheet(p)
        assert rated_count(rows2) == 2  # row 1 + row 2

        rows2[2] = apply_rating(rows2[2], "Y", "0", "copy", "bad")
        save_sheet(p, rows2)

        final = load_sheet(p)
        assert rated_count(final) == 3
        assert final[2]["accuracy_hand"] == "0"
