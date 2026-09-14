"""
Tests for code.fetch.discover_rules.

Uses tests/fixtures/manual.json — a trimmed copy of the real manual
record with three child sections across two groups.
"""

import json
import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from code.fetch.discover_rules import discover, write_plan

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def manual_data():
    with open(FIXTURES / "manual.json") as f:
        return json.load(f)


class TestDiscover:

    def test_plan_lists_three_base_paths(self, manual_data, tmp_path):
        """The trimmed manual has 3 child sections; the plan CSV lists all three."""
        with patch("code.fetch.discover_rules.get_json", return_value=manual_data):
            sections = discover("/guidance/immigration-rules")

        assert len(sections) == 3

        plan_path = write_plan(sections, tmp_path)
        with open(plan_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3
        paths = [r["base_path"] for r in rows]
        assert "/guidance/immigration-rules/immigration-rules-part-1-leave-to-enter-or-stay-in-the-uk" in paths
        assert "/guidance/immigration-rules/immigration-rules-part-2-transitional-provisions" in paths
        assert "/guidance/immigration-rules/immigration-rules-appendix-a-attributes" in paths

    def test_titles_preserved(self, manual_data, tmp_path):
        """Titles from the manual record appear in the plan CSV."""
        with patch("code.fetch.discover_rules.get_json", return_value=manual_data):
            sections = discover()

        plan_path = write_plan(sections, tmp_path)
        with open(plan_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        titles = [r["title"] for r in rows]
        assert "Part 1: leave to enter or stay in the UK" in titles
        assert "Appendix A: attributes" in titles

    def test_empty_on_failure(self):
        """Returns empty list when the manual cannot be fetched."""
        with patch("code.fetch.discover_rules.get_json", return_value=None):
            sections = discover()
        assert sections == []
