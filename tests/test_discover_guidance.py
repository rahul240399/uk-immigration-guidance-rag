"""
Tests for code.fetch.discover_guidance.

Uses tests/fixtures/routes.yaml with two routes sharing one path
and one cross-cutting entry.
"""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from code.fetch.discover_guidance import load_routes, validate_and_enrich, write_plan

FIXTURES = Path(__file__).parent / "fixtures"
ROUTES_YAML = str(FIXTURES / "routes.yaml")


def _fake_publication(bp: str, doc_type: str = "guidance",
                      has_html: bool = True, has_pdf: bool = False):
    """Build a minimal publication API record."""
    attachments = []
    if has_html:
        attachments.append({"attachment_type": "html", "url": bp + "/html"})
    if has_pdf:
        attachments.append({
            "attachment_type": "file",
            "content_type": "application/pdf",
            "url": bp + "/doc.pdf",
        })
    return {
        "base_path": bp,
        "title": bp.split("/")[-1].replace("-", " ").title(),
        "document_type": doc_type,
        "public_updated_at": "2025-06-01T00:00:00+00:00",
        "details": {"attachments": attachments},
    }


class TestLoadRoutes:

    def test_dedup_join(self):
        """shared-guidance appears once with routes joined by ';'."""
        entries = load_routes(ROUTES_YAML)
        by_path = {e["base_path"]: e["routes"] for e in entries}
        assert by_path["/government/publications/shared-guidance"] == "skilled_worker;student"

    def test_counts_without_optional(self):
        """Without --include-optional: 3 unique paths (sw, student, shared, cross-cutting)."""
        entries = load_routes(ROUTES_YAML, include_optional=False)
        assert len(entries) == 4  # sw, shared, student, cross-cutting

    def test_counts_with_optional(self):
        """With --include-optional: adds one more."""
        entries = load_routes(ROUTES_YAML, include_optional=True)
        assert len(entries) == 5

    def test_cross_cutting_tagged(self):
        entries = load_routes(ROUTES_YAML)
        by_path = {e["base_path"]: e["routes"] for e in entries}
        assert by_path["/government/publications/cross-cutting-one"] == "cross_cutting"


class TestValidateAndEnrich:

    def _mock_get_json(self, bp):
        return _fake_publication(bp)

    def test_csv_columns(self, tmp_path):
        """Plan CSV has the expected columns."""
        entries = [
            {"base_path": "/government/publications/test-one", "routes": "r1"},
        ]
        with patch("code.fetch.discover_guidance.get_json",
                    side_effect=lambda bp: _fake_publication(bp)):
            rows = validate_and_enrich(entries)

        plan = write_plan(rows, tmp_path)
        with open(plan, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {
                "route", "base_path", "title", "document_type",
                "public_updated_at", "has_html_attachment", "has_pdf_attachment",
            }

    def test_non_guidance_type_fails(self):
        """A document_type != 'guidance' causes sys.exit(1)."""
        entries = [
            {"base_path": "/government/publications/bad-type", "routes": "r1"},
        ]
        with patch("code.fetch.discover_guidance.get_json",
                    return_value=_fake_publication(
                        "/government/publications/bad-type",
                        doc_type="detailed_guide")):
            with pytest.raises(SystemExit) as exc_info:
                validate_and_enrich(entries)
            assert exc_info.value.code == 1

    def test_404_fails(self):
        """A path returning None (404) causes sys.exit(1)."""
        entries = [
            {"base_path": "/government/publications/missing", "routes": "r1"},
        ]
        with patch("code.fetch.discover_guidance.get_json", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                validate_and_enrich(entries)
            assert exc_info.value.code == 1

    def test_attachment_flags(self, tmp_path):
        """has_html and has_pdf flags are set correctly."""
        entries = [
            {"base_path": "/government/publications/with-both", "routes": "r1"},
        ]
        with patch("code.fetch.discover_guidance.get_json",
                    return_value=_fake_publication(
                        "/government/publications/with-both",
                        has_html=True, has_pdf=True)):
            rows = validate_and_enrich(entries)

        assert rows[0]["has_html_attachment"] is True
        assert rows[0]["has_pdf_attachment"] is True
