"""
Tests for the immigration rules parser.

Fixtures:
  - synthetic-section.json: exercises three identifier families, hyperlinks,
    nested items, and table rows
  - immigration-rules-appendix-a-attributes.json: real Appendix A (OGL)
  - ecaa-settlement.json: real ECAA Settlement appendix (OGL)
"""

import json
import pytest
from pathlib import Path

from code.parse.parse_rules import RulesParser

FIXTURES = Path(__file__).parent / "fixtures"
TEST_CONFIG = str(FIXTURES / "paths.yaml")


@pytest.fixture
def parser():
    """Parser wired to the test fixtures directory."""
    p = RulesParser(config_path=TEST_CONFIG)
    return p


# ── Synthetic section ────────────────────────────────────────────────

class TestSyntheticSection:
    """Parse synthetic-section.json and verify identifier families."""

    @pytest.fixture(autouse=True)
    def _parse(self, parser):
        path = FIXTURES / "synthetic-section.json"
        self.records = parser.parse_section(path)
        parser.records = self.records  # for post-parse if needed

    def test_appendix_family_matched(self):
        """SYN 1.1, SYN 1.2, SYN 1.3 are appendix-family."""
        appendix_rules = [r for r in self.records
                          if r.rule_family == 'appendix' and r.text_type in ('rule', 'deleted')]
        assert len(appendix_rules) >= 3, f"Expected ≥3 appendix rules, got {len(appendix_rules)}"

    def test_part_family_matched(self):
        """Numeric rules 1, 2 are part-family."""
        part_rules = [r for r in self.records
                      if r.rule_family == 'part' and r.text_type in ('rule', 'deleted')]
        assert len(part_rules) >= 2, f"Expected ≥2 part rules, got {len(part_rules)}"

    def test_dotted_family_matched(self):
        """Dotted rules 1.1, 1.2 are dotted-family."""
        dotted_rules = [r for r in self.records
                        if r.rule_family == 'dotted' and r.text_type in ('rule', 'deleted')]
        assert len(dotted_rules) >= 2, f"Expected ≥2 dotted rules, got {len(dotted_rules)}"

    def test_hyperlink_captured(self):
        """The <a href> to Appendix A is captured in links_out."""
        all_links = [lk for r in self.records for lk in r.links_out]
        hyperlinks = [lk for lk in all_links if lk.get('source') == 'hyperlink'
                      or 'appendix-a' in lk.get('target', '')]
        assert len(hyperlinks) >= 1, "No hyperlink to Appendix A found"

    def test_nested_items_separate(self):
        """Subparagraphs (a), (b), (b)(i) are separate records."""
        subparas = [r for r in self.records if r.text_type == 'subparagraph']
        refs = {r.rule_ref for r in subparas}
        assert 'SYN 1.3(a)' in refs or 'SYN 1.2(a)' in refs, f"Missing subpara (a), got {refs}"
        assert 'SYN 1.3(b)' in refs or 'SYN 1.2(b)' in refs, f"Missing subpara (b), got {refs}"
        assert 'SYN 1.3(b)(i)' in refs or 'SYN 1.2(b)(i)' in refs, f"Missing compound (b)(i), got {refs}"

    def test_one_record_per_table_row(self):
        """Table has 2 data rows → 2 table_row records."""
        table_rows = [r for r in self.records if r.text_type == 'table_row']
        assert len(table_rows) == 2, f"Expected 2 table rows, got {len(table_rows)}"


# ── Appendix A ───────────────────────────────────────────────────────

class TestAppendixA:
    """Parse the real Appendix A and check structural counts."""

    @pytest.fixture(autouse=True)
    def _parse(self, parser):
        path = FIXTURES / "immigration-rules-appendix-a-attributes.json"
        self.records = parser.parse_section(path)

    def test_top_level_identifiers(self):
        """210 top-level identifiers."""
        rules = [r for r in self.records
                 if r.text_type in ('rule', 'deleted') and r.rule_ref and not r.attached_to]
        assert len(rules) == 210, f"Expected 210 identifiers, got {len(rules)}"

    def test_deleted_count(self):
        """171 records are deleted."""
        deleted = [r for r in self.records if r.text_type == 'deleted']
        assert len(deleted) == 171, f"Expected 171 deleted, got {len(deleted)}"

    def test_table_5_rows(self):
        """Table 5 has four data rows with points 20, 20, 15, 20."""
        table5_rows = [r for r in self.records
                       if r.text_type == 'table_row' and 'Table 5' in ' '.join(r.heading_path)]
        assert len(table5_rows) == 4, f"Expected 4 Table 5 rows, got {len(table5_rows)}"
        # Points are in the text as "Points: <N>"
        points = []
        for row in table5_rows:
            # Extract last number after "Points:" or the last cell value
            import re
            m = re.findall(r'Points:\s*(\d+)', row.text)
            if m:
                points.append(int(m[-1]))
        assert points == [20, 20, 15, 20], f"Expected [20, 20, 15, 20], got {points}"

    def test_no_child_text_inside_parent(self):
        """No record with text ≥40 chars has its text inside its attached_to parent."""
        text_by_id = {r.paragraph_id: r.text for r in self.records}
        for r in self.records:
            if len(r.text.strip()) >= 40 and r.attached_to:
                parent_text = text_by_id.get(r.attached_to, "")
                assert r.text.strip() not in parent_text, (
                    f"Child text inside parent: {r.paragraph_id}")


# ── ECAA Settlement ──────────────────────────────────────────────────

class TestECAA:
    """Parse the real ECAA Settlement appendix."""

    @pytest.fixture(autouse=True)
    def _parse(self, parser):
        path = FIXTURES / "-immigration-rules-appendix-ecaa-settlement-ecaa-nationals-and-settlement.json"
        self.records = parser.parse_section(path)

    def test_identifier_count(self):
        """22 rule/deleted records with identifiers."""
        id_records = [r for r in self.records
                      if r.text_type in ('rule', 'deleted') and r.rule_ref and not r.attached_to]
        assert len(id_records) == 22, f"Expected 22 identifiers, got {len(id_records)}"

    def test_ecaa_compound_present(self):
        """ECAA 7.1(b)(i) is present as a subparagraph."""
        all_refs = {r.rule_ref for r in self.records if r.rule_ref}
        assert 'ECAA 7.1(b)(i)' in all_refs, f"Missing ECAA 7.1(b)(i), got {all_refs}"

    def test_deleted_count(self):
        """3 deleted records: ECAA 1.1, bare DELETED, ECAA 7.1(g)."""
        deleted = [r for r in self.records if r.text_type == 'deleted']
        assert len(deleted) == 3, f"Expected 3 deleted, got {len(deleted)}"
