"""
Tests for code.common.link_resolution.

Synthetic fixtures only — no data files, no network, no git.
"""

import pytest

from code.common.link_resolution import (
    resolve_section_name,
    rebuild_section_links,
)


SECTION_PATHS = {
    "/guidance/immigration-rules/immigration-rules-part-8-family-members",
    "/guidance/immigration-rules/immigration-rules-part-5-working-in-the-uk",
    "/guidance/immigration-rules/immigration-rules-appendix-v-visitor",
}

NAME_INDEX = {
    "family members": "/guidance/immigration-rules/immigration-rules-part-8-family-members",
}

ALIASES = {
    "appendix": {
        "v": "/guidance/immigration-rules/immigration-rules-appendix-v-visitor",
        "visitor": "/guidance/immigration-rules/immigration-rules-appendix-v-visitor",
    },
    "part": {
        "8": "/guidance/immigration-rules/immigration-rules-part-8-family-members",
        "5": "/guidance/immigration-rules/immigration-rules-part-5-working-in-the-uk",
    },
    "absent": {
        "appendix": ["w", "f"],
        "part": [],
    },
}


class TestResolveSectionName:

    def test_t1_part_8_resolves(self):
        """'Part 8' -> resolved to the part-8 path."""
        status, rt = resolve_section_name("8", "part", NAME_INDEX, ALIASES, SECTION_PATHS)
        assert status == "resolved"
        assert rt == "/guidance/immigration-rules/immigration-rules-part-8-family-members"

    def test_t2_appendix_w_absent(self):
        """'Appendix W' -> absent."""
        status, rt = resolve_section_name("W", "appendix", NAME_INDEX, ALIASES, SECTION_PATHS)
        assert status == "absent"
        assert rt is None

    def test_t3_part_9_unresolved(self):
        """'Part 9' -> unresolved (not in any block)."""
        status, rt = resolve_section_name("9", "part", NAME_INDEX, ALIASES, SECTION_PATHS)
        assert status == "unresolved"
        assert rt is None

    def test_t4_appendix_8_unresolved(self):
        """'Appendix 8' -> unresolved (8 is a part alias, not appendix)."""
        status, rt = resolve_section_name("8", "appendix", NAME_INDEX, ALIASES, SECTION_PATHS)
        assert status == "unresolved"
        assert rt is None


class TestRebuildSectionLinks:

    def test_t5_two_part_mentions_plus_paragraph(self):
        """Rebuild on a record with two Part mentions gives two part links
        in text order and leaves its paragraph link untouched."""
        record = {
            "text": "See Part 8 and Part 5 and paragraph 320 for details.",
            "text_type": "narrative",
            "links_out": [
                {"target": "320", "target_type": "paragraph", "source": "regex",
                 "resolved": True, "resolved_to": "some:00042"},
                {"target": "FM", "target_type": "appendix", "source": "regex",
                 "resolved": True, "resolved_to": "/old/path"},
            ],
        }
        new_links = rebuild_section_links(record)

        # Paragraph link kept
        para_links = [lk for lk in new_links if lk["target_type"] == "paragraph"]
        assert len(para_links) == 1
        assert para_links[0]["resolved_to"] == "some:00042"

        # Two part links in text order
        part_links = [lk for lk in new_links if lk["target_type"] == "part"]
        assert len(part_links) == 2
        assert part_links[0]["target"] == "8"
        assert part_links[1]["target"] == "5"

        # Old appendix link dropped
        app_links = [lk for lk in new_links if lk["target_type"] == "appendix"]
        assert len(app_links) == 0

    def test_t6_table_row_gets_no_links(self):
        """A table_row record gets no links."""
        record = {
            "text": "See Part 8 and Appendix V",
            "text_type": "table_row",
            "links_out": [
                {"target": "8", "target_type": "part", "source": "regex",
                 "resolved": True},
            ],
        }
        new_links = rebuild_section_links(record)
        assert new_links == []

    def test_t7_unresolved_never_gets_resolved_to(self):
        """An unresolved name never gets resolved_to."""
        record = {
            "text": "See Part 9 for details.",
            "text_type": "narrative",
            "links_out": [],
        }
        new_links = rebuild_section_links(record)
        part_links = [lk for lk in new_links if lk["target_type"] == "part"]
        assert len(part_links) == 1
        assert part_links[0]["resolved"] is False
        assert "resolved_to" not in part_links[0]


# ── t8: hyperlink to section with no identifiers resolves ────────────

class TestHyperlinkNoIdentifiers:

    def test_t8_hyperlink_resolves_to_section_without_ids(self):
        """A section with no identifiers (e.g. salary list) still resolves hyperlinks."""
        from code.common.link_resolution import resolve_links

        section_base_paths = {
            "/guidance/immigration-rules/immigration-rules-appendix-immigration-salary-list",
            "/guidance/immigration-rules/immigration-rules-part-8-family-members",
        }
        records = [{
            "paragraph_id": "test:00001",
            "section_base_path": "/guidance/immigration-rules/immigration-rules-part-8-family-members",
            "links_out": [{
                "target": "/guidance/immigration-rules/immigration-rules-appendix-immigration-salary-list",
                "target_type": "hyperlink",
                "source": "hyperlink",
                "resolved": False,
            }],
        }]
        resolve_links(records, {}, {"appendix": {}, "part": {}, "absent": {"appendix": [], "part": []}},
                       section_base_paths)
        lk = records[0]["links_out"][0]
        assert lk["resolved"] is True
        assert lk["resolved_to"] == "/guidance/immigration-rules/immigration-rules-appendix-immigration-salary-list"


# ── t9: "FM- SE" normalises to "FM-SE" ──────────────────────────────

class TestHyphenNormalisation:

    def test_t9_fm_dash_se_resolves(self):
        """'Appendix FM- SE' resolves to the FM-SE section via hyphen collapse."""
        name_index = {
            "fmse": "/guidance/immigration-rules/immigration-rules-appendix-fm-se-family-members-specified-evidence",
        }
        status, rt = resolve_section_name(
            "FM- SE", "appendix", name_index, ALIASES, SECTION_PATHS)
        assert status == "resolved"
        assert "fm-se" in rt


# ── t10: duplicate identifier keys resolve to lowest seq ─────────────

class TestDuplicateIdentifierKeys:

    def test_t10_lowest_seq_wins(self):
        """When two records share a (section, ref), lowest seq is kept."""
        from code.common.link_resolution import resolve_links

        records = [
            {"paragraph_id": "s:00010", "seq": 10, "rule_ref": "X1",
             "text_type": "rule", "section_base_path": "/sec",
             "links_out": []},
            {"paragraph_id": "s:00020", "seq": 20, "rule_ref": "X1",
             "text_type": "deleted", "section_base_path": "/sec",
             "links_out": []},
            {"paragraph_id": "s:00030", "seq": 30, "rule_ref": None,
             "text_type": "narrative", "section_base_path": "/sec",
             "links_out": [{"target": "X1", "target_type": "paragraph",
                            "source": "regex", "resolved": False}]},
        ]
        resolve_links(records, {}, {"appendix": {}, "part": {}, "absent": {"appendix": [], "part": []}},
                       {"/sec"})
        lk = records[2]["links_out"][0]
        assert lk["resolved"] is True
        assert lk["resolved_to"] == "s:00010"  # lowest seq
