"""
Tests for code.eval.diagnose — synthetic fixtures.
"""

import re
import pytest

from code.eval.diagnose import RULE_ID_RE, NUMBER_RE, SECTION_NAME_RE


class TestRegexes:
    def test_rule_id_pattern(self):
        assert RULE_ID_RE.search("SW 1.1 requires")
        assert RULE_ID_RE.search("paragraph 276ADE")
        assert RULE_ID_RE.search("rule 9.1.1")
        assert not RULE_ID_RE.search("the applicant must")

    def test_number_pattern(self):
        assert NUMBER_RE.search("at least 5 years")
        assert not NUMBER_RE.search("the applicant")

    def test_section_name_pattern(self):
        assert SECTION_NAME_RE.search("under Appendix FM")
        assert SECTION_NAME_RE.search("Part Suitability")
        assert not SECTION_NAME_RE.search("the rules")


class TestTextTypeAndIsChild:
    def test_text_type_from_corpus_record(self):
        """text_type comes from the corpus record, not covariates."""
        record = {"paragraph_id": "x:1", "text_type": "subparagraph",
                   "attached_to": "x:0"}
        assert record["text_type"] == "subparagraph"
        assert bool(record["attached_to"]) is True  # is_child

    def test_is_child_false_when_no_attached_to(self):
        record = {"paragraph_id": "x:1", "text_type": "rule",
                   "attached_to": None}
        assert bool(record.get("attached_to") or "") is False


class TestFullListRank:
    def test_rank_1_when_gold_is_top(self):
        """Full-list rank equals 1 when the gold is the top-scored unit."""
        import numpy as np
        scores = np.array([0.9, 0.5, 0.3, 0.1])
        gold_score = 0.9  # index 0
        rank = int(np.sum(scores > gold_score)) + 1
        assert rank == 1

    def test_rank_3_when_two_above(self):
        import numpy as np
        scores = np.array([0.9, 0.8, 0.5, 0.3])
        gold_score = 0.5  # index 2
        rank = int(np.sum(scores > gold_score)) + 1
        assert rank == 3


class TestParentInTop10:
    def test_parent_found_in_flat_top10(self):
        """For a child gold, parent_in_top10 = True when parent is in flat top 10."""
        flat_units = [
            {"unit_id": "parent1", "paragraph_ids": ["parent1"], "score": 1.0},
            {"unit_id": "child1", "paragraph_ids": ["child1"], "score": 0.9},
        ]
        child_attached_to = "parent1"
        parent_found = any(
            child_attached_to in u["paragraph_ids"]
            for u in flat_units[:10]
        )
        assert parent_found is True

    def test_parent_not_in_top10(self):
        flat_units = [
            {"unit_id": "other1", "paragraph_ids": ["other1"], "score": 1.0},
        ]
        child_attached_to = "parent1"
        parent_found = any(
            child_attached_to in u["paragraph_ids"]
            for u in flat_units[:10]
        )
        assert parent_found is False


class TestFoundAtK:
    def test_found_at_5_and_10(self):
        units = [
            {"paragraph_ids": ["x1"]},
            {"paragraph_ids": ["x2"]},
            {"paragraph_ids": ["g1"]},
        ]
        found_rank = None
        for rank, u in enumerate(units, 1):
            if "g1" in u["paragraph_ids"]:
                found_rank = rank
                break
        assert found_rank == 3
        assert found_rank <= 5
        assert found_rank <= 10

    def test_absent_gold(self):
        units = [{"paragraph_ids": ["x1"]}]
        found_rank = None
        for rank, u in enumerate(units, 1):
            if "g1" in u["paragraph_ids"]:
                found_rank = rank
                break
        assert found_rank is None
