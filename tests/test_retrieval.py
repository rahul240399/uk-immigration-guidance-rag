"""
Tests for retrieval grid — synthetic fixtures only.
"""

import json

import pytest
from collections import defaultdict

from code.eval.run_grid import rrf, apply_architecture
from code.eval.score import recall_at_k, mrr, ndcg_at_k, budget_recall


class TestRRF:
    def test_rrf_ordering(self):
        """RRF puts the item ranked high in both lists first."""
        list_a = [("c1", 10), ("c2", 8), ("c3", 5)]
        list_b = [("c1", 9), ("c3", 7), ("c2", 3)]
        result = rrf([list_a, list_b], k=60)
        ids = [cid for cid, _ in result]
        assert ids[0] == "c1"  # Top in both


class TestRecallMRR:
    def test_unit_rank_semantics(self):
        """A unit at rank 2 carrying 10 pids including one gold: R@10=1, MRR=0.5."""
        units = [
            {"paragraph_ids": ["x1", "x2", "x3"], "token_count": 50},
            {"paragraph_ids": ["x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "g1"],
             "token_count": 50},
        ]
        gold = {"g1"}
        assert recall_at_k(units, gold, 10) == 1.0
        assert mrr(units, gold) == pytest.approx(0.5)

    def test_recall_union(self):
        """Recall = union of paragraph_ids over first k units ∩ gold / |gold|."""
        units = [
            {"paragraph_ids": ["g1", "x1"], "token_count": 50},
            {"paragraph_ids": ["g2", "x2"], "token_count": 50},
        ]
        gold = {"g1", "g2"}
        assert recall_at_k(units, gold, 1) == pytest.approx(0.5)
        assert recall_at_k(units, gold, 2) == pytest.approx(1.0)

    def test_ndcg_new_hits(self):
        """nDCG gain at rank r = gold ids not covered by earlier units."""
        units = [
            {"paragraph_ids": ["g1"], "token_count": 50},
            {"paragraph_ids": ["g1", "g2"], "token_count": 50},  # g1 already covered
        ]
        gold = {"g1", "g2"}
        val = ndcg_at_k(units, gold, 10)
        # rank 1: gain=1 (g1), rank 2: gain=1 (g2 new)
        # dcg = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
        # idcg = 1/log2(2) + 1/log2(3) = 1.631
        assert val == pytest.approx(1.0, abs=0.01)


class TestBudgetRecall:
    def test_budget_truncation(self):
        """Budget recall stops when tokens exceed budget."""
        units = [
            {"paragraph_ids": ["g1"], "token_count": 300},
            {"paragraph_ids": ["g2"], "token_count": 300},
            {"paragraph_ids": ["g3"], "token_count": 300},
        ]
        gold = {"g1", "g2", "g3"}
        assert budget_recall(units, gold, 640) == pytest.approx(2 / 3)


class TestPCReturn:
    def test_child_always_in_set(self):
        """Child is always included in pcreturn carried set."""
        chunks_by_id = {
            "c1": {"chunk_id": "c1", "paragraph_ids": ["c1"], "token_count": 50,
                   "parent_id": "p1"},
        }
        parents = {
            "p1": {"parent_id": "p1", "unit_ids": ["c1", "c2", "c3"],
                   "section_base_path": "/sec", "heading": "H", "text": "...",
                   "token_count": 150},
        }
        units_by_id = {
            "c1": {"unit_id": "c1", "token_count": 40, "unit_token_count": 50},
            "c2": {"unit_id": "c2", "token_count": 40, "unit_token_count": 50},
            "c3": {"unit_id": "c3", "token_count": 40, "unit_token_count": 50},
        }
        ranked = [("c1", 1.0)]
        result = apply_architecture(ranked, "pcreturn", chunks_by_id, parents,
                                     None, units_by_id, 20)
        assert "c1" in result[0]["paragraph_ids"]

    def test_large_parent_caps_at_512(self):
        """A parent of 900 tokens with child last: caps within 512."""
        # 9 siblings, each 100 record tokens, heading = 10 tokens
        sibs = [f"s{i}" for i in range(9)]
        chunks_by_id = {sid: {"chunk_id": sid, "paragraph_ids": [sid],
                              "token_count": 110, "parent_id": "p1"} for sid in sibs}
        parents = {
            "p1": {"parent_id": "p1", "unit_ids": sibs,
                   "section_base_path": "/sec", "heading": "H", "text": "...",
                   "token_count": 900},
        }
        units_by_id = {sid: {"unit_id": sid, "token_count": 100,
                             "unit_token_count": 110} for sid in sibs}
        # Child is the last sibling (s8)
        ranked = [("s8", 1.0)]
        result = apply_architecture(ranked, "pcreturn", chunks_by_id, parents,
                                     None, units_by_id, 20)
        carried = result[0]["paragraph_ids"]
        tc = result[0]["token_count"]
        assert "s8" in carried
        assert tc <= 512
        # Should include child (100) + heading (10) = 110, plus ~4 more siblings
        assert len(carried) >= 2

    def test_small_parent_whole(self):
        """A parent under 512 tokens yields the whole parent."""
        sibs = ["c1", "c2", "c3"]
        chunks_by_id = {sid: {"chunk_id": sid, "paragraph_ids": [sid],
                              "token_count": 60, "parent_id": "p1"} for sid in sibs}
        parents = {
            "p1": {"parent_id": "p1", "unit_ids": sibs,
                   "section_base_path": "/sec", "heading": "H", "text": "...",
                   "token_count": 180},
        }
        units_by_id = {sid: {"unit_id": sid, "token_count": 50,
                             "unit_token_count": 60} for sid in sibs}
        ranked = [("c2", 1.0)]
        result = apply_architecture(ranked, "pcreturn", chunks_by_id, parents,
                                     None, units_by_id, 20)
        assert set(result[0]["paragraph_ids"]) == set(sibs)


class TestLinkexp:
    def test_appends_resolved_targets(self):
        """linkexp appends resolved paragraph link targets."""
        chunks_by_id = {
            "c1": {"chunk_id": "c1", "paragraph_ids": ["p1"], "token_count": 50},
        }
        links = {"p1": ["t1", "t2"]}
        units_by_id = {
            "t1": {"unit_id": "t1", "unit_token_count": 30, "token_count": 25},
            "t2": {"unit_id": "t2", "unit_token_count": 40, "token_count": 35},
        }
        ranked = [("c1", 1.0)]
        result = apply_architecture(ranked, "linkexp", chunks_by_id, None,
                                     links, units_by_id, 20)
        assert set(result[0]["appended_targets"]) == {"t1", "t2"}
        assert result[0]["appended_tokens"] == 70


class TestQuestionOrderAssertion:
    def test_permuted_fails(self):
        """Different question order gives different sha256."""
        from code.eval.run_grid import _sha256_str
        sha_a = _sha256_str(";".join(["Q1", "Q2", "Q3"]))
        sha_b = _sha256_str(";".join(["Q2", "Q1", "Q3"]))
        assert sha_a != sha_b


class TestLinkexpRecallGEFlat:
    def test_linkexp_recall_gte_flat(self):
        """linkexp recall >= flat: paragraph_ids includes appended_targets."""
        # Flat: unit carries ["p1"]
        flat_units = [
            {"paragraph_ids": ["p1"], "token_count": 50},
            {"paragraph_ids": ["p2"], "token_count": 50},
        ]
        # Linkexp: unit carries ["p1", "t1"] (t1 appended)
        link_units = [
            {"paragraph_ids": ["p1", "t1"], "token_count": 50},
            {"paragraph_ids": ["p2"], "token_count": 50},
        ]
        gold = {"p1", "t1"}

        flat_r = recall_at_k(flat_units, gold, 10)
        link_r = recall_at_k(link_units, gold, 10)

        assert link_r >= flat_r
        assert link_r == 1.0
        assert flat_r == 0.5


class TestPCReturnRecallGEFlat:
    def test_pcreturn_recall_gte_flat(self):
        """pcreturn recall >= flat with same ranking."""
        flat_units = [
            {"paragraph_ids": ["c1"], "token_count": 50},
            {"paragraph_ids": ["c2"], "token_count": 50},
        ]
        pc_units = [
            {"paragraph_ids": ["c1", "c3"], "token_count": 100},
            {"paragraph_ids": ["c2"], "token_count": 50},
        ]
        gold = {"c1", "c3"}

        flat_r = recall_at_k(flat_units, gold, 10)
        pc_r = recall_at_k(pc_units, gold, 10)

        assert pc_r >= flat_r
        assert pc_r == 1.0
        assert flat_r == 0.5


class TestPCReturnRankingIdentical:
    def test_ranking_order_same_as_flat(self):
        """pcreturn unit_ids and order must equal flat."""
        chunks_by_id = {
            "c1": {"chunk_id": "c1", "paragraph_ids": ["c1"], "token_count": 60,
                   "parent_id": "p1"},
            "c2": {"chunk_id": "c2", "paragraph_ids": ["c2"], "token_count": 60,
                   "parent_id": "p1"},
        }
        parents = {
            "p1": {"parent_id": "p1", "unit_ids": ["c1", "c2"],
                   "section_base_path": "/sec", "heading": "H", "text": "...",
                   "token_count": 120},
        }
        units_by_id = {
            "c1": {"unit_id": "c1", "token_count": 50, "unit_token_count": 60},
            "c2": {"unit_id": "c2", "token_count": 50, "unit_token_count": 60},
        }
        ranked = [("c1", 1.0), ("c2", 0.5)]

        flat_result = apply_architecture(ranked, "flat", chunks_by_id, None, None, None, 20)
        pc_result = apply_architecture(ranked, "pcreturn", chunks_by_id, parents, None, units_by_id, 20)

        flat_ids = [r["unit_id"] for r in flat_result]
        pc_ids = [r["unit_id"] for r in pc_result]
        assert flat_ids == pc_ids


class TestScorerHandBuilt:
    def test_hand_built_retrieved_jsonl(self, tmp_path):
        """Hand-built retrieved.jsonl scores under D33 unit-rank semantics."""
        from code.eval.score import score_run
        # Q1: gold={g1,g2}; unit1 carries [g1], unit2 carries [x1], unit3 carries [g2]
        #   R@5 = 2/2 = 1.0 (both found in first 3 units)
        #   MRR = 1/1 = 1.0 (unit 1 carries g1)
        # Q2: gold={g3}; unit1 carries [x1], unit2 carries [x2], unit3 carries [g3]
        #   R@5 = 1/1 = 1.0
        #   MRR = 1/3 (unit 3 is first with g3)
        retrievals = [
            {"question_id": "Q1", "retrieved": [
                {"paragraph_ids": ["g1"], "token_count": 50, "score": 1.0, "unit_id": "u1"},
                {"paragraph_ids": ["x1"], "token_count": 50, "score": 0.9, "unit_id": "u2"},
                {"paragraph_ids": ["g2"], "token_count": 50, "score": 0.8, "unit_id": "u3"},
            ]},
            {"question_id": "Q2", "retrieved": [
                {"paragraph_ids": ["x1"], "token_count": 100, "score": 1.0, "unit_id": "u4"},
                {"paragraph_ids": ["x2"], "token_count": 100, "score": 0.9, "unit_id": "u5"},
                {"paragraph_ids": ["g3"], "token_count": 100, "score": 0.8, "unit_id": "u6"},
            ]},
        ]
        gold = {"Q1": {"g1", "g2"}, "Q2": {"g3"}}
        per_q = score_run(retrievals, gold)

        assert per_q[0]["recall_10"] == pytest.approx(1.0)
        assert per_q[0]["mrr"] == pytest.approx(1.0)
        assert per_q[1]["recall_10"] == pytest.approx(1.0)
        assert per_q[1]["mrr"] == pytest.approx(1 / 3, abs=0.01)
        assert per_q[0]["budget_640"] == pytest.approx(1.0)
        assert per_q[1]["budget_640"] == pytest.approx(1.0)

    def test_budget_truncation_scorer(self):
        from code.eval.score import score_run
        retrievals = [{"question_id": "Q1", "retrieved": [
            {"paragraph_ids": ["g1"], "token_count": 400, "score": 1.0, "unit_id": "u1"},
            {"paragraph_ids": ["g2"], "token_count": 400, "score": 0.9, "unit_id": "u2"},
        ]}]
        gold = {"Q1": {"g1", "g2"}}
        per_q = score_run(retrievals, gold)
        assert per_q[0]["budget_640"] == pytest.approx(0.5)

    def test_unit_rank2_ten_pids(self):
        """Unit at rank 2 carrying 10 pids including one gold: R@10=1, MRR=0.5."""
        from code.eval.score import score_run
        retrievals = [{"question_id": "Q1", "retrieved": [
            {"paragraph_ids": ["x" + str(i) for i in range(10)], "token_count": 50, "unit_id": "u1"},
            {"paragraph_ids": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "g1"],
             "token_count": 50, "unit_id": "u2"},
        ]}]
        gold = {"Q1": {"g1"}}
        per_q = score_run(retrievals, gold)
        assert per_q[0]["recall_10"] == pytest.approx(1.0)
        assert per_q[0]["mrr"] == pytest.approx(0.5)


class TestVerify:
    def test_verify_passes(self, tmp_path):
        """--verify passes when results.json matches retrieved.jsonl."""
        from code.eval.score import score_run, verify

        retrievals = [{"question_id": "Q1", "retrieved": [
            {"paragraph_ids": ["g1"], "token_count": 50, "score": 1.0, "unit_id": "u1"},
        ]}]
        gold = {"Q1": {"g1"}}
        per_q = score_run(retrievals, gold)

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with open(run_dir / "retrieved.jsonl", "w") as f:
            for r in retrievals:
                f.write(json.dumps(r) + "\n")
        (run_dir / "results.json").write_text(json.dumps({
            "summary": {}, "per_question": per_q}))

        assert verify(run_dir, gold) is True

    def test_verify_fails_on_edit(self, tmp_path):
        """--verify fails when results.json is tampered."""
        from code.eval.score import score_run, verify

        retrievals = [{"question_id": "Q1", "retrieved": [
            {"paragraph_ids": ["g1"], "token_count": 50, "score": 1.0, "unit_id": "u1"},
        ]}]
        gold = {"Q1": {"g1"}}
        per_q = score_run(retrievals, gold)

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with open(run_dir / "retrieved.jsonl", "w") as f:
            for r in retrievals:
                f.write(json.dumps(r) + "\n")
        # Tamper: set recall_10 to 0.0
        per_q[0]["recall_10"] = 0.0
        (run_dir / "results.json").write_text(json.dumps({
            "summary": {}, "per_question": per_q}))

        assert verify(run_dir, gold) is False
