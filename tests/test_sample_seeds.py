"""
Tests for code.evalset.sample_seeds — synthetic fixtures, no data files.
"""

import csv
import json
import random
from pathlib import Path

import pytest
import yaml


def _write_corpus(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_links(path: Path, links: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "from_paragraph_id", "target", "target_type", "source",
            "status", "resolved_to", "candidates"])
        w.writeheader()
        w.writerows(links)


def _make_record(pid, section, text_type="rule", text="A rule with more than fifteen words to pass the minimum word filter easily", 
                 confidence="high", rule_ref=None, attached_to=None):
    return {
        "paragraph_id": pid,
        "seq": int(pid.split(":")[-1]),
        "rule_ref": rule_ref or pid.split(":")[-1],
        "rule_family": "appendix",
        "section_base_path": section,
        "section_title": "Test",
        "heading_path": [],
        "text": text,
        "text_type": text_type,
        "attached_to": attached_to,
        "links_out": [],
        "content_id": "x",
        "public_updated_at": "2025-01-01",
        "snapshot_date": "2025-01-01",
        "licence": "OGL-3.0",
        "raw_html": "",
        "raw_html_sha256": "",
        "source_line": None,
        "source_pos": None,
        "parse_confidence": confidence,
    }


SEC_A = "/guidance/immigration-rules/sec-a"
SEC_B = "/guidance/immigration-rules/sec-b"
SEC_CC = "/guidance/immigration-rules/sec-cc"


@pytest.fixture
def setup(tmp_path):
    """Build minimal corpus, links, evalset, routes, paths."""
    processed = tmp_path / "processed"
    processed.mkdir()
    evalset_dir = tmp_path / "evalset"
    evalset_dir.mkdir()
    runs = tmp_path / "runs"

    # 10 rules in sec-a, 5 in sec-b
    records = []
    for i in range(1, 11):
        records.append(_make_record(f"a:{i:05d}", SEC_A, rule_ref=f"A{i}"))
        # Add 2 children for first 5 rules
        if i <= 5:
            records.append(_make_record(f"a:{100+i*2:05d}", SEC_A, text_type="subparagraph",
                                        attached_to=f"a:{i:05d}", rule_ref=f"A{i}(a)"))
            records.append(_make_record(f"a:{100+i*2+1:05d}", SEC_A, text_type="subparagraph",
                                        attached_to=f"a:{i:05d}", rule_ref=f"A{i}(b)"))
    for i in range(1, 6):
        records.append(_make_record(f"b:{i:05d}", SEC_B, rule_ref=f"B{i}"))
    # Add a deleted and low-confidence record
    records.append(_make_record(f"a:00099", SEC_A, text_type="deleted"))
    records.append(_make_record(f"a:00098", SEC_A, confidence="low"))
    # Cross-cutting
    records.append(_make_record(f"cc:00001", SEC_CC, rule_ref="CC1"))

    corpus_path = processed / "rules-paragraphs_v1.jsonl"
    _write_corpus(corpus_path, records)

    # Links: a:00001 -> b:00001 (cross-section)
    links = [
        {"from_paragraph_id": "a:00001", "target": "B1", "target_type": "paragraph",
         "source": "regex", "status": "resolved", "resolved_to": "b:00001", "candidates": ""},
        {"from_paragraph_id": "a:00002", "target": "B2", "target_type": "paragraph",
         "source": "regex", "status": "resolved", "resolved_to": "b:00002", "candidates": ""},
    ]
    links_path = processed / "rules-crossrefs_v2.csv"
    _write_links(links_path, links)

    evalset_yaml = {
        "seed": 42,
        "corpus": "rules-paragraphs_v1.jsonl",
        "links": "rules-crossrefs_v2.csv",
        "exclude_text_types": ["deleted", "list_item", "table_row"],
        "exclude_confidence": ["low"],
        "min_words": 5,
        "counts": {"route_a": {"T1": 3, "T2": 2, "T3": 2}},
        "tiers": {},
        "llm": {"model": "test", "options": {}},
        "embeddings": {"model": "test"},
    }
    evalset_path = tmp_path / "evalset.yaml"
    evalset_path.write_text(yaml.dump(evalset_yaml))

    routes_yaml = {
        "routes": {
            "route_a": {"rules_sections": [SEC_A, SEC_B]},
        },
        "cross_cutting": {"rules_sections": [SEC_CC]},
    }
    routes_path = tmp_path / "routes.yaml"
    routes_path.write_text(yaml.dump(routes_yaml))

    paths_yaml = {
        "raw_rules": str(tmp_path / "raw"),
        "raw_guidance": str(tmp_path / "raw-g"),
        "interim": str(tmp_path / "interim"),
        "processed": str(processed),
        "runs": str(runs),
        "index": str(tmp_path / "index"),
        "evalset": str(evalset_dir),
        "results": str(tmp_path / "results"),
        "logs": str(tmp_path / "logs"),
        "manifests": {"rules": str(corpus_path), "guidance": str(corpus_path)},
    }
    paths_path = tmp_path / "paths.yaml"
    paths_path.write_text(yaml.dump(paths_yaml))

    return {
        "tmp_path": tmp_path,
        "records": records,
        "corpus_path": corpus_path,
        "links_path": links_path,
        "evalset_path": evalset_path,
        "routes_path": routes_path,
        "paths_path": paths_path,
        "evalset_dir": evalset_dir,
    }


def _run_sampler(setup):
    """Run the sampler in-process and return the CSV rows."""
    import sys
    from unittest.mock import patch

    sys.argv = [
        "sample_seeds",
        "--config", str(setup["paths_path"]),
        "--evalset", str(setup["evalset_path"]),
        "--routes", str(setup["routes_path"]),
    ]
    from code.evalset.sample_seeds import main
    main()

    csv_files = list(setup["evalset_dir"].glob("*_seeds_v0.csv"))
    assert len(csv_files) == 1
    with open(csv_files[0], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestDeterminism:
    def test_same_seed_same_output(self, setup):
        rows1 = _run_sampler(setup)
        # Clean output for second run
        for p in setup["evalset_dir"].iterdir():
            p.unlink()
        rows2 = _run_sampler(setup)
        assert rows1 == rows2


class TestGoldIds:
    def test_gold_ids_exist_and_not_excluded(self, setup):
        rows = _run_sampler(setup)
        records_by_id = {}
        with open(setup["corpus_path"], encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                records_by_id[r["paragraph_id"]] = r

        for row in rows:
            for pid in row["gold_paragraph_ids"].split(";"):
                assert pid in records_by_id, f"Gold id {pid} not in corpus"
                r = records_by_id[pid]
                assert r["text_type"] not in ("deleted", "list_item", "table_row")
                assert r["parse_confidence"] != "low"


class TestStratumCounts:
    def test_counts_match_config(self, setup):
        rows = _run_sampler(setup)
        from collections import Counter
        c = Counter((r["route"], r["tier"]) for r in rows)
        # Config asks for T1=3, T2=2, T3=2 for route_a
        assert c[("route_a", "T1")] == 3
        assert c[("route_a", "T2")] == 2
        assert c[("route_a", "T3")] == 2


class TestAbsentSection:
    def test_absent_section_aborts(self, setup):
        import yaml as _yaml
        routes = _yaml.safe_load(setup["routes_path"].read_text())
        routes["routes"]["route_a"]["rules_sections"].append("/nonexistent")
        setup["routes_path"].write_text(_yaml.dump(routes))

        with pytest.raises(SystemExit):
            _run_sampler(setup)


class TestExcludedTypes:
    def test_no_excluded_types_in_gold(self, setup):
        rows = _run_sampler(setup)
        records_by_id = {}
        with open(setup["corpus_path"], encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                records_by_id[r["paragraph_id"]] = r

        for row in rows:
            for pid in row["gold_paragraph_ids"].split(";"):
                assert records_by_id[pid]["text_type"] not in ("deleted", "list_item", "table_row")


class TestT3SpansSections:
    def test_t3_two_sections(self, setup):
        rows = _run_sampler(setup)
        for row in rows:
            if row["tier"] == "T3":
                sections = row["sections"].split(";")
                assert len(sections) == 2, f"T3 should span 2 sections, got {sections}"


class TestT2GroupSize:
    def test_t2_two_to_six(self, setup):
        rows = _run_sampler(setup)
        for row in rows:
            if row["tier"] == "T2":
                ids = row["gold_paragraph_ids"].split(";")
                assert 2 <= len(ids) <= 6, f"T2 group size {len(ids)} not in [2,6]"
