"""Tests for make_results.py (T7) with synthetic fixtures."""
import json
from pathlib import Path

import numpy as np
import pytest

from code.eval.make_results import (
    BASELINE,
    METRICS,
    SEED,
    bootstrap_ci,
    build_t1,
    build_t2,
    holm_correction,
    median_diff_ci,
    paired_wilcoxon,
    parse_config_name,
)


# ── synthetic per-question data ──────────────────────────────────────

def _make_results(configs=None, n_questions=10):
    """Build a synthetic all_results dict."""
    if configs is None:
        configs = [
            "para-bge-bm25-flat-dryrun",
            "para-bge-dense-flat-dryrun",
            "para-bge-hybrid-linkexp-dryrun",
            "parentchild-bge-hybrid-pcreturn-dryrun",
        ]
    rng = np.random.default_rng(42)
    results = {}
    for ci, name in enumerate(configs):
        pq = []
        for qi in range(n_questions):
            row = {"question_id": f"Q{qi:04d}"}
            for m in METRICS:
                row[m] = round(float(rng.uniform(0.1, 1.0)), 4)
            # Make later configs slightly better for deterministic ordering
            for m in METRICS:
                row[m] = round(row[m] + ci * 0.05, 4)
            pq.append(row)
        results[name] = {"per_question": pq, "summary": {}}
    return results


# ── bootstrap ────────────────────────────────────────────────────────

class TestBootstrap:
    def test_deterministic(self):
        vals = [0.3, 0.5, 0.7, 0.9, 0.4]
        m1, lo1, hi1 = bootstrap_ci(vals, seed=SEED)
        m2, lo2, hi2 = bootstrap_ci(vals, seed=SEED)
        assert m1 == m2
        assert lo1 == lo2
        assert hi1 == hi2

    def test_different_seed(self):
        vals = [0.3, 0.5, 0.7, 0.9, 0.4]
        _, lo1, _ = bootstrap_ci(vals, seed=SEED)
        _, lo2, _ = bootstrap_ci(vals, seed=99999)
        # Very unlikely to be identical
        assert lo1 != lo2

    def test_ci_contains_mean(self):
        vals = [0.3, 0.5, 0.7, 0.9, 0.4]
        m, lo, hi = bootstrap_ci(vals)
        assert lo <= m <= hi

    def test_single_value(self):
        m, lo, hi = bootstrap_ci([0.5])
        assert m == 0.5
        assert lo == hi == 0.5


# ── Wilcoxon ─────────────────────────────────────────────────────────

class TestWilcoxon:
    def test_identical_gives_p1(self):
        a = [0.5, 0.6, 0.7, 0.8]
        stat, p, n, tied = paired_wilcoxon(a, a)
        assert p == 1.0
        assert tied == 4

    def test_different_gives_low_p(self):
        a = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        b = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        stat, p, n, tied = paired_wilcoxon(a, b)
        assert p < 0.05
        assert tied == 0

    def test_n_correct(self):
        a = [0.5, 0.6, 0.7]
        b = [0.4, 0.5, 0.6]
        _, _, n, _ = paired_wilcoxon(a, b)
        assert n == 3


# ── Holm correction ──────────────────────────────────────────────────

class TestHolm:
    def test_ordering_preserved(self):
        """Holm-adjusted p-values maintain the rank order of raw p-values."""
        raw = [0.01, 0.04, 0.03, 0.10]
        adj = holm_correction(raw)
        # Sorted raw: 0.01, 0.03, 0.04, 0.10
        # The smallest raw should get the smallest adjusted
        raw_order = sorted(range(len(raw)), key=lambda i: raw[i])
        adj_ordered = [adj[i] for i in raw_order]
        assert adj_ordered == sorted(adj_ordered)

    def test_hand_computed(self):
        """Hand-computed Holm on 3 p-values.

        raw = [0.04, 0.01, 0.05]
        sorted: (1, 0.01), (0, 0.04), (2, 0.05)
        step 0: 0.01 * 3 = 0.03
        step 1: 0.04 * 2 = 0.08, cummax = max(0.03, 0.08) = 0.08
        step 2: 0.05 * 1 = 0.05, cummax = max(0.08, 0.05) = 0.08
        adjusted[1] = 0.03, adjusted[0] = 0.08, adjusted[2] = 0.08
        """
        raw = [0.04, 0.01, 0.05]
        adj = holm_correction(raw)
        assert adj[1] == 0.03
        assert adj[0] == 0.08
        assert adj[2] == 0.08

    def test_empty(self):
        assert holm_correction([]) == []

    def test_single(self):
        assert holm_correction([0.05]) == [0.05]

    def test_caps_at_1(self):
        raw = [0.9, 0.8]
        adj = holm_correction(raw)
        assert all(a <= 1.0 for a in adj)


# ── marginal means (t2) ─────────────────────────────────────────────

class TestMarginalMeans:
    def test_factors_present(self):
        results = _make_results()
        t2 = build_t2(results)
        factors = {r["factor"] for r in t2}
        assert factors == {"chunking", "retrieval", "architecture"}

    def test_levels_from_configs(self):
        results = _make_results()
        t2 = build_t2(results)
        chunking_levels = {r["level"] for r in t2 if r["factor"] == "chunking"}
        assert "para" in chunking_levels
        assert "parentchild" in chunking_levels


# ── table shapes ─────────────────────────────────────────────────────

class TestTableShapes:
    def test_t1_row_per_config(self):
        results = _make_results()
        t1 = build_t1(results)
        assert len(t1) == 4  # 4 configs

    def test_t1_has_ci_columns(self):
        results = _make_results()
        t1 = build_t1(results)
        for m in METRICS:
            assert f"{m}_mean" in t1[0]
            assert f"{m}_ci_lo" in t1[0]
            assert f"{m}_ci_hi" in t1[0]

    def test_t2_has_primary_metric(self):
        results = _make_results()
        t2 = build_t2(results)
        for r in t2:
            assert "primary_metric" in r


# ── t3 uniqueness and Holm count ─────────────────────────────────────

class TestT3:
    def test_unique_config_metric_vs(self):
        from code.eval.make_results import build_t3
        results = _make_results(configs=[
            "para-bge-bm25-flat-dryrun",
            "para-bge-dense-flat-dryrun",
            "para-bge-hybrid-linkexp-dryrun",
            "parentchild-bge-hybrid-pcreturn-dryrun",
        ])
        t3 = build_t3(results)
        triples = [(r["config"], r["metric"], r["vs"]) for r in t3]
        assert len(triples) == len(set(triples))

    def test_holm_count_equals_rows(self):
        from code.eval.make_results import build_t3
        results = _make_results(configs=[
            "para-bge-bm25-flat-dryrun",
            "para-bge-dense-flat-dryrun",
            "para-bge-hybrid-linkexp-dryrun",
        ])
        t3 = build_t3(results)
        assert all("p_holm" in r for r in t3)
        # Holm applied to exactly len(t3) tests
        holm_count = sum(1 for r in t3 if "p_holm" in r)
        assert holm_count == len(t3)

    def test_has_mean_diff_columns(self):
        from code.eval.make_results import build_t3
        results = _make_results(configs=[
            "para-bge-bm25-flat-dryrun",
            "para-bge-dense-flat-dryrun",
        ])
        t3 = build_t3(results)
        if t3:
            assert "mean_diff" in t3[0]
            assert "mean_diff_ci_lo" in t3[0]
            assert "mean_diff_ci_hi" in t3[0]
            assert "n_nonzero" in t3[0]


# ── parse_config_name ────────────────────────────────────────────────

class TestParseConfig:
    def test_flat(self):
        p = parse_config_name("para-bge-bm25-flat-dryrun")
        assert p == {"chunking": "para", "retrieval": "bm25",
                     "architecture": "flat"}

    def test_linkexp(self):
        p = parse_config_name("para-bge-hybrid-linkexp-dryrun")
        assert p == {"chunking": "para", "retrieval": "hybrid",
                     "architecture": "linkexp"}

    def test_pcreturn(self):
        p = parse_config_name("parentchild-bge-dense-pcreturn")
        assert p == {"chunking": "parentchild", "retrieval": "dense",
                     "architecture": "pcreturn"}


# ── missing run folder ──────────────────────────────────────────────

class TestMissingRunFolder:
    def test_missing_results_json(self, tmp_path):
        """main() returns 1 when a run folder lacks results.json."""
        from code.eval.make_results import main
        # Write a grid manifest pointing to a nonexistent folder
        manifest = {"run_folders": {"test-config": str(tmp_path / "nope")}}
        mp = tmp_path / "manifest.json"
        mp.write_text(json.dumps(manifest))
        # Write minimal evalset and covariates
        ep = tmp_path / "evalset.csv"
        ep.write_text("question_id,tier\nQ0001,T1\n")
        cp = tmp_path / "covariates.csv"
        cp.write_text("question_id\nQ0001\n")
        # Write paths.yaml
        pp = tmp_path / "paths.yaml"
        paths_data = {
            "raw_rules": str(tmp_path), "raw_guidance": str(tmp_path),
            "interim": str(tmp_path), "processed": str(tmp_path),
            "runs": str(tmp_path / "runs"), "index": str(tmp_path),
            "evalset": str(tmp_path), "results": str(tmp_path / "results"),
            "logs": str(tmp_path), "manifests": {
                "rules": str(tmp_path / "manifest_v1.json"),
                "guidance": str(tmp_path / "guidance-manifest_v1.json"),
            },
        }
        pp.write_text(json.dumps(paths_data))
        (tmp_path / "runs").mkdir()
        (tmp_path / "results").mkdir()

        ret = main(["--config", str(pp), "--grid-manifest", str(mp),
                     "--evalset", str(ep), "--covariates", str(cp),
                     "--out", str(tmp_path / "out")])
        assert ret == 1
