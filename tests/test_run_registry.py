"""
Tests for code.common.run_registry.

All tests use tmp_path — no network, no git, no real data.
"""

import csv
import json
from pathlib import Path

import pytest

from code.common.run_registry import (
    load_paths,
    start_run,
    index_dir,
    REGISTRY_COLUMNS,
)

MODULE_PATH = Path("code/common/run_registry.py")


def _make_paths(tmp_path: Path, manifests: dict | None = None) -> dict:
    """Build a minimal paths dict rooted in tmp_path."""
    if manifests is None:
        manifests = {
            "rules": str(tmp_path / "rules-manifest.json"),
            "guidance": str(tmp_path / "guidance-manifest.json"),
        }
    return {
        "raw_rules": str(tmp_path / "raw-rules"),
        "raw_guidance": str(tmp_path / "raw-guidance"),
        "interim": str(tmp_path / "interim"),
        "processed": str(tmp_path / "processed"),
        "runs": str(tmp_path / "runs"),
        "index": str(tmp_path / "index"),
        "evalset": str(tmp_path / "evalset"),
        "results": str(tmp_path / "results"),
        "logs": str(tmp_path / "logs"),
        "manifests": manifests,
    }


def _make_manifest(path: Path, **overrides) -> None:
    """Write a small manifest JSON at path."""
    data = {
        "snapshot_date": "2026-09-14T12:00:00Z",
        "date_frozen": "2026-09-14T13:00:00Z",
        "parser_version": "abc123",
        "total_records": 100,
    }
    data.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ── t1: start_run creates folder with config.json and run.log ────────

class TestStartRun:

    def test_creates_folder_with_config_and_log(self, tmp_path):
        paths = _make_paths(tmp_path)
        ctx = start_run("parse", "test-cfg", {"k": 1}, paths,
                        repo_root=tmp_path)
        assert ctx.run_dir.exists()
        assert (ctx.run_dir / "config.json").exists()
        assert (ctx.run_dir / "run.log").exists()
        ctx.finish("ok", "done")


# ── t2: two calls in same second → distinct folders ──────────────────

class TestDistinctFolders:

    def test_two_calls_same_second(self, tmp_path):
        paths = _make_paths(tmp_path)
        ctx1 = start_run("s", "c", {}, paths, repo_root=tmp_path)
        ctx2 = start_run("s", "c", {}, paths, repo_root=tmp_path)
        assert ctx1.run_dir != ctx2.run_dir
        assert ctx1.run_dir.exists()
        assert ctx2.run_dir.exists()
        ctx1.finish("ok", "a")
        ctx2.finish("ok", "b")


# ── t3: finish appends exactly one row with 13 columns ──────────────

class TestFinishRow:

    def test_one_row_13_columns(self, tmp_path):
        paths = _make_paths(tmp_path)
        ctx = start_run("eval", "cfg1", {}, paths, repo_root=tmp_path)
        ctx.log("hello")
        ctx.finish("ok", "test headline")

        registry = Path(paths["runs"]) / "registry.csv"
        assert registry.exists()

        with open(registry, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert set(rows[0].keys()) == set(REGISTRY_COLUMNS)
        assert len(REGISTRY_COLUMNS) == 13


# ── t4: index_dir raises / renames ───────────────────────────────────

class TestIndexDir:

    def test_raises_when_exists(self, tmp_path):
        paths = _make_paths(tmp_path)
        idx = index_dir("myindex", paths)
        (idx / "build.log").write_text("done")

        with pytest.raises(FileExistsError, match="index exists"):
            index_dir("myindex", paths, rebuild=False)

    def test_renames_on_rebuild(self, tmp_path):
        paths = _make_paths(tmp_path)
        idx = index_dir("myindex", paths)
        (idx / "build.log").write_text("done")

        new_idx = index_dir("myindex", paths, rebuild=True)
        assert new_idx.exists()
        assert new_idx.name == "myindex"

        idx_root = Path(paths["index"])
        replaced = [p for p in idx_root.iterdir()
                     if p.name.startswith("myindex_replaced_")]
        assert len(replaced) == 1


# ── t5: corpus entries carry manifest sha256 (mapping) ───────────────

class TestCorpusEntries:

    def test_manifest_sha256_mapping(self, tmp_path):
        m_rules = tmp_path / "rules-m.json"
        m_guidance = tmp_path / "guidance-m.json"
        _make_manifest(m_rules, total_records=500)
        _make_manifest(m_guidance, total_records=200)

        paths = _make_paths(tmp_path, manifests={
            "rules": str(m_rules),
            "guidance": str(m_guidance),
        })
        ctx = start_run("test", "cfg", {}, paths, repo_root=tmp_path)

        config = json.loads((ctx.run_dir / "config.json").read_text())
        corpus = config["corpus"]

        assert isinstance(corpus, dict)
        assert "rules" in corpus
        assert "guidance" in corpus
        assert corpus["rules"]["sha256"] is not None
        assert len(corpus["rules"]["sha256"]) == 64
        assert corpus["guidance"]["sha256"] is not None
        assert len(corpus["guidance"]["sha256"]) == 64
        assert corpus["rules"]["total_records"] == 500
        assert corpus["guidance"]["total_records"] == 200
        ctx.finish("ok", "done")


# ── t6: no forbidden calls in module source ──────────────────────────

class TestNoSubprocess:

    def test_source_clean(self):
        source = MODULE_PATH.read_text()
        assert "subprocess" not in source
        assert "os.system" not in source


# ── t7: registry row sha256 values match config.json corpus entries ──

class TestRegistryManifestSha:

    def test_registry_sha_matches_config(self, tmp_path):
        m_rules = tmp_path / "rules-m.json"
        m_guidance = tmp_path / "guidance-m.json"
        _make_manifest(m_rules, total_records=15822)
        _make_manifest(m_guidance, total_records=9099)

        paths = _make_paths(tmp_path, manifests={
            "rules": str(m_rules),
            "guidance": str(m_guidance),
        })
        ctx = start_run("eval", "cfg", {}, paths, repo_root=tmp_path)
        ctx.finish("ok", "verified")

        # Read config.json corpus
        config = json.loads((ctx.run_dir / "config.json").read_text())
        rules_sha = config["corpus"]["rules"]["sha256"]
        guidance_sha = config["corpus"]["guidance"]["sha256"]

        # Read registry row
        registry = Path(paths["runs"]) / "registry.csv"
        with open(registry, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        row = rows[0]
        assert row["rules_manifest_sha256"] == rules_sha
        assert row["guidance_manifest_sha256"] == guidance_sha
        assert row["rules_manifest_sha256"] != ""
        assert row["guidance_manifest_sha256"] != ""
