"""
Run registry — tracks every pipeline and experiment run.

Usage (dry-run CLI)::

    python -m code.common.run_registry --config config/paths.yaml --dry-run
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ── Paths ────────────────────────────────────────────────────────────

_REQUIRED_KEYS = [
    "raw_rules", "raw_guidance", "interim", "processed",
    "runs", "index", "evalset", "results", "logs", "manifests",
]

_REQUIRED_MANIFEST_KEYS = ["rules", "guidance"]


def load_paths(config_path: str) -> dict:
    """Load paths.yaml; raise KeyError for any missing required key."""
    with open(config_path) as f:
        paths = yaml.safe_load(f) or {}
    for key in _REQUIRED_KEYS:
        if key not in paths:
            raise KeyError(key)
    manifests = paths.get("manifests")
    if not isinstance(manifests, dict):
        raise KeyError("manifests must be a mapping with keys: rules, guidance")
    for mk in _REQUIRED_MANIFEST_KEYS:
        if mk not in manifests:
            raise KeyError(f"manifests.{mk}")
    return paths


# ── Git info (reads .git files directly) ─────────────────────────────

def _read_git_info(repo_root: Path) -> tuple[str, str]:
    """Read commit hash and ref from .git/HEAD by reading files directly."""
    git_dir = repo_root / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.exists():
        return "unknown", "unknown"

    head = head_file.read_text().strip()
    if head.startswith("ref: "):
        ref_name = head[5:]
        ref_path = git_dir / ref_name
        if ref_path.exists():
            commit = ref_path.read_text().strip()
        else:
            packed = git_dir / "packed-refs"
            commit = "unknown"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(ref_name):
                        commit = line.split()[0]
                        break
        return commit, ref_name
    else:
        return head, "HEAD"


# ── Code tree hash ───────────────────────────────────────────────────

def _code_tree_sha256(repo_root: Path) -> str:
    """SHA256 over sorted relative paths+contents of code/**/*.py and config/*.yaml,
    excluding config/paths.yaml."""
    h = hashlib.sha256()
    entries: list[tuple[str, bytes]] = []

    for pattern in ["code/**/*.py", "config/*.yaml"]:
        for p in sorted(repo_root.glob(pattern)):
            rel = str(p.relative_to(repo_root))
            if rel == "config/paths.yaml":
                continue
            entries.append((rel, p.read_bytes()))

    for rel, content in sorted(entries):
        h.update(rel.encode())
        h.update(content)

    return h.hexdigest()


# ── Package versions ─────────────────────────────────────────────────

def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


_TRACKED_PACKAGES = [
    "numpy", "pandas", "scipy", "rank_bm25",
    "sentence-transformers", "torch", "ragas",
]


def _package_versions() -> dict:
    return {n: _pkg_version(n) for n in _TRACKED_PACKAGES}


# ── Manifest reading ────────────────────────────────────────────────

def _read_manifest_entry(path_str: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        return {"path": path_str, "sha256": None, "snapshot_date": None,
                "date_frozen": None, "parser_version": None, "total_records": None}
    raw = p.read_bytes()
    data = json.loads(raw)
    return {
        "path": path_str,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "snapshot_date": data.get("snapshot_date"),
        "date_frozen": data.get("date_frozen"),
        "parser_version": data.get("parser_version"),
        "total_records": data.get("total_records"),
    }


# ── RunContext ───────────────────────────────────────────────────────

REGISTRY_COLUMNS = [
    "run_id", "stage", "config_name", "status", "start_time", "end_time",
    "headline", "git_commit", "code_tree_sha256",
    "rules_manifest_sha256", "guidance_manifest_sha256",
    "evalset_sha256", "run_dir",
]


class RunContext:
    """Context manager-style object for a single run."""

    def __init__(self, run_dir: Path, run_id: str, stage: str,
                 config_name: str, config_data: dict,
                 registry_csv: Path):
        self.run_dir = run_dir
        self.run_id = run_id
        self.stage = stage
        self.config_name = config_name
        self.config_data = config_data
        self.start_time = config_data["start_time"]
        self._registry_csv = registry_csv
        self._log_fh = open(run_dir / "run.log", "a", encoding="utf-8")

    def log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._log_fh.write(f"{ts}  {msg}\n")
        self._log_fh.flush()

    def finish(self, status: str, headline: str,
               metrics: Optional[dict] = None) -> None:
        end_time = datetime.now(timezone.utc).isoformat()

        result = {"end_time": end_time, "status": status,
                  "headline": headline, "metrics": metrics}
        (self.run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        self._log_fh.close()

        # Read sha256 from the named corpus keys — never select by filename
        corpus = self.config_data.get("corpus", {})
        rules_sha = (corpus.get("rules") or {}).get("sha256") or ""
        guidance_sha = (corpus.get("guidance") or {}).get("sha256") or ""

        evalset = self.config_data.get("evalset") or {}
        evalset_sha = evalset.get("sha256", "")

        row = {
            "run_id": self.run_id,
            "stage": self.stage,
            "config_name": self.config_name,
            "status": status,
            "start_time": self.start_time,
            "end_time": end_time,
            "headline": headline,
            "git_commit": self.config_data.get("code", {}).get("git_commit", ""),
            "code_tree_sha256": self.config_data.get("code", {}).get("code_tree_sha256", ""),
            "rules_manifest_sha256": rules_sha,
            "guidance_manifest_sha256": guidance_sha,
            "evalset_sha256": evalset_sha,
            "run_dir": str(self.run_dir),
        }

        write_header = not self._registry_csv.exists()
        with open(self._registry_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ── start_run ────────────────────────────────────────────────────────

def start_run(
    stage: str,
    config_name: str,
    params: dict,
    paths: dict,
    evalset_path: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> RunContext:
    """Create a timestamped run folder and return a RunContext."""
    if repo_root is None:
        repo_root = Path(".")

    runs_dir = Path(paths["runs"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    base_name = f"{stamp}_{stage}_{config_name}"
    run_dir = runs_dir / base_name

    if run_dir.exists():
        counter = 2
        while (runs_dir / f"{base_name}_{counter}").exists():
            counter += 1
        run_dir = runs_dir / f"{base_name}_{counter}"

    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name

    git_commit, git_ref = _read_git_info(repo_root)
    tree_sha = _code_tree_sha256(repo_root)

    # Corpus manifests — mapping {rules: path, guidance: path}
    manifest_map = paths.get("manifests", {})
    corpus = {
        key: _read_manifest_entry(mpath)
        for key, mpath in manifest_map.items()
    }

    evalset_entry = None
    if evalset_path:
        ep = Path(evalset_path)
        if ep.exists():
            evalset_entry = {
                "path": evalset_path,
                "sha256": hashlib.sha256(ep.read_bytes()).hexdigest(),
            }

    start_time = datetime.now(timezone.utc).isoformat()

    config_data = {
        "run_id": run_id,
        "stage": stage,
        "config_name": config_name,
        "params": params,
        "start_time": start_time,
        "code": {
            "git_commit": git_commit,
            "git_ref": git_ref,
            "code_tree_sha256": tree_sha,
        },
        "corpus": corpus,
        "evalset": evalset_entry,
        "python_version": platform.python_version(),
        "packages": _package_versions(),
    }

    (run_dir / "config.json").write_text(
        json.dumps(config_data, ensure_ascii=False, indent=2), encoding="utf-8")

    registry_csv = runs_dir / "registry.csv"
    ctx = RunContext(run_dir, run_id, stage, config_name,
                     config_data, registry_csv)
    return ctx


# ── index_dir ────────────────────────────────────────────────────────

def index_dir(config_name: str, paths: dict, rebuild: bool = False) -> Path:
    """Return the index directory for a config, creating it if needed."""
    idx_root = Path(paths["index"])
    idx_root.mkdir(parents=True, exist_ok=True)
    target = idx_root / config_name

    if target.exists() and (target / "build.log").exists():
        if not rebuild:
            raise FileExistsError("index exists; pass --rebuild")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        renamed = idx_root / f"{config_name}_replaced_{stamp}"
        target.rename(renamed)

    target.mkdir(parents=True, exist_ok=True)
    return target


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run registry dry-run")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = load_paths(args.config)

    if args.dry_run:
        ctx = start_run("dryrun", "registry-test",
                        {"note": "dry run"}, paths)
        ctx.log("dry run executed")
        ctx.finish("ok", "dry run")

        config = json.loads((ctx.run_dir / "config.json").read_text())

        registry_path = Path(paths["runs"]) / "registry.csv"
        with open(registry_path) as f:
            reader = csv.reader(f)
            next(reader)
            row_count = sum(1 for _ in reader)

        print(f"run folder: {ctx.run_dir.name}")
        print(f"registry rows (excl header): {row_count}")
        print(f"config.json keys: {sorted(config.keys())}")
        corpus = config.get("corpus", {})
        for key, entry in corpus.items():
            print(f"corpus.{key}: sha256={entry['sha256']}")


if __name__ == "__main__":
    main()
