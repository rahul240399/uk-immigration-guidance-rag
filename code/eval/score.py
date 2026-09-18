"""
Score retrieval runs (D33 unit-rank semantics).

k counts retrieved units, not flattened paragraph ids.

Usage::

    python -m code.eval.score --config config/paths.yaml --evalset <csv>
    python -m code.eval.score --verify <run-folder> --evalset <csv>
"""

import argparse
import csv
import json
import logging
import math
from datetime import date
from pathlib import Path

import numpy as np

from code.common.run_registry import load_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Metric functions (D33: k = unit rank) ────────────────────────────

def recall_at_k(units: list[dict], gold: set[str], k: int) -> float:
    """Union of paragraph_ids over the first k units ∩ gold / |gold|."""
    if not gold:
        return 0.0
    found = set()
    for u in units[:k]:
        for pid in u.get("paragraph_ids", []):
            if pid in gold:
                found.add(pid)
    return len(found) / len(gold)


def mrr(units: list[dict], gold: set[str]) -> float:
    """1 / rank of the first unit whose paragraph_ids contain a gold id."""
    for rank, u in enumerate(units, 1):
        if any(pid in gold for pid in u.get("paragraph_ids", [])):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(units: list[dict], gold: set[str], k: int) -> float:
    """Gain at rank r = gold ids in unit r not covered by earlier units."""
    if not gold:
        return 0.0
    covered = set()
    dcg = 0.0
    for rank, u in enumerate(units[:k], 1):
        new_hits = set()
        for pid in u.get("paragraph_ids", []):
            if pid in gold and pid not in covered:
                new_hits.add(pid)
                covered.add(pid)
        gain = len(new_hits)
        if gain > 0:
            dcg += gain / math.log2(rank + 1)
    ideal_total = len(gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_total + 1))
    return dcg / idcg if idcg > 0 else 0.0


def budget_recall(units: list[dict], gold: set[str], budget: int) -> float:
    """Recall within a token budget. Uses token_count from the unit."""
    if not gold:
        return 0.0
    tokens_used = 0
    found = set()
    for u in units:
        tc = u.get("token_count", 0)
        if tokens_used + tc > budget:
            break
        tokens_used += tc
        for pid in u.get("paragraph_ids", []):
            if pid in gold:
                found.add(pid)
    return len(found) / len(gold)


def bootstrap_ci(values, n_resamples=1000, ci=0.95, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.array(values)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
             for _ in range(n_resamples)]
    return float(np.percentile(means, (1 - ci) / 2 * 100)), \
           float(np.percentile(means, (1 + ci) / 2 * 100))


# ── Score one run ────────────────────────────────────────────────────

def score_run(retrievals: list[dict], qid_to_gold: dict) -> list[dict]:
    """Score a run. Pure function of retrieved.jsonl."""
    per_q = []
    for ret in retrievals:
        qid = ret["question_id"]
        gold = qid_to_gold.get(qid, set())
        units = ret.get("retrieved", [])
        per_q.append({
            "question_id": qid,
            "recall_5": recall_at_k(units, gold, 5),
            "recall_10": recall_at_k(units, gold, 10),
            "mrr": mrr(units, gold),
            "ndcg_10": ndcg_at_k(units, gold, 10),
            "budget_640": budget_recall(units, gold, 640),
            "gold": list(gold),
        })
    return per_q


# ── Verify (independent implementation from D33 sentence) ────────────

def verify(run_folder: Path, qid_to_gold: dict) -> bool:
    """Recompute Recall@10 independently and compare to results.json."""
    with open(run_folder / "retrieved.jsonl", encoding="utf-8") as f:
        retrievals = [json.loads(l) for l in f]
    with open(run_folder / "results.json") as f:
        saved = json.load(f)

    # Independent ten-line recomputation per D33:
    # Recall@k = |union of paragraph_ids over the first k units ∩ gold| / |gold|
    recomputed = []
    for ret in retrievals:
        gold = qid_to_gold.get(ret["question_id"], set())
        covered = set()
        for u in ret.get("retrieved", [])[:10]:
            covered.update(pid for pid in u.get("paragraph_ids", []) if pid in gold)
        recomputed.append(len(covered) / len(gold) if gold else 0.0)

    saved_values = [q["recall_10"] for q in saved["per_question"]]
    ok = True
    for i, (r, s) in enumerate(zip(recomputed, saved_values)):
        if abs(r - s) > 1e-6:
            log.error("  Q%d: recomputed=%.6f saved=%.6f", i, r, s)
            ok = False
    return ok


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Score retrieval runs (D33)")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--verify", default=None, metavar="RUN_FOLDER")
    args = ap.parse_args()

    with open(args.evalset, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    qid_to_gold = {q["question_id"]: set(q["gold_paragraph_ids"].split(";"))
                    for q in questions}

    if args.verify:
        ok = verify(Path(args.verify), qid_to_gold)
        print(f"VERIFY {'OK' if ok else 'FAIL'}: {Path(args.verify).name}")
        if not ok:
            exit(1)
        return

    paths = load_paths(args.config)
    runs_dir = Path(paths["runs"])
    results_dir = Path(paths["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    grid_runs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and "grid" in d.name])

    latest: dict[str, Path] = {}
    for run_dir in grid_runs:
        cfg_p = run_dir / "config.json"
        ret_p = run_dir / "retrieved.jsonl"
        if not cfg_p.exists() or not ret_p.exists():
            continue
        with open(cfg_p) as f:
            name = json.load(f).get("config_name", "")
        latest[name] = run_dir

    log.info("Scoring %d configs (latest per name)", len(latest))
    all_results = {}

    for config_name, run_dir in sorted(latest.items()):
        with open(run_dir / "retrieved.jsonl", encoding="utf-8") as f:
            retrievals = [json.loads(l) for l in f]

        per_q = score_run(retrievals, qid_to_gold)

        metrics = ["recall_5", "recall_10", "mrr", "ndcg_10", "budget_640"]
        summary = {"config_name": config_name, "n": len(per_q)}
        for m in metrics:
            vals = [q[m] for q in per_q]
            summary[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            lo, hi = bootstrap_ci(vals)
            summary[f"{m}_ci_lo"] = round(lo, 4)
            summary[f"{m}_ci_hi"] = round(hi, 4)

        (run_dir / "results.json").write_text(
            json.dumps({"summary": summary, "per_question": per_q}, indent=2))
        all_results[config_name] = {"summary": summary, "run_dir": str(run_dir)}

    today = date.today().isoformat()
    summary_path = results_dir / f"{today}_grid_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fields = ["config_name", "n"] + [f"{m}_{s}" for m in metrics
                  for s in ["mean", "ci_lo", "ci_hi"]]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name in sorted(all_results):
            writer.writerow(all_results[name]["summary"])

    manifest_path = results_dir / f"{today}_grid_manifest.json"
    manifest_path.write_text(json.dumps({
        "run_folders": {k: v["run_dir"] for k, v in all_results.items()},
        "evalset": str(args.evalset), "n_configs": len(all_results),
    }, indent=2))

    log.info("Summary: %s", summary_path)
    print(f"\n{'Config':<45s} {'n':>3s} {'R@10':>6s}")
    print("-" * 58)
    for name in sorted(all_results):
        s = all_results[name]["summary"]
        print(f"{name:<45s} {s['n']:>3d} {s['recall_10_mean']:>6.3f}")


if __name__ == "__main__":
    main()
