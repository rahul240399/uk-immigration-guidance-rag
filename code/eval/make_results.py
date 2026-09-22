"""
Produce dissertation results tables from v1 pipeline runs (T7).

Eight tables (each csv + markdown), a results manifest, and --freeze.

Usage::

    python -m code.eval.make_results --config config/paths.yaml \\
        --grid-manifest <json> --evalset <csv> --covariates <csv> \\
        [--judge <agreement json> --judgements <folders>] \\
        --out results/<date>_results_v1/
"""
import argparse, csv, hashlib, json, shutil, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import wilcoxon
import yaml

from code.common.run_registry import load_paths, start_run

SEED = 20260918
N_BOOT = 1000
BASELINE = "para-bge-bm25-flat"
METRICS = ["recall_5", "recall_10", "mrr", "ndcg_10", "budget_640"]

# D51 primary metric per factor
PRIMARY = {"chunking": "recall_10", "retrieval": "recall_10",
           "architecture": "budget_640"}

DRYRUN_NOTE = ("v1 run folders carry a '-dryrun' suffix from a labelling "
               "defect; identified by evalset sha256 705efffc… and "
               "qid sha256 0887559a…")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ── bootstrap ────────────────────────────────────────────────────────

def bootstrap_ci(values, n=N_BOOT, ci=0.95, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = np.array([float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
                      for _ in range(n)])
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return round(float(np.mean(arr)), 4), round(lo, 4), round(hi, 4)


# ── Wilcoxon + Holm ──────────────────────────────────────────────────

def paired_wilcoxon(a, b):
    """Wilcoxon signed-rank test. Returns (stat, p, n, n_tied)."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    diff = a - b
    tied = int(np.sum(diff == 0))
    nonzero = diff[diff != 0]
    if len(nonzero) < 1:
        return 0.0, 1.0, len(a), tied
    stat, p = wilcoxon(nonzero)
    return float(stat), float(p), len(a), tied


def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adj = min(adj, 1.0)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = round(cummax, 6)
    return adjusted


def median_diff_ci(a, b, n=N_BOOT, ci=0.95, seed=SEED):
    """Median paired difference with bootstrap CI."""
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    med = float(np.median(diff))
    rng = np.random.default_rng(seed)
    meds = [float(np.median(rng.choice(diff, size=len(diff), replace=True)))
            for _ in range(n)]
    lo = float(np.percentile(meds, (1 - ci) / 2 * 100))
    hi = float(np.percentile(meds, (1 + ci) / 2 * 100))
    return round(med, 4), round(lo, 4), round(hi, 4)


# ── config parsing ───────────────────────────────────────────────────

def parse_config_name(name: str) -> dict:
    """Extract chunking, retrieval, architecture from config name."""
    clean = name.replace("-dryrun", "")
    parts = clean.split("-")
    # <chunker>-bge-<retrieval>-<arch>
    chunker = parts[0] if parts else ""
    retrieval = parts[2] if len(parts) > 2 else ""
    arch = parts[3] if len(parts) > 3 else "flat"
    return {"chunking": chunker, "retrieval": retrieval, "architecture": arch}


def _write_table(out_dir: Path, name: str, rows: list[dict], fields: list[str]):
    """Write a table as CSV and markdown."""
    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / f"{name}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join("---" for _ in fields) + " |\n")
        for r in rows:
            vals = [str(r.get(k, "")) for k in fields]
            f.write("| " + " | ".join(vals) + " |\n")


# ── table builders ───────────────────────────────────────────────────

def build_t1(all_results: dict) -> list[dict]:
    """t1_grid: one row per config with bootstrap CIs."""
    rows = []
    for name in sorted(all_results):
        pq = all_results[name]["per_question"]
        r = {"config": name, "n": len(pq)}
        for m in METRICS:
            vals = [q[m] for q in pq]
            mean, lo, hi = bootstrap_ci(vals)
            r[f"{m}_mean"] = mean
            r[f"{m}_ci_lo"] = lo
            r[f"{m}_ci_hi"] = hi
        rows.append(r)
    return rows


def build_t2(all_results: dict) -> list[dict]:
    """t2_factors: marginal means per level of each factor."""
    factor_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name, data in all_results.items():
        parsed = parse_config_name(name)
        for factor in ["chunking", "retrieval", "architecture"]:
            level = parsed[factor]
            for q in data["per_question"]:
                for m in METRICS:
                    factor_vals[factor][level][m].append(q[m])

    rows = []
    for factor in ["chunking", "retrieval", "architecture"]:
        primary = PRIMARY[factor]
        for level in sorted(factor_vals[factor]):
            vals = factor_vals[factor][level]
            r = {"factor": factor, "level": level,
                 "primary_metric": primary, "n": len(vals[primary])}
            for m in METRICS:
                r[f"{m}_mean"] = round(float(np.mean(vals[m])), 4)
            rows.append(r)
    return rows


def build_t3(all_results: dict) -> list[dict]:
    """t3_paired: Wilcoxon against baseline and best per metric, deduplicated.

    One row per (config, metric, vs). Comparators = baseline and the best
    configuration under that metric. No duplicates; Holm over the whole table.
    """
    if BASELINE not in all_results and (BASELINE + "-dryrun") not in all_results:
        baseline_key = next((k for k in all_results if BASELINE in k), None)
    else:
        baseline_key = BASELINE if BASELINE in all_results else BASELINE + "-dryrun"

    if not baseline_key:
        return []

    base_pq = {q["question_id"]: q for q in all_results[baseline_key]["per_question"]}

    # Find best config per metric
    best_by_metric = {}
    for m in METRICS:
        best_name, best_val = None, -1.0
        for name, data in all_results.items():
            mean_val = float(np.mean([q[m] for q in data["per_question"]]))
            if mean_val > best_val:
                best_val = mean_val
                best_name = name
        best_by_metric[m] = best_name

    def _make_test(name, m, vs_name, vs_pq):
        pq = {q["question_id"]: q for q in all_results[name]["per_question"]}
        qids = sorted(set(pq) & set(vs_pq))
        a = [pq[qid][m] for qid in qids]
        b = [vs_pq[qid][m] for qid in qids]
        stat, p, n, tied = paired_wilcoxon(a, b)
        med, med_lo, med_hi = median_diff_ci(a, b)
        # mean diff with bootstrap CI
        diff = np.array(a, dtype=float) - np.array(b, dtype=float)
        mean_d = round(float(np.mean(diff)), 4)
        rng = np.random.default_rng(SEED)
        boot_means = [float(np.mean(rng.choice(diff, size=len(diff), replace=True)))
                      for _ in range(N_BOOT)]
        md_lo = round(float(np.percentile(boot_means, 2.5)), 4)
        md_hi = round(float(np.percentile(boot_means, 97.5)), 4)
        n_nonzero = int(np.sum(diff != 0))
        return {
            "config": name, "metric": m, "vs": vs_name,
            "stat": round(stat, 2), "p_raw": round(p, 6),
            "median_diff": med, "diff_ci_lo": med_lo, "diff_ci_hi": med_hi,
            "mean_diff": mean_d, "mean_diff_ci_lo": md_lo, "mean_diff_ci_hi": md_hi,
            "n": n, "tied": tied, "n_nonzero": n_nonzero,
        }

    seen = set()
    tests = []
    for name in sorted(all_results):
        if name == baseline_key:
            continue
        for m in METRICS:
            # vs baseline
            key = (name, m, baseline_key)
            if key not in seen:
                seen.add(key)
                tests.append(_make_test(name, m, baseline_key, base_pq))

            # vs best for this metric
            best_name = best_by_metric[m]
            if best_name and best_name != name:
                best_pq = {q["question_id"]: q
                           for q in all_results[best_name]["per_question"]}
                key2 = (name, m, best_name)
                if key2 not in seen:
                    seen.add(key2)
                    tests.append(_make_test(name, m, best_name, best_pq))

    # Assert uniqueness
    trip = [(t["config"], t["metric"], t["vs"]) for t in tests]
    assert len(trip) == len(set(trip)), "Duplicate (config, metric, vs) in t3"

    # Holm correction over the deduplicated table
    raw_ps = [t["p_raw"] for t in tests]
    adj_ps = holm_correction(raw_ps)
    for t, ap in zip(tests, adj_ps):
        t["p_holm"] = ap

    return tests


def build_t4(all_results: dict, evalset: list[dict]) -> list[dict]:
    """t4_tiers: R@10 and budget_640 by tier for 5 best + baseline."""
    qid_tier = {q["question_id"]: q.get("tier", "") for q in evalset}

    # Find 5 best by budget_640
    config_means = []
    for name, data in all_results.items():
        vals = [q["budget_640"] for q in data["per_question"]]
        config_means.append((name, float(np.mean(vals))))
    config_means.sort(key=lambda x: x[1], reverse=True)
    best5 = [c[0] for c in config_means[:5]]

    baseline_key = next((k for k in all_results if BASELINE in k), None)
    selected = list(dict.fromkeys(best5 + ([baseline_key] if baseline_key else [])))

    rows = []
    for name in selected:
        pq = all_results[name]["per_question"]
        by_tier = defaultdict(list)
        for q in pq:
            t = qid_tier.get(q["question_id"], "?")
            by_tier[t].append(q)
        for tier in sorted(by_tier):
            qs = by_tier[tier]
            r10 = round(float(np.mean([q["recall_10"] for q in qs])), 4)
            b640 = round(float(np.mean([q["budget_640"] for q in qs])), 4)
            rows.append({"config": name, "tier": tier, "n": len(qs),
                         "recall_10": r10, "budget_640": b640})
    return rows


def build_t5(all_results: dict, evalset: list[dict],
             covariates: list[dict]) -> list[dict]:
    """t5_style_easy: split by drafting_style and lexically_easy."""
    qid_style = {c["question_id"]: c.get("drafting_style", "")
                 for c in covariates}
    qid_easy = {c["question_id"]: c.get("lexically_easy", "")
                for c in covariates}

    baseline_key = next((k for k in all_results if BASELINE in k), None)
    config_means = []
    for name, data in all_results.items():
        vals = [q["budget_640"] for q in data["per_question"]]
        config_means.append((name, float(np.mean(vals))))
    config_means.sort(key=lambda x: x[1], reverse=True)
    best5 = [c[0] for c in config_means[:5]]
    selected = list(dict.fromkeys(best5 + ([baseline_key] if baseline_key else [])))

    rows = []
    for name in selected:
        pq = all_results[name]["per_question"]
        # By drafting style
        by_style = defaultdict(list)
        for q in pq:
            s = qid_style.get(q["question_id"], "?")
            by_style[s].append(q)
        for style in sorted(by_style):
            qs = by_style[style]
            rows.append({"config": name, "split": f"style:{style}",
                         "n": len(qs),
                         "recall_10": round(float(np.mean([q["recall_10"] for q in qs])), 4),
                         "budget_640": round(float(np.mean([q["budget_640"] for q in qs])), 4)})
        # By lexically_easy
        by_easy = defaultdict(list)
        for q in pq:
            e = qid_easy.get(q["question_id"], "?")
            by_easy[e].append(q)
        for easy in sorted(by_easy):
            qs = by_easy[easy]
            rows.append({"config": name, "split": f"easy:{easy}",
                         "n": len(qs),
                         "recall_10": round(float(np.mean([q["recall_10"] for q in qs])), 4),
                         "budget_640": round(float(np.mean([q["budget_640"] for q in qs])), 4)})
    return rows


def build_t6(all_results: dict, run_folders: dict) -> list[dict]:
    """t6_cost: linkexp appended, pcreturn tokens, context tokens at k=5."""
    rows = []
    for name in sorted(all_results):
        pq = all_results[name]["per_question"]
        # Mean context tokens from retrieved units (first 5)
        folder = run_folders.get(name)
        ret_data = []
        if folder:
            ret_p = Path(folder) / "retrieved.jsonl"
            if ret_p.exists():
                with ret_p.open() as f:
                    ret_data = [json.loads(l) for l in f if l.strip()]

        parsed = parse_config_name(name)
        arch = parsed["architecture"]

        ctx_tokens_5 = []
        appended_counts = []
        appended_tokens = []
        returned_tokens = []

        for rd in ret_data:
            units = rd.get("retrieved", [])[:5]
            ctx_tokens_5.append(sum(u.get("token_count", 0) for u in units))
            if arch == "linkexp":
                appended_counts.append(sum(len(u.get("appended_targets", []))
                                           for u in units))
                appended_tokens.append(sum(u.get("appended_tokens", 0)
                                           for u in units))
            elif arch == "pcreturn":
                returned_tokens.append(sum(u.get("token_count", 0)
                                           for u in units))

        r = {"config": name, "architecture": arch, "n": len(ret_data)}
        if ctx_tokens_5:
            r["mean_context_tokens_k5"] = round(float(np.mean(ctx_tokens_5)), 1)
        if arch == "linkexp" and appended_counts:
            r["mean_paragraphs_added"] = round(float(np.mean(appended_counts)), 2)
            r["mean_tokens_added"] = round(float(np.mean(appended_tokens)), 1)
        if arch == "pcreturn" and returned_tokens:
            r["mean_returned_tokens"] = round(float(np.mean(returned_tokens)), 1)
        rows.append(r)
    return rows


def build_t7(judge_folders: list[str], agreement_path: Optional[str]) -> list[dict]:
    """t7_rag: per D49 config faithfulness/accuracy with CIs, nulls, not-found."""
    if not judge_folders:
        return []

    rows = []
    agreement = {}
    if agreement_path:
        agreement = json.loads(Path(agreement_path).read_text())

    for jdir in judge_folders:
        jp = Path(jdir)
        cp = jp / "config.json"
        cn = json.loads(cp.read_text()).get("config_name", "") if cp.exists() else ""
        clean_cn = cn.replace("-dryrun", "")

        jfile = jp / "judgements.jsonl"
        if not jfile.exists():
            candidates = list(jp.glob("*judgements.jsonl"))
            jfile = candidates[0] if candidates else jfile
        if not jfile.exists():
            continue

        judgements = [json.loads(l) for l in jfile.open() if l.strip()]
        n = len(judgements)

        faith_vals = [j["faithfulness"] for j in judgements
                      if j.get("faithfulness") is not None]
        acc_vals = [j["accuracy"] for j in judgements
                    if j.get("accuracy") is not None]
        null_count = sum(1 for j in judgements if j.get("flag"))
        not_found = sum(1 for j in judgements if j.get("not_found_case"))

        r = {"config": clean_cn, "n": n}
        if faith_vals:
            fm, flo, fhi = bootstrap_ci(faith_vals)
            r.update({"faith_mean": fm, "faith_ci_lo": flo, "faith_ci_hi": fhi})
        if acc_vals:
            am, alo, ahi = bootstrap_ci(acc_vals)
            r.update({"acc_mean": am, "acc_ci_lo": alo, "acc_ci_hi": ahi})
        r["null_count"] = null_count
        r["not_found_rate"] = round(not_found / n, 4) if n else 0

        # Agreement stats — try both original and clean config name
        per_config = agreement.get("per_config", {})
        ca = per_config.get(cn) or per_config.get(clean_cn) or {}
        if ca:
            r["acc_kappa"] = ca.get("accuracy", {}).get("kappa")
            r["faith_kappa"] = ca.get("faithfulness", {}).get("kappa")
            r["acc_unvalidated"] = ca.get("accuracy", {}).get("unvalidated")
            r["faith_unvalidated"] = ca.get("faithfulness", {}).get("unvalidated")

        rows.append(r)
    return rows


def build_t8(run_folders: dict, grid_manifest: dict) -> list[dict]:
    """t8_runs: provenance table."""
    rows = []
    for name, folder in sorted(run_folders.items()):
        fp = Path(folder)
        cp = fp / "config.json"
        if not cp.exists():
            continue
        cfg = json.loads(cp.read_text())
        code = cfg.get("code", {})
        rows.append({
            "config": name,
            "run_id": cfg.get("run_id", ""),
            "git_commit": code.get("git_commit", ""),
            "code_tree_sha256": code.get("code_tree_sha256", "")[:16],
            "run_folder": fp.name,
            "note": DRYRUN_NOTE,
        })
    return rows


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build results tables (T7)")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--grid-manifest", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--judge", default=None, help="Agreement JSON")
    ap.add_argument("--judgements", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args(argv)

    paths = load_paths(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load grid manifest ───────────────────────────────────────────
    grid_manifest = json.loads(Path(args.grid_manifest).read_text())
    run_folders = grid_manifest.get("run_folders", {})

    # ── load per-question results from each run ──────────────────────
    all_results = {}
    for name, folder in run_folders.items():
        rp = Path(folder) / "results.json"
        if not rp.exists():
            print(f"ERROR: results.json missing in {folder}", file=sys.stderr)
            return 1
        data = json.loads(rp.read_text())
        all_results[name] = data

    # ── load evalset + covariates ────────────────────────────────────
    with open(args.evalset, newline="", encoding="utf-8") as f:
        evalset = list(csv.DictReader(f))
    with open(args.covariates, newline="", encoding="utf-8") as f:
        covariates = list(csv.DictReader(f))

    # ── build tables ─────────────────────────────────────────────────
    t1 = build_t1(all_results)
    t1_fields = ["config", "n"] + [f"{m}_{s}" for m in METRICS
                                    for s in ["mean", "ci_lo", "ci_hi"]]
    _write_table(out_dir, "t1_grid", t1, t1_fields)

    t2 = build_t2(all_results)
    t2_fields = ["factor", "level", "primary_metric", "n"] + \
                [f"{m}_mean" for m in METRICS]
    _write_table(out_dir, "t2_factors", t2, t2_fields)

    t3 = build_t3(all_results)
    t3_fields = ["config", "metric", "vs", "stat", "p_raw", "p_holm",
                 "median_diff", "diff_ci_lo", "diff_ci_hi",
                 "mean_diff", "mean_diff_ci_lo", "mean_diff_ci_hi",
                 "n", "tied", "n_nonzero"]
    _write_table(out_dir, "t3_paired", t3, t3_fields)

    t4 = build_t4(all_results, evalset)
    _write_table(out_dir, "t4_tiers", t4,
                 ["config", "tier", "n", "recall_10", "budget_640"])

    t5 = build_t5(all_results, evalset, covariates)
    _write_table(out_dir, "t5_style_easy", t5,
                 ["config", "split", "n", "recall_10", "budget_640"])

    t6 = build_t6(all_results, run_folders)
    t6_fields = ["config", "architecture", "n", "mean_context_tokens_k5",
                 "mean_paragraphs_added", "mean_tokens_added",
                 "mean_returned_tokens"]
    _write_table(out_dir, "t6_cost", t6, t6_fields)

    t7 = build_t7(args.judgements, args.judge)
    if t7:
        t7_fields = ["config", "n", "faith_mean", "faith_ci_lo", "faith_ci_hi",
                     "acc_mean", "acc_ci_lo", "acc_ci_hi",
                     "null_count", "not_found_rate",
                     "acc_kappa", "faith_kappa",
                     "acc_unvalidated", "faith_unvalidated"]
        _write_table(out_dir, "t7_rag", t7, t7_fields)

    t8 = build_t8(run_folders, grid_manifest)
    _write_table(out_dir, "t8_runs", t8,
                 ["config", "run_id", "git_commit", "code_tree_sha256",
                  "run_folder", "note"])

    # ── manifest ─────────────────────────────────────────────────────
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "n_configs": len(all_results),
        "n_questions": len(evalset),
        "inputs": {
            "grid_manifest": {"path": args.grid_manifest,
                              "sha256": _sha256(Path(args.grid_manifest))},
            "evalset": {"path": args.evalset,
                        "sha256": _sha256(Path(args.evalset))},
            "covariates": {"path": args.covariates,
                           "sha256": _sha256(Path(args.covariates))},
        },
        "run_folders": run_folders,
        "tables": ["t1_grid", "t2_factors", "t3_paired", "t4_tiers",
                   "t5_style_easy", "t6_cost"] +
                  (["t7_rag"] if t7 else []) + ["t8_runs"],
        "bootstrap": {"n": N_BOOT, "seed": SEED, "ci": 0.95},
        "note": DRYRUN_NOTE,
    }
    if args.judge:
        manifest["inputs"]["agreement"] = {
            "path": args.judge, "sha256": _sha256(Path(args.judge))}

    manifest_path = out_dir / "RESULTS_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # ── freeze ───────────────────────────────────────────────────────
    if args.freeze:
        frozen = Path(paths["results"]) / "results_v1"
        if frozen.exists():
            shutil.rmtree(frozen)
        shutil.copytree(out_dir, frozen)
        print(f"Frozen → {frozen}")

    print(f"Tables: {len(manifest['tables'])} written to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
