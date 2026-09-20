"""Freeze the evaluation set after verdict review (T2c-4).

Usage::

    python -m code.evalset.freeze_evalset \\
        --config config/paths.yaml \\
        --candidate <csv> --neardup <verdict csv> --covariates <csv> \\
        --out evalset/<date>_evalset_v1.csv

Verdict handling:
  - ``duplicate``: drop the row with the later question_id of the pair.
  - ``reworded``: replace question_b's text with reworded_question_b.
  - ``distinct``: keep both.
  - Any pair with an empty verdict stops the freeze.

Floor check per route × tier against fallback_floor in config/evalset.yaml
(T3 synthetic and hand counted together as T3); strata below the floor are
listed, not fatal; the manifest records them as limitations.

No corpus text is printed.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

from code.common.run_registry import load_paths, start_run

# ── constants ────────────────────────────────────────────────────────

from code.evalset.merge_questions import COLUMNS  # D53 columns

VALID_VERDICTS = {"duplicate", "reworded", "distinct"}


# ── helpers ──────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strata_key(route: str, tier: str) -> str:
    """Route × tier key for floor check (T3 synthetic+hand counted as T3)."""
    return f"{route}_{tier}"


def _detailed_key(route: str, tier: str, link_type: str) -> str:
    lt = link_type or ""
    return f"{route}_{tier}_{lt}".rstrip("_")


# ── verdict application (testable) ──────────────────────────────────

def apply_verdicts(
    candidate_rows: list[dict],
    neardup_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Apply near-duplicate verdicts.

    Returns (kept_rows, removed_pairs) where removed_pairs is a list of
    dicts with pair_id, qid_dropped, verdict, reason.

    Raises ValueError on empty verdicts or missing reworded text.
    """
    # Check for empty verdicts
    for nd in neardup_rows:
        v = nd.get("verdict", "").strip()
        if not v:
            raise ValueError(
                f"Empty verdict for pair {nd['pair_id']} "
                f"({nd['qid_a']}/{nd['qid_b']}); cannot freeze")
        if v not in VALID_VERDICTS:
            raise ValueError(
                f"Invalid verdict {v!r} for pair {nd['pair_id']}")

    # Build a lookup from question_id -> row
    rows_by_id = {r["question_id"]: r for r in candidate_rows}

    # Collect ids to drop and rewrites
    to_drop: set[str] = set()
    rewrites: dict[str, str] = {}  # qid -> new question text
    removed_pairs: list[dict] = []

    for nd in neardup_rows:
        v = nd["verdict"].strip()
        qid_a = nd["qid_a"]
        qid_b = nd["qid_b"]

        if v == "duplicate":
            # Drop the row with the later question_id
            later = max(qid_a, qid_b)
            to_drop.add(later)
            removed_pairs.append({
                "pair_id": nd["pair_id"],
                "qid_dropped": later,
                "verdict": "duplicate",
                "reason": f"duplicate of {min(qid_a, qid_b)}",
            })

        elif v == "reworded":
            new_text = nd.get("reworded_question_b", "").strip()
            if not new_text:
                raise ValueError(
                    f"Verdict 'reworded' for pair {nd['pair_id']} but "
                    f"reworded_question_b is empty")
            rewrites[qid_b] = new_text

        # distinct: keep both, nothing to do

    # Apply
    kept: list[dict] = []
    for r in candidate_rows:
        qid = r["question_id"]
        if qid in to_drop:
            continue
        if qid in rewrites:
            r = dict(r)
            r["question"] = rewrites[qid]
        kept.append(r)

    return kept, removed_pairs


def floor_check(
    rows: list[dict],
    fallback_floor: dict,
) -> list[dict]:
    """Check strata against fallback_floor.

    Returns a list of shortfall dicts: {route, tier, floor_key, floor, actual, shortfall}.
    """
    # Count per route × tier (T3 synthetic + hand counted together)
    counts: dict[str, int] = {}
    for r in rows:
        key = _strata_key(r["route"], r["tier"])
        counts[key] = counts.get(key, 0) + 1

    shortfalls: list[dict] = []
    for route, tiers in fallback_floor.items():
        for floor_key, floor_val in tiers.items():
            if floor_val == 0:
                continue
            # Map floor_key to the tier used in the data
            if floor_key.startswith("T3"):
                tier = "T3"
            else:
                tier = floor_key
            strata_key = _strata_key(route, tier)
            actual = counts.get(strata_key, 0)
            if actual < floor_val:
                shortfalls.append({
                    "route": route,
                    "tier": tier,
                    "floor_key": floor_key,
                    "floor": floor_val,
                    "actual": actual,
                    "shortfall": floor_val - actual,
                })

    return shortfalls


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Freeze the evaluation set (T2c-4)")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--evalset-config", default="config/evalset.yaml")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--neardup", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    paths = load_paths(args.config)

    cand_path = Path(args.candidate)
    nd_path = Path(args.neardup)
    cov_path = Path(args.covariates)
    out_path = Path(args.out)

    for label, p in [("candidate", cand_path), ("neardup", nd_path),
                     ("covariates", cov_path)]:
        if not p.exists():
            print(f"ERROR: {label} file not found: {p}", file=sys.stderr)
            return 1

    # ── load evalset config for floor check ──────────────────────────
    with open(args.evalset_config) as f:
        evalset_cfg = yaml.safe_load(f)
    fallback_floor = evalset_cfg.get("fallback_floor", {})

    # ── read inputs ──────────────────────────────────────────────────
    with cand_path.open(encoding="utf-8", newline="") as f:
        candidate_rows = list(csv.DictReader(f))
    with nd_path.open(encoding="utf-8", newline="") as f:
        neardup_rows = list(csv.DictReader(f))
    with cov_path.open(encoding="utf-8", newline="") as f:
        cov_rows = list(csv.DictReader(f))

    # ── register run ─────────────────────────────────────────────────
    params = {
        "candidate": args.candidate,
        "candidate_sha256": _sha256(cand_path),
        "neardup": args.neardup,
        "neardup_sha256": _sha256(nd_path),
        "covariates": args.covariates,
        "covariates_sha256": _sha256(cov_path),
    }
    ctx = start_run("evalset", "freeze-v1", params, paths)
    ctx.log(f"candidate: {len(candidate_rows)} rows")
    ctx.log(f"neardup: {len(neardup_rows)} pairs")

    # ── apply verdicts ───────────────────────────────────────────────
    try:
        kept, removed_pairs = apply_verdicts(candidate_rows, neardup_rows)
    except ValueError as exc:
        ctx.log(f"FAILED: {exc}")
        ctx.finish("failed", str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ctx.log(f"after verdicts: {len(kept)} rows, {len(removed_pairs)} removed")

    # ── floor check ──────────────────────────────────────────────────
    shortfalls = floor_check(kept, fallback_floor)

    # ── write frozen CSV atomically ──────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(kept)
    tmp.replace(out_path)

    output_sha = _sha256(out_path)

    # ── copy covariates restricted to frozen ids ─────────────────────
    frozen_ids = {r["question_id"] for r in kept}
    frozen_cov_rows = [r for r in cov_rows if r["question_id"] in frozen_ids]

    cov_frozen_name = str(out_path).replace("_evalset_v1.csv",
                                            "_evalset_covariates_v1_frozen.csv")
    cov_frozen_path = Path(cov_frozen_name)
    cov_tmp = cov_frozen_path.with_suffix(".tmp")
    if frozen_cov_rows:
        cov_cols = list(frozen_cov_rows[0].keys())
        with cov_tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cov_cols)
            w.writeheader()
            w.writerows(frozen_cov_rows)
        cov_tmp.replace(cov_frozen_path)

    cov_frozen_sha = _sha256(cov_frozen_path) if cov_frozen_path.exists() else ""

    # ── counts per route × tier × link_type × source ────────────────
    from collections import Counter
    strata_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for r in kept:
        dk = _detailed_key(r["route"], r["tier"], r.get("link_type", ""))
        strata_counts[dk] = strata_counts.get(dk, 0) + 1
        s = r.get("source", "")
        source_counts[s] = source_counts.get(s, 0) + 1

    # ── manifest ─────────────────────────────────────────────────────
    manifest = {
        "frozen_rows": len(kept),
        "removed_pairs": removed_pairs,
        "counts_per_stratum": strata_counts,
        "counts_per_source": source_counts,
        "floor_check": {
            "shortfalls": shortfalls,
            "limitations": [
                f"{s['route']} {s['floor_key']}: {s['actual']}/{s['floor']} "
                f"(shortfall {s['shortfall']})"
                for s in shortfalls
            ],
        },
        "input_sha256": {
            "candidate": params["candidate_sha256"],
            "neardup": params["neardup_sha256"],
            "covariates": params["covariates_sha256"],
        },
        "output_sha256": output_sha,
        "covariates_frozen_sha256": cov_frozen_sha,
    }

    manifest_path = Path(str(out_path).replace(".csv", "_manifest.json"))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx.log(f"frozen {len(kept)} rows → {out_path}")
    ctx.finish("ok", f"frozen {len(kept)} rows", metrics=manifest)

    # ── stdout ───────────────────────────────────────────────────────
    # Verdict counts
    from collections import Counter as C2
    vcs = C2(nd["verdict"] for nd in neardup_rows)
    print(f"verdicts: {dict(vcs)}")
    print(f"rows removed: {len(removed_pairs)}")
    for rp in removed_pairs:
        print(f"  {rp['pair_id']}: dropped {rp['qid_dropped']} ({rp['verdict']}: {rp['reason']})")
    print(f"frozen rows: {len(kept)}")
    print("counts per route × tier × link_type:")
    for k, v in sorted(strata_counts.items()):
        print(f"  {k}: {v}")
    if shortfalls:
        print("floor check — below floor:")
        for s in shortfalls:
            print(f"  {s['route']} {s['floor_key']}: "
                  f"{s['actual']}/{s['floor']} (shortfall {s['shortfall']})")
    else:
        print("floor check: all strata at or above floor")
    print(f"output: {out_path}")
    print(f"output sha256: {output_sha}")
    print(f"covariates frozen: {cov_frozen_path}")
    print(f"run folder: {ctx.run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
