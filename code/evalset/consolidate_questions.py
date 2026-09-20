"""
Consolidate overlapping generation runs into one clean CSV.

Usage::
    python -m code.evalset.consolidate_questions --config config/paths.yaml \
        --evalset config/evalset.yaml \
        --input data/evalset/2026-09-19_evalset_questions-synthetic_v1.csv \
        --seeds data/evalset/2026-09-19_evalset_seeds_v1.csv
"""
import argparse, csv, hashlib, json, re
from collections import Counter
from pathlib import Path
import yaml
from code.common.run_registry import load_paths, start_run

DISPLAY = {"skilled_worker":"Skilled Worker","student":"Student","graduate":"Graduate","visitor":"Visitor","family":"Family"}
VALID_ROUTES = set(DISPLAY.keys())
B6_RE = re.compile(r'\b(Appendix|Part\s+\d|paragraph\s+\d|para\.?\s*\d|[A-Z]{1,6}\s?\d+\.\d+|[A-Z]+-[A-Z]+\.\d|\([A-Z]\))', re.IGNORECASE)

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate_row(row, seen_q):
    vs = []
    q=row["question"]; a=row["reference_answer"]; dn=DISPLAY.get(row["route"],"")
    if B6_RE.search(q): vs.append("B6_regex")
    if dn not in q: vs.append("no_route_name")
    if q.lower().count(dn.lower())>1: vs.append("route_name_twice")
    if len(q.split())<8: vs.append("q_short")
    if len(a.split())<10: vs.append("a_short")
    if a.rstrip().endswith(":"): vs.append("a_colon")
    if "the following" in a.lower(): vs.append("a_the_following")
    if "listed in" in a.lower(): vs.append("a_listed_in")
    if q.lower().strip() in seen_q: vs.append("duplicate")
    return vs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--evalset", default="config/evalset.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--seeds", required=True)
    args = ap.parse_args()

    paths = load_paths(args.config)
    with open(args.evalset) as f: ecfg = yaml.safe_load(f)
    evalset_dir = Path(paths["evalset"])

    # Load seeds: gold -> seed_id
    with open(args.seeds, newline="") as f:
        seeds = list(csv.DictReader(f))
    seed_gold_to_id = {s["gold_paragraph_ids"]: s["seed_id"] for s in seeds}
    seed_golds = set(seed_gold_to_id.keys())

    # Load input rows
    with open(args.input, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    log_rules = {"fragment":0, "outside_seeds":0, "dup_gold":0, "dup_question":0, "kept":0}
    per_row_rule = []

    # Rule a: drop fragments and outside-seeds
    valid_rows = []
    for r in raw_rows:
        if r["route"] not in VALID_ROUTES:
            log_rules["fragment"] += 1; per_row_rule.append("dropped:fragment"); continue
        if r["gold_paragraph_ids"] not in seed_golds:
            log_rules["outside_seeds"] += 1; per_row_rule.append("dropped:outside_seeds"); continue
        valid_rows.append(r)
        per_row_rule.append("candidate")

    # Rule b: one row per gold set — keep last-written that passes validator
    by_gold = {}
    for r in valid_rows:
        g = r["gold_paragraph_ids"]
        by_gold.setdefault(g, []).append(r)

    deduped = []
    for g, group in by_gold.items():
        if len(group) > 1: log_rules["dup_gold"] += len(group) - 1
        # Try last-written passing row first
        chosen = None
        for r in reversed(group):
            if not validate_row(r, set()):
                chosen = r; break
        if not chosen:
            chosen = group[-1]  # keep last if none passes
        deduped.append(chosen)

    # Rule c: drop duplicate questions (case-insensitive)
    seen_q = set(); final = []
    for r in deduped:
        ql = r["question"].lower().strip()
        if ql in seen_q:
            log_rules["dup_question"] += 1; continue
        seen_q.add(ql)
        final.append(r)

    # Rule d: question_id from seed_id, sort by seed_id
    for r in final:
        sid = seed_gold_to_id.get(r["gold_paragraph_ids"], "S0000")
        num = sid[1:]  # strip "S"
        r["question_id"] = f"Q{num}"
    final.sort(key=lambda r: r["question_id"])
    log_rules["kept"] = len(final)

    # Check validator failures
    seen_q2 = set(); v_fails = 0
    for r in final:
        vs = validate_row(r, seen_q2)
        seen_q2.add(r["question"].lower().strip())
        if vs: v_fails += 1

    # Stratum counts vs D52
    counts = ecfg["counts"]
    stratum_target = {}
    for rn in counts:
        for t in counts[rn]:
            tc = counts[rn][t]
            if tc > 0:
                if t.startswith("T3_"):
                    stratum_target[(rn,"T3",t[3:])] = tc
                else:
                    stratum_target[(rn,t,"")] = tc

    stratum_actual = Counter()
    for r in final:
        stratum_actual[(r["route"],r["tier"],r.get("link_type",""))] += 1

    shortfalls = {}
    for sk, target in sorted(stratum_target.items()):
        actual = stratum_actual.get(sk, 0)
        if actual < target:
            shortfalls[f"{sk[0]}_{sk[1]}_{sk[2]}"] = target - actual

    # Write output
    out_path = evalset_dir / "2026-09-20_evalset_questions-synthetic_v1.csv"
    fields = ["question_id","route","tier","link_type","question","reference_answer",
              "gold_paragraph_ids","source","generator_model","embedding_model",
              "validated_by_student","notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(final)

    # Register run
    ctx = start_run("evalset", "synthetic-v1-consolidate", {
        "input": str(args.input), "seeds": str(args.seeds),
        "evalset_sha256": _sha(args.evalset),
    }, paths)
    ctx.log(f"input={len(raw_rows)} kept={len(final)} v_fails={v_fails}")
    ctx.finish("ok", f"consolidated: {len(final)} rows, {v_fails} validator failures")

    # Write consolidation manifest
    manifest = {
        "rules_applied": log_rules,
        "validator_failures": v_fails,
        "stratum_counts": {f"{k[0]}_{k[1]}_{k[2]}": v for k,v in stratum_actual.items()},
        "shortfalls": shortfalls,
        "total_rows": len(final),
    }
    mp = evalset_dir / "2026-09-20_evalset_questions-synthetic_v1_consolidation.json"
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"Input: {len(raw_rows)} rows")
    print(f"Rules: {log_rules}")
    print(f"Kept: {len(final)} rows")
    print(f"Validator failures: {v_fails}")
    print(f"\nPer stratum:")
    for sk in sorted(stratum_target):
        t = stratum_target[sk]; a = stratum_actual.get(sk, 0)
        sf = t - a
        flag = f" SHORTFALL {sf}" if sf > 0 else ""
        print(f"  {str(sk):50s} target={t:2d} actual={a:2d}{flag}")
    print(f"\nOutput: {out_path}")

if __name__ == "__main__": main()
