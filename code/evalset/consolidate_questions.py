"""Consolidate synthetic question rows from overlapping generation runs (decision D62).

Rules, applied in order:
  a. drop rows whose route is not a known route (fragments) and rows whose
     gold_paragraph_ids (exact semicolon-joined string) is not a seeds v1 group;
  b. one row per gold set: the last-written row that passes the validator; if none
     passes, the gold set is dropped;
  c. drop any row whose question text (case-insensitive, stripped) equals an earlier kept row;
  d. question_id = "Q" + seed number; route, tier, link_type taken from the seed row.

Usage:
  python -m code.evalset.consolidate_questions --blended <csv> [--topup <jsonl or csv> ...]
      --seeds <seeds v1 csv> --out <csv>
Writes <out> and <out minus .csv>_consolidation.json. No corpus text is printed.
"""
import argparse, csv, hashlib, json, re, sys
from pathlib import Path

ROUTES = {"skilled_worker": "Skilled Worker", "student": "Student", "graduate": "Graduate",
          "visitor": "Visitor", "family": "Family"}
COLUMNS = ["question_id", "route", "tier", "link_type", "question", "reference_answer",
           "gold_paragraph_ids", "source", "generator_model", "embedding_model",
           "validated_by_student", "notes"]
B6 = re.compile(r"\b(Appendix|Part\s+\d|paragraph\s+\d|para\.?\s*\d|[A-Z]{1,6}\s?\d+\.\d+|"
                r"[A-Z]+-[A-Z]+\.\d|\([A-Z]\))", re.I)


def passes(row):
    q, a = row.get("question") or "", row.get("reference_answer") or ""
    name = ROUTES[row["route"]]
    return (not B6.search(q)
            and len(re.findall(re.escape(name), q, re.I)) == 1
            and len(q.split()) >= 8
            and len(a.split()) >= 10
            and not a.rstrip().endswith(":")
            and not re.search(r"the following|listed in", a, re.I))


def read_rows(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return list(csv.DictReader(p.open(encoding="utf-8", newline="")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blended", required=True)
    ap.add_argument("--topup", nargs="*", default=[])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = {s["gold_paragraph_ids"]: s for s in csv.DictReader(open(args.seeds, encoding="utf-8", newline=""))}
    rows = read_rows(args.blended)
    for t in args.topup:
        rows += read_rows(t)

    report = {"input_rows": len(rows), "dropped": {"a_fragment": [], "a_outside_seeds": [],
              "b_no_passing_row": [], "b_superseded": [], "c_duplicate_question": []}}
    # rule a
    kept_a = []
    for i, r in enumerate(rows):
        if r.get("route") not in ROUTES:
            report["dropped"]["a_fragment"].append(i); continue
        if r.get("gold_paragraph_ids") not in seeds:
            report["dropped"]["a_outside_seeds"].append({"row": i, "gold": r.get("gold_paragraph_ids")}); continue
        kept_a.append((i, r))
    # rule b
    by_gold = {}
    for i, r in kept_a:
        by_gold.setdefault(r["gold_paragraph_ids"], []).append((i, r))
    kept_b = []
    for g, items in by_gold.items():
        good = [(i, r) for i, r in items if passes(r)]
        if not good:
            report["dropped"]["b_no_passing_row"].append({"gold": g, "rows": [i for i, _ in items]}); continue
        keep_i, keep_r = good[-1]
        report["dropped"]["b_superseded"] += [i for i, _ in items if i != keep_i]
        kept_b.append((keep_i, keep_r))
    kept_b.sort(key=lambda x: x[0])
    # rule c
    seen, kept_c = set(), []
    for i, r in kept_b:
        key = r["question"].strip().lower()
        if key in seen:
            report["dropped"]["c_duplicate_question"].append(i); continue
        seen.add(key); kept_c.append(r)
    # rule d
    out = []
    for r in kept_c:
        s = seeds[r["gold_paragraph_ids"]]
        o = {c: (r.get(c) or "") for c in COLUMNS}
        o.update({"question_id": "Q" + s["seed_id"][1:], "route": s["route"], "tier": s["tier"],
                  "link_type": s.get("link_type", ""), "source": r.get("source") or "prompt-v2",
                  "validated_by_student": "N"})
        out.append(o)
    out.sort(key=lambda o: o["question_id"])

    outp = Path(args.out)
    tmp = outp.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(out)
    tmp.replace(outp)
    strata = {}
    for o in out:
        k = f'{o["route"]}_{o["tier"]}_{o["link_type"]}'
        strata[k] = strata.get(k, 0) + 1
    report.update({"kept": len(out), "strata": strata,
                   "ids_unique": len({o["question_id"] for o in out}) == len(out),
                   "validator_failures": sum(1 for o in out if not passes(o)),
                   "output_sha256": hashlib.sha256(outp.read_bytes()).hexdigest(),
                   "seeds_sha256": hashlib.sha256(Path(args.seeds).read_bytes()).hexdigest()})
    Path(str(outp)[:-4] + "_consolidation.json").write_text(json.dumps(report, indent=2))
    print(f"kept {len(out)} rows; ids_unique={report['ids_unique']}; "
          f"validator_failures={report['validator_failures']}; sha256={report['output_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
