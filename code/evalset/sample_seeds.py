"""
Sample seed groups for the evaluation set.

Usage::

    python -m code.evalset.sample_seeds \\
        --config config/paths.yaml --evalset config/evalset.yaml
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Sample seed groups for evalset")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--evalset", default="config/evalset.yaml")
    ap.add_argument("--routes", default="config/routes.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        paths = yaml.safe_load(f)
    with open(args.evalset) as f:
        evalcfg = yaml.safe_load(f)
    with open(args.routes) as f:
        routecfg = yaml.safe_load(f)

    processed = Path(paths["processed"])
    evalset_dir = Path(paths["evalset"])
    evalset_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = processed / evalcfg["corpus"]
    links_path = processed / evalcfg["links"]

    seed = evalcfg["seed"]
    exclude_types = set(evalcfg.get("exclude_text_types", []))
    exclude_conf = set(evalcfg.get("exclude_confidence", []))
    min_words = evalcfg.get("min_words", 15)
    counts = evalcfg["counts"]

    # Build route -> sections mapping
    route_sections: dict[str, set[str]] = {}
    for rname, rcfg in routecfg.get("routes", {}).items():
        route_sections[rname] = set(rcfg.get("rules_sections", []))

    cc_sections = set(routecfg.get("cross_cutting", {}).get("rules_sections", []))

    # All route sections (for seed eligibility)
    all_route_sections = set()
    for secs in route_sections.values():
        all_route_sections |= secs

    # Load corpus
    records_by_id: dict[str, dict] = {}
    records_by_section: dict[str, list[dict]] = defaultdict(list)
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            records_by_id[r["paragraph_id"]] = r
            records_by_section[r["section_base_path"]].append(r)

    # Verify all configured sections exist
    for rname, secs in route_sections.items():
        for s in secs:
            if s not in records_by_section:
                print(f"ABORT: configured section absent from corpus: {s} (route {rname})")
                sys.exit(1)
    for s in cc_sections:
        if s not in records_by_section:
            print(f"ABORT: cross_cutting section absent from corpus: {s}")
            sys.exit(1)

    # Load links (v2 CSV with status column)
    para_links: dict[str, list[dict]] = defaultdict(list)  # paragraph_id -> resolved para links
    with open(links_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["target_type"] == "paragraph"
                    and row.get("status") == "resolved"
                    and row.get("resolved_to")):
                para_links[row["from_paragraph_id"]].append(row)

    # Eligibility filter
    def _eligible(r: dict) -> bool:
        if r["text_type"] in exclude_types:
            return False
        if r.get("parse_confidence") in exclude_conf:
            return False
        if len(r.get("text", "").split()) < min_words:
            return False
        return True

    # Build pools per (route, tier)
    # Section -> route mapping (a section can belong to one route for seeding)
    section_to_route: dict[str, str] = {}
    for rname, secs in route_sections.items():
        for s in secs:
            section_to_route[s] = rname

    # T1 pool: one eligible rule or narrative record per route section
    t1_pool: dict[str, list] = defaultdict(list)  # route -> list of seed dicts
    for rname, secs in route_sections.items():
        for sec in secs:
            for r in records_by_section[sec]:
                if r["text_type"] in ("rule", "narrative") and _eligible(r):
                    t1_pool[rname].append({
                        "gold_paragraph_ids": [r["paragraph_id"]],
                        "sections": [sec],
                    })

    # T2 pool: a rule with attached children, 2-6 records after exclusions
    t2_pool: dict[str, list] = defaultdict(list)
    children_of: dict[str, list[dict]] = defaultdict(list)
    for r in records_by_id.values():
        if r.get("attached_to"):
            children_of[r["attached_to"]].append(r)

    for rname, secs in route_sections.items():
        for sec in secs:
            for r in records_by_section[sec]:
                if r["text_type"] != "rule" or not _eligible(r):
                    continue
                kids = [c for c in children_of.get(r["paragraph_id"], [])
                        if _eligible(c)]
                group = [r] + kids
                if 2 <= len(group) <= 6:
                    t2_pool[rname].append({
                        "gold_paragraph_ids": [g["paragraph_id"] for g in group],
                        "sections": [sec],
                    })

    # T3 pool: eligible record with a resolved para link to a different section
    t3_pool: dict[str, list] = defaultdict(list)
    all_eligible_sections = all_route_sections | cc_sections
    for rname, secs in route_sections.items():
        for sec in secs:
            for r in records_by_section[sec]:
                if not _eligible(r):
                    continue
                links = para_links.get(r["paragraph_id"], [])
                for lk in links:
                    target_id = lk["resolved_to"]
                    target_rec = records_by_id.get(target_id)
                    if not target_rec:
                        continue
                    target_sec = target_rec["section_base_path"]
                    if target_sec == sec:
                        continue
                    if target_sec not in all_eligible_sections:
                        continue
                    if not _eligible(target_rec):
                        continue
                    t3_pool[rname].append({
                        "gold_paragraph_ids": [r["paragraph_id"], target_id],
                        "sections": sorted(set([sec, target_sec])),
                        "_link": lk,  # for random choice
                    })

    pools = {"T1": t1_pool, "T2": t2_pool, "T3": t3_pool}

    # Sample
    rng = random.Random(seed)
    rows = []
    manifest_strata = {}
    seed_counter = 0

    for rname in counts:
        tier_counts = counts[rname]
        for tier in ["T1", "T2", "T3"]:
            n = tier_counts.get(tier, 0)
            pool = pools[tier].get(rname, [])
            stratum_key = f"{rname}_{tier}"
            manifest_strata[stratum_key] = {"pool": len(pool), "drawn": min(n, len(pool))}

            if len(pool) < n:
                print(f"WARNING: {stratum_key} pool={len(pool)} < requested={n}")

            sampled = rng.sample(pool, min(n, len(pool)))

            for item in sampled:
                seed_counter += 1
                # For T3, choose one link at random from the record's eligible links
                rows.append({
                    "seed_id": f"S{seed_counter:04d}",
                    "route": rname,
                    "tier": tier,
                    "gold_paragraph_ids": ";".join(item["gold_paragraph_ids"]),
                    "sections": ";".join(item["sections"]),
                })

    # Write CSV
    today = date.today().isoformat()
    csv_path = evalset_dir / f"{today}_evalset_seeds_v0.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed_id", "route", "tier",
                                                "gold_paragraph_ids", "sections"])
        writer.writeheader()
        writer.writerows(rows)

    # Write manifest
    manifest = {
        "seed": seed,
        "evalset_yaml_sha256": _sha256_file(Path(args.evalset)),
        "corpus_sha256": _sha256_file(corpus_path),
        "links_sha256": _sha256_file(links_path),
        "strata": manifest_strata,
    }
    manifest_path = evalset_dir / f"{today}_evalset_seeds_v0.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Seeds: {len(rows)} rows → {csv_path}")
    print(f"Manifest: {manifest_path}")
    print(f"\nPer-stratum pool/drawn:")
    for k in sorted(manifest_strata):
        s = manifest_strata[k]
        print(f"  {k:25s}  pool={s['pool']:5d}  drawn={s['drawn']}")
    print(f"\nsha256 seeds CSV:     {_sha256_file(csv_path)}")
    print(f"sha256 manifest JSON: {_sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
