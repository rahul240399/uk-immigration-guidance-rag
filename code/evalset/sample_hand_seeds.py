"""
Build hand-seed CSV for section-level cross-reference questions.

Usage::

    python -m code.evalset.sample_hand_seeds --config config/paths.yaml --evalset config/evalset.yaml
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Sample hand seeds for section-level xrefs")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--evalset", default="config/evalset.yaml")
    ap.add_argument("--routes", default="config/routes.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        paths = yaml.safe_load(f)
    with open(args.evalset) as f:
        ecfg = yaml.safe_load(f)
    with open(args.routes) as f:
        rcfg = yaml.safe_load(f)

    processed = Path(paths["processed"])
    evalset_dir = Path(paths["evalset"])
    evalset_dir.mkdir(parents=True, exist_ok=True)

    seed = ecfg["seed"]
    exclude_types = set(ecfg.get("exclude_text_types", []))
    exclude_conf = set(ecfg.get("exclude_confidence", []))
    min_words = ecfg.get("min_words", 15)

    # Read hand_seeds config
    hs = ecfg["hand_seeds"]
    hand_counts = hs["counts"]
    link_order = hs["order"]  # [rules_section_xcut, rules_section_other]
    overdraw = hs["overdraw"]

    route_sections = {}
    for rname, rc in rcfg.get("routes", {}).items():
        route_sections[rname] = set(rc.get("rules_sections", []))
    cc_sections = set(rcfg.get("cross_cutting", {}).get("rules_sections", []))

    sec_to_route = {}
    for rname, secs in route_sections.items():
        for s in secs:
            sec_to_route[s] = rname

    # Load corpus
    corpus_sections = set()
    section_titles = {}
    records = {}
    with open(processed / "rules-paragraphs_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus_sections.add(r["section_base_path"])
            section_titles[r["section_base_path"]] = r.get("section_title", "")
            records[r["paragraph_id"]] = r

    # Eligible sources
    eligible = {}
    for pid, r in records.items():
        if r["text_type"] in exclude_types or r["text_type"] == "deleted":
            continue
        if r.get("parse_confidence") in exclude_conf:
            continue
        if len(r.get("text", "").split()) < min_words:
            continue
        if r["section_base_path"] in sec_to_route:
            eligible[pid] = r

    # Load resolved section links
    source_targets: dict[str, list[str]] = defaultdict(list)
    with open(processed / "rules-crossrefs_v2.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] != "resolved":
                continue
            if row["target_type"] not in ("appendix", "part"):
                continue
            src = row["from_paragraph_id"]
            target_bp = row["resolved_to"]
            if not target_bp or target_bp not in corpus_sections:
                continue
            src_rec = eligible.get(src)
            if not src_rec:
                continue
            if target_bp == src_rec["section_base_path"]:
                continue
            source_targets[src].append(target_bp)

    for src in source_targets:
        source_targets[src] = sorted(set(source_targets[src]))

    # Map link_order names to classification logic
    # rules_section_xcut = target in cross_cutting
    # rules_section_other = any other
    def classify(targets):
        if any(t in cc_sections for t in targets):
            return "rules_section_xcut"
        return "rules_section_other"

    rows = []
    manifest_pools = {}
    seed_counter = 0

    for rname in ["skilled_worker", "student", "graduate", "visitor", "family"]:
        secs = route_sections[rname]
        hand = hand_counts[rname]
        draw_target = overdraw * hand

        # Build pools per link_type
        pools: dict[str, list[str]] = {lt: [] for lt in link_order}
        for src, targets in source_targets.items():
            rec = eligible.get(src)
            if not rec or rec["section_base_path"] not in secs:
                continue
            lt = classify(targets)
            pools[lt].append(src)

        # Sort and shuffle each pool with stratum-specific seed
        # Use short keys (e1/e2) for the RNG to match the original draw
        lt_to_rng_key = {link_order[0]: "e1", link_order[1]: "e2"}
        for lt in link_order:
            pools[lt].sort()
            rng = random.Random(f"{seed}:{rname}:{lt_to_rng_key[lt]}")
            rng.shuffle(pools[lt])
            manifest_pools[f"{rname}_{lt}"] = {"pool": len(pools[lt]), "drawn": 0}

        # Draw: first link_type first, then next fills remainder
        drawn = []
        drawn_ids = set()

        for lt in link_order:
            for src in pools[lt]:
                if len(drawn) >= draw_target:
                    break
                if src in drawn_ids:
                    continue
                drawn.append((src, lt))
                drawn_ids.add(src)
                manifest_pools[f"{rname}_{lt}"]["drawn"] += 1

        for rank, (src, link_type) in enumerate(drawn, 1):
            seed_counter += 1
            rec = eligible[src]
            targets = source_targets[src]
            target_secs = ";".join(targets)
            target_titles = ";".join(section_titles.get(t, "") for t in targets)

            rows.append({
                "seed_id": f"H{seed_counter:04d}",
                "route": rname,
                "link_type": link_type,
                "source_paragraph_id": src,
                "source_rule_ref": rec.get("rule_ref", ""),
                "source_section": rec["section_base_path"],
                "target_section": target_secs,
                "target_section_title": target_titles,
                "draw_rank": rank,
            })

    # Write CSV
    csv_path = evalset_dir / "2026-09-19_evalset_seeds-hand_v0.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed_id", "route", "link_type", "source_paragraph_id",
            "source_rule_ref", "source_section", "target_section",
            "target_section_title", "draw_rank"])
        writer.writeheader()
        writer.writerows(rows)

    # Manifest
    evalset_sha = _sha256_file(Path(args.evalset))
    manifest = {
        "seed": seed,
        "evalset_yaml_sha256": evalset_sha,
        "corpus_sha256": _sha256_file(processed / "rules-paragraphs_v1.jsonl"),
        "links_sha256": _sha256_file(processed / "rules-crossrefs_v2.csv"),
        "pools": manifest_pools,
        "total_rows": len(rows),
    }
    manifest_path = evalset_dir / "2026-09-19_evalset_seeds-hand_v0_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Report
    print(f"evalset_yaml_sha256: {evalset_sha}")
    print(f"Rows: {len(rows)}")
    route_type_counts = Counter()
    for r in rows:
        route_type_counts[(r["route"], r["link_type"])] += 1
    for rname in ["skilled_worker", "student", "graduate", "visitor", "family"]:
        xcut = route_type_counts.get((rname, "rules_section_xcut"), 0)
        other = route_type_counts.get((rname, "rules_section_other"), 0)
        print(f"  {rname}: xcut={xcut} other={other} total={xcut+other}")

    print(f"\nPools:")
    for key in sorted(manifest_pools):
        p = manifest_pools[key]
        print(f"  {key:40s} pool={p['pool']:3d} drawn={p['drawn']:3d}")

    # Compare with old run
    import json as _json
    try:
        with open("/tmp/old_hand_ids.json") as f:
            old_ids = _json.load(f)
        new_ids = [r["source_paragraph_id"] for r in rows]
        if old_ids == new_ids:
            print("\nSource ID sequence: IDENTICAL to previous run")
        else:
            diffs = sum(1 for a, b in zip(old_ids, new_ids) if a != b)
            print(f"\nSource ID sequence: {diffs} differences out of {len(old_ids)}")
    except FileNotFoundError:
        print("\nNo previous run to compare")


if __name__ == "__main__":
    main()
