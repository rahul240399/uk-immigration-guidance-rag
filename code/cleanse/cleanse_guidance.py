"""
Cleanse guidance cross-references: resolve with typed aliases (no rebuild),
write v2 crossrefs CSV.

Usage::

    python -m code.cleanse.cleanse_guidance --config config/paths.yaml
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from code.common.link_resolution import (
    load_typed_aliases,
    resolve_links,
)
from code.common.run_registry import load_paths, start_run
from code.validate.freeze_rules import build_name_index


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _v1_status(link: dict) -> str:
    target = link.get("target", "")
    rt = link.get("resolved_to", "")
    if target == "absent" or rt == "absent":
        return "absent"
    if link.get("resolved", False):
        return "resolved"
    return "unresolved"


def main():
    ap = argparse.ArgumentParser(description="Cleanse guidance cross-references")
    ap.add_argument("--config", default="config/paths.yaml")
    args = ap.parse_args()

    paths = load_paths(args.config)
    processed = Path(paths["processed"])

    jsonl_path = processed / "guidance-paragraphs_v1.jsonl"
    v1_csv_path = processed / "guidance-crossrefs_v1.csv"
    # Use rules identifier table for resolving paragraph refs in guidance
    id_table_path = processed / "rules-identifier-table_v1.json"
    aliases_path = Path("config/section_aliases.yaml")

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    with open(id_table_path) as f:
        id_table = json.load(f)

    aliases = load_typed_aliases(str(aliases_path))
    # Build section set from rules manifest (101 sections), not from id table (93)
    rules_manifest_path = processed / "manifest_v1.json"
    with open(rules_manifest_path) as f:
        rules_manifest = json.load(f)
    section_base_paths = set(rules_manifest.get("record_counts", {}).get("by_section", {}).keys())

    # Build name index from rules records (guidance resolves against rules)
    rules_jsonl = processed / "rules-paragraphs_v1.jsonl"
    rules_records = []
    with open(rules_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rules_records.append(json.loads(line))
    name_index = build_name_index(rules_records)

    before = Counter()
    for r in records:
        for lk in r.get("links_out", []):
            before[(lk["target_type"], _v1_status(lk))] += 1

    params = {
        "links_input": str(v1_csv_path),
        "aliases_sha256": _sha256_file(aliases_path),
        "rebuild": False,
    }
    ctx = start_run("cleanse", "guidance-links-v2", params, paths)
    ctx.log(f"records: {len(records)}")

    # No rebuild for guidance — names were preserved
    # Resolve with typed aliases
    resolve_links(records, name_index, aliases, section_base_paths, id_table)

    after = Counter()
    for r in records:
        for lk in r.get("links_out", []):
            status = lk.get("status", "unresolved")
            after[(lk["target_type"], status)] += 1

    part_per_section = Counter()
    for r in records:
        for lk in r.get("links_out", []):
            if lk["target_type"] == "part" and lk.get("status") == "resolved":
                part_per_section[lk.get("resolved_to", "")] += 1

    unresolved_names = Counter()
    for r in records:
        for lk in r.get("links_out", []):
            if lk.get("status") == "unresolved" and lk["target_type"] in ("appendix", "part"):
                unresolved_names[lk["target"]] += 1

    v2_csv_path = processed / "guidance-crossrefs_v2.csv"
    with open(v2_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["from_paragraph_id", "target", "target_type",
                         "source", "status", "resolved_to", "candidates", "cites_rules"])
        for r in records:
            cites = r.get("cites_rules", False)
            for lk in r.get("links_out", []):
                writer.writerow([
                    r["paragraph_id"],
                    lk["target"],
                    lk["target_type"],
                    lk.get("source", ""),
                    lk.get("status", "unresolved"),
                    lk.get("resolved_to", ""),
                    ";".join(lk.get("candidates", [])),
                    cites,
                ])

    report = {
        "before": {f"{k[0]}_{k[1]}": v for k, v in sorted(before.items())},
        "after": {f"{k[0]}_{k[1]}": v for k, v in sorted(after.items())},
        "part_resolved_per_section": dict(part_per_section.most_common()),
        "unresolved_names": dict(unresolved_names.most_common()),
        "input_sha256": {
            "jsonl": _sha256_file(jsonl_path),
            "v1_csv": _sha256_file(v1_csv_path),
            "id_table": _sha256_file(id_table_path),
            "aliases": _sha256_file(aliases_path),
        },
        "output_sha256": {
            "v2_csv": _sha256_file(v2_csv_path),
        },
    }
    report_path = processed / "guidance-cleanse-report_v1.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    ctx.log(f"v2 csv: {v2_csv_path}")
    ctx.finish("ok", f"guidance links v2: {sum(after.values())} links, "
               f"{after.get(('part','resolved'),0)} part resolved")

    print(f"run folder: {ctx.run_dir.name}")
    print(f"\nBefore (v1):")
    for k in sorted(before):
        print(f"  {k[0]:12s} {k[1]:12s} {before[k]}")
    print(f"\nAfter (v2):")
    for k in sorted(after):
        print(f"  {k[0]:12s} {k[1]:12s} {after[k]}")
    print(f"\nPart resolved per section:")
    for bp, n in part_per_section.most_common():
        print(f"  {n:4d}  {bp}")
    print(f"\nUnresolved names ({len(unresolved_names)}):")
    for name, n in unresolved_names.most_common(20):
        print(f"  {n:4d}  {name}")
    print(f"\nOutputs:")
    print(f"  {v2_csv_path} sha256={report['output_sha256']['v2_csv']}")
    print(f"  {report_path}")
    print(f"\nV1 CSV unchanged: sha256={report['input_sha256']['v1_csv']}")


if __name__ == "__main__":
    main()
