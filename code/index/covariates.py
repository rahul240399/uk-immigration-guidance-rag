"""
Compute per-record covariates for Rules records.

Usage::

    python -m code.index.covariates --config config/paths.yaml
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from transformers import AutoTokenizer

TOKENIZER_NAME = "BAAI/bge-base-en-v1.5"
NEW_PATTERN = re.compile(r"^[A-Z]{1,6}\s?\d+\.\d+")


def main():
    ap = argparse.ArgumentParser(description="Compute rules covariates")
    ap.add_argument("--config", default="config/paths.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        paths = yaml.safe_load(f)

    processed = Path(paths["processed"])
    rules_path = processed / "rules-paragraphs_v1.jsonl"
    links_path = processed / "rules-crossrefs_v2.csv"

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Load records
    records = []
    by_section: dict[str, list[dict]] = defaultdict(list)
    with open(rules_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            records.append(r)
            by_section[r["section_base_path"]].append(r)

    # Load v2 links
    links_from: dict[str, int] = Counter()
    links_resolved: dict[str, int] = Counter()
    with open(links_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["from_paragraph_id"]
            links_from[pid] += 1
            if row.get("status") == "resolved":
                links_resolved[pid] += 1

    # Drafting style per section (D37)
    section_style: dict[str, str] = {}
    for sec, sec_records in by_section.items():
        rule_refs = [r["rule_ref"] for r in sec_records
                     if r["text_type"] in ("rule", "deleted") and r.get("rule_ref")]
        if not rule_refs:
            section_style[sec] = "list"
        else:
            new_count = sum(1 for ref in rule_refs if NEW_PATTERN.match(ref))
            section_style[sec] = "new" if new_count >= len(rule_refs) / 2 else "old"

    # Siblings per heading
    heading_siblings: dict[str, int] = Counter()
    for sec, sec_records in by_section.items():
        for r in sec_records:
            hp = tuple(r.get("heading_path", []))
            key = sec + "|" + str(hp)
            heading_siblings[key] += 1

    # Write CSV
    out_path = processed / "rules-covariates_v1.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paragraph_id", "section_base_path", "drafting_style",
            "token_count", "section_records", "siblings_same_heading",
            "links_out_v2", "links_resolved_v2",
        ])
        writer.writeheader()
        for r in records:
            hp = tuple(r.get("heading_path", []))
            heading_key = r["section_base_path"] + "|" + str(hp)

            text = (" > ".join(r.get("heading_path", [])) or r.get("section_title", "")) + "\n" + r["text"]
            toks = tok.encode(text, add_special_tokens=False)

            writer.writerow({
                "paragraph_id": r["paragraph_id"],
                "section_base_path": r["section_base_path"],
                "drafting_style": section_style.get(r["section_base_path"], "list"),
                "token_count": len(toks),
                "section_records": len(by_section[r["section_base_path"]]),
                "siblings_same_heading": heading_siblings[heading_key],
                "links_out_v2": links_from.get(r["paragraph_id"], 0),
                "links_resolved_v2": links_resolved.get(r["paragraph_id"], 0),
            })

    # Style counts
    style_counts = Counter(section_style.values())
    print(f"Covariates: {len(records)} rows → {out_path}")
    print(f"Section style counts: {dict(style_counts)}")
    for style in sorted(style_counts):
        secs = [s for s, st in section_style.items() if st == style]
        print(f"  {style}: {len(secs)} sections")


if __name__ == "__main__":
    main()
