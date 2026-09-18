"""
Build the unit table from processed corpora.

Usage::

    python -m code.index.units --config config/paths.yaml
"""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import yaml
from transformers import AutoTokenizer

TOKENIZER_NAME = "BAAI/bge-base-en-v1.5"


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def heading_line(record: dict) -> str:
    hp = record.get("heading_path", [])
    if hp:
        return " > ".join(hp)
    return record.get("section_title", "")


def unit_text(record: dict) -> str:
    return heading_line(record) + "\n" + record.get("text", "")


def build_route_map(routes_path: str) -> dict[str, str]:
    """Map section_base_path -> route name for rules sections."""
    with open(routes_path) as f:
        cfg = yaml.safe_load(f)
    mapping: dict[str, str] = {}
    for rname, rcfg in (cfg.get("routes") or {}).items():
        for sec in rcfg.get("rules_sections", []):
            mapping[sec] = rname
    for sec in (cfg.get("cross_cutting") or {}).get("rules_sections", []):
        mapping[sec] = "cross_cutting"
    return mapping


def main():
    ap = argparse.ArgumentParser(description="Build unit table")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--routes", default="config/routes.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        paths = yaml.safe_load(f)

    processed = Path(paths["processed"])
    index_dir = Path(paths["index"])
    index_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    route_map = build_route_map(args.routes)

    rules_path = processed / "rules-paragraphs_v1.jsonl"
    guidance_path = processed / "guidance-paragraphs_v1.jsonl"

    units = []
    counts = {"rules": 0, "guidance": 0}

    # Rules
    with open(rules_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r["text_type"] == "deleted":
                continue
            text = unit_text(r)
            record_only = r.get("text", "")
            record_toks = tok.encode(record_only, add_special_tokens=False)
            unit_toks = tok.encode(text, add_special_tokens=False)
            route = route_map.get(r["section_base_path"], "other")
            units.append({
                "unit_id": r["paragraph_id"],
                "corpus": "rules",
                "section_base_path": r["section_base_path"],
                "heading_path": r.get("heading_path", []),
                "seq": r["seq"],
                "route": route,
                "text": text,
                "token_count": len(record_toks),
                "unit_token_count": len(unit_toks),
            })
            counts["rules"] += 1

    # Guidance
    with open(guidance_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r["text_type"] == "deleted":
                continue
            text = unit_text(r)
            record_only = r.get("text", "")
            record_toks = tok.encode(record_only, add_special_tokens=False)
            unit_toks = tok.encode(text, add_special_tokens=False)
            route_field = r.get("route", [])
            route = route_field[0] if isinstance(route_field, list) and route_field else "other"
            units.append({
                "unit_id": r["paragraph_id"],
                "corpus": "guidance",
                "section_base_path": r["section_base_path"],
                "heading_path": r.get("heading_path", []),
                "seq": r["seq"],
                "route": route,
                "text": text,
                "token_count": len(record_toks),
                "unit_token_count": len(unit_toks),
            })
            counts["guidance"] += 1

    # Median of unit_token_count for rules (expected 44)
    rules_unit_tokens = [u["unit_token_count"] for u in units if u["corpus"] == "rules"]
    median_tc = statistics.median(rules_unit_tokens)
    window_size = max(64, round(median_tc / 64) * 64)

    # Write JSONL
    out_path = index_dir / "units_v1.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    # Manifest
    manifest = {
        "counts": counts,
        "total_units": len(units),
        "median_rules_token_count": median_tc,
        "window_size_d34": window_size,
        "tokenizer": TOKENIZER_NAME,
        "input_sha256": {
            "rules": _sha256_file(rules_path),
            "guidance": _sha256_file(guidance_path),
        },
        "output_sha256": _sha256_file(out_path),
    }
    manifest_path = index_dir / "units_v1.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"Units: {len(units)} (rules={counts['rules']}, guidance={counts['guidance']})")
    print(f"Median token count (rules): {median_tc}")
    print(f"Window size (D34): {window_size}")
    if window_size != 64:
        print("ERROR: D34 expects window size 64")
        exit(1)
    print(f"Output: {out_path}")
    print(f"sha256: {manifest['output_sha256']}")


if __name__ == "__main__":
    main()
