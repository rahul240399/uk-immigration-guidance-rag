#!/usr/bin/env python3
"""
Rules validation and freezing script.
Validates parsed rules data, re-resolves all cross-reference links,
and creates frozen v1 outputs.

Usage:
    python -m code.validate.freeze_rules <rules-parse-folder>

Checks:
    1. Schema: all fields present, paragraph_id unique
    2. Text: raw_html re-derived text matches stored text
    3. Containment: no record's text (40+ chars) appears inside its attached_to parent

On failure: prints errors and exits 1 before writing any output.
On success: writes four files to processed/ and manifests/.
"""

import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, Counter
import subprocess
import re
from bs4 import BeautifulSoup, NavigableString
import yaml

from code.parse.common import (
    normalize_reference_key,
    normalize_whitespace,
    normalize_name,
    strip_trailing_connectors,
    longest_prefix_match,
    extract_text_to_first_list,
)


def get_parser_version() -> str:
    """Get a version identifier for the parser. Uses git HEAD if available, else file hash."""
    rules_path = Path('code/parse/parse_rules.py')
    if not rules_path.exists():
        print("ERROR: code/parse/parse_rules.py does not exist")
        sys.exit(1)

    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        commit_hash = result.stdout.strip()
        if commit_hash:
            return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: sha256 of the parser file itself
    import hashlib as _hl
    return "file:" + _hl.sha256(rules_path.read_bytes()).hexdigest()


# ── Name index / alias helpers ───────────────────────────────────────

def build_name_index(records):
    """Build name_index from section_title values in the records."""
    name_index = {}  # normalized name -> section_base_path
    seen = set()

    for r in records:
        bp = r['section_base_path']
        if bp in seen:
            continue
        seen.add(bp)
        title = r['section_title']

        for prefix in ['Appendix ', 'Part ', 'appendix ', 'part ']:
            pos = title.lower().find(prefix.lower())
            if pos >= 0:
                after = title[pos + len(prefix):]
                name = after.split(':', 1)[0].strip()
                # Drop bracketed suffixes
                name = re.sub(r'\s*\([^)]*\)\s*', ' ', name).strip()
                norm = normalize_name(name)
                if norm and len(norm) >= 2:
                    name_index[norm] = bp
                break

    return name_index


def load_section_aliases():
    alias_path = Path("config/section_aliases.yaml")
    if alias_path.exists():
        with open(alias_path) as f:
            return yaml.safe_load(f) or {}
    return {}




def resolve_section_name(raw_name: str, name_index: dict, aliases: dict,
                         section_base_paths: set):
    """Resolve an appendix or part name to a section_base_path or 'absent'."""
    clean = strip_trailing_connectors(raw_name)
    norm = normalize_name(clean)

    # Longest prefix match
    match = longest_prefix_match(norm, name_index)
    if match:
        return match

    # Aliases
    if norm in aliases:
        val = aliases[norm]
        if val == "absent":
            return "absent"
        if val in section_base_paths:
            return val

    return None


# ── Link resolution pass ─────────────────────────────────────────────

def resolve_all_links(records, name_index, aliases, section_base_paths):
    """Re-resolve every link in every record. Mutates links in place."""

    # Build identifier table from all records (normalised keys)
    id_by_section = defaultdict(dict)   # section_bp -> {norm_ref -> paragraph_id}
    global_ref = defaultdict(list)      # norm_ref -> [(section_bp, paragraph_id)]

    for r in records:
        if r.get('rule_ref') and r.get('text_type') in ['rule', 'deleted', 'subparagraph']:
            norm = normalize_reference_key(r['rule_ref'])
            bp = r['section_base_path']
            id_by_section[bp][norm] = r['paragraph_id']
            global_ref[norm].append((bp, r['paragraph_id']))

    # Unique corpus-wide: only refs in exactly one section
    unique_global = {}
    for norm, entries in global_ref.items():
        sections_set = set(e[0] for e in entries)
        if len(sections_set) == 1:
            unique_global[norm] = entries[0][1]  # paragraph_id

    for r in records:
        record_bp = r['section_base_path']
        links = r.get('links_out', [])

        # Pre-compute which sections other links in the record mention
        # (for paragraph resolution step 2)
        mentioned_sections = set()
        for lk in links:
            if lk['target_type'] in ('appendix', 'part'):
                resolved = resolve_section_name(
                    lk['target'], name_index, aliases, section_base_paths)
                if resolved and resolved != 'absent':
                    mentioned_sections.add(resolved)

        for link in links:
            tt = link['target_type']
            src = link.get('source', '')
            resolved_to = None
            candidates = None

            if tt == 'paragraph':
                norm_target = normalize_reference_key(link['target'])

                # 1. Same section
                if norm_target in id_by_section.get(record_bp, {}):
                    resolved_to = id_by_section[record_bp][norm_target]

                # 2. Section named by another link in same record
                if not resolved_to:
                    for sbp in mentioned_sections:
                        if norm_target in id_by_section.get(sbp, {}):
                            resolved_to = id_by_section[sbp][norm_target]
                            break

                # 3. Unique corpus-wide
                if not resolved_to:
                    if norm_target in unique_global:
                        resolved_to = unique_global[norm_target]
                    else:
                        # Multiple sections?
                        entries = global_ref.get(norm_target, [])
                        secs = sorted(set(e[0] for e in entries))
                        if len(secs) > 1:
                            candidates = secs

            elif tt in ('appendix', 'part'):
                resolved_to = resolve_section_name(
                    link['target'], name_index, aliases, section_base_paths)

            elif src == 'hyperlink':
                href = link['target']
                # Check if the href base path exists in corpus
                if href in section_base_paths:
                    resolved_to = href

            # Write back
            if resolved_to:
                link['resolved'] = True
                link['resolved_to'] = resolved_to
                # Remove stale candidates
                link.pop('candidates', None)
            else:
                link['resolved'] = False
                link.pop('resolved_to', None)
                if candidates:
                    link['candidates'] = candidates
                else:
                    link.pop('candidates', None)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Validate and freeze parsed rules")
    ap.add_argument("run_folder", help="Path to the rules-parse run folder")
    ap.add_argument("--config", default="config/paths.yaml", help="Path to paths.yaml")
    args = ap.parse_args()

    parse_folder = Path(args.run_folder)
    if not parse_folder.exists():
        print(f"Error: folder {parse_folder} does not exist")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_rules_path = Path(config["raw_rules"])
    processed_path = Path(config["processed"])

    rules_jsonl = parse_folder / "rules-paragraphs.jsonl"
    raw_manifest_path = raw_rules_path / "manifest.json"

    if not rules_jsonl.exists():
        print(f"Error: {rules_jsonl} does not exist"); sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────
    print("Loading data...")
    records = []
    with open(rules_jsonl, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records")

    with open(raw_manifest_path, 'r', encoding='utf-8') as f:
        raw_manifest_data = f.read()
        raw_manifest_sha256 = hashlib.sha256(raw_manifest_data.encode()).hexdigest()
        raw_manifest_json = json.loads(raw_manifest_data)

    # ── Resolution pass ──────────────────────────────────────────────
    print("Building indexes and resolving links...")
    section_base_paths = set(r['section_base_path'] for r in records)
    name_index = build_name_index(records)
    aliases = load_section_aliases()

    resolve_all_links(records, name_index, aliases, section_base_paths)

    # ── CHECK 1: Schema ──────────────────────────────────────────────
    print("\n=== VALIDATION CHECKS ===")
    print("1. Schema check...")

    required_fields = {
        'paragraph_id', 'seq', 'rule_ref', 'rule_family', 'section_base_path',
        'section_title', 'heading_path', 'text', 'text_type', 'attached_to',
        'links_out', 'content_id', 'public_updated_at', 'snapshot_date',
        'licence', 'raw_html', 'raw_html_sha256', 'source_line', 'source_pos',
        'parse_confidence'
    }

    schema_errors = []
    paragraph_ids = set()

    for i, r in enumerate(records):
        missing = required_fields - set(r.keys())
        if missing:
            schema_errors.append(f"Record {i}: Missing {missing}")
        pid = r.get('paragraph_id')
        if pid:
            if pid in paragraph_ids:
                schema_errors.append(f"Record {i}: Duplicate paragraph_id '{pid}'")
            paragraph_ids.add(pid)

    schema_pass = len(schema_errors) == 0
    print(f"   Schema: {'PASS' if schema_pass else 'FAIL'}")
    if schema_errors:
        for e in schema_errors[:10]:
            print(f"     {e}")
        if len(schema_errors) > 10:
            print(f"     ... and {len(schema_errors) - 10} more")

    # ── CHECK 2: Text ────────────────────────────────────────────────
    print("2. Text check...")
    text_mismatches = []

    for r in records:
        raw_html = r.get('raw_html', '')
        expected = r.get('text', '')
        tt = r.get('text_type')
        if not raw_html:
            continue

        try:
            soup = BeautifulSoup(raw_html, 'html.parser')
            root = soup.find()
            if not root:
                continue

            if tt in ('rule', 'narrative', 'continuation', 'deleted',
                      'subparagraph', 'list_item'):
                extracted = extract_text_to_first_list(root)
            elif tt == 'table_row':
                continue   # synthesised from context outside raw_html
            else:
                extracted = root.get_text()

            if normalize_whitespace(extracted) != normalize_whitespace(expected):
                text_mismatches.append(r.get('paragraph_id'))
        except Exception as e:
            text_mismatches.append(r.get('paragraph_id'))

    text_pass = len(text_mismatches) == 0
    print(f"   Text: {'PASS' if text_pass else 'FAIL'} ({len(text_mismatches)} mismatches)")
    if text_mismatches:
        for pid in text_mismatches[:10]:
            print(f"     {pid}")

    # ── CHECK 3: Containment ─────────────────────────────────────────
    print("3. Containment check...")
    pid_text = {r['paragraph_id']: r['text'] for r in records}
    containment = []
    for r in records:
        txt = r['text'].strip()
        att = r.get('attached_to')
        if len(txt) >= 40 and att:
            ptxt = pid_text.get(att, "")
            if ptxt and len(ptxt) > len(txt) and txt in ptxt:
                containment.append(r['paragraph_id'])

    containment_pass = len(containment) == 0
    print(f"   Containment: {'PASS' if containment_pass else 'FAIL'} ({len(containment)} violations)")

    # ── Gate ─────────────────────────────────────────────────────────
    if not (schema_pass and text_pass and containment_pass):
        print("\nValidation FAILED — no output written.")
        sys.exit(1)

    # ── Write outputs ────────────────────────────────────────────────
    print("\n=== GENERATING OUTPUTS ===")
    processed_path.mkdir(parents=True, exist_ok=True)

    # 1. rules-paragraphs_v1.jsonl (with updated links_out)
    out_jsonl = processed_path / "rules-paragraphs_v1.jsonl"
    print(f"Writing {out_jsonl}...")
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 2. Identifier table
    by_section = defaultdict(dict)
    ref_sections = defaultdict(set)
    for r in records:
        if r.get('rule_ref') and r.get('text_type') in ('rule', 'deleted', 'subparagraph'):
            bp = r['section_base_path']
            norm = normalize_reference_key(r['rule_ref'])
            by_section[bp][norm] = r['paragraph_id']
            ref_sections[norm].add(bp)

    unique_global = {}
    for norm, secs in ref_sections.items():
        if len(secs) == 1:
            bp = next(iter(secs))
            unique_global[norm] = by_section[bp][norm]

    id_table = {
        'by_section': {k: dict(v) for k, v in by_section.items()},
        'unique_global': unique_global
    }
    id_path = processed_path / "rules-identifier-table_v1.json"
    print(f"Writing {id_path}...")
    with open(id_path, 'w', encoding='utf-8') as f:
        json.dump(id_table, f, ensure_ascii=False, indent=2)

    # 3. crossrefs_v1.csv
    csv_path = processed_path / "crossrefs_v1.csv"
    print(f"Writing {csv_path}...")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("from_paragraph_id,target,target_type,source,resolved,resolved_to,candidates\n")
        for r in records:
            for lk in r.get('links_out', []):
                cands = ';'.join(lk.get('candidates', []))
                rt = lk.get('resolved_to', '')
                f.write(
                    f"{r['paragraph_id']},"
                    f"{lk['target']},"
                    f"{lk['target_type']},"
                    f"{lk['source']},"
                    f"{lk.get('resolved', False)},"
                    f"{rt},"
                    f"{cands}\n"
                )

    # 4. Manifest
    print("Generating manifest...")
    type_counts = Counter(r['text_type'] for r in records)
    section_counts = Counter(r['section_base_path'] for r in records)

    res_stats = defaultdict(lambda: {'total': 0, 'resolved': 0})
    for r in records:
        for lk in r.get('links_out', []):
            tt = lk['target_type']
            res_stats[tt]['total'] += 1
            if lk.get('resolved', False):
                res_stats[tt]['resolved'] += 1

    res_rates = {}
    for tt, s in res_stats.items():
        if s['total'] > 0:
            res_rates[tt] = round(s['resolved'] / s['total'] * 100, 2)

    parser_version = get_parser_version()
    snapshot_date = raw_manifest_json.get('snapshot_date', '')
    date_frozen = datetime.now(timezone.utc).isoformat()

    manifest = {
        'snapshot_date': snapshot_date,
        'raw_manifest_sha256': raw_manifest_sha256,
        'parser_version': parser_version,
        'record_counts': {
            'total': len(records),
            'by_type': dict(type_counts),
            'by_section': dict(section_counts)
        },
        'resolution_rates_by_type': res_rates,
        'validation_results': {
            'schema_check': schema_pass,
            'text_check': text_pass,
            'containment_check': containment_pass
        },
        'run_folder_used': str(parse_folder),
        'date_frozen': date_frozen,
        'sections_count': len(section_counts),
        'total_records': len(records)
    }

    mp = processed_path / "manifest_v1.json"
    print(f"Writing {mp}...")
    with open(mp, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    md = Path("manifests")
    md.mkdir(parents=True, exist_ok=True)
    mr = md / "manifest_v1.json"
    print(f"Writing {mr}...")
    with open(mr, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    print(f"Records: {len(records)}")
    print(f"Sections: {len(section_counts)}")
    print(f"Schema:      PASS")
    print(f"Text:        PASS (0 mismatches)")
    print(f"Containment: PASS (0 violations)")
    print(f"Parser ver:  {parser_version[:12]}...")
    print(f"Snapshot:    {snapshot_date}")
    print(f"Frozen:      {date_frozen}")

    total_links = sum(len(r.get('links_out', [])) for r in records)
    print(f"\nTotal links: {total_links}")
    for tt in sorted(res_stats):
        s = res_stats[tt]
        print(f"  {tt}: {s['resolved']}/{s['total']} = {res_rates.get(tt, 0):.1f}%")

    print("\nFreeze complete!")


if __name__ == '__main__':
    main()
