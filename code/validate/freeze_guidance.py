"""
Guidance validation and freezing script.
Mirrors the rules freeze (freeze_rules.py) with equivalent checks.

Checks:
    1. Schema: all fields present, paragraph_id unique, guidance-specific
       fields (publication_base_path, attachment_id, route, source,
       page_number, cites_rules) present.
    2. Text reconstruction:
       - HTML records: re-parse raw_html, extract text, compare.
       - PDF records: flag-only (no raw_html to reconstruct from).
    3. Containment: no child record's text (40+ chars) inside its
       attached_to parent's text.
    4. Cross-reference consistency: resolved=True ↔ resolved_to non-empty.

On failure: prints errors and exits 1 before writing any output.
On success: writes four files to processed/.

Usage::

    python -m code.validate.freeze_guidance <guidance-parse-folder>
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from bs4 import BeautifulSoup

from code.parse.common import (
    normalize_reference_key,
    normalize_whitespace,
    extract_text_to_first_list,
)

# ── Helpers ──────────────────────────────────────────────────────────


def get_parser_version() -> str:
    rules_path = Path('code/parse/parse_guidance.py')
    if not rules_path.exists():
        return "unknown"
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "file:" + hashlib.sha256(rules_path.read_bytes()).hexdigest()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Validate and freeze parsed guidance records")
    ap.add_argument("run_folder", help="Path to the guidance-parse run folder")
    ap.add_argument("--config", default="config/paths.yaml")
    args = ap.parse_args()

    parse_folder = Path(args.run_folder)
    if not parse_folder.exists():
        print(f"Error: folder {parse_folder} does not exist")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    processed_path = Path(config["processed"])
    raw_guidance_path = Path(config["raw_guidance"])

    guidance_jsonl = parse_folder / "guidance-paragraphs.jsonl"
    if not guidance_jsonl.exists():
        print(f"Error: {guidance_jsonl} does not exist")
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────
    print("Loading data...")
    records = []
    with open(guidance_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records")

    # Load raw guidance manifest for sha256
    raw_manifest_path = raw_guidance_path / "manifest.json"
    raw_manifest_sha256 = ""
    if raw_manifest_path.exists():
        raw_manifest_sha256 = hashlib.sha256(
            raw_manifest_path.read_bytes()).hexdigest()

    # ── CHECK 1: Schema ──────────────────────────────────────────────
    print("\n=== VALIDATION CHECKS ===")
    print("1. Schema check...")

    required_fields = {
        'paragraph_id', 'seq', 'rule_ref', 'rule_family', 'section_base_path',
        'section_title', 'heading_path', 'text', 'text_type', 'attached_to',
        'links_out', 'content_id', 'public_updated_at', 'snapshot_date',
        'licence', 'raw_html', 'raw_html_sha256', 'source_line', 'source_pos',
        'parse_confidence',
        # Guidance-specific
        'publication_base_path', 'attachment_id', 'route', 'source',
        'page_number', 'cites_rules',
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

    # ── CHECK 2: Text reconstruction ─────────────────────────────────
    print("2. Text check...")
    text_mismatches = []
    html_checked = 0
    pdf_skipped = 0

    for r in records:
        source = r.get('source', '')
        raw_html = r.get('raw_html', '')
        expected = r.get('text', '')
        tt = r.get('text_type')

        # PDF records: no raw_html to reconstruct from
        if source == 'pdf' or not raw_html:
            pdf_skipped += 1
            continue

        # table_row: synthesised from context outside raw_html (same as rules)
        if tt == 'table_row':
            continue

        html_checked += 1
        try:
            soup = BeautifulSoup(raw_html, 'html.parser')
            root = soup.find()
            if not root:
                continue

            if tt in ('rule', 'narrative', 'continuation', 'deleted',
                      'subparagraph', 'list_item'):
                extracted = extract_text_to_first_list(root)
            else:
                extracted = root.get_text()

            if normalize_whitespace(extracted) != normalize_whitespace(expected):
                text_mismatches.append(r.get('paragraph_id'))
        except Exception:
            text_mismatches.append(r.get('paragraph_id'))

    text_pass = len(text_mismatches) == 0
    print(f"   Text: {'PASS' if text_pass else 'FAIL'} "
          f"({html_checked} HTML checked, {len(text_mismatches)} mismatches, "
          f"{pdf_skipped} PDF skipped)")
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
    print(f"   Containment: {'PASS' if containment_pass else 'FAIL'} "
          f"({len(containment)} violations)")

    # ── CHECK 4: Cross-reference consistency ─────────────────────────
    print("4. Cross-ref consistency check...")
    xref_errors = 0
    for r in records:
        for lk in r.get('links_out', []):
            resolved = lk.get('resolved', False)
            resolved_to = lk.get('resolved_to', '')
            if resolved and not resolved_to:
                xref_errors += 1
            if not resolved and resolved_to:
                xref_errors += 1

    xref_pass = xref_errors == 0
    print(f"   Cross-refs: {'PASS' if xref_pass else 'FAIL'} "
          f"({xref_errors} inconsistencies)")

    # ── Gate ─────────────────────────────────────────────────────────
    all_pass = schema_pass and text_pass and containment_pass and xref_pass
    if not all_pass:
        print("\nValidation FAILED — no output written.")
        sys.exit(1)

    # ── Write outputs ────────────────────────────────────────────────
    print("\n=== GENERATING OUTPUTS ===")
    processed_path.mkdir(parents=True, exist_ok=True)

    # 1. guidance-paragraphs_v1.jsonl (records unchanged)
    out_jsonl = processed_path / "guidance-paragraphs_v1.jsonl"
    print(f"Writing {out_jsonl}...")
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 2. guidance-identifier-table_v1.json
    print("Building identifier table...")
    by_pub = defaultdict(dict)
    ref_pub_count = defaultdict(set)

    for r in records:
        if r.get('rule_ref') and r.get('text_type') in ('rule', 'deleted', 'subparagraph'):
            pub = r['publication_base_path']
            norm = normalize_reference_key(r['rule_ref'])
            by_pub[pub][norm] = r['paragraph_id']
            ref_pub_count[norm].add(pub)

    unique_global = {}
    for norm, pubs in ref_pub_count.items():
        if len(pubs) == 1:
            pub = next(iter(pubs))
            unique_global[norm] = by_pub[pub][norm]

    id_table = {
        'by_publication': {k: dict(v) for k, v in by_pub.items()},
        'unique_global': unique_global,
    }
    id_path = processed_path / "guidance-identifier-table_v1.json"
    print(f"Writing {id_path}...")
    with open(id_path, 'w', encoding='utf-8') as f:
        json.dump(id_table, f, ensure_ascii=False, indent=2)

    # 3. guidance-crossrefs_v1.csv
    csv_path = processed_path / "guidance-crossrefs_v1.csv"
    print(f"Writing {csv_path}...")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        f.write("from_paragraph_id,target,target_type,source,resolved,"
                "resolved_to,candidates,cites_rules\n")
        for r in records:
            cites = r.get('cites_rules', False)
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
                    f"{cands},"
                    f"{cites}\n"
                )

    # 4. guidance-manifest_v1.json
    print("Generating manifest...")
    type_counts = Counter(r['text_type'] for r in records)
    source_counts = Counter(r['source'] for r in records)
    route_counts = Counter()
    for r in records:
        for rt in r.get('route', []):
            route_counts[rt] += 1
    pub_counts = Counter(r['publication_base_path'] for r in records)

    # Resolution rates by type
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

    # Confidence breakdown
    conf_counts = Counter(r['parse_confidence'] for r in records)
    cites_count = sum(1 for r in records if r.get('cites_rules'))

    parser_version = get_parser_version()
    date_frozen = datetime.now(timezone.utc).isoformat()

    manifest = {
        'snapshot_date': records[0]['snapshot_date'] if records else '',
        'raw_manifest_sha256': raw_manifest_sha256,
        'parser_version': parser_version,
        'date_frozen': date_frozen,
        'record_counts': {
            'total': len(records),
            'by_type': dict(type_counts),
            'by_source': dict(source_counts),
            'by_route': dict(route_counts),
            'by_publication': dict(pub_counts),
        },
        'resolution_rates_by_type': res_rates,
        'confidence_distribution': dict(conf_counts),
        'records_citing_rules': cites_count,
        'validation_results': {
            'schema_check': schema_pass,
            'text_check': text_pass,
            'text_html_checked': html_checked,
            'text_pdf_skipped': pdf_skipped,
            'text_mismatches': len(text_mismatches),
            'containment_check': containment_pass,
            'xref_consistency_check': xref_pass,
        },
        'run_folder_used': str(parse_folder),
        'publications_count': len(pub_counts),
        'total_records': len(records),
    }

    mp = processed_path / "guidance-manifest_v1.json"
    print(f"Writing {mp}...")
    with open(mp, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    print(f"Records:       {len(records)}")
    print(f"  HTML:        {source_counts.get('html', 0)}")
    print(f"  PDF:         {source_counts.get('pdf', 0)}")
    print(f"Publications:  {len(pub_counts)}")
    print(f"Citing rules:  {cites_count}")
    print(f"Schema:        PASS")
    print(f"Text:          PASS ({html_checked} HTML verified, "
          f"{pdf_skipped} PDF skipped)")
    print(f"Containment:   PASS")
    print(f"Cross-refs:    PASS")
    print(f"Confidence:    {dict(conf_counts)}")
    print(f"Resolution:    {res_rates}")
    print(f"Parser ver:    {parser_version[:12]}...")
    print(f"Frozen:        {date_frozen}")

    total_links = sum(len(r.get('links_out', [])) for r in records)
    print(f"\nTotal links:   {total_links}")
    for tt in sorted(res_stats):
        s = res_stats[tt]
        print(f"  {tt}: {s['resolved']}/{s['total']} = "
              f"{res_rates.get(tt, 0):.1f}%")

    print(f"\nOutputs:")
    print(f"  {out_jsonl}")
    print(f"  {id_path}")
    print(f"  {csv_path}")
    print(f"  {mp}")
    print("\nFreeze complete!")


if __name__ == '__main__':
    main()
