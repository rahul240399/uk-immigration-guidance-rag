"""
Typed link resolution and rebuild for v2 cross-reference tables.

resolve_section_name uses the typed alias blocks (appendix, part, absent)
so that numeric names like "8" resolve correctly by ref_type.
"""

from pathlib import Path
from typing import Optional

import yaml

from code.parse.common import (
    APPENDIX_REF,
    PART_REF,
    PARAGRAPH_REF,
    normalize_name,
    strip_trailing_connectors,
    longest_prefix_match,
    normalize_reference_key,
)


# ── Alias loading ────────────────────────────────────────────────────

def load_typed_aliases(
    alias_path: str = "config/section_aliases.yaml",
) -> dict:
    """Load the structured alias file.

    Returns::

        {
            "appendix": {norm_name: section_path, ...},
            "part":     {norm_name: section_path, ...},
            "absent":   {"appendix": [norm_name, ...], "part": [...]},
        }
    """
    p = Path(alias_path)
    if not p.exists():
        return {"appendix": {}, "part": {}, "absent": {"appendix": [], "part": []}}
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    return {
        "appendix": raw.get("appendix") or {},
        "part": raw.get("part") or {},
        "absent": raw.get("absent") or {"appendix": [], "part": []},
    }


# ── Name normalisation helper ────────────────────────────────────────

import re as _re

def _collapse_hyphen_spaces(s: str) -> str:
    """Collapse whitespace around hyphens: 'FM- SE' and 'FM -SE' -> 'FM-SE'."""
    return _re.sub(r'\s*-\s*', '-', s)


# ── Typed section resolver ───────────────────────────────────────────

def resolve_section_name(
    raw_name: str,
    ref_type: str,
    name_index: dict,
    aliases: dict,
    section_base_paths: set,
) -> tuple[str, Optional[str]]:
    """Resolve an appendix or part name.

    Returns (status, resolved_to) where status is one of
    ``"resolved"``, ``"absent"``, ``"unresolved"``.
    """
    clean = strip_trailing_connectors(raw_name)
    # Collapse whitespace around hyphens: "FM- SE" -> "FM-SE"
    clean = _collapse_hyphen_spaces(clean)
    norm = normalize_name(clean)

    # 1. Longest-prefix match on the title index
    match = longest_prefix_match(norm, name_index)
    if match:
        return "resolved", match

    # 2. Typed alias block
    typed_block = aliases.get(ref_type, {})
    if norm in typed_block:
        val = typed_block[norm]
        if val in section_base_paths:
            return "resolved", val
        # Path not in corpus but alias exists — still mark resolved
        return "resolved", val

    # 3. Typed absent list
    absent_list = (aliases.get("absent") or {}).get(ref_type, [])
    if norm in absent_list:
        return "absent", None

    return "unresolved", None


# ── Link rebuild ─────────────────────────────────────────────────────

def rebuild_section_links(record: dict) -> list[dict]:
    """Re-extract appendix and part links from record text.

    Drops existing appendix/part links and re-extracts from text using
    APPENDIX_REF and PART_REF, one link per match in text order.
    Paragraph and hyperlink links are kept unchanged.
    table_row records get no links (decision D40).
    """
    if record.get("text_type") == "table_row":
        return []

    # Keep paragraph and hyperlink links
    kept = [
        lk for lk in record.get("links_out", [])
        if lk["target_type"] in ("paragraph", "hyperlink")
    ]

    text = record.get("text", "")
    new_links: list[dict] = []

    # Collect all matches with their positions for text-order output
    matches: list[tuple[int, str, str]] = []  # (pos, target, target_type)
    for m in APPENDIX_REF.finditer(text):
        matches.append((m.start(), m.group(1), "appendix"))
    for m in PART_REF.finditer(text):
        matches.append((m.start(), m.group(1), "part"))
    matches.sort(key=lambda x: x[0])

    for _, target, tt in matches:
        new_links.append({
            "target": target,
            "target_type": tt,
            "source": "regex",
            "resolved": False,
        })

    return kept + new_links


# ── Full resolution (paragraph links) ───────────────────────────────
# Reuses the identifier-table approach from freeze_rules.resolve_all_links.

def resolve_links(
    records: list[dict],
    name_index: dict,
    aliases: dict,
    section_base_paths: set,
    id_table: dict | None = None,
) -> None:
    """Resolve all links in-place.

    *   paragraph links: same-section → mentioned-section → unique-global
    *   appendix / part links: typed resolver
    *   hyperlinks: resolved if href is a known section

    If *id_table* is provided (``{by_section, unique_global}``), it is
    used directly; otherwise one is built from the records.
    """
    from collections import defaultdict

    if id_table is not None:
        by_section = id_table.get("by_section", {})
        unique_global = id_table.get("unique_global", {})
        # Build global_ref for candidate lookup
        global_ref: dict[str, list] = defaultdict(list)
        for bp, refs in by_section.items():
            for norm, pid in refs.items():
                global_ref[norm].append((bp, pid))
    else:
        by_section: dict = defaultdict(dict)
        global_ref = defaultdict(list)
        # Track seq to keep lowest on duplicate keys
        _seq_by_key: dict = defaultdict(lambda: defaultdict(lambda: float('inf')))
        for r in records:
            if r.get("rule_ref") and r.get("text_type") in ("rule", "deleted", "subparagraph"):
                norm = normalize_reference_key(r["rule_ref"])
                bp = r["section_base_path"]
                seq = r.get("seq", 0)
                if seq < _seq_by_key[bp][norm]:
                    _seq_by_key[bp][norm] = seq
                    by_section[bp][norm] = r["paragraph_id"]
                global_ref[norm].append((bp, r["paragraph_id"]))
        unique_global = {}
        for norm, entries in global_ref.items():
            secs = set(e[0] for e in entries)
            if len(secs) == 1:
                unique_global[norm] = by_section[next(iter(secs))][norm]

    for r in records:
        record_bp = r["section_base_path"]
        links = r.get("links_out", [])

        # Pre-compute mentioned sections from appendix/part links
        mentioned: set[str] = set()
        for lk in links:
            if lk["target_type"] in ("appendix", "part"):
                status, rt = resolve_section_name(
                    lk["target"], lk["target_type"],
                    name_index, aliases, section_base_paths)
                if status == "resolved" and rt:
                    mentioned.add(rt)

        for lk in links:
            tt = lk["target_type"]
            src = lk.get("source", "")

            if tt == "paragraph":
                norm_target = normalize_reference_key(lk["target"])
                resolved_to = None
                candidates = None

                if norm_target in by_section.get(record_bp, {}):
                    resolved_to = by_section[record_bp][norm_target]
                if not resolved_to:
                    for sbp in mentioned:
                        if norm_target in by_section.get(sbp, {}):
                            resolved_to = by_section[sbp][norm_target]
                            break
                if not resolved_to:
                    if norm_target in unique_global:
                        resolved_to = unique_global[norm_target]
                    else:
                        entries = global_ref.get(norm_target, [])
                        secs = sorted(set(e[0] for e in entries))
                        if len(secs) > 1:
                            candidates = secs

                if resolved_to:
                    lk["resolved"] = True
                    lk["resolved_to"] = resolved_to
                    lk["status"] = "resolved"
                    lk.pop("candidates", None)
                else:
                    lk["resolved"] = False
                    lk.pop("resolved_to", None)
                    lk["status"] = "unresolved"
                    if candidates:
                        lk["candidates"] = candidates

            elif tt in ("appendix", "part"):
                status, resolved_to = resolve_section_name(
                    lk["target"], tt, name_index, aliases, section_base_paths)
                lk["status"] = status
                if status == "resolved" and resolved_to:
                    lk["resolved"] = True
                    lk["resolved_to"] = resolved_to
                elif status == "absent":
                    lk["resolved"] = False
                    lk.pop("resolved_to", None)
                else:
                    lk["resolved"] = False
                    lk.pop("resolved_to", None)
                lk.pop("candidates", None)

            elif src == "hyperlink":
                href = lk["target"]
                if href in section_base_paths:
                    lk["resolved"] = True
                    lk["resolved_to"] = href
                    lk["status"] = "resolved"
                else:
                    lk["resolved"] = False
                    lk.pop("resolved_to", None)
                    lk["status"] = "unresolved"
