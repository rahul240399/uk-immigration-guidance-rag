"""
Shared helpers for the immigration rules pipeline.
Imported by parse_rules.py and freeze_rules.py — never duplicated.
"""

import re
from typing import Dict, Optional

from bs4 import NavigableString


def normalize_reference_key(rule_ref: str) -> str:
    """Normalize rule reference for lookup: upper-case, remove spaces and full stops."""
    if not rule_ref:
        return ""
    return re.sub(r'[\s.]', '', rule_ref.upper())


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_name(raw: str) -> str:
    """Lower-case, remove punctuation except spaces, collapse whitespace."""
    n = re.sub(r'[^\w\s]', '', raw.lower()).strip()
    return re.sub(r'\s+', ' ', n)


def strip_trailing_connectors(name: str) -> str:
    """Strip trailing connector words from appendix/part names."""
    connectors = {'with', 'of', 'and', 'for', 'the'}
    words = name.split()
    while words and words[-1].lower() in connectors:
        words.pop()
    return ' '.join(words)


def longest_prefix_match(target_norm: str, name_index: Dict[str, str]) -> Optional[str]:
    """Find longest prefix match in name index (min 3 chars)."""
    if len(target_norm) < 3:
        return None
    best, best_len = None, 0
    for idx_name, bp in name_index.items():
        if target_norm.startswith(idx_name) or idx_name.startswith(target_norm):
            ml = min(len(target_norm), len(idx_name))
            if ml > best_len and ml >= 3:
                best, best_len = bp, ml
    return best


def extract_text_to_first_list(element) -> str:
    """Extract element text up to first ol or ul child."""
    parts = []
    for child in element.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif hasattr(child, 'name'):
            if child.name in ['ol', 'ul']:
                break
            else:
                parts.append(child.get_text())
    return ''.join(parts)


# Identifier regex families (no IGNORECASE)
APPENDIX_PATTERN = re.compile(
    r'^(?P<ref>[A-Z]{1,6}(?:\s?\([A-Z]{1,4}\))?(?:-[A-Z]{1,6})*\.?\s?[A-Z]?\d+[A-Z]?'
    r'(?:\.\d+[A-Z]?)*(?:-SD)?)\.?(?=\s|[A-Z(]|$)')
PART_PATTERN = re.compile(
    r'^(?P<ref>\d{1,3}[A-Z]{0,3}\d?(?:-SD)?)\.?(?=\s|[A-Z(]|$)')
RANGE_PATTERN = re.compile(
    r'^(?P<ref>\d{1,3}[A-Z]{0,3}\s?-\s?\d{1,3}[A-Z]{0,3})\.?(?=\s|[A-Z]|$)')
DOTTED_PATTERN = re.compile(
    r'^(?P<ref>\d{1,3}(?:\.\d+)+)\.?(?=\s|[A-Z(]|$)')
SUBPARA_PATTERN = re.compile(
    r'^(?P<label>\((?:[a-z]{1,2}|[ivx]+|\d{1,2})\)(?:\([a-z0-9]+\))*)\s')

# Cross-reference patterns
APPENDIX_REF = re.compile(
    r'Appendix\s+([A-Z][\w-]*(?:\s(?:[A-Z][\w-]*|with|of|and|for|the)){0,5})')
PART_REF = re.compile(r'Part\s+(\d{1,2}[A-Z]?|[A-Z][a-z]+)')
PARAGRAPH_REF = re.compile(
    r'paragraphs?\s+((?:[A-Z]{1,4}\s?)?\d+[A-Z]{0,3}(?:-SD)?(?:\.\d+)*)')
