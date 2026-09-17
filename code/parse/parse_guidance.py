"""
Caseworker Guidance Parser
--------------------------
Parses HTML and PDF guidance attachments to structured paragraph records,
resolving cross-references against the frozen Rules identifier table.

Input:  raw guidance folder (config/paths.yaml → raw_guidance),
        Rules identifier table (processed/rules-identifier-table_v1.json),
        fetch plan CSV (manifests/<date>_corpus_guidance-fetch-plan.csv).
Output: interim/<stamp>_guidance-parse/guidance-paragraphs.jsonl
        interim/<stamp>_guidance-parse/guidance-parse-report.json

Licence: Contains public sector information licensed under the
         Open Government Licence v3.0.
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from bs4 import BeautifulSoup, NavigableString, Tag

from code.parse.common import (
    normalize_reference_key,
    normalize_whitespace,
    normalize_name,
    strip_trailing_connectors,
    longest_prefix_match,
    extract_text_to_first_list,
    APPENDIX_PATTERN, PART_PATTERN, RANGE_PATTERN, DOTTED_PATTERN,
    SUBPARA_PATTERN, APPENDIX_REF, PART_REF, PARAGRAPH_REF,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Data contract ────────────────────────────────────────────────────

@dataclass
class GuidanceRecord:
    paragraph_id: str
    seq: int
    rule_ref: Optional[str]
    rule_family: Optional[str]
    section_base_path: str        # publication base_path
    section_title: str
    heading_path: List[str]
    text: str
    text_type: str
    attached_to: Optional[str]
    links_out: List[Dict]
    content_id: str
    public_updated_at: str
    snapshot_date: str
    licence: str
    raw_html: str
    raw_html_sha256: str
    source_line: Optional[int]
    source_pos: Optional[int]
    parse_confidence: str
    # Guidance-specific fields
    publication_base_path: str
    attachment_id: str
    route: List[str]
    source: str                   # 'html' or 'pdf'
    page_number: Optional[int]
    cites_rules: bool


# ── Rules cross-reference resolver ───────────────────────────────────

class RulesResolver:
    """Look up rule references against the frozen Rules identifier table."""

    def __init__(self, id_table_path: str, section_list: list):
        with open(id_table_path) as f:
            table = json.load(f)
        self.by_section = table.get("by_section", {})
        self.unique_global = table.get("unique_global", {})
        self.section_paths = set(section_list)

        # Build name index from section paths
        self.name_index: Dict[str, str] = {}
        for bp in self.section_paths:
            title_part = bp.rstrip("/").split("/")[-1]
            # Extract after 'immigration-rules-appendix-' or 'immigration-rules-part-'
            for prefix in ["immigration-rules-appendix-", "immigration-rules-part-"]:
                if prefix in title_part:
                    name = title_part.split(prefix, 1)[-1]
                    norm = normalize_name(name.replace("-", " "))
                    if norm and len(norm) >= 3:
                        self.name_index[norm] = bp
                    break

        # Load section aliases
        self.aliases: Dict[str, str] = {}
        try:
            with open("config/section_aliases.yaml") as f:
                self.aliases = yaml.safe_load(f) or {}
        except FileNotFoundError:
            pass

    def resolve_paragraph(self, ref: str, record_text: str = "") -> Optional[str]:
        """Resolve a paragraph reference to a Rules paragraph_id."""
        norm = normalize_reference_key(ref)

        # unique_global first
        if norm in self.unique_global:
            return self.unique_global[norm]

        # Try section named in the same sentence
        for pattern in [APPENDIX_REF, PART_REF]:
            for m in pattern.finditer(record_text):
                section_name = strip_trailing_connectors(m.group(1))
                section_bp = self._resolve_section_name(section_name)
                if section_bp and section_bp in self.by_section:
                    section_refs = self.by_section[section_bp]
                    if norm in section_refs:
                        return section_refs[norm]

        return None

    def resolve_section(self, name: str) -> Optional[str]:
        """Resolve an appendix or part name to a section base_path."""
        return self._resolve_section_name(name)

    def _resolve_section_name(self, raw_name: str) -> Optional[str]:
        clean = strip_trailing_connectors(raw_name)
        norm = normalize_name(clean)
        match = longest_prefix_match(norm, self.name_index)
        if match:
            return match
        if norm in self.aliases:
            val = self.aliases[norm]
            if val == "absent":
                return "absent"
            if val in self.section_paths:
                return val
        return None

    def is_rules_section(self, href: str) -> Optional[str]:
        """If href points to an immigration rules section, return the base_path."""
        for prefix in ["/guidance/immigration-rules/", "https://www.gov.uk/guidance/immigration-rules/",
                       "http://www.gov.uk/guidance/immigration-rules/"]:
            if prefix in href:
                bp = "/guidance/immigration-rules/" + href.split("/guidance/immigration-rules/")[-1]
                bp = bp.split("#")[0].split("?")[0]  # strip fragment/query
                if bp in self.section_paths:
                    return bp
        return None


# ── HTML parsing ─────────────────────────────────────────────────────

class GuidanceHTMLParser:
    """Parse an HTML guidance attachment body into GuidanceRecords."""

    def __init__(self, resolver: RulesResolver, snapshot_date: str):
        self.resolver = resolver
        self.snapshot_date = snapshot_date
        self.seq_counter = 0

    def parse_html_body(self, html: str, section_data: dict,
                        pub_base_path: str, attachment_id: str,
                        route: list, heading_root: str) -> list[GuidanceRecord]:
        """Parse details.body HTML into records."""
        soup = BeautifulSoup(html, "html.parser")

        # Drop govspeak accessibility notice blocks
        for div in soup.find_all("div", class_="application-notice"):
            div.decompose()
        for div in soup.find_all("div", class_="call-to-action"):
            div.decompose()

        records: list[GuidanceRecord] = []
        heading_path = [heading_root]
        last_rule_record: Optional[GuidanceRecord] = None
        slug = section_data["base_path"].rstrip("/").split("/")[-1] or "index"

        for element in soup.find_all(["h2", "h3", "h4", "h5", "h6", "p", "table",
                                       "div", "ol", "ul"]):
            # Headings
            if element.name in ["h2", "h3", "h4", "h5", "h6"]:
                level = int(element.name[1]) - 2  # h2 = 0
                text = normalize_whitespace(element.get_text())
                if level < len(heading_path) - 1:  # -1 because [0] is root
                    heading_path = heading_path[:level + 1]
                while len(heading_path) <= level + 1:
                    heading_path.append("")
                heading_path[level + 1] = text
                # Trim trailing empties
                while heading_path and heading_path[-1] == "":
                    heading_path.pop()
                last_rule_record = None
                continue

            # Tables
            if element.name == "table":
                table_records = self._process_table(element, heading_path, section_data,
                                                     slug, pub_base_path, attachment_id, route)
                records.extend(table_records)
                continue

            # Legislative lists
            if element.name == "div" and "legislative-list-wrapper" in element.get("class", []):
                for ol_ul in element.find_all(["ol", "ul"], recursive=False):
                    list_recs, last_rule_record = self._process_list(
                        ol_ul, last_rule_record.rule_ref if last_rule_record else "",
                        0, section_data, heading_path, last_rule_record,
                        slug, pub_base_path, attachment_id, route)
                    records.extend(list_recs)
                continue

            if element.name in ["ol", "ul"] and "legislative-list" in element.get("class", []):
                if not element.find_parent("div", class_="legislative-list-wrapper"):
                    list_recs, last_rule_record = self._process_list(
                        element, last_rule_record.rule_ref if last_rule_record else "",
                        0, section_data, heading_path, last_rule_record,
                        slug, pub_base_path, attachment_id, route)
                    records.extend(list_recs)
                continue

            # Paragraphs
            if element.name == "p":
                p_text = normalize_whitespace(extract_text_to_first_list(element))
                if not p_text.strip():
                    continue

                raw_html = str(element)
                source_line = getattr(element, "sourceline", None)
                source_pos = getattr(element, "sourcepos", None)

                # Check for rule identifier
                rule_ref, rule_family, match_end = self._identify_rule(p_text)
                if rule_ref:
                    is_del = self._is_deleted(p_text, match_end)
                    links, cites = self._extract_links(element, p_text, section_data["base_path"])
                    record = self._make_record(
                        section_data, slug, p_text,
                        "deleted" if is_del else "rule",
                        rule_ref=rule_ref, rule_family=rule_family,
                        heading_path=heading_path.copy(), links_out=links,
                        raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                        parse_confidence="high" if heading_path else "medium",
                        pub_base_path=pub_base_path, attachment_id=attachment_id,
                        route=route, cites_rules=cites)
                    records.append(record)
                    last_rule_record = record
                    continue

                # Narrative / continuation
                text_type = "narrative" if heading_path else "continuation"
                links, cites = self._extract_links(element, p_text, section_data["base_path"])
                record = self._make_record(
                    section_data, slug, p_text, text_type,
                    heading_path=heading_path.copy(),
                    attached_to=last_rule_record.paragraph_id if last_rule_record and text_type == "continuation" else None,
                    links_out=links,
                    raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                    parse_confidence="medium",
                    pub_base_path=pub_base_path, attachment_id=attachment_id,
                    route=route, cites_rules=cites)
                records.append(record)
                continue

        return records

    # ── helpers ───────────────────────────────────────────────────────

    def _identify_rule(self, text: str) -> tuple:
        text = text.strip()
        for pattern, family in [(APPENDIX_PATTERN, "appendix"), (PART_PATTERN, "part"),
                                 (RANGE_PATTERN, "range"), (DOTTED_PATTERN, "dotted")]:
            m = pattern.match(text)
            if m:
                ref = re.sub(r"\s+", " ", m.group("ref").strip()).rstrip(".")
                return ref, family, m.end()
        return None, None, None

    def _is_deleted(self, text: str, match_end: int | None = None) -> bool:
        clean = text[match_end:].strip() if match_end else text.strip()
        return bool(re.match(r"^DELETED[.!]?$", clean))

    def _extract_links(self, element: Tag, record_text: str,
                       section_base_path: str) -> tuple[list, bool]:
        """Extract links and resolve against Rules. Returns (links, cites_rules)."""
        links = []
        cites = False

        # Hyperlinks
        for a in element.find_all("a", href=True):
            href = a.get("href", "")
            rules_bp = self.resolver.is_rules_section(href)
            if rules_bp:
                links.append({"target": rules_bp, "target_type": "hyperlink",
                              "source": "hyperlink", "resolved": True,
                              "resolved_to": rules_bp})
                cites = True

        # Textual references
        for pattern, ref_type in [(APPENDIX_REF, "appendix"), (PART_REF, "part"),
                                   (PARAGRAPH_REF, "paragraph")]:
            for m in pattern.finditer(record_text):
                target = m.group(1)
                link = {"target": target, "target_type": ref_type,
                        "source": "regex", "resolved": False}

                if ref_type == "paragraph":
                    resolved_to = self.resolver.resolve_paragraph(target, record_text)
                    if resolved_to:
                        link["resolved"] = True
                        link["resolved_to"] = resolved_to
                        cites = True
                elif ref_type in ("appendix", "part"):
                    resolved_to = self.resolver.resolve_section(target)
                    if resolved_to:
                        link["resolved"] = True
                        link["resolved_to"] = resolved_to
                        if resolved_to != "absent":
                            cites = True

                links.append(link)

        return links, cites

    def _process_table(self, table, heading_path, section_data, slug,
                       pub_base_path, attachment_id, route):
        records = []
        caption = ""
        cap_elem = table.find("caption")
        if cap_elem:
            caption = normalize_whitespace(cap_elem.get_text())
        elif heading_path:
            caption = heading_path[-1]

        headers = []
        header_row = table.find("tr")
        if header_row:
            headers = [normalize_whitespace(th.get_text()) for th in header_row.find_all("th")]

        for row in table.find_all("tr")[1:]:
            cells = [normalize_whitespace(td.get_text()) for td in row.find_all(["td", "th"])]
            if not cells:
                continue
            row_text = caption
            if headers and len(cells) >= len(headers):
                pairs = [f"{h}: {c}" for h, c in zip(headers, cells)]
                row_text += " | " + " | ".join(pairs)
            else:
                row_text += " | " + " | ".join(cells)

            raw_html = str(row)
            record = self._make_record(
                section_data, slug, row_text, "table_row",
                heading_path=(heading_path + [caption]) if caption else heading_path,
                links_out=[], raw_html=raw_html,
                source_line=getattr(row, "sourceline", None),
                source_pos=getattr(row, "sourcepos", None),
                parse_confidence="medium",
                pub_base_path=pub_base_path, attachment_id=attachment_id,
                route=route, cites_rules=False)
            records.append(record)
        return records

    def _process_list(self, ol_ul, parent_ref, depth, section_data, heading_path,
                      last_rule_record, slug, pub_base_path, attachment_id, route):
        records = []
        current_last = last_rule_record

        for li in ol_ul.find_all("li", recursive=False):
            li_text = normalize_whitespace(extract_text_to_first_list(li))
            raw_html = str(li)
            source_line = getattr(li, "sourceline", None)
            source_pos = getattr(li, "sourcepos", None)

            match = SUBPARA_PATTERN.match(li_text)

            # Depth 0: check for rule identifier (not labeled)
            if depth == 0 and not match:
                rule_ref, rule_family, match_end = self._identify_rule(li_text)
                if rule_ref:
                    is_del = self._is_deleted(li_text, match_end)
                    links, cites = self._extract_links(li, li_text, section_data["base_path"])
                    record = self._make_record(
                        section_data, slug, li_text,
                        "deleted" if is_del else "rule",
                        rule_ref=rule_ref, rule_family=rule_family,
                        heading_path=heading_path, links_out=links,
                        raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                        parse_confidence="high",
                        pub_base_path=pub_base_path, attachment_id=attachment_id,
                        route=route, cites_rules=cites)
                    records.append(record)
                    current_last = record

                    for nested in li.find_all(["ol", "ul"], recursive=False):
                        nr, _ = self._process_list(nested, rule_ref, depth + 1,
                                                    section_data, heading_path, record,
                                                    slug, pub_base_path, attachment_id, route)
                        records.extend(nr)
                    continue

            # Subparagraph
            if match:
                label = match.group("label")
                compound_ref = f"{parent_ref}{label}" if parent_ref else label
                is_del = self._is_deleted(li_text, match.end())
                links, cites = self._extract_links(li, li_text, section_data["base_path"])

                attached_to = current_last.paragraph_id if current_last else None
                conf = "high" if current_last else "low"

                record = self._make_record(
                    section_data, slug, li_text,
                    "deleted" if is_del else "subparagraph",
                    rule_ref=compound_ref, heading_path=heading_path,
                    attached_to=attached_to, links_out=links,
                    raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                    parse_confidence=conf,
                    pub_base_path=pub_base_path, attachment_id=attachment_id,
                    route=route, cites_rules=cites)
                records.append(record)

                for nested in li.find_all(["ol", "ul"], recursive=False):
                    nr, _ = self._process_list(nested, compound_ref, depth + 1,
                                                section_data, heading_path, current_last,
                                                slug, pub_base_path, attachment_id, route)
                    records.extend(nr)
                continue

            # Plain list item
            links, cites = self._extract_links(li, li_text, section_data["base_path"])
            record = self._make_record(
                section_data, slug, li_text, "list_item",
                heading_path=heading_path,
                attached_to=current_last.paragraph_id if current_last else None,
                links_out=links,
                raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                parse_confidence="medium" if current_last else "low",
                pub_base_path=pub_base_path, attachment_id=attachment_id,
                route=route, cites_rules=cites)
            records.append(record)

        return records, current_last

    def _make_record(self, section_data, slug, text, text_type, *,
                     rule_ref=None, rule_family=None, heading_path,
                     attached_to=None, links_out, raw_html,
                     source_line=None, source_pos=None, parse_confidence,
                     pub_base_path, attachment_id, route, cites_rules) -> GuidanceRecord:
        self.seq_counter += 1
        pid = f"{slug}:{self.seq_counter:05d}"
        sha = hashlib.sha256(raw_html.encode("utf-8")).hexdigest()
        return GuidanceRecord(
            paragraph_id=pid, seq=self.seq_counter,
            rule_ref=rule_ref, rule_family=rule_family,
            section_base_path=section_data["base_path"],
            section_title=section_data["title"],
            heading_path=heading_path, text=text, text_type=text_type,
            attached_to=attached_to, links_out=links_out,
            content_id=section_data["content_id"],
            public_updated_at=section_data["public_updated_at"],
            snapshot_date=self.snapshot_date, licence="OGL-3.0",
            raw_html=raw_html, raw_html_sha256=sha,
            source_line=source_line, source_pos=source_pos,
            parse_confidence=parse_confidence,
            publication_base_path=pub_base_path,
            attachment_id=attachment_id, route=route,
            source="html", page_number=None, cites_rules=cites_rules)


# ── PDF parsing ──────────────────────────────────────────────────────

class GuidancePDFParser:
    """Parse a PDF guidance attachment into GuidanceRecords."""

    def __init__(self, resolver: RulesResolver, snapshot_date: str,
                 seq_start: int = 0):
        self.resolver = resolver
        self.snapshot_date = snapshot_date
        self.seq_counter = seq_start

    def parse_pdf(self, pdf_path: Path, section_data: dict,
                  pub_base_path: str, attachment_id: str,
                  route: list) -> tuple[list[GuidanceRecord], bool]:
        """Parse PDF. Returns (records, excluded_flag)."""
        import pdfplumber

        slug = section_data["base_path"].rstrip("/").split("/")[-1] or "index"
        slug = f"{slug}__{attachment_id}"

        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)

            # ── Step 1: extract raw text per page ────────────────────
            page_texts: list[list[str]] = []  # list of (page_num, lines)
            for page in pdf.pages:
                h, w = page.height, page.width
                body = page.crop((0, h * 0.08, w, h * 0.93))
                raw = body.extract_text(x_tolerance=3, y_tolerance=3) or ""
                page_texts.append(raw.split("\n"))

            # ── Step 2: detect header/footer lines ───────────────────
            # Normalize digits to N, find lines on >50% of pages
            def _norm_digits(s): return re.sub(r"\d+", "N", s.strip())
            line_freq: Counter = Counter()
            for lines in page_texts:
                seen = set()
                for line in lines:
                    nk = _norm_digits(line)
                    if nk and nk not in seen:
                        seen.add(nk)
                        line_freq[nk] += 1
            threshold = n_pages / 2
            hf_patterns = {k for k, v in line_freq.items() if v > threshold}

            # ── Step 3: detect TOC pages ─────────────────────────────
            toc_pages: set[int] = set()
            for pi, lines in enumerate(page_texts):
                non_empty = [l for l in lines if l.strip()]
                if not non_empty:
                    continue
                dot_leader_count = sum(
                    1 for l in non_empty
                    if re.search(r"\.{5,}", l) or re.search(r"\d+\s*$", l.strip())
                )
                if dot_leader_count > len(non_empty) / 2:
                    toc_pages.add(pi)

            # ── Step 4: detect font sizes for headings ───────────────
            font_sizes: Counter = Counter()
            for page in pdf.pages:
                for char in page.chars:
                    font_sizes[round(char.get("size", 0), 1)] += 1
            body_size = font_sizes.most_common(1)[0][0] if font_sizes else 12
            heading_threshold = body_size + 3

            # ── Step 5: extract tables per page ──────────────────────
            page_tables: dict[int, list] = {}
            for pi, page in enumerate(pdf.pages):
                if pi in toc_pages:
                    continue
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                })
                if tables:
                    page_tables[pi] = tables

            # ── Step 6: build records ────────────────────────────────
            records: list[GuidanceRecord] = []
            heading_path: list[str] = []
            last_narrative: Optional[GuidanceRecord] = None
            last_rule: Optional[GuidanceRecord] = None
            empty_pages = 0

            trailer_re = re.compile(
                r"^(Related content|Contents|Related external links)$")
            redaction_re = re.compile(
                r"^Official\s*[–-]\s*sensitive:\s*(start|end)\s+of\s+section$",
                re.IGNORECASE)

            for pi, raw_lines in enumerate(page_texts):
                page_num = pi + 1

                if pi in toc_pages:
                    continue

                # Filter lines
                clean_lines: list[str] = []
                for line in raw_lines:
                    stripped = line.strip()
                    if not stripped:
                        clean_lines.append("")
                        continue
                    if _norm_digits(stripped) in hf_patterns:
                        continue
                    if trailer_re.match(stripped):
                        continue
                    if redaction_re.match(stripped):
                        continue
                    clean_lines.append(stripped)

                # Check for empty page
                if not any(l.strip() for l in clean_lines):
                    empty_pages += 1
                    continue

                # Detect headings from font sizes
                page_obj = pdf.pages[pi]
                heading_chars: dict[int, float] = {}  # char index → size
                for char in page_obj.chars:
                    if round(char.get("size", 0), 1) >= heading_threshold:
                        heading_chars[id(char)] = char.get("size", 0)

                # Process tables for this page
                if pi in page_tables:
                    for table_data in page_tables[pi]:
                        headers = []
                        if table_data and table_data[0]:
                            headers = [re.sub(r"\s+", " ", (c or "").strip())
                                       for c in table_data[0]]
                        for row_cells in table_data[1:]:
                            cells = [re.sub(r"\s+", " ", (c or "").strip())
                                     for c in row_cells]
                            if not any(cells):
                                continue
                            if headers and len(cells) >= len(headers):
                                pairs = [f"{h}: {c}" for h, c in zip(headers, cells)]
                                row_text = " | ".join(pairs)
                            else:
                                row_text = " | ".join(c for c in cells if c)

                            record = self._make_record(
                                section_data, slug, row_text, "table_row",
                                heading_path=heading_path.copy(), page_number=page_num,
                                pub_base_path=pub_base_path, attachment_id=attachment_id,
                                route=route)
                            records.append(record)

                # Process text lines into paragraphs
                current_para: list[str] = []

                def _flush_para():
                    nonlocal last_narrative, last_rule
                    if not current_para:
                        return
                    text = " ".join(current_para)
                    current_para.clear()

                    # Check for rule identifier
                    rule_ref, rule_family, match_end = self._identify_rule(text)
                    if rule_ref:
                        is_del = self._is_deleted(text, match_end)
                        links, cites = self._resolve_text_refs(text)
                        record = self._make_record(
                            section_data, slug, text,
                            "deleted" if is_del else "rule",
                            rule_ref=rule_ref, rule_family=rule_family,
                            heading_path=heading_path.copy(),
                            links_out=links, page_number=page_num,
                            pub_base_path=pub_base_path, attachment_id=attachment_id,
                            route=route, cites_rules=cites)
                        records.append(record)
                        last_rule = record
                        last_narrative = record
                        return

                    # Check for subparagraph
                    m = SUBPARA_PATTERN.match(text)
                    if m:
                        label = m.group("label")
                        base_ref = last_rule.rule_ref if last_rule else ""
                        compound = f"{base_ref}{label}" if base_ref else label
                        links, cites = self._resolve_text_refs(text)
                        record = self._make_record(
                            section_data, slug, text, "subparagraph",
                            rule_ref=compound, heading_path=heading_path.copy(),
                            attached_to=last_rule.paragraph_id if last_rule else None,
                            links_out=links, page_number=page_num,
                            pub_base_path=pub_base_path, attachment_id=attachment_id,
                            route=route, cites_rules=cites)
                        records.append(record)
                        return

                    # Narrative
                    links, cites = self._resolve_text_refs(text)
                    record = self._make_record(
                        section_data, slug, text, "narrative",
                        heading_path=heading_path.copy(),
                        links_out=links, page_number=page_num,
                        pub_base_path=pub_base_path, attachment_id=attachment_id,
                        route=route, cites_rules=cites)
                    records.append(record)
                    last_narrative = record

                for line in clean_lines:
                    if not line:
                        _flush_para()
                        continue

                    # Bullets and numbered steps → list_item
                    if re.match(r"^[•o]\s+", line) or re.match(r"^\d+\.\s+", line):
                        _flush_para()
                        text = re.sub(r"^[•o]\s+", "", line).strip()
                        text = re.sub(r"^\d+\.\s+", "", line).strip() if not text else text
                        links, cites = self._resolve_text_refs(line)
                        record = self._make_record(
                            section_data, slug, line, "list_item",
                            heading_path=heading_path.copy(),
                            attached_to=last_narrative.paragraph_id if last_narrative else None,
                            links_out=links, page_number=page_num,
                            pub_base_path=pub_base_path, attachment_id=attachment_id,
                            route=route, cites_rules=cites)
                        records.append(record)
                        continue

                    # Heading detection (heuristic: short line, not a sentence)
                    # Use font-size based detection for first few words
                    if len(line) < 80 and not line.endswith(".") and not line.endswith(","):
                        # Check if this looks like a heading
                        if re.match(r"^(Annex|Part|Section|Chapter|Appendix)\s", line):
                            _flush_para()
                            heading_path = [line]
                            last_rule = None
                            continue
                        if re.match(r"^\d+\.\s+[A-Z]", line) and len(line) < 60:
                            _flush_para()
                            if len(heading_path) < 2:
                                heading_path.append(line)
                            else:
                                heading_path[1] = line
                            continue

                    current_para.append(line)

                _flush_para()

        excluded = empty_pages > 0.2 * n_pages

        return records, excluded

    def _identify_rule(self, text):
        text = text.strip()
        for pattern, family in [(APPENDIX_PATTERN, "appendix"), (PART_PATTERN, "part"),
                                 (RANGE_PATTERN, "range"), (DOTTED_PATTERN, "dotted")]:
            m = pattern.match(text)
            if m:
                ref = re.sub(r"\s+", " ", m.group("ref").strip()).rstrip(".")
                return ref, family, m.end()
        return None, None, None

    def _is_deleted(self, text, match_end=None):
        clean = text[match_end:].strip() if match_end else text.strip()
        return bool(re.match(r"^DELETED[.!]?$", clean))

    def _resolve_text_refs(self, text: str) -> tuple[list, bool]:
        links = []
        cites = False
        for pattern, ref_type in [(APPENDIX_REF, "appendix"), (PART_REF, "part"),
                                   (PARAGRAPH_REF, "paragraph")]:
            for m in pattern.finditer(text):
                target = m.group(1)
                link = {"target": target, "target_type": ref_type,
                        "source": "regex", "resolved": False}
                if ref_type == "paragraph":
                    resolved_to = self.resolver.resolve_paragraph(target, text)
                    if resolved_to:
                        link["resolved"] = True
                        link["resolved_to"] = resolved_to
                        cites = True
                elif ref_type in ("appendix", "part"):
                    resolved_to = self.resolver.resolve_section(target)
                    if resolved_to:
                        link["resolved"] = True
                        link["resolved_to"] = resolved_to
                        if resolved_to != "absent":
                            cites = True
                links.append(link)
        return links, cites

    def _make_record(self, section_data, slug, text, text_type, *,
                     rule_ref=None, rule_family=None, heading_path=None,
                     attached_to=None, links_out=None, page_number=None,
                     pub_base_path="", attachment_id="", route=None,
                     cites_rules=False) -> GuidanceRecord:
        self.seq_counter += 1
        pid = f"{slug}:{self.seq_counter:05d}"
        raw_html = ""  # PDF records have no HTML
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return GuidanceRecord(
            paragraph_id=pid, seq=self.seq_counter,
            rule_ref=rule_ref, rule_family=rule_family,
            section_base_path=section_data["base_path"],
            section_title=section_data["title"],
            heading_path=heading_path or [], text=text, text_type=text_type,
            attached_to=attached_to, links_out=links_out or [],
            content_id=section_data["content_id"],
            public_updated_at=section_data["public_updated_at"],
            snapshot_date=self.snapshot_date, licence="OGL-3.0",
            raw_html=raw_html, raw_html_sha256=sha,
            source_line=None, source_pos=None,
            parse_confidence="medium",  # PDF capped at medium
            publication_base_path=pub_base_path,
            attachment_id=attachment_id, route=route or [],
            source="pdf", page_number=page_number,
            cites_rules=cites_rules)


# ── Report generation ────────────────────────────────────────────────

def generate_report(records: list[GuidanceRecord],
                    flagged_pdfs: list[str]) -> dict:
    """Build the parse report."""
    by_route: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    by_pub: dict[str, dict] = defaultdict(lambda: {
        "type_counts": defaultdict(int), "source_counts": defaultdict(int),
        "headings": 0, "xrefs_found": 0, "xrefs_resolved": 0,
    })

    total_low = 0
    low_tokens: list[str] = []

    for r in records:
        for rt in r.route:
            by_route[rt][r.text_type] += 1
            by_route[rt][f"source_{r.source}"] += 1

        pub = r.publication_base_path
        by_pub[pub]["type_counts"][r.text_type] += 1
        by_pub[pub]["source_counts"][r.source] += 1
        if r.heading_path:
            by_pub[pub]["headings"] += 1
        for link in r.links_out:
            by_pub[pub]["xrefs_found"] += 1
            if link.get("resolved"):
                by_pub[pub]["xrefs_resolved"] += 1

        if r.parse_confidence == "low":
            total_low += 1
            tokens = r.text.split()
            if tokens:
                low_tokens.append(tokens[0])

    overall_low_share = total_low / len(records) if records else 0
    common_low_tokens = Counter(low_tokens).most_common(20)

    return {
        "total_records": len(records),
        "by_route": {k: dict(v) for k, v in by_route.items()},
        "by_publication": {k: {
            "type_counts": dict(v["type_counts"]),
            "source_counts": dict(v["source_counts"]),
            "headings": v["headings"],
            "xrefs_found": v["xrefs_found"],
            "xrefs_resolved": v["xrefs_resolved"],
        } for k, v in by_pub.items()},
        "flagged_pdfs": flagged_pdfs,
        "overall_low_confidence_share": overall_low_share,
        "common_low_confidence_tokens": common_low_tokens,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Parse caseworker guidance attachments")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--plan", required=True, help="Guidance fetch-plan CSV")
    ap.add_argument("--id-table", default="data/processed/rules-identifier-table_v1.json",
                    help="Rules identifier table JSON")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_guidance = Path(config["raw_guidance"])
    interim_path = Path(config["interim"])

    # Timestamped output
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = interim_path / f"{stamp}_guidance-parse"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load plan
    with open(args.plan, newline="", encoding="utf-8") as f:
        plan = list(csv.DictReader(f))
    log.info("Plan: %d publications", len(plan))

    # Build section list from raw-rules manifest
    rules_manifest_path = Path(config.get("raw_rules", "data/raw-rules")) / "manifest.json"
    section_list = []
    if rules_manifest_path.exists():
        with open(rules_manifest_path) as f:
            rm = json.load(f)
        # section paths from file_hashes keys
        for fname in rm.get("file_hashes", {}):
            slug = fname.replace(".json", "")
            section_list.append(f"/guidance/immigration-rules/{slug}")

    # Load identifier table
    resolver = RulesResolver(args.id_table, section_list)
    log.info("Resolver: %d sections, %d unique global refs",
             len(resolver.section_paths), len(resolver.unique_global))

    # Get snapshot date from raw guidance manifest
    gm_path = raw_guidance / "manifest.json"
    snapshot_date = ""
    if gm_path.exists():
        with open(gm_path) as f:
            snapshot_date = json.load(f).get("snapshot_date", "")

    html_parser = GuidanceHTMLParser(resolver, snapshot_date)
    all_records: list[GuidanceRecord] = []
    flagged_pdfs: list[str] = []
    html_att_count = 0
    pdf_att_count = 0
    pub_count = 0

    for row in plan:
        bp = row["base_path"]
        route_str = row["route"]
        routes = route_str.split(";")
        first_route = routes[0]
        slug = bp.rstrip("/").split("/")[-1] or "index"
        has_html = row.get("has_html_attachment", "").strip().lower() == "true"

        pub_file = raw_guidance / first_route / f"{slug}.json"
        if not pub_file.exists():
            log.warning("Missing publication: %s", pub_file)
            continue

        with open(pub_file) as f:
            pub_data = json.load(f)

        section_data = {
            "base_path": pub_data.get("base_path", bp),
            "title": pub_data.get("title", ""),
            "content_id": pub_data.get("content_id", ""),
            "public_updated_at": pub_data.get("public_updated_at", ""),
        }
        pub_count += 1

        attachments = pub_data.get("details", {}).get("attachments", [])

        if has_html:
            html_atts = [a for a in attachments if a.get("attachment_type") == "html"]
            for att in html_atts:
                att_id = att.get("id", "unknown")
                att_file = raw_guidance / first_route / f"{slug}__{att_id}.json"
                if not att_file.exists():
                    log.warning("Missing HTML attachment: %s", att_file)
                    continue

                with open(att_file) as f:
                    att_data = json.load(f)

                body = att_data.get("details", {}).get("body", "")
                heading_root = att.get("title", section_data["title"])

                recs = html_parser.parse_html_body(
                    body, section_data, bp, att_id, routes, heading_root)
                all_records.extend(recs)
                html_att_count += 1
                log.info("  HTML %s/%s: %d records", first_route, slug, len(recs))
        else:
            pdf_atts = [a for a in attachments
                        if a.get("attachment_type") == "file"
                        and ("pdf" in a.get("content_type", "").lower()
                             or str(a.get("url", "")).lower().endswith(".pdf"))]
            for att in pdf_atts:
                att_id = att.get("id", "unknown")
                pdf_file = raw_guidance / first_route / f"{slug}__{att_id}.pdf"
                if not pdf_file.exists():
                    log.warning("Missing PDF: %s", pdf_file)
                    continue

                pdf_parser = GuidancePDFParser(
                    resolver, snapshot_date, html_parser.seq_counter)
                recs, excluded = pdf_parser.parse_pdf(
                    pdf_file, section_data, bp, att_id, routes)
                html_parser.seq_counter = pdf_parser.seq_counter
                all_records.extend(recs)
                pdf_att_count += 1

                if excluded:
                    flagged_pdfs.append(str(pdf_file))
                log.info("  PDF  %s/%s: %d records%s", first_route, slug,
                         len(recs), " [FLAGGED]" if excluded else "")

    # Write JSONL
    jsonl_path = out_dir / "guidance-paragraphs.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_records:
            d = {
                "paragraph_id": r.paragraph_id, "seq": r.seq,
                "rule_ref": r.rule_ref, "rule_family": r.rule_family,
                "section_base_path": r.section_base_path,
                "section_title": r.section_title,
                "heading_path": r.heading_path, "text": r.text,
                "text_type": r.text_type, "attached_to": r.attached_to,
                "links_out": r.links_out,
                "content_id": r.content_id,
                "public_updated_at": r.public_updated_at,
                "snapshot_date": r.snapshot_date, "licence": r.licence,
                "raw_html": r.raw_html, "raw_html_sha256": r.raw_html_sha256,
                "source_line": r.source_line, "source_pos": r.source_pos,
                "parse_confidence": r.parse_confidence,
                "publication_base_path": r.publication_base_path,
                "attachment_id": r.attachment_id,
                "route": r.route, "source": r.source,
                "page_number": r.page_number,
                "cites_rules": r.cites_rules,
            }
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Report
    report = generate_report(all_records, flagged_pdfs)
    report_path = out_dir / "guidance-parse-report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("\nParsing complete:")
    log.info("  Publications: %d", pub_count)
    log.info("  HTML attachments: %d", html_att_count)
    log.info("  PDF attachments: %d", pdf_att_count)
    log.info("  Total records: %d", len(all_records))
    log.info("  Low confidence: %.1f%%", report["overall_low_confidence_share"] * 100)
    log.info("  Output: %s", jsonl_path)
    log.info("  Report: %s", report_path)

    if report["overall_low_confidence_share"] > 0.10:
        log.error("Gate FAILED: low confidence %.1f%% > 10%%",
                  report["overall_low_confidence_share"] * 100)
        exit(1)


if __name__ == "__main__":
    main()
