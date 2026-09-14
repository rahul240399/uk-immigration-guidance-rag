"""
Immigration Rules Parser v1
---------------------------
Parses raw immigration rules JSON files to structured paragraph records.

Output:
- rules-paragraphs.jsonl: One record per rule/paragraph/element
- parse-report.json: Processing statistics and confidence metrics

Licence: Contains public sector information licensed under the
         Open Government Licence v3.0.
"""

import hashlib
import json
import re
import yaml
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

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


@dataclass
class ParseRecord:
    """Structure for parsed immigration rule records."""
    paragraph_id: str
    seq: int
    rule_ref: Optional[str]
    rule_family: Optional[str]
    section_base_path: str
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


class RulesParser:
    """Parser for immigration rules documents."""
    
    def __init__(self, config_path: str = "config/paths.yaml"):
        """Initialize parser with configuration."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.raw_rules_path = Path(config["raw_rules"])
        self.interim_path = Path(config["interim"])
        self.interim_path.mkdir(parents=True, exist_ok=True)
        
        # Load manifest
        manifest_path = self.raw_rules_path / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        
        # Initialize tracking
        self.identifier_table = {}  # (section_base_path, rule_ref) -> paragraph_id mapping
        self.section_titles = {}    # base_path -> title mapping
        self.section_base_paths = set()  # All valid section base paths
        self.name_index = {}        # normalized name -> base_path mapping
        self.parse_stats = defaultdict(lambda: defaultdict(int))
        self.records = []
        self.seq_counter = 0
        self.current_section_rules = {}  # rule_ref -> paragraph_id for current section
        self.unmatched_tokens = []  # For reporting unmatched first tokens
        self.unresolved_names = []  # For tracking unresolved reference names
        self.section_aliases = {}   # Will be loaded from config
        self.global_rule_index = defaultdict(list)  # rule_ref -> [(section_path, paragraph_id), ...]
    
    def load_section_titles(self):
        """Build section title mapping and name index from all files."""
        # Try to load section aliases
        try:
            alias_path = Path("config/section_aliases.yaml")
            if alias_path.exists():
                with open(alias_path) as f:
                    self.section_aliases = yaml.safe_load(f) or {}
        except Exception:
            pass
            
        for file_path in self.raw_rules_path.glob("*.json"):
            if file_path.name in ['manifest.json', 'fetch-log.csv']:
                continue
                
            with open(file_path) as f:
                data = json.load(f)
            
            base_path = data.get("base_path", "")
            title = data.get("title", "")
            if base_path and title:
                self.section_titles[base_path] = title.strip()
                self.section_base_paths.add(base_path)
                
                # Build name index
                self._add_to_name_index(title, base_path)
    
    def _add_to_name_index(self, title: str, base_path: str):
        """Add title to name index with normalized names."""
        # Extract name after 'Appendix ' or 'Part ' (case-insensitive)
        for prefix in ['Appendix ', 'Part ', 'appendix ', 'part ']:
            if prefix.lower() in title.lower():
                # Find the prefix position
                prefix_pos = title.lower().find(prefix.lower())
                after_prefix = title[prefix_pos + len(prefix):]
                
                # Take up to ':' or end
                name = after_prefix.split(':', 1)[0].strip()
                
                # Drop bracketed suffixes
                name = re.sub(r'\s*\([^)]*\)\s*', ' ', name).strip()
                
                # Normalize: lowercase, remove punctuation, collapse whitespace
                normalized = normalize_name(name)
                
                if normalized and len(normalized) >= 3:
                    self.name_index[normalized] = base_path
                break
    
    def get_slug_from_path(self, base_path: str) -> str:
        """Extract slug from base path."""
        return base_path.rstrip("/").split("/")[-1] or "index"
    
    def normalize_text(self, text: str) -> str:
        """Normalize whitespace in text."""
        return normalize_whitespace(text)
    
    def normalize_rule_ref(self, rule_ref: str) -> str:
        """Normalize rule reference by collapsing whitespace and removing trailing dots."""
        normalized = re.sub(r'\s+', ' ', rule_ref.strip())
        if normalized.endswith('.'):
            normalized = normalized[:-1]
        return normalized
    
    def normalize_reference_key(self, rule_ref: str) -> str:
        """Normalize rule reference for lookup: uppercase, remove spaces and full stops."""
        return normalize_reference_key(rule_ref)
    
    def strip_trailing_connectors(self, name: str) -> str:
        """Strip trailing connector words from appendix names."""
        return strip_trailing_connectors(name)
    
    def longest_prefix_match(self, target: str) -> Optional[str]:
        """Find longest prefix match in name index."""
        target_norm = normalize_name(target)
        return longest_prefix_match(target_norm, self.name_index)
    
    def extract_text_to_first_list(self, element: Tag) -> str:
        """Extract element text up to first ol or ul child."""
        return normalize_whitespace(extract_text_to_first_list(element))
    
    def identify_rule(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Identify rule reference and family from text. Returns (rule_ref, family, match_end)."""
        text = text.strip()
        
        # Try appendix style first
        match = APPENDIX_PATTERN.match(text)
        if match:
            return self.normalize_rule_ref(match.group('ref')), 'appendix', match.end()
        
        # Try part style
        match = PART_PATTERN.match(text)
        if match:
            return self.normalize_rule_ref(match.group('ref')), 'part', match.end()
        
        # Try range style
        match = RANGE_PATTERN.match(text)
        if match:
            return self.normalize_rule_ref(match.group('ref')), 'range', match.end()
        
        # Try dotted style
        match = DOTTED_PATTERN.match(text)
        if match:
            return self.normalize_rule_ref(match.group('ref')), 'dotted', match.end()
        
        return None, None, None
    
    def is_deleted(self, text: str, match_end: Optional[int] = None) -> bool:
        """Check if text represents a deleted rule."""
        if match_end is not None:
            # Strip the matched identifier
            clean_text = text[match_end:].strip()
        else:
            # For bare DELETED paragraphs
            clean_text = text.strip()
        
        return bool(re.match(r'^DELETED[.!]?$', clean_text))
    
    def extract_links(self, element: Tag, record_text: str, section_base_path: str) -> List[Dict]:
        """Extract cross-references from element and text."""
        links = []
        
        # Hyperlinks
        for link in element.find_all('a', href=True):
            href = link.get('href', '')
            if '/guidance/immigration-rules/' in href:
                # Extract base path from href
                if href.startswith('/guidance/immigration-rules/'):
                    base_path = href
                else:
                    # Handle relative or absolute URLs
                    base_path = '/guidance/immigration-rules/' + href.split('/guidance/immigration-rules/')[-1]
                
                # Only mark as resolved if the base path exists in our section list
                resolved = base_path in self.section_base_paths
                
                links.append({
                    'target': base_path,
                    'target_type': 'hyperlink',
                    'source': 'hyperlink',
                    'resolved': resolved
                })
        
        # Textual references
        for pattern, ref_type in [
            (APPENDIX_REF, 'appendix'),
            (PART_REF, 'part'),
            (PARAGRAPH_REF, 'paragraph')
        ]:
            for match in pattern.finditer(record_text):
                target = match.group(1)
                resolved_target, candidates = self.resolve_reference(target, ref_type, record_text, section_base_path)
                
                link_dict = {
                    'target': target,
                    'target_type': ref_type,
                    'source': 'regex',
                    'resolved': resolved_target is not None
                }
                
                if resolved_target == "absent":
                    link_dict['target'] = "absent"
                    link_dict['resolved'] = True
                elif candidates:
                    link_dict['candidates'] = candidates
                
                links.append(link_dict)
        
        return links
    
    def resolve_reference(self, target: str, ref_type: str, record_text: str = "", 
                         section_base_path: str = "") -> Tuple[Optional[str], Optional[List[str]]]:
        """Resolve cross-reference target. Returns (resolved_target, candidate_sections)."""
        if ref_type in ['appendix', 'part']:
            # Strip trailing connectors
            clean_target = self.strip_trailing_connectors(target)
            
            # Try longest prefix match first
            match = self.longest_prefix_match(clean_target)
            if match:
                return match, None
            
            # Try section aliases
            target_norm = normalize_name(clean_target)
            
            if target_norm in self.section_aliases:
                alias_target = self.section_aliases[target_norm]
                if alias_target == "absent":
                    return "absent", None
                elif alias_target in self.section_base_paths:
                    return alias_target, None
            
            # Mark as unresolved for reporting
            self.unresolved_names.append(target)
            return None, None
            
        elif ref_type == 'paragraph':
            # First try current section
            normalized_target = self.normalize_reference_key(target)
            key = (section_base_path, normalized_target)
            if key in self.identifier_table:
                return self.identifier_table[key], None
            
            # Then try to find section mentioned in same sentence
            sentence_sections = []
            for pattern, section_type in [(APPENDIX_REF, 'appendix'), (PART_REF, 'part')]:
                for match in pattern.finditer(record_text):
                    section_name = match.group(1).strip()
                    resolved_section, _ = self.resolve_reference(section_name, section_type, record_text, section_base_path)
                    if resolved_section and resolved_section != "absent":
                        sentence_sections.append(resolved_section)
            
            # Try paragraph resolution in mentioned sections
            for section_bp in sentence_sections:
                key = (section_bp, normalized_target)
                if key in self.identifier_table:
                    return self.identifier_table[key], None
            
            # Finally, try global resolution
            candidates = self.global_rule_index.get(normalized_target, [])
            if len(candidates) == 1:
                # Exactly one match corpus-wide
                return candidates[0][1], None  # return paragraph_id
            elif len(candidates) > 1:
                # Multiple matches - return candidates
                candidate_sections = sorted(set(c[0] for c in candidates))  # section paths
                return None, candidate_sections
        
        return None, None
    
    def calculate_confidence(self, record: ParseRecord) -> str:
        """Calculate parse confidence for record."""
        if record.text_type in ['rule', 'deleted'] and record.rule_ref:
            return 'high'
        elif record.text_type == 'subparagraph':
            return 'high'  # All subparagraphs are high confidence if properly labeled
        elif record.text_type == 'table_row':
            return 'high'  # Table rows are structured data
        elif record.text_type == 'narrative' and record.heading_path:
            return 'medium'  # Narrative with heading context
        elif record.text_type in ['continuation'] and record.attached_to:
            return 'medium'
        elif record.text_type == 'list_item':
            # List items attached to rules are medium, standalone are low
            return 'medium' if record.attached_to else 'low'
        else:
            return 'low'
    
    def get_element_provenance(self, element: Tag) -> Tuple[str, str, Optional[int], Optional[int]]:
        """Get provenance information for an element."""
        raw_html = str(element)
        raw_html_sha256 = hashlib.sha256(raw_html.encode('utf-8')).hexdigest()
        source_line = getattr(element, 'sourceline', None)
        source_pos = getattr(element, 'sourcepos', None)
        return raw_html, raw_html_sha256, source_line, source_pos
    
    def make_record(self, section_data: Dict, seq: int, text: str, text_type: str, *,
                    rule_ref: Optional[str] = None, rule_family: Optional[str] = None,
                    heading_path: List[str], attached_to: Optional[str] = None,
                    links_out: List[Dict], raw_html: str, source_line: Optional[int],
                    source_pos: Optional[int], parse_confidence: str) -> ParseRecord:
        """Factory: build a ParseRecord with derived fields filled in one place."""
        slug = self.get_slug_from_path(section_data['base_path'])
        paragraph_id = f"{slug}:{seq:05d}"
        raw_html_sha256 = hashlib.sha256(raw_html.encode('utf-8')).hexdigest()
        return ParseRecord(
            paragraph_id=paragraph_id,
            seq=seq,
            rule_ref=rule_ref,
            rule_family=rule_family,
            section_base_path=section_data['base_path'],
            section_title=section_data['title'],
            heading_path=heading_path,
            text=text,
            text_type=text_type,
            attached_to=attached_to,
            links_out=links_out,
            content_id=section_data['content_id'],
            public_updated_at=section_data['public_updated_at'],
            snapshot_date=self.manifest['snapshot_date'],
            licence='OGL-3.0',
            raw_html=raw_html,
            raw_html_sha256=raw_html_sha256,
            source_line=source_line,
            source_pos=source_pos,
            parse_confidence=parse_confidence,
        )
    
    def build_compound_label(self, base_ref: str, labels: List[str]) -> str:
        """Build compound subparagraph label like ECAA 7.1(b)(i)."""
        result = base_ref
        for label in labels:
            result += f"({label})"
        return result
    
    def process_table_rows(self, table: Tag, heading_path: List[str], section_data: Dict) -> List[ParseRecord]:
        """Process table rows into records."""
        records = []
        caption = ""
        
        # Get table caption
        caption_elem = table.find('caption')
        if caption_elem:
            caption = self.normalize_text(caption_elem.get_text())
        elif heading_path:
            caption = heading_path[-1]
        
        # Get headers
        headers = []
        header_row = table.find('tr')
        if header_row:
            for th in header_row.find_all('th'):
                headers.append(self.normalize_text(th.get_text()))
        
        # Process data rows
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = [self.normalize_text(td.get_text()) for td in row.find_all(['td', 'th'])]
            
            if cells:
                # Create header:cell pairs
                row_text = caption
                if headers and len(cells) >= len(headers):
                    pairs = [f"{h}: {c}" for h, c in zip(headers, cells)]
                    row_text += " | " + " | ".join(pairs)
                else:
                    row_text += " | " + " | ".join(cells)
                
                raw_html, raw_html_sha256, source_line, source_pos = self.get_element_provenance(row)
                
                self.seq_counter += 1
                record = self.make_record(
                    section_data, self.seq_counter, row_text, 'table_row',
                    heading_path=heading_path + [caption] if caption else heading_path,
                    links_out=[],
                    raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                    parse_confidence='medium',
                )
                
                records.append(record)
        
        return records
    
    def process_list_items(self, ol_ul: Tag, parent_ref: str, depth: int, section_data: Dict, 
                          heading_path: List[str], last_rule_record: Optional[ParseRecord], 
                          label_stack: List[str] = None) -> Tuple[List[ParseRecord], Optional[ParseRecord]]:
        """Process list items recursively. Returns (records, updated_last_rule_record)."""
        records = []
        current_last_rule = last_rule_record
        
        if label_stack is None:
            label_stack = []
        
        for li in ol_ul.find_all('li', recursive=False):
            li_text = self.extract_text_to_first_list(li)
            raw_html, raw_html_sha256, source_line, source_pos = self.get_element_provenance(li)
            
            # At depth 0, check for rule identifier first (but not for labeled items)
            match = SUBPARA_PATTERN.match(li_text)
            if depth == 0 and not match:
                rule_ref, rule_family, match_end = self.identify_rule(li_text)
                if rule_ref:
                    # This is a rule record
                    self.seq_counter += 1
                    
                    text_type = 'deleted' if self.is_deleted(li_text, match_end) else 'rule'
                    
                    record = self.make_record(
                        section_data, self.seq_counter, li_text, text_type,
                        rule_ref=rule_ref, rule_family=rule_family,
                        heading_path=heading_path,
                        links_out=self.extract_links(li, li_text, section_data['base_path']),
                        raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                        parse_confidence='high',
                    )
                    
                    records.append(record)
                    current_last_rule = record  # Update for later items
                    key = (section_data['base_path'], self.normalize_reference_key(rule_ref))
                    self.identifier_table[key] = record.paragraph_id
                    self.current_section_rules[rule_ref] = record.paragraph_id
                    # Add to global index
                    self.global_rule_index[self.normalize_reference_key(rule_ref)].append((section_data['base_path'], record.paragraph_id))
                    
                    # Process nested lists within this rule
                    nested_lists = li.find_all(['ol', 'ul'], recursive=False)
                    for nested_ol_ul in nested_lists:
                        nested_records, _ = self.process_list_items(
                            nested_ol_ul, rule_ref, depth + 1, section_data, heading_path, record, []
                        )
                        records.extend(nested_records)
                    
                    continue
            
            # Check for subparagraph label
            if match:
                label = match.group('label')  # This now captures full compound label
                
                # Build rule reference with compound label
                if parent_ref:
                    compound_ref = f"{parent_ref}{label}"
                else:
                    compound_ref = label
                
                self.seq_counter += 1
                
                # Check if this is deleted
                is_deleted_subpara = self.is_deleted(li_text, match.end())
                text_type = 'deleted' if is_deleted_subpara else 'subparagraph'
                
                # Determine attachment - at depth 0, always attach to last_rule_record
                attached_to = None
                parse_confidence = 'low'
                if depth == 0 and current_last_rule:
                    attached_to = current_last_rule.paragraph_id
                    parse_confidence = 'high'
                elif current_last_rule:
                    attached_to = current_last_rule.paragraph_id
                    parse_confidence = 'high'
                
                record = self.make_record(
                    section_data, self.seq_counter, li_text, text_type,
                    rule_ref=compound_ref,
                    heading_path=heading_path,
                    attached_to=attached_to,
                    links_out=self.extract_links(li, li_text, section_data['base_path']),
                    raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                    parse_confidence=parse_confidence,
                )
                
                records.append(record)
                key = (section_data['base_path'], self.normalize_reference_key(compound_ref))
                self.identifier_table[key] = record.paragraph_id
                
                # Process nested lists
                nested_lists = li.find_all(['ol', 'ul'], recursive=False)
                for nested_ol_ul in nested_lists:
                    nested_records, _ = self.process_list_items(
                        nested_ol_ul, compound_ref, depth + 1, section_data, heading_path, 
                        current_last_rule, []
                    )
                    records.extend(nested_records)
            
            else:
                # Regular list item - collect for unmatched token analysis
                if li_text:
                    first_token = li_text.split()[0] if li_text.split() else ""
                    self.unmatched_tokens.append(first_token)
                
                self.seq_counter += 1
                
                record = self.make_record(
                    section_data, self.seq_counter, li_text, 'list_item',
                    heading_path=heading_path,
                    attached_to=current_last_rule.paragraph_id if current_last_rule else None,
                    links_out=self.extract_links(li, li_text, section_data['base_path']),
                    raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                    parse_confidence='medium' if current_last_rule else 'low',
                )
                
                records.append(record)
        
        return records, current_last_rule
    
    def parse_section(self, file_path: Path) -> List[ParseRecord]:
        """Parse a single section file."""
        with open(file_path) as f:
            data = json.load(f)
        
        section_data = {
            'base_path': data.get('base_path', ''),
            'title': data.get('title', ''),
            'content_id': data.get('content_id', ''),
            'public_updated_at': data.get('public_updated_at', ''),
        }
        
        html_body = data.get('details', {}).get('body', '')
        if not html_body:
            return []
        
        soup = BeautifulSoup(html_body, 'html.parser')
        records = []
        heading_path = []
        last_rule_record = None
        self.current_section_rules = {}  # Reset for each section
        
        # Walk through elements in document order
        for element in soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table', 'div', 'ol', 'ul']):
            
            # Handle headings
            if element.name in ['h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1]) - 2  # h2 = level 0
                heading_text = self.normalize_text(element.get_text())
                
                # Update heading path
                if level < len(heading_path):
                    heading_path = heading_path[:level]
                if level >= len(heading_path):
                    heading_path.extend([''] * (level - len(heading_path) + 1))
                heading_path[level] = heading_text
                
                # Reset last_rule_record when entering new heading section
                last_rule_record = None
                continue
            
            # Handle tables
            elif element.name == 'table':
                table_records = self.process_table_rows(element, heading_path, section_data)
                records.extend(table_records)
                continue
            
            # Handle legislative list wrappers
            elif element.name == 'div' and 'legislative-list-wrapper' in element.get('class', []):
                for ol_ul in element.find_all(['ol', 'ul'], recursive=False):
                    list_records, updated_last_rule = self.process_list_items(
                        ol_ul, last_rule_record.rule_ref if last_rule_record else "", 0,
                        section_data, heading_path, last_rule_record
                    )
                    records.extend(list_records)
                    # Update last_rule_record if we found new rules
                    if updated_last_rule and updated_last_rule != last_rule_record:
                        last_rule_record = updated_last_rule
                continue
            
            # Handle top-level lists (ol/ul with legislative-list class)
            elif element.name in ['ol', 'ul'] and 'legislative-list' in element.get('class', []):
                # Check if this list is not already processed as part of a wrapper
                if not element.find_parent(['div'], class_='legislative-list-wrapper'):
                    list_records, updated_last_rule = self.process_list_items(
                        element, last_rule_record.rule_ref if last_rule_record else "", 0,
                        section_data, heading_path, last_rule_record
                    )
                    records.extend(list_records)
                    # Update last_rule_record if we found new rules
                    if updated_last_rule and updated_last_rule != last_rule_record:
                        last_rule_record = updated_last_rule
                continue
            
            # Handle paragraphs
            elif element.name == 'p':
                p_text = self.extract_text_to_first_list(element)
                if not p_text.strip():
                    continue
                
                raw_html, raw_html_sha256, source_line, source_pos = self.get_element_provenance(element)
                
                # Check if this is a rule
                rule_ref, rule_family, match_end = self.identify_rule(p_text)
                
                if rule_ref:
                    # This is a rule record
                    self.seq_counter += 1
                    
                    text_type = 'deleted' if self.is_deleted(p_text, match_end) else 'rule'
                    
                    record = self.make_record(
                        section_data, self.seq_counter, p_text, text_type,
                        rule_ref=rule_ref, rule_family=rule_family,
                        heading_path=heading_path.copy(),
                        links_out=self.extract_links(element, p_text, section_data['base_path']),
                        raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                        parse_confidence='high' if heading_path else 'medium',
                    )
                    
                    records.append(record)
                    last_rule_record = record
                    key = (section_data['base_path'], self.normalize_reference_key(rule_ref))
                    self.identifier_table[key] = record.paragraph_id
                    self.current_section_rules[rule_ref] = record.paragraph_id
                    # Add to global index
                    self.global_rule_index[self.normalize_reference_key(rule_ref)].append((section_data['base_path'], record.paragraph_id))
                
                elif self.is_deleted(p_text):
                    # Bare deleted paragraph
                    self.seq_counter += 1
                    
                    record = self.make_record(
                        section_data, self.seq_counter, p_text, 'deleted',
                        heading_path=heading_path.copy(),
                        links_out=[],
                        raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                        parse_confidence='medium',
                    )
                    
                    records.append(record)
                
                else:
                    # Unnumbered paragraph with no identifier and not deleted
                    if p_text:
                        first_token = p_text.split()[0] if p_text.split() else ""
                        self.unmatched_tokens.append(first_token)
                    
                    self.seq_counter += 1
                    
                    # Determine if this should be continuation or narrative
                    text_type = 'narrative' if heading_path else 'continuation'
                    
                    record = self.make_record(
                        section_data, self.seq_counter, p_text, text_type,
                        heading_path=heading_path.copy(),
                        attached_to=last_rule_record.paragraph_id if last_rule_record and text_type == 'continuation' else None,
                        links_out=self.extract_links(element, p_text, section_data['base_path']),
                        raw_html=raw_html, source_line=source_line, source_pos=source_pos,
                        parse_confidence='medium' if text_type == 'narrative' else ('medium' if last_rule_record else 'low'),
                    )
                    
                    records.append(record)
        
        return records
    
    def generate_report(self) -> Dict:
        """Generate parsing report with statistics."""
        section_stats = {}
        total_stats = defaultdict(int)
        zero_record_sections = []
        duplicate_containment_count = 0
        unresolved_by_type = defaultdict(int)
        
        # Group records by section
        sections = defaultdict(list)
        for record in self.records:
            sections[record.section_base_path].append(record)
        
        # Check for duplicate containment - records whose full text (40+ chars) appears inside parent's text
        # Build a lookup of paragraph_id to record text for efficiency
        paragraph_text_lookup = {record.paragraph_id: record.text.strip() for record in self.records}
        
        for record in self.records:
            record_text = record.text.strip()
            if len(record_text) >= 40 and record.attached_to:
                # Check if this record's text appears inside the attached_to record's text
                parent_text = paragraph_text_lookup.get(record.attached_to, "")
                if parent_text and len(parent_text) > len(record_text) and record_text in parent_text:
                    duplicate_containment_count += 1
        
        # Count unresolved links by type
        for record in self.records:
            for link in record.links_out:
                if not link.get('resolved', False):
                    unresolved_by_type[link['target_type']] += 1
        
        # Check for sections with zero records
        for base_path in self.section_base_paths:
            if base_path not in sections:
                # Try to get body length for this section
                slug = self.get_slug_from_path(base_path)
                section_file = None
                for file_path in self.raw_rules_path.glob("*.json"):
                    if file_path.stem == slug or slug in file_path.name:
                        section_file = file_path
                        break
                
                body_length = 0
                if section_file:
                    try:
                        with open(section_file) as f:
                            data = json.load(f)
                        body = data.get('details', {}).get('body', '')
                        body_length = len(body)
                    except:
                        pass
                
                zero_record_sections.append({
                    'base_path': base_path,
                    'body_length': body_length
                })
        
        for base_path, section_records in sections.items():
            type_counts = Counter(r.text_type for r in section_records)
            rule_identifiers = [r.rule_ref for r in section_records if r.rule_ref]
            deleted_count = type_counts.get('deleted', 0)
            low_confidence = sum(1 for r in section_records if r.parse_confidence == 'low')
            low_confidence_share = low_confidence / len(section_records) if section_records else 0
            
            # Count records over 2000 characters
            long_records = sum(1 for r in section_records if len(r.text) > 2000)
            
            # Count unresolved cross-references
            unresolved_refs = 0
            for record in section_records:
                unresolved_refs += sum(1 for link in record.links_out if not link['resolved'])
            
            section_stats[base_path] = {
                'type_counts': dict(type_counts),
                'rule_identifiers_count': len(rule_identifiers),
                'deleted_count': deleted_count,
                'low_confidence_share': low_confidence_share,
                'unresolved_cross_references': unresolved_refs,
                'records_over_2000_chars': long_records
            }
            
            # Add to totals
            for text_type, count in type_counts.items():
                total_stats[text_type] += count
            total_stats['total_records'] += len(section_records)
            total_stats['total_unresolved_refs'] += unresolved_refs
            total_stats['total_long_records'] += long_records
        
        # Calculate overall low-confidence share
        total_low_confidence = sum(1 for r in self.records if r.parse_confidence == 'low')
        overall_low_confidence_share = total_low_confidence / len(self.records) if self.records else 0
        
        # Analyze unmatched tokens
        continuation_list_tokens = []
        for record in self.records:
            if record.text_type in ['continuation', 'list_item', 'narrative'] and record.text:
                first_token = record.text.split()[0] if record.text.split() else ""
                if first_token:
                    continuation_list_tokens.append(first_token)
        
        # Get most common token shapes
        common_token_shapes = Counter(continuation_list_tokens).most_common(20)
        
        # Calculate reference resolution rates
        total_appendix_refs = sum(1 for link in [l for r in self.records for l in r.links_out] if link['target_type'] == 'appendix')
        total_part_refs = sum(1 for link in [l for r in self.records for l in r.links_out] if link['target_type'] == 'part')  
        total_paragraph_refs = sum(1 for link in [l for r in self.records for l in r.links_out] if link['target_type'] == 'paragraph')
        
        resolved_appendix = sum(1 for link in [l for r in self.records for l in r.links_out] 
                               if link['target_type'] == 'appendix' and link.get('resolved', False))
        resolved_part = sum(1 for link in [l for r in self.records for l in r.links_out] 
                           if link['target_type'] == 'part' and link.get('resolved', False))
        resolved_paragraph = sum(1 for link in [l for r in self.records for l in r.links_out] 
                                if link['target_type'] == 'paragraph' and link.get('resolved', False))
        
        appendix_resolution_rate = (resolved_appendix / total_appendix_refs * 100) if total_appendix_refs else 0
        part_resolution_rate = (resolved_part / total_part_refs * 100) if total_part_refs else 0
        paragraph_resolution_rate = (resolved_paragraph / total_paragraph_refs * 100) if total_paragraph_refs else 0
        
        # Get most common unresolved names
        common_unresolved = Counter(self.unresolved_names).most_common(20)
        
        return {
            'sections': section_stats,
            'totals': dict(total_stats),
            'overall_low_confidence_share': overall_low_confidence_share,
            'total_sections': len(sections),
            'zero_record_sections': zero_record_sections,
            'common_unmatched_token_shapes': common_token_shapes,
            'duplicate_containment_count': duplicate_containment_count,
            'unresolved_by_type': dict(unresolved_by_type),
            'common_unresolved_names': common_unresolved,
            'resolution_rates': {
                'appendix': appendix_resolution_rate,
                'part': part_resolution_rate,  
                'paragraph': paragraph_resolution_rate
            },
            'reference_counts': {
                'total_appendix_refs': total_appendix_refs,
                'total_part_refs': total_part_refs,
                'total_paragraph_refs': total_paragraph_refs,
                'resolved_appendix': resolved_appendix,
                'resolved_part': resolved_part,
                'resolved_paragraph': resolved_paragraph
            }
        }
    
    def post_parse_link_resolution(self) -> None:
        """Re-resolve all unresolved paragraph links using complete identifier tables."""
        # Build complete identifier table from all records (normalized keys)
        complete_identifier_table = {}  # (section_base_path, normalized_rule_ref) -> paragraph_id
        complete_global_index = defaultdict(list)  # normalized_rule_ref -> [(section_path, paragraph_id), ...]
        
        for record in self.records:
            if record.rule_ref and record.text_type in ['rule', 'deleted', 'subparagraph']:
                normalized_ref = self.normalize_reference_key(record.rule_ref)
                key = (record.section_base_path, normalized_ref)
                complete_identifier_table[key] = record.paragraph_id
                complete_global_index[normalized_ref].append((record.section_base_path, record.paragraph_id))
        
        # Track resolution statistics
        before_resolved = 0
        after_resolved = 0
        total_paragraph_links = 0
        
        # Re-resolve all unresolved paragraph links
        for record in self.records:
            for link in record.links_out:
                if link['target_type'] == 'paragraph':
                    total_paragraph_links += 1
                    if link.get('resolved', False):
                        before_resolved += 1
                        after_resolved += 1
                        continue
                    
                    # Try resolution
                    normalized_target = self.normalize_reference_key(link['target'])
                    resolved_to = None
                    candidates = None
                    
                    # 1. Try same section
                    key = (record.section_base_path, normalized_target)
                    if key in complete_identifier_table:
                        resolved_to = complete_identifier_table[key]
                    
                    # 2. Try sections mentioned in same record's other links
                    if not resolved_to:
                        mentioned_sections = set()
                        for other_link in record.links_out:
                            if other_link.get('resolved_to') and other_link['target_type'] in ['appendix', 'part']:
                                # Extract base path from resolved_to if it's a section reference
                                if other_link['resolved_to'] != 'absent':
                                    mentioned_sections.add(other_link['resolved_to'])
                        
                        for section_bp in mentioned_sections:
                            key = (section_bp, normalized_target)
                            if key in complete_identifier_table:
                                resolved_to = complete_identifier_table[key]
                                break
                    
                    # 3. Try unique corpus-wide
                    if not resolved_to:
                        candidates_list = complete_global_index.get(normalized_target, [])
                        if len(candidates_list) == 1:
                            resolved_to = candidates_list[0][1]  # paragraph_id
                        elif len(candidates_list) > 1:
                            candidates = sorted(set(c[0] for c in candidates_list))  # section paths
                    
                    # Update link
                    if resolved_to:
                        link['resolved'] = True
                        link['resolved_to'] = resolved_to
                        after_resolved += 1
                    elif candidates:
                        link['candidates'] = candidates
        
        # Print resolution statistics
        if total_paragraph_links > 0:
            before_rate = (before_resolved / total_paragraph_links) * 100
            after_rate = (after_resolved / total_paragraph_links) * 100
            print(f"  Paragraph link resolution: {before_rate:.1f}% -> {after_rate:.1f}% ({after_resolved - before_resolved} additional resolved)")
    
    def run(self) -> None:
        """Run the complete parsing process."""
        from datetime import datetime
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        timestamped_output_dir = self.interim_path / f"{timestamp}_rules-parse"
        timestamped_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading section titles...")
        self.load_section_titles()
        
        print(f"Processing {len(list(self.raw_rules_path.glob('*.json'))) - 1} sections...")
        
        # Process each section file
        for file_path in sorted(self.raw_rules_path.glob("*.json")):
            if file_path.name in ['manifest.json']:
                continue
            
            print(f"  Parsing {file_path.name}...")
            section_records = self.parse_section(file_path)
            self.records.extend(section_records)
        
        # Update confidence scores after all parsing
        for record in self.records:
            record.parse_confidence = self.calculate_confidence(record)
        
        # Post-parse link resolution pass
        print(f"Post-parsing link resolution...")
        self.post_parse_link_resolution()
        
        # Write output files
        output_jsonl = timestamped_output_dir / "rules-paragraphs.jsonl"
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for record in self.records:
                record_dict = {
                    'paragraph_id': record.paragraph_id,
                    'seq': record.seq,
                    'rule_ref': record.rule_ref,
                    'rule_family': record.rule_family,
                    'section_base_path': record.section_base_path,
                    'section_title': record.section_title,
                    'heading_path': record.heading_path,
                    'text': record.text,
                    'text_type': record.text_type,
                    'attached_to': record.attached_to,
                    'links_out': record.links_out,
                    'content_id': record.content_id,
                    'public_updated_at': record.public_updated_at,
                    'snapshot_date': record.snapshot_date,
                    'licence': record.licence,
                    'raw_html': record.raw_html,
                    'raw_html_sha256': record.raw_html_sha256,
                    'source_line': record.source_line,
                    'source_pos': record.source_pos,
                    'parse_confidence': record.parse_confidence
                }
                f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')
        
        # Generate and write report
        report = self.generate_report()
        report_path = timestamped_output_dir / "parse-report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\nParsing complete:")
        print(f"  Records: {len(self.records)}")
        print(f"  Output: {output_jsonl}")
        print(f"  Report: {report_path}")
        print(f"  Low confidence share: {report['overall_low_confidence_share']:.1%}")
        
        # Print resolution rates
        print(f"\nResolution rates:")
        print(f"  Appendix references: {report['resolution_rates']['appendix']:.1f}%")
        print(f"  Part references: {report['resolution_rates']['part']:.1f}%") 
        print(f"  Paragraph references: {report['resolution_rates']['paragraph']:.1f}%")
        
        # Print zero record sections
        if report['zero_record_sections']:
            print(f"\nSections with zero records:")
            for section in report['zero_record_sections']:
                print(f"  {section['base_path']} (body length: {section['body_length']} chars)")
        
        # Print duplicate containment count
        print(f"\nDuplicate containment count: {report['duplicate_containment_count']}")
        
        # Print unresolved by type
        if report['unresolved_by_type']:
            print(f"\nUnresolved references by type:")
            for ref_type, count in report['unresolved_by_type'].items():
                print(f"  {ref_type}: {count}")
        
        # Print common unmatched token shapes
        if report['common_unmatched_token_shapes']:
            print(f"\n20 most common first-token shapes in continuation/list_item/narrative records:")
            for token, count in report['common_unmatched_token_shapes'][:10]:  # Show fewer
                print(f"  '{token}': {count}")
        
        # Print common unresolved names for alias creation
        if report['common_unresolved_names']:
            print(f"\n10 most common unresolved reference names (for config/section_aliases.yaml):")
            for name, count in report['common_unresolved_names'][:10]:  # Show fewer
                print(f"  '{name}': {count}")
        
        # Exit with error if low confidence is too high
        if report['overall_low_confidence_share'] > 0.05:  # 5% threshold
            print(f"ERROR: Low confidence share {report['overall_low_confidence_share']:.1%} exceeds 5%")
            exit(1)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Parse immigration rules to structured records")
    ap.add_argument("--config", default="config/paths.yaml", help="Path to paths.yaml")
    args = ap.parse_args()
    parser = RulesParser(config_path=args.config)
    parser.run()


if __name__ == "__main__":
    main()