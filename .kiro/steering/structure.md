# Project Structure Guidelines

Stage outputs are written to interim/<YYYY-MM-DD_HHMM>_<stage>/; existing folders are never overwritten.

## Rules Parser Data Contract

Each record in rules-paragraphs.jsonl must contain these fields:
- `paragraph_id`: string (unique across file)
- `seq`: integer 
- `rule_ref`: string or null
- `rule_family`: string or null
- `section_base_path`: string
- `section_title`: string
- `heading_path`: array of strings
- `text`: string
- `text_type`: string
- `attached_to`: string or null
- `links_out`: array of objects
- `content_id`: string
- `public_updated_at`: string (ISO datetime)
- `snapshot_date`: string (ISO datetime)
- `licence`: string
- `raw_html`: string
- `raw_html_sha256`: string
- `source_line`: integer or null
- `source_pos`: integer or null
- `parse_confidence`: string