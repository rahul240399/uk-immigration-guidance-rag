# UK Immigration API Discovery

Clean implementation for discovering UK Immigration Rules and Caseworker Guidance endpoints using GOV.UK Content API.

## Features

- **Immigration Rules Discovery**: Extracts endpoints from manual structure navigation
- **Caseworker Guidance Discovery**: Explores collections recursively for guidance documents  
- **Accessibility Validation**: Tests each endpoint and logs metadata
- **Rate Limiting**: Compliant 2 requests/second to respect GOV.UK API limits
- **Clean Logging**: Precise progress reporting and error handling

## Usage

### Run Discovery
```bash
python run_discovery.py
```

### Output Files

1. **discovered_endpoints.json** - Clean endpoint lists:
```json
{
  "immigration_rules": ["/guidance/immigration-rules/..."],
  "caseworker_guidance": ["/government/publications/..."], 
  "summary": {
    "immigration_rules_count": 101,
    "caseworker_guidance_count": 581,
    "total_count": 682,
    "discovered_at": "2026-09-13T11:30:00.000000"
  }
}
```

2. **accessibility_metadata_YYYYMMDD_HHMMSS.json** - Validation results:
```json
{
  "metadata": [
    {
      "endpoint_path": "/guidance/immigration-rules/immigration-rules-index",
      "api_url": "https://www.gov.uk/api/content/...",
      "accessible": true,
      "status_code": 200,
      "content_id": "b60fae57-7bb7-4183-b753-a06f77375762",
      "title": "Immigration Rules: Index", 
      "document_type": "manual_section",
      "validated_at": "2026-09-13T11:30:15.123456"
    }
  ],
  "summary": {
    "total_endpoints": 682,
    "accessible_endpoints": 682,
    "validation_timestamp": "2026-09-13T11:35:00.000000"
  }
}
```

## API Structure

### Immigration Rules
- **Source**: `/guidance/immigration-rules` (manual document)
- **Method**: Extract from `child_section_groups[].child_sections[].base_path`
- **Pattern**: `/guidance/immigration-rules/*`

### Caseworker Guidance  
- **Source**: `/government/collections/visas-and-immigration-operational-guidance`
- **Method**: Recursive collection exploration
- **Patterns**: `/government/publications/*`, `/guidance/*`
- **Filters**: Excludes statistics, news, speeches, consultations

## Requirements

- Python 3.7+
- requests>=2.31.0

## Implementation Details

- **Rate Limiting**: 500ms between requests (2/sec)
- **Error Handling**: Continues on individual endpoint failures
- **Progress Logging**: Reports every 10 validated endpoints
- **Timeout**: 10 seconds per API request
- **Clean Code**: ~150 lines, well-documented, minimal dependencies
