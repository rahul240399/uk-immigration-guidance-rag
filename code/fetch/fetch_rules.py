"""
Immigration Rules Raw Data Extractor
------------------------------------
Fetches each immigration rules endpoint from GOV.UK Content API and saves
the complete API response unchanged as <slug>.json files.

Output:
- raw_rules/<slug>.json: Complete API response for each endpoint
- fetch-log.csv: Request log with timestamps, status, metadata
- manifest.json: Snapshot metadata with file hashes and completeness info

Licence: Contains public sector information licensed under the
         Open Government Licence v3.0.
         https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
"""

import csv
import hashlib
import json
import logging
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── configuration ──────────────────────────────────────────────────────────────
BASE_API       = "https://www.gov.uk/api/content"
RATE_LIMIT     = 0.5          # seconds between requests (2 req/sec)
RETRY_WAITS    = [2, 5, 10]   # back-off schedule (seconds)
ENDPOINTS_FILE = "discovered_endpoints.json"
USER_AGENT     = "WarwickMScDissertation/1.0 (u5757819@live.warwick.ac.uk)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_output_path(config_path: str = "config/paths.yaml") -> Path:
    """Load output path from config."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    return Path(config["raw_rules"])


def slug_from_path(api_path: str) -> str:
    """Turn /guidance/immigration-rules/foo-bar → foo-bar"""
    return api_path.rstrip("/").split("/")[-1] or "index"


def fetch_with_retry(session: requests.Session, api_path: str) -> tuple[dict | None, int, int]:
    """Fetch one endpoint with retries. Returns (parsed_json, status_code, response_bytes)."""
    url = BASE_API + api_path
    for attempt, wait in enumerate([0] + RETRY_WAITS, start=1):
        if wait:
            time.sleep(wait)
        try:
            r = session.get(url, timeout=20)
            response_bytes = len(r.content)
            
            if r.status_code == 200:
                return r.json(), r.status_code, response_bytes
            if r.status_code == 404:
                log.warning("404 %s – skipping", api_path)
                return None, r.status_code, response_bytes
            log.warning("HTTP %s on attempt %d for %s", r.status_code, attempt, api_path)
            
        except requests.RequestException as exc:
            log.warning("Request error attempt %d for %s: %s", attempt, api_path, exc)
    
    log.error("All retries exhausted for %s", api_path)
    return None, 0, 0


def calculate_sha256(data: dict) -> str:
    """Calculate SHA256 hash of JSON response"""
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fetch raw immigration rules from GOV.UK API")
    ap.add_argument("--config", default="config/paths.yaml", help="Path to paths.yaml")
    args = ap.parse_args()

    # Load configuration
    output_dir = load_output_path(args.config)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load endpoints
    endpoints_path = Path(ENDPOINTS_FILE)
    if not endpoints_path.exists():
        log.error("Endpoints file not found: %s", ENDPOINTS_FILE)
        return

    with endpoints_path.open() as f:
        endpoints = json.load(f)

    immigration_rules = endpoints.get("immigration_rules", [])
    total = len(immigration_rules)
    
    log.info("Starting extraction of %d immigration rules documents", total)
    log.info("Output directory: %s", output_dir)

    # Setup session
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })

    # Initialize tracking
    fetch_log_path = output_dir / "fetch-log.csv"
    manifest_path = output_dir / "manifest.json"
    
    fetch_log = []
    file_hashes = {}
    fetched_count = 0
    missing_list = []
    last_request = 0.0
    
    snapshot_date = datetime.now(timezone.utc).isoformat()

    # Process each endpoint
    for i, api_path in enumerate(immigration_rules, 1):
        # Rate limiting
        elapsed = time.time() - last_request
        if elapsed < RATE_LIMIT:
            time.sleep(RATE_LIMIT - elapsed)
        last_request = time.time()

        # Fetch data
        request_time = datetime.now(timezone.utc)
        data, status_code, response_bytes = fetch_with_retry(session, api_path)
        
        url = BASE_API + api_path
        content_id = data.get("content_id", "") if data else ""
        public_updated_at = data.get("public_updated_at", "") if data else ""
        
        # Calculate hash and log request
        if data:
            response_hash = calculate_sha256(data)
            slug = slug_from_path(api_path)
            
            # Save complete API response
            output_file = output_dir / f"{slug}.json"
            output_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            
            file_hashes[f"{slug}.json"] = response_hash
            fetched_count += 1
        else:
            response_hash = ""
            missing_list.append(api_path)

        # Log the request
        fetch_log.append({
            "timestamp": request_time.isoformat(),
            "url": url,
            "status": status_code,
            "bytes": response_bytes,
            "content_id": content_id,
            "public_updated_at": public_updated_at,
            "sha256": response_hash
        })

        if i % 10 == 0 or i == total:
            log.info("%d/%d done (fetched=%d)", i, total, fetched_count)

    # Write fetch log
    with open(fetch_log_path, 'w', newline='', encoding='utf-8') as f:
        if fetch_log:
            writer = csv.DictWriter(f, fieldnames=fetch_log[0].keys())
            writer.writeheader()
            writer.writerows(fetch_log)

    # Write manifest
    manifest = {
        "snapshot_date": snapshot_date,
        "licence": "OGL-3.0",
        "licence_url": "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
        "sections_planned": total,
        "sections_fetched": fetched_count,
        "file_hashes": file_hashes,
        "missing_list": missing_list,
        "user_agent": USER_AGENT,
        "source_endpoints_file": str(endpoints_path.resolve())
    }
    
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    log.info("Extraction complete:")
    log.info("  Files saved: %d/%d", fetched_count, total)
    log.info("  Missing: %d", len(missing_list))
    log.info("  Fetch log: %s", fetch_log_path)
    log.info("  Manifest: %s", manifest_path)
    
    # Verify all files have details.body key
    files_with_body = 0
    for file_path in output_dir.glob("*.json"):
        if file_path.name in ['manifest.json', 'fetch-log.csv']:
            continue
        try:
            with open(file_path) as f:
                data = json.load(f)
                if "details" in data and "body" in data["details"]:
                    files_with_body += 1
        except Exception as e:
            log.warning("Error checking %s: %s", file_path, e)
    
    log.info("  Files with details.body: %d/%d", files_with_body, fetched_count)


if __name__ == "__main__":
    main()