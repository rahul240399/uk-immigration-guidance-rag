"""
Fetch raw immigration rules from GOV.UK Content API.

Reads a plan CSV (base_path, title) produced by ``discover_rules``,
fetches each section's full API response unchanged, and writes
``fetch-log.csv`` and ``manifest.json`` alongside the saved files.

Usage::

    python -m code.fetch.fetch_rules \\
        --plan manifests/2026-09-14_corpus_rules-fetch-plan.csv \\
        --out  data/raw-rules

Licence: Contains public sector information licensed under the
         Open Government Licence v3.0.
"""

import argparse
import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from code.fetch.http_client import get_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def slug_from_path(api_path: str) -> str:
    """``/guidance/immigration-rules/foo-bar`` → ``foo-bar``"""
    return api_path.rstrip("/").split("/")[-1] or "index"


def load_plan(plan_path: Path) -> list[dict]:
    """Read the fetch-plan CSV into a list of {base_path, title} dicts."""
    with open(plan_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch raw immigration rules sections")
    ap.add_argument("--plan", required=True,
                    help="Path to the rules-fetch-plan CSV")
    ap.add_argument("--out", required=True,
                    help="Output directory for raw JSON files")
    args = ap.parse_args()

    plan = load_plan(Path(args.plan))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(plan)
    log.info("Fetching %d sections → %s", total, out_dir)

    # Fetch log — one row per HTTP attempt
    fetch_log_path = out_dir / "fetch-log.csv"
    fetch_log_fh = open(fetch_log_path, "w", newline="", encoding="utf-8")
    log_fields = ["timestamp", "url", "status", "bytes",
                  "content_id", "public_updated_at", "sha256"]
    log_writer = csv.DictWriter(fetch_log_fh, fieldnames=log_fields)
    log_writer.writeheader()

    def write_log_row(ts, url, status, nbytes, cid, pub, sha):
        log_writer.writerow({
            "timestamp": ts, "url": url, "status": status,
            "bytes": nbytes, "content_id": cid,
            "public_updated_at": pub, "sha256": sha,
        })

    snapshot_date = datetime.now(timezone.utc).isoformat()
    file_hashes: dict[str, str] = {}
    missing: list[str] = []
    fetched = 0

    for i, row in enumerate(plan, 1):
        bp = row["base_path"]
        data = get_json(bp, log_writer=write_log_row)

        if data is None:
            missing.append(bp)
        else:
            slug = slug_from_path(bp)
            dest = out_dir / f"{slug}.json"
            body = json.dumps(data, ensure_ascii=False, indent=2)
            dest.write_text(body, encoding="utf-8")
            sha = hashlib.sha256(
                json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            file_hashes[f"{slug}.json"] = sha
            fetched += 1

        if i % 10 == 0 or i == total:
            log.info("%d/%d done (fetched=%d)", i, total, fetched)

    fetch_log_fh.close()

    # Manifest
    manifest = {
        "snapshot_date": snapshot_date,
        "licence": "OGL-3.0",
        "licence_url": "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
        "sections_planned": total,
        "sections_fetched": fetched,
        "file_hashes": file_hashes,
        "missing_list": missing,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Done: %d/%d fetched, %d missing", fetched, total, len(missing))
    log.info("Log: %s  Manifest: %s", fetch_log_path, manifest_path)


if __name__ == "__main__":
    main()
