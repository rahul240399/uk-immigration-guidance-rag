"""
Fetch caseworker guidance publications from GOV.UK Content API.

Reads a plan CSV produced by ``discover_guidance``, fetches each
publication's full API record and its attachments (HTML preferred,
PDF fallback), and writes fetch-log.csv and manifest.json.

Usage::

    python -m code.fetch.fetch_guidance \\
        --plan manifests/2026-09-14_corpus_guidance-fetch-plan.csv \\
        --config config/paths.yaml

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

import yaml

from code.fetch.http_client import get_json, get_bytes

MAX_PDF_MB = 25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def slug_from_path(api_path: str) -> str:
    return api_path.rstrip("/").split("/")[-1] or "index"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(data: dict) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch caseworker guidance publications")
    ap.add_argument("--plan", required=True, help="Path to guidance-fetch-plan CSV")
    ap.add_argument("--config", default="config/paths.yaml", help="Path to paths.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    out_root = Path(config["raw_guidance"])
    out_root.mkdir(parents=True, exist_ok=True)

    # Read plan
    with open(args.plan, newline="", encoding="utf-8") as f:
        plan = list(csv.DictReader(f))

    total = len(plan)
    log.info("Fetching %d publications → %s", total, out_root)

    # Fetch log
    log_path = out_root / "fetch-log.csv"
    log_fh = open(log_path, "w", newline="", encoding="utf-8")
    log_fields = ["timestamp", "url", "status", "bytes",
                  "content_id", "public_updated_at", "sha256"]
    log_writer = csv.DictWriter(log_fh, fieldnames=log_fields)
    log_writer.writeheader()

    def write_log(ts, url, status, nbytes, cid, pub, sha):
        log_writer.writerow({
            "timestamp": ts, "url": url, "status": status,
            "bytes": nbytes, "content_id": cid,
            "public_updated_at": pub, "sha256": sha,
        })

    snapshot_date = datetime.now(timezone.utc).isoformat()
    file_hashes: dict[str, str] = {}
    route_counts: dict[str, int] = {}
    html_count = 0
    pdf_count = 0
    skipped_count = 0
    flagged_licences: list[str] = []
    seen_slugs: set[str] = set()

    for i, row in enumerate(plan, 1):
        bp = row["base_path"]
        # First route listed (for folder placement)
        route = row["route"].split(";")[0]
        slug = slug_from_path(bp)
        has_html = row.get("has_html_attachment", "").strip().lower() == "true"

        # Deduplicate: a document listed under several routes is fetched once
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        route_dir = out_root / route
        route_dir.mkdir(parents=True, exist_ok=True)

        # Fetch the publication record
        data = get_json(bp, log_writer=write_log)
        if data is None:
            log.warning("Skipping %s — could not fetch", bp)
            skipped_count += 1
            continue

        # Save full API record
        pub_path = route_dir / f"{slug}.json"
        pub_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        pub_path.write_bytes(pub_bytes)
        file_hashes[f"{route}/{slug}.json"] = sha256_json(data)
        route_counts[route] = route_counts.get(route, 0) + 1

        # Check licence
        licence_line = data.get("details", {}).get("body", "") or ""
        pub_licence = data.get("licence", "")
        if "OGL" not in licence_line and "open government licence" not in licence_line.lower():
            # Check top-level licence field or links
            links_licence = ""
            for lnk in data.get("links", {}).get("available_translations", []):
                pass  # not useful
            # If no OGL reference anywhere, flag it
            if "OGL" not in pub_licence and "ogl" not in str(data.get("links", {})).lower():
                flagged_licences.append(bp)

        # Process attachments
        attachments = data.get("details", {}).get("attachments", [])

        if has_html:
            # Fetch HTML attachment records
            html_atts = [a for a in attachments if a.get("attachment_type") == "html"]
            for att in html_atts:
                att_id = att.get("id", "unknown")
                url_path = att.get("url", "")
                api_path = (url_path if url_path.startswith("/")
                            else url_path.replace("https://www.gov.uk", ""))

                att_data = get_json(api_path, log_writer=write_log)
                if att_data is None:
                    log.warning("  Skipping HTML attachment %s for %s", att_id, slug)
                    skipped_count += 1
                    continue

                att_path = route_dir / f"{slug}__{att_id}.json"
                att_bytes = json.dumps(att_data, ensure_ascii=False, indent=2).encode("utf-8")
                att_path.write_bytes(att_bytes)
                file_hashes[f"{route}/{slug}__{att_id}.json"] = sha256_json(att_data)
                html_count += 1
        else:
            # Fetch PDF attachments
            pdf_atts = [a for a in attachments
                        if a.get("attachment_type") == "file"
                        and ("pdf" in a.get("content_type", "").lower()
                             or str(a.get("url", "")).lower().endswith(".pdf"))]
            for att in pdf_atts:
                att_id = att.get("id", "unknown")
                pdf_url = att.get("url", "")
                if not pdf_url.startswith("http"):
                    pdf_url = "https://www.gov.uk" + pdf_url

                pdf_bytes = get_bytes(pdf_url)
                if pdf_bytes is None:
                    log.warning("  Skipping PDF %s for %s — unavailable", att_id, slug)
                    skipped_count += 1
                    continue

                size_mb = len(pdf_bytes) / (1024 * 1024)
                if size_mb > MAX_PDF_MB:
                    log.warning("  Skipping PDF %s (%.1fMB > %dMB)", att_id, size_mb, MAX_PDF_MB)
                    skipped_count += 1
                    continue

                pdf_path = route_dir / f"{slug}__{att_id}.pdf"
                pdf_path.write_bytes(pdf_bytes)
                file_hashes[f"{route}/{slug}__{att_id}.pdf"] = sha256_bytes(pdf_bytes)
                pdf_count += 1

        if i % 5 == 0 or i == total:
            log.info("%d/%d done (html=%d pdf=%d skipped=%d)",
                     i, total, html_count, pdf_count, skipped_count)

    log_fh.close()

    # Manifest
    manifest = {
        "snapshot_date": snapshot_date,
        "licence": "OGL-3.0",
        "licence_url": "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
        "publications_planned": total,
        "publications_fetched": len(seen_slugs) - skipped_count,
        "route_counts": route_counts,
        "html_attachment_count": html_count,
        "pdf_attachment_count": pdf_count,
        "skipped_count": skipped_count,
        "flagged_non_ogl_licence": flagged_licences,
        "file_hashes": file_hashes,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Done: %d publications, %d HTML, %d PDF, %d skipped, %d flagged",
             len(seen_slugs), html_count, pdf_count, skipped_count, len(flagged_licences))
    log.info("Log: %s  Manifest: %s", log_path, manifest_path)


if __name__ == "__main__":
    main()
