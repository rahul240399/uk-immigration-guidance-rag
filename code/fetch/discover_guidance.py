"""
Build a caseworker guidance fetch plan from config/routes.yaml.

Default mode reads explicit endpoint lists from the config, validates
each via the API, and writes a fetch-plan CSV with attachment metadata.

The ``--list-subcollections`` mode walks the GOV.UK collection tree
and prints what it finds (useful for populating routes.yaml).

Usage::

    # Default: build plan from routes.yaml
    python -m code.fetch.discover_guidance --out manifests/

    # Optional routes included
    python -m code.fetch.discover_guidance --out manifests/ --include-optional

    # Explore the collection tree (no plan written)
    python -m code.fetch.discover_guidance --list-subcollections

Licence: Contains public sector information licensed under the
         Open Government Licence v3.0.
"""

import argparse
import csv
import logging
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

import yaml

from code.fetch.http_client import get_json

COLLECTION_PATH = "/government/collections/visas-and-immigration-operational-guidance"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Plan from routes.yaml ────────────────────────────────────────────

def load_routes(config_path: str, include_optional: bool = False) -> list[dict]:
    """
    Collect base_paths from config/routes.yaml.

    Returns a deduplicated list of ``{"base_path": ..., "routes": ...}``
    dicts.  A path listed under several routes gets one entry with the
    route names joined by ``';'``.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # OrderedDict to preserve insertion order and accumulate routes
    seen: OrderedDict[str, list[str]] = OrderedDict()

    # routes.<name>.guidance
    for route_name, route_cfg in (cfg.get("routes") or {}).items():
        for bp in route_cfg.get("guidance") or []:
            seen.setdefault(bp, []).append(route_name)

    # cross_cutting.guidance
    for bp in (cfg.get("cross_cutting") or {}).get("guidance") or []:
        seen.setdefault(bp, []).append("cross_cutting")

    # optional.guidance (only if flag set)
    if include_optional:
        for bp in (cfg.get("optional") or {}).get("guidance") or []:
            seen.setdefault(bp, []).append("optional")

    return [{"base_path": bp, "routes": ";".join(routes)}
            for bp, routes in seen.items()]


def validate_and_enrich(entries: list[dict]) -> list[dict]:
    """
    Fetch each publication's API record, validate it, and add metadata.

    Fails the run if any path returns 404 or has a non-guidance
    document_type.
    """
    enriched = []
    failures = []

    for entry in entries:
        bp = entry["base_path"]
        data = get_json(bp)

        if data is None:
            failures.append(f"  404 or unreachable: {bp}")
            continue

        doc_type = data.get("document_type", "")
        if doc_type != "guidance":
            failures.append(f"  document_type={doc_type!r} (expected 'guidance'): {bp}")
            continue

        attachments = data.get("details", {}).get("attachments", [])
        has_html = any(a.get("attachment_type") == "html" for a in attachments)
        has_pdf = any(
            "pdf" in a.get("content_type", "").lower()
            or str(a.get("url", "")).lower().endswith(".pdf")
            for a in attachments
        )

        enriched.append({
            "route": entry["routes"],
            "base_path": bp,
            "title": data.get("title", ""),
            "document_type": doc_type,
            "public_updated_at": data.get("public_updated_at", ""),
            "has_html_attachment": has_html,
            "has_pdf_attachment": has_pdf,
        })

    if failures:
        log.error("Validation failed for %d path(s):", len(failures))
        for msg in failures:
            log.error(msg)
        sys.exit(1)

    log.info("Validated %d publications", len(enriched))
    return enriched


def write_plan(rows: list[dict], out_dir: Path) -> Path:
    """Write the fetch-plan CSV and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    plan_path = out_dir / f"{today}_corpus_guidance-fetch-plan.csv"

    fieldnames = ["route", "base_path", "title", "document_type",
                  "public_updated_at", "has_html_attachment", "has_pdf_attachment"]

    with open(plan_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote plan (%d rows) to %s", len(rows), plan_path)
    return plan_path


# ── Collection-walking mode (--list-subcollections) ──────────────────

def list_subcollections(collection_path: str = COLLECTION_PATH) -> None:
    """Walk the collection tree and print every sub-collection and document."""
    data = get_json(collection_path)
    if data is None:
        log.error("Could not fetch collection at %s", collection_path)
        return

    for doc in data.get("links", {}).get("documents", []):
        bp = doc.get("base_path", "")
        doc_type = doc.get("document_type", "")
        title = doc.get("title", "")

        if doc_type == "document_collection":
            sub_data = get_json(bp)
            if sub_data:
                sub_docs = [
                    d.get("base_path", "")
                    for d in sub_data.get("links", {}).get("documents", [])
                    if d.get("document_type") != "document_collection"
                ]
                print(f"[collection] {title}: {len(sub_docs)} documents")
                for sbp in sub_docs:
                    print(f"  {sbp}")
            else:
                print(f"[collection] {title}: FETCH FAILED")
        else:
            print(f"[document]   {bp}  ({title})")


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a caseworker guidance fetch plan from routes.yaml")
    ap.add_argument("--out", default="manifests",
                    help="Directory for the plan CSV (default: manifests/)")
    ap.add_argument("--routes", default="config/routes.yaml",
                    help="Path to routes.yaml (default: config/routes.yaml)")
    ap.add_argument("--include-optional", action="store_true",
                    help="Include paths from the optional section")
    ap.add_argument("--list-subcollections", action="store_true",
                    help="Walk the GOV.UK collection tree and print results (no plan written)")
    args = ap.parse_args()

    if args.list_subcollections:
        list_subcollections()
        return

    entries = load_routes(args.routes, include_optional=args.include_optional)
    log.info("Loaded %d paths from %s", len(entries), args.routes)

    rows = validate_and_enrich(entries)
    write_plan(rows, Path(args.out))


if __name__ == "__main__":
    main()
