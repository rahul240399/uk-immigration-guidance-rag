"""
Discover caseworker guidance endpoints from the GOV.UK collection.

Reads ``/government/collections/visas-and-immigration-operational-guidance``,
recursively explores sub-collections, and writes a fetch-plan CSV.

Usage::

    python -m code.fetch.discover_guidance --out manifests/

Licence: Contains public sector information licensed under the
         Open Government Licence v3.0.
"""

import argparse
import csv
import logging
from datetime import date
from pathlib import Path

from code.fetch.http_client import get_json

COLLECTION_PATH = "/government/collections/visas-and-immigration-operational-guidance"

EXCLUDE_PATTERNS = ["/statistics/", "/news/", "/speeches/", "/consultations/"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _is_guidance_document(path: str) -> bool:
    """Check if path is a guidance document (not statistics, news, etc.)."""
    return (
        path.startswith("/government/publications/")
        or path.startswith("/guidance/")
    ) and not any(ex in path for ex in EXCLUDE_PATTERNS)


def _extract_documents(collection_data: dict) -> list[dict]:
    """Extract guidance documents from a collection's links.documents."""
    docs = []
    for doc in collection_data.get("links", {}).get("documents", []):
        bp = doc.get("base_path", "")
        if _is_guidance_document(bp) and doc.get("document_type") != "document_collection":
            docs.append({"base_path": bp, "title": doc.get("title", "")})
    return docs


def discover(collection_path: str = COLLECTION_PATH) -> list[dict]:
    """
    Fetch the guidance collection and return a list of
    ``{"base_path": ..., "title": ...}`` dicts for every guidance document.
    """
    data = get_json(collection_path)
    if data is None:
        log.error("Could not fetch collection at %s", collection_path)
        return []

    sections: list[dict] = []

    for doc in data.get("links", {}).get("documents", []):
        bp = doc.get("base_path", "")
        doc_type = doc.get("document_type", "")

        if doc_type == "document_collection":
            # Recurse into sub-collection
            sub_data = get_json(bp)
            if sub_data:
                sub_docs = _extract_documents(sub_data)
                sections.extend(sub_docs)
                log.info("  Sub-collection %s: %d documents",
                         doc.get("title", bp), len(sub_docs))
            else:
                log.warning("  Failed to fetch sub-collection: %s", bp)
        elif _is_guidance_document(bp):
            sections.append({"base_path": bp, "title": doc.get("title", "")})

    # Deduplicate by base_path, preserving order
    seen = set()
    unique = []
    for s in sections:
        if s["base_path"] not in seen:
            seen.add(s["base_path"])
            unique.append(s)

    log.info("Discovered %d guidance documents from %s", len(unique), collection_path)
    return unique


def write_plan(sections: list[dict], out_dir: Path) -> Path:
    """Write the fetch-plan CSV and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    plan_path = out_dir / f"{today}_corpus_guidance-fetch-plan.csv"
    with open(plan_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["base_path", "title"])
        writer.writeheader()
        writer.writerows(sections)
    log.info("Wrote plan to %s", plan_path)
    return plan_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Discover caseworker guidance endpoints and write a fetch plan")
    ap.add_argument("--out", default="manifests",
                    help="Directory for the plan CSV (default: manifests/)")
    args = ap.parse_args()

    sections = discover()
    if sections:
        write_plan(sections, Path(args.out))


if __name__ == "__main__":
    main()
