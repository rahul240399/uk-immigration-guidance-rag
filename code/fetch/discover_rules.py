"""
Discover immigration rules sections from the GOV.UK manual record.

Reads ``/guidance/immigration-rules``, extracts
``details.child_section_groups[].child_sections[].base_path`` and titles,
and writes a fetch-plan CSV.

Usage::

    python -m code.fetch.discover_rules --out manifests/

Licence: Contains public sector information licensed under the
         Open Government Licence v3.0.
"""

import argparse
import csv
import logging
from datetime import date
from pathlib import Path

from code.fetch.http_client import get_json

MANUAL_PATH = "/guidance/immigration-rules"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def discover(manual_path: str = MANUAL_PATH) -> list[dict]:
    """
    Fetch the manual record and return a list of
    ``{"base_path": ..., "title": ...}`` dicts for every child section.
    """
    data = get_json(manual_path)
    if data is None:
        log.error("Could not fetch manual record at %s", manual_path)
        return []

    sections = []
    for group in data.get("details", {}).get("child_section_groups", []):
        for child in group.get("child_sections", []):
            bp = child.get("base_path", "")
            title = child.get("title", "")
            if bp:
                sections.append({"base_path": bp, "title": title})

    log.info("Discovered %d sections from %s", len(sections), manual_path)
    return sections


def write_plan(sections: list[dict], out_dir: Path) -> Path:
    """Write the fetch-plan CSV and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    plan_path = out_dir / f"{today}_corpus_rules-fetch-plan.csv"
    with open(plan_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["base_path", "title"])
        writer.writeheader()
        writer.writerows(sections)
    log.info("Wrote plan to %s", plan_path)
    return plan_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Discover immigration rules sections and write a fetch plan")
    ap.add_argument("--out", default="manifests",
                    help="Directory for the plan CSV (default: manifests/)")
    args = ap.parse_args()

    sections = discover()
    if sections:
        write_plan(sections, Path(args.out))


if __name__ == "__main__":
    main()
