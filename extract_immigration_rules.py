"""
Immigration Rules Extractor
----------------------------
Fetches each immigration rules endpoint from GOV.UK Content API and
writes one JSON file per document to raw_documents/immigration_rules/

Output structure per file:
{
  "title":          "...",
  "content_id":     "...",
  "description":    "...",
  "base_path":      "...",
  "change_history": [ { "date": "...", "note": "..." }, ... ],
  "rules":          "full natural text with subheadings preserved"
}

Licence: Open Government Licence v3.0
https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
"""

import json, re, time, logging
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# ── config ─────────────────────────────────────────────────────────────────────
BASE_API       = "https://www.gov.uk/api/content"
RATE_LIMIT     = 0.5          # seconds between requests (2 req/sec)
RETRY_WAITS    = [2, 5, 10]
ENDPOINTS_FILE = "discovered_endpoints.json"
OUT_DIR        = Path("raw_documents/immigration_rules")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── HTML → natural text ────────────────────────────────────────────────────────

def html_to_natural_text(html: str) -> str:
    """
    Convert immigration rules HTML to natural text.
    - h2/h3/h4 headings are kept inline as plain text lines
    - legislative-list items are extracted with their numbering
    - paragraphs are separated by blank lines
    - no HTML tags in output
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Remove any <script> or <style> noise
    for tag in soup(["script", "style"]):
        tag.decompose()

    lines = []

    def process(node):
        if isinstance(node, NavigableString):
            # bare text nodes at top level – usually whitespace, skip
            return

        if not isinstance(node, Tag):
            return

        tag = node.name

        # ── headings: emit as plain text line ─────────────────────────────
        if tag in ("h2", "h3", "h4", "h5", "h6"):
            text = node.get_text(" ", strip=True)
            if text:
                lines.append("")          # blank line before heading
                lines.append(text)
            return

        # ── legislative list wrapper: just recurse ─────────────────────────
        if "legislative-list-wrapper" in node.get("class", []):
            for child in node.children:
                process(child)
            return

        # ── ordered / unordered list ───────────────────────────────────────
        if tag in ("ol", "ul"):
            for li in node.find_all("li", recursive=False):
                text = li.get_text(" ", strip=True)
                # collapse internal whitespace runs
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    lines.append(text)
            return

        # ── paragraph ─────────────────────────────────────────────────────
        if tag == "p":
            text = node.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                lines.append(text)
                lines.append("")          # blank line after paragraph
            return

        # ── div / section / article: recurse ──────────────────────────────
        if tag in ("div", "section", "article", "main"):
            for child in node.children:
                process(child)
            return

        # ── anything else: extract text ────────────────────────────────────
        text = node.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            lines.append(text)

    for child in soup.children:
        process(child)

    # Collapse runs of more than one blank line into a single blank line
    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ── fetch with retry ────────────────────────────────────────────────────────────

def fetch(session: requests.Session, api_path: str) -> dict | None:
    url = BASE_API + api_path
    for attempt, wait in enumerate([0] + RETRY_WAITS, start=1):
        if wait:
            time.sleep(wait)
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                log.warning("404 – skipping %s", api_path)
                return None
            log.warning("HTTP %s on attempt %d: %s", r.status_code, attempt, api_path)
        except requests.RequestException as exc:
            log.warning("Network error attempt %d for %s: %s", attempt, api_path, exc)
    log.error("All retries exhausted for %s", api_path)
    return None


# ── build document ──────────────────────────────────────────────────────────────

def build_document(data: dict) -> dict:
    details = data.get("details", {})

    change_history = [
        {"date": c.get("public_timestamp", ""), "note": c.get("note", "")}
        for c in details.get("change_history", [])
    ]

    rules_text = html_to_natural_text(details.get("body", ""))

    return {
        "title":          data.get("title", ""),
        "content_id":     data.get("content_id", ""),
        "description":    data.get("description", ""),
        "base_path":      data.get("base_path", ""),
        "change_history": change_history,
        "rules":          rules_text,
    }


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    with open(ENDPOINTS_FILE) as f:
        endpoints = json.load(f)

    immigration_rules = endpoints.get("immigration_rules", [])
    total = len(immigration_rules)
    log.info("Extracting %d immigration rules documents", total)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "UK-Immigration-Data-Extraction/1.0",
        "Accept": "application/json",
    })

    ok = fail = 0
    last_req = 0.0

    for i, api_path in enumerate(immigration_rules, 1):
        # rate limit
        gap = time.time() - last_req
        if gap < RATE_LIMIT:
            time.sleep(RATE_LIMIT - gap)
        last_req = time.time()

        data = fetch(session, api_path)
        if data is None:
            fail += 1
            continue

        doc  = build_document(data)
        slug = api_path.rstrip("/").split("/")[-1] or "index"
        dest = OUT_DIR / f"{slug}.json"
        dest.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        ok += 1

        if i % 10 == 0 or i == total:
            log.info("%d/%d done (ok=%d fail=%d)", i, total, ok, fail)

    log.info("Complete — saved: %d  failed: %d", ok, fail)


if __name__ == "__main__":
    main()
