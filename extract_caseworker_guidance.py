"""
Caseworker Guidance Extractor
-------------------------------
For each of the 589 caseworker guidance publications:
  - If an accessible HTML version exists → extract via Content API (clean, tables preserved)
  - If PDF only → extract with pdfplumber (strips headers/footers, structured tables)

Output: raw_documents/caseworker_guidance/<slug>.json

{
  "title":          "...",
  "content_id":     "...",
  "description":    "...",
  "base_path":      "...",
  "collection":     "...",
  "change_history": [ { "date": "...", "note": "..." } ],
  "documents": [
    {
      "attachment_id": "...",
      "title":         "...",
      "source":        "html" | "pdf",
      "source_url":    "...",
      "content":       "natural text, tables as Col1 | Col2 rows"
    }
  ]
}

Licence: Open Government Licence v3.0
https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
"""

import io, json, re, time, logging
from pathlib import Path

import requests
import pdfplumber
from bs4 import BeautifulSoup, NavigableString, Tag

# ── config ─────────────────────────────────────────────────────────────────────
BASE_API       = "https://www.gov.uk/api/content"
RATE_LIMIT     = 0.5
RETRY_WAITS    = [2, 5, 10]
ENDPOINTS_FILE = "discovered_endpoints.json"
OUT_DIR        = Path("raw_documents/caseworker_guidance")
MAX_PDF_MB     = 25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── HTML → natural text ────────────────────────────────────────────────────────

def html_to_text(html: str) -> str:
    """Same approach as immigration rules extractor. Tables become Col1 | Col2 rows."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    lines = []

    def process(node):
        if isinstance(node, NavigableString):
            return
        if not isinstance(node, Tag):
            return
        name = node.name

        if name in ("h2", "h3", "h4", "h5", "h6"):
            text = node.get_text(" ", strip=True)
            if text:
                lines.append("")
                lines.append(text)
            return

        if name in ("ol", "ul"):
            # skip legislative-list-wrapper div — just recurse
            if name == "div" and "legislative-list-wrapper" in node.get("class", []):
                for c in node.children:
                    process(c)
                return
            for li in node.find_all("li", recursive=False):
                text = re.sub(r"\s+", " ", li.get_text(" ", strip=True)).strip()
                if text:
                    lines.append(text)
            return

        if name == "p":
            text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
            if text:
                lines.append(text)
                lines.append("")
            return

        if name == "table":
            for row in node.find_all("tr"):
                cells = [
                    re.sub(r"\s+", " ", c.get_text(" ", strip=True)).strip()
                    for c in row.find_all(["th", "td"])
                ]
                row_text = " | ".join(c for c in cells if c)
                if row_text:
                    lines.append(row_text)
            lines.append("")
            return

        if name in ("div", "section", "article", "main"):
            for c in node.children:
                process(c)
            return

        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if text:
            lines.append(text)

    for child in soup.children:
        process(child)

    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ── PDF → natural text with pdfplumber ────────────────────────────────────────

# Patterns to strip from PDF output
_NOISE = re.compile(
    r"^(page\s+\d+\s+of\s+\d+|published for home office staff|"
    r"uncontrolled if printed|official[\s\-]sensitive|"
    r"version\s+\d+[\.\d]*\s*$|v\d+\.\d+\s*$|"
    r"[\.\s]{10,}[\d]+\s*$)",   # TOC dot leaders:  Introduction ......... 3
    re.IGNORECASE,
)

def _clean_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    return "" if _NOISE.match(line) else line


def pdf_to_text(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF using pdfplumber.
    - Crops top 8% and bottom 7% of each page to remove headers/footers
    - Extracts tables first as 'Col1 | Col2' rows
    - Extracts remaining body text
    - Strips noise lines (page numbers, version stamps, TOC leaders)
    """
    lines = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                h, w = page.height, page.width

                # crop margins: remove top 8% and bottom 7%
                body = page.crop((0, h * 0.08, w, h * 0.93))

                # ── tables ────────────────────────────────────────────────
                tables = body.extract_tables(
                    table_settings={
                        "vertical_strategy":   "lines",
                        "horizontal_strategy": "lines",
                    }
                )

                table_bboxes = []
                for table in tables:
                    # get bounding box so we can exclude table area from text
                    try:
                        tobj = body.find_tables()[0]
                        table_bboxes.append(tobj.bbox)
                    except Exception:
                        pass

                    for row in table:
                        cells = [
                            re.sub(r"\s+", " ", (cell or "").strip())
                            for cell in row
                        ]
                        row_text = " | ".join(c for c in cells if c)
                        if row_text:
                            lines.append(row_text)
                    if table:
                        lines.append("")

                # ── body text (all text — pdfplumber deduplicates vs tables) ──
                text = body.extract_text(x_tolerance=3, y_tolerance=3) or ""
                for raw_line in text.split("\n"):
                    clean = _clean_line(raw_line)
                    if clean:
                        lines.append(clean)
                lines.append("")

    except Exception as exc:
        log.warning("pdfplumber error: %s", exc)
        return f"[PDF extraction failed: {exc}]"

    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ── network helpers ────────────────────────────────────────────────────────────

_last = [0.0]

def _rate():
    gap = time.time() - _last[0]
    if gap < RATE_LIMIT:
        time.sleep(RATE_LIMIT - gap)
    _last[0] = time.time()


def fetch_json(session, path: str) -> dict | None:
    url = BASE_API + path
    for attempt, wait in enumerate([0] + RETRY_WAITS, start=1):
        if wait:
            time.sleep(wait)
        try:
            _rate()
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                log.warning("404 %s", path)
                return None
            log.warning("HTTP %s attempt %d: %s", r.status_code, attempt, path)
        except requests.RequestException as e:
            log.warning("Network error attempt %d %s: %s", attempt, path, e)
    return None


def fetch_bytes(session, url: str) -> bytes | None:
    for attempt, wait in enumerate([0] + RETRY_WAITS, start=1):
        if wait:
            time.sleep(wait)
        try:
            _rate()
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                size_mb = len(r.content) / (1024 * 1024)
                if size_mb > MAX_PDF_MB:
                    log.warning("PDF too large (%.1fMB), skipping: %s", size_mb, url)
                    return None
                return r.content
            if r.status_code == 404:
                return None
            log.warning("HTTP %s attempt %d: %s", r.status_code, attempt, url)
        except requests.RequestException as e:
            log.warning("PDF fetch error attempt %d: %s", attempt, e)
    return None


# ── attachment processing ──────────────────────────────────────────────────────

def _normalise(title: str) -> str:
    """Strip '(accessible)' suffix for matching HTML↔PDF pairs."""
    return re.sub(r"\s*\(accessible[^)]*\)\s*$", "", title, flags=re.I).strip().lower()


def process_attachments(attachments: list, session) -> list:
    html_atts = [a for a in attachments if a.get("attachment_type") == "html"]
    pdf_atts  = [a for a in attachments if a.get("attachment_type") == "file"
                 and "pdf" in a.get("content_type", "").lower()]

    # titles already covered by HTML version
    html_titles = {_normalise(a.get("title", "")) for a in html_atts}

    documents = []

    # ── HTML attachments: fetch body from Content API ──────────────────────
    for att in html_atts:
        url_path   = att.get("url", "")
        api_path   = url_path if url_path.startswith("/") else url_path.replace("https://www.gov.uk", "")
        source_url = "https://www.gov.uk" + api_path

        data = fetch_json(session, api_path)
        if not data:
            log.warning("Could not fetch HTML attachment: %s", api_path)
            continue

        content = html_to_text(data.get("details", {}).get("body", ""))
        documents.append({
            "attachment_id": att.get("id", ""),
            "title":         att.get("title", ""),
            "source":        "html",
            "source_url":    source_url,
            "content":       content,
        })

    # ── PDF attachments: only if no HTML version covers same document ──────
    for att in pdf_atts:
        title = att.get("title", "")
        if _normalise(title) in html_titles:
            log.debug("Skipping PDF, HTML version exists: %s", title)
            continue

        pdf_url = att.get("url", "")
        if not pdf_url.startswith("http"):
            pdf_url = "https://www.gov.uk" + pdf_url

        pdf_bytes = fetch_bytes(session, pdf_url)
        if pdf_bytes is None:
            content = "[PDF unavailable]"
        else:
            log.debug("Extracting PDF (%dKB): %s",
                      len(pdf_bytes) // 1024, att.get("title", ""))
            content = pdf_to_text(pdf_bytes)

        documents.append({
            "attachment_id": att.get("id", ""),
            "title":         title,
            "source":        "pdf",
            "source_url":    pdf_url,
            "num_pages":     att.get("number_of_pages"),
            "file_size_kb":  att.get("file_size", 0) // 1024,
            "content":       content,
        })

    return documents


# ── build document ─────────────────────────────────────────────────────────────

def build_document(data: dict, session) -> dict:
    details = data.get("details", {})
    links   = data.get("links", {})

    change_history = [
        {"date": c.get("public_timestamp", ""), "note": c.get("note", "")}
        for c in details.get("change_history", [])
    ]

    collections = links.get("document_collections", [])
    collection  = collections[0].get("title", "") if collections else ""

    documents = process_attachments(details.get("attachments", []), session)

    return {
        "title":          data.get("title", ""),
        "content_id":     data.get("content_id", ""),
        "description":    data.get("description", ""),
        "base_path":      data.get("base_path", ""),
        "collection":     collection,
        "change_history": change_history,
        "documents":      documents,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    with open(ENDPOINTS_FILE) as f:
        endpoints = json.load(f)

    paths = endpoints.get("caseworker_guidance", [])
    total = len(paths)
    log.info("Extracting %d caseworker guidance documents", total)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "UK-Immigration-Data-Extraction/1.0",
        "Accept":     "application/json",
    })

    ok = fail = 0

    for i, api_path in enumerate(paths, 1):
        data = fetch_json(session, api_path)
        if data is None:
            fail += 1
            continue

        doc  = build_document(data, session)
        slug = api_path.rstrip("/").split("/")[-1] or "index"
        dest = OUT_DIR / f"{slug}.json"
        dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        ok += 1

        if i % 25 == 0 or i == total:
            log.info("%d/%d  ok=%d  fail=%d", i, total, ok, fail)

    log.info("Done — saved: %d  failed: %d", ok, fail)


if __name__ == "__main__":
    main()
