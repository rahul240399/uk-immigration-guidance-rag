"""
Reusable attachment helpers for caseworker guidance fetching.

Kept from the original extract_caseworker_guidance.py for reuse in
the guidance fetch stage (3b). Parsing functions (html_to_text,
pdf_to_text) belong in code/parse and are not included here.
"""

import logging
import re

from code.fetch.http_client import get_json, get_bytes

log = logging.getLogger(__name__)


def normalise_title(title: str) -> str:
    """Strip '(accessible)' suffix for matching HTML↔PDF pairs."""
    return re.sub(r"\s*\(accessible[^)]*\)\s*$", "", title, flags=re.I).strip().lower()


def process_attachments(attachments: list) -> list:
    """
    Process a publication's attachments list.

    - Fetches HTML attachments via Content API (preferred).
    - Falls back to PDF attachments only if no HTML version covers the
      same title (matched after stripping '(accessible)' suffix).
    - Returns raw content without parsing — parsing belongs in code/parse.

    Returns a list of dicts with keys:
        attachment_id, title, source ('html'|'pdf'), source_url, content
    """
    html_atts = [a for a in attachments if a.get("attachment_type") == "html"]
    pdf_atts = [a for a in attachments
                if a.get("attachment_type") == "file"
                and "pdf" in a.get("content_type", "").lower()]

    html_titles = {normalise_title(a.get("title", "")) for a in html_atts}
    documents = []

    # HTML attachments: fetch body from Content API
    for att in html_atts:
        url_path = att.get("url", "")
        api_path = (url_path if url_path.startswith("/")
                    else url_path.replace("https://www.gov.uk", ""))
        source_url = "https://www.gov.uk" + api_path

        data = get_json(api_path)
        if not data:
            log.warning("Could not fetch HTML attachment: %s", api_path)
            continue

        raw_body = data.get("details", {}).get("body", "")
        documents.append({
            "attachment_id": att.get("id", ""),
            "title": att.get("title", ""),
            "source": "html",
            "source_url": source_url,
            "content": raw_body,
        })

    # PDF attachments: only if no HTML version covers same title
    for att in pdf_atts:
        title = att.get("title", "")
        if normalise_title(title) in html_titles:
            continue

        pdf_url = att.get("url", "")
        if not pdf_url.startswith("http"):
            pdf_url = "https://www.gov.uk" + pdf_url

        pdf_content = get_bytes(pdf_url)

        documents.append({
            "attachment_id": att.get("id", ""),
            "title": title,
            "source": "pdf",
            "source_url": pdf_url,
            "content": pdf_content,  # raw bytes or None; caller parses
        })

    return documents
