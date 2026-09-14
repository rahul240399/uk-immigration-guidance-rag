"""
Immigration Rules HTML Parser
-----------------------------
Converts immigration rules HTML to natural text while preserving structure.
"""

import re
from bs4 import BeautifulSoup, NavigableString, Tag


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