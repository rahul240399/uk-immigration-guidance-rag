"""
Shared HTTP client for the GOV.UK Content API.

Every fetch module imports get_json from here; no other module calls
requests directly.

Features:
- Identifying User-Agent header
- 1 second between requests (rate limiting)
- 3 retries on 429 and 5xx
- One log row per request via the caller-supplied log_writer
"""

import hashlib
import json
import logging
import time
from typing import Callable, Optional

import requests

BASE_API = "https://www.gov.uk/api/content"
USER_AGENT = "WarwickMScDissertation/1.0 (u5757819@live.warwick.ac.uk)"
RATE_LIMIT = 1.0          # seconds between requests
RETRY_WAITS = [2, 5, 10]  # back-off schedule on 429 / 5xx

log = logging.getLogger(__name__)

_session: Optional[requests.Session] = None
_last_request: float = 0.0


def _get_session() -> requests.Session:
    """Lazy singleton session with the identifying headers."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
    return _session


def _rate_wait() -> None:
    """Block until at least RATE_LIMIT seconds since the last request."""
    global _last_request
    gap = time.time() - _last_request
    if gap < RATE_LIMIT:
        time.sleep(RATE_LIMIT - gap)
    _last_request = time.time()


LogWriter = Callable[..., None]
"""Signature: log_writer(timestamp, url, status, resp_bytes,
                         content_id, public_updated_at, sha256)"""


def get_json(
    path_or_url: str,
    *,
    log_writer: Optional[LogWriter] = None,
) -> Optional[dict]:
    """
    Fetch a GOV.UK Content API endpoint and return parsed JSON.

    Parameters
    ----------
    path_or_url : str
        Either an absolute URL or an API path (e.g. ``/guidance/immigration-rules``).
        Paths are prefixed with ``BASE_API``.
    log_writer : callable, optional
        Called once per HTTP attempt with
        ``(timestamp, url, status, bytes, content_id, public_updated_at, sha256)``.

    Returns
    -------
    dict or None
        Parsed JSON body on success, ``None`` on 404 or exhausted retries.
    """
    url = path_or_url if path_or_url.startswith("http") else BASE_API + path_or_url
    session = _get_session()

    for attempt, wait in enumerate([0] + RETRY_WAITS, start=1):
        if wait:
            time.sleep(wait)

        _rate_wait()
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        try:
            r = session.get(url, timeout=20)
        except requests.RequestException as exc:
            log.warning("Request error attempt %d for %s: %s", attempt, url, exc)
            if log_writer:
                log_writer(ts, url, 0, 0, "", "", "")
            continue

        resp_bytes = len(r.content)

        if r.status_code == 200:
            data = r.json()
            content_id = data.get("content_id", "")
            pub_date = data.get("public_updated_at", "")
            body_bytes = json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
            sha = hashlib.sha256(body_bytes).hexdigest()
            if log_writer:
                log_writer(ts, url, 200, resp_bytes, content_id, pub_date, sha)
            return data

        if r.status_code == 404:
            log.warning("404 %s — skipping", url)
            if log_writer:
                log_writer(ts, url, 404, resp_bytes, "", "", "")
            return None

        # Retry on 429 and 5xx
        if r.status_code == 429 or r.status_code >= 500:
            log.warning("HTTP %s attempt %d for %s — retrying",
                        r.status_code, attempt, url)
            if log_writer:
                log_writer(ts, url, r.status_code, resp_bytes, "", "", "")
            continue

        # Other client errors — don't retry
        log.warning("HTTP %s for %s — not retrying", r.status_code, url)
        if log_writer:
            log_writer(ts, url, r.status_code, resp_bytes, "", "", "")
        return None

    log.error("All retries exhausted for %s", url)
    return None


# ── Binary fetch (PDFs, etc.) ────────────────────────────────────────

MAX_BINARY_MB = 25


def get_bytes(
    url: str,
    *,
    timeout: int = 60,
    max_mb: int = MAX_BINARY_MB,
) -> bytes | None:
    """
    Fetch binary content (e.g. PDFs) with retry and size limit.

    Uses the shared session and rate limiter. Retries on 429/5xx.
    Returns raw bytes on success, None on failure.
    """
    session = _get_session()

    for attempt, wait in enumerate([0] + RETRY_WAITS, start=1):
        if wait:
            time.sleep(wait)

        _rate_wait()

        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                size_mb = len(r.content) / (1024 * 1024)
                if size_mb > max_mb:
                    log.warning("File too large (%.1fMB > %dMB), skipping: %s",
                                size_mb, max_mb, url)
                    return None
                return r.content
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                log.warning("HTTP %s attempt %d for %s", r.status_code, attempt, url)
                continue
            log.warning("HTTP %s for %s — not retrying", r.status_code, url)
            return None
        except requests.RequestException as e:
            log.warning("Fetch error attempt %d: %s", attempt, e)

    return None
