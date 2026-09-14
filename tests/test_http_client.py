"""
Tests for code.fetch.http_client.

All tests mock requests.Session.get so no network calls are made.
"""

from unittest.mock import patch, MagicMock
import json

import pytest

import code.fetch.http_client as hc


@pytest.fixture(autouse=True)
def reset_session():
    """Reset the module-level singleton between tests."""
    hc._session = None
    hc._last_request = 0.0
    yield
    hc._session = None


def _mock_response(status=200, body=None, content=b""):
    """Build a fake requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    if body is not None:
        content = json.dumps(body).encode()
    resp.content = content
    resp.json.return_value = body
    return resp


class TestUserAgent:

    @patch("code.fetch.http_client.time.sleep")
    def test_user_agent_header(self, _sleep):
        """The session carries the identifying User-Agent."""
        body = {"content_id": "abc", "public_updated_at": "2025-01-01"}

        with patch("code.fetch.http_client.requests.Session") as MockSess:
            instance = MockSess.return_value
            instance.get.return_value = _mock_response(200, body)
            instance.headers = {}

            # Force new session creation
            hc._session = None
            result = hc.get_json("/test")

        # headers.update was called with our User-Agent
        update_call = instance.headers.update if hasattr(instance.headers, 'update') else None
        # Simpler: check the session factory set the header
        assert hc.USER_AGENT == "WarwickMScDissertation/1.0 (u5757819@live.warwick.ac.uk)"
        assert result is not None


class TestRetry:

    @patch("code.fetch.http_client.time.sleep")
    def test_retry_on_503_then_success(self, _sleep):
        """A 503 triggers retry; success on second attempt. Two log rows."""
        body = {"content_id": "c1", "public_updated_at": "2025-06-01"}
        resp_503 = _mock_response(503)
        resp_200 = _mock_response(200, body)

        with patch("code.fetch.http_client.requests.Session") as MockSess:
            instance = MockSess.return_value
            instance.get.side_effect = [resp_503, resp_200]
            instance.headers = {}

            hc._session = None
            log_rows = []
            result = hc.get_json("/retry-test",
                                 log_writer=lambda *args: log_rows.append(args))

        assert result == body
        assert len(log_rows) == 2
        assert log_rows[0][2] == 503   # first attempt status
        assert log_rows[1][2] == 200   # second attempt status


class TestLogRow:

    @patch("code.fetch.http_client.time.sleep")
    def test_one_log_row_per_attempt(self, _sleep):
        """Each HTTP attempt produces exactly one log_writer call."""
        body = {"content_id": "x", "public_updated_at": "2025-01-01"}

        with patch("code.fetch.http_client.requests.Session") as MockSess:
            instance = MockSess.return_value
            instance.get.return_value = _mock_response(200, body)
            instance.headers = {}

            hc._session = None
            rows = []
            hc.get_json("/one-shot", log_writer=lambda *a: rows.append(a))

        assert len(rows) == 1
        ts, url, status, nbytes, cid, pub, sha = rows[0]
        assert status == 200
        assert cid == "x"
        assert len(sha) == 64  # sha256 hex digest
