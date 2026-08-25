"""Unit tests for OpenSkyClient retry / rate-limit handling (no real HTTP)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from alt_data.ingestion.fetcher import DataFetcher
from alt_data.ingestion.opensky_client import OpenSkyClient, OpenSkyQuotaError


def _response(status_code: int, headers: dict | None = None, payload=None):
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {}
    response.json.return_value = payload if payload is not None else []
    return response


def _client(**kwargs) -> OpenSkyClient:
    return OpenSkyClient(
        base_url="https://opensky.test/api",
        token_manager=MagicMock(headers=MagicMock(return_value={})),
        max_retries=3,
        max_retry_after=300,
        **kwargs,
    )


class TestRetryAfter:
    def test_429_honors_retry_after_header(self):
        client = _client()
        client._session.get = MagicMock(
            side_effect=[
                _response(429, {"X-Rate-Limit-Retry-After-Seconds": "7"}),
                _response(200, payload=[{"icao24": "abc123"}]),
            ]
        )

        with patch("alt_data.ingestion.opensky_client.time.sleep") as sleep:
            rows = client.get_arrivals_by_airport("KJFK", 1_704_067_200, 1_704_153_599)

        assert rows == [{"icao24": "abc123"}]
        assert any(call.args[0] == 8 for call in sleep.call_args_list)  # 7 + 1s margin

    def test_429_beyond_cap_raises_quota_error(self):
        client = _client()
        client._session.get = MagicMock(
            return_value=_response(429, {"X-Rate-Limit-Retry-After-Seconds": "86400"})
        )

        with pytest.raises(OpenSkyQuotaError, match="quota"):
            client.get_arrivals_by_airport("KJFK", 1_704_067_200, 1_704_153_599)
        # Fail fast: no retries after the server said "come back tomorrow".
        assert client._session.get.call_count == 1

    def test_429_without_header_falls_back_to_backoff(self):
        client = _client()
        client._session.get = MagicMock(
            side_effect=[_response(429), _response(200, payload=[])]
        )

        with patch("alt_data.ingestion.opensky_client.time.sleep") as sleep:
            rows = client.get_arrivals_by_airport("KJFK", 1_704_067_200, 1_704_153_599)

        assert rows == []
        assert any(call.args[0] == 2.0 for call in sleep.call_args_list)

    def test_retry_after_parses_standard_header(self):
        response = _response(429, {"Retry-After": "42"})
        assert OpenSkyClient._retry_after_seconds(response) == 42.0

    def test_retry_after_none_when_missing(self):
        assert OpenSkyClient._retry_after_seconds(_response(429)) is None


class TestFetcherQuotaAbort:
    def test_fetcher_aborts_run_on_quota_error(self):
        client = MagicMock()
        client.get_arrivals_by_airport.side_effect = OpenSkyQuotaError("quota gone")

        fetcher = DataFetcher(client=client)
        with pytest.raises(OpenSkyQuotaError):
            fetcher.fetch(
                "KJFK",
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 10, 23, 59, 59, tzinfo=timezone.utc),
            )
        # Aborted on the first window instead of grinding all ten days.
        assert client.get_arrivals_by_airport.call_count == 1
        client.get_departures_by_airport.assert_not_called()
