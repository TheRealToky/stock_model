"""Thin HTTP client wrapping the OpenSky Network REST API.

We deliberately don't depend on ``opensky-api`` (the community Python
client) because it is sync-only, untyped, and raises on empty results.
A small ``requests`` wrapper keeps full control over retries,
back-off, rate limiting, and auth.
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from alt_data.config.settings import alt_settings
from alt_data.utils.logging import get_logger

logger = get_logger(__name__)


class OpenSkyClient:
    """Rate-limited, retrying HTTP client for OpenSky.

    Public methods mirror the two endpoints we use:

    * :meth:`get_arrivals_by_airport`
    * :meth:`get_departures_by_airport`

    The client is safe to share between threads -- the rate-limit
    window is protected by a lock.
    """

    def __init__(
        self,
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        max_requests_per_minute: int | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        cfg = alt_settings.opensky
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.username = username if username is not None else cfg.username
        self.password = password if password is not None else cfg.password
        self.max_rpm = max_requests_per_minute or cfg.max_requests_per_minute
        self.timeout = timeout or cfg.request_timeout_seconds
        self.max_retries = max_retries or alt_settings.pipeline.max_retries

        self._session = requests.Session()
        if self.username and self.password:
            self._session.auth = HTTPBasicAuth(self.username, self.password)

        self._window: deque[float] = deque(maxlen=self.max_rpm)
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def get_arrivals_by_airport(
        self,
        airport_icao: str,
        begin: int,
        end: int,
    ) -> list[dict[str, Any]]:
        """Return the raw list of arrival rows for [begin, end] (UNIX s)."""
        return self._get(
            "/flights/arrival",
            params={"airport": airport_icao.upper(), "begin": begin, "end": end},
        )

    def get_departures_by_airport(
        self,
        airport_icao: str,
        begin: int,
        end: int,
    ) -> list[dict[str, Any]]:
        """Return the raw list of departure rows for [begin, end] (UNIX s)."""
        return self._get(
            "/flights/departure",
            params={"airport": airport_icao.upper(), "begin": begin, "end": end},
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """GET *path* with exponential back-off retries and rate limiting."""
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self._rate_limit()
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "OpenSky request error on attempt {}/{}: {}",
                    attempt,
                    self.max_retries,
                    exc,
                )
                self._sleep_backoff(attempt)
                continue

            # OpenSky returns 404 when no flights are found in the window.
            # That's not an error -- treat it as an empty list.
            if response.status_code == 404:
                logger.debug("OpenSky 404 (no data) for {} {}", path, params)
                return []

            # 429 = rate limit hit; 5xx = transient -> retry with backoff.
            if response.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    "OpenSky {} on attempt {}/{} for {} {}",
                    response.status_code,
                    attempt,
                    self.max_retries,
                    path,
                    params,
                )
                self._sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                # Client errors that aren't 404/429 are not retried.
                response.raise_for_status()

            try:
                data = response.json()
            except ValueError as exc:
                last_exc = exc
                logger.warning("OpenSky returned non-JSON payload: {}", exc)
                self._sleep_backoff(attempt)
                continue

            if not isinstance(data, list):
                logger.warning("OpenSky returned non-list payload: {!r}", type(data))
                return []
            return data

        raise RuntimeError(
            f"OpenSky {path} failed after {self.max_retries} attempts: "
            f"{last_exc!r}"
        )

    def _rate_limit(self) -> None:
        """Block until we are under ``max_rpm`` requests in the last 60s."""
        with self._lock:
            now = time.monotonic()
            if len(self._window) == self.max_rpm:
                elapsed = now - self._window[0]
                if elapsed < 60:
                    wait = 60.0 - elapsed
                    logger.debug("OpenSky rate limit: sleeping {:.2f}s", wait)
                    time.sleep(wait)
                    now = time.monotonic()
            self._window.append(now)

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        """Exponential back-off: 2s, 4s, 8s, ..., capped at 60s."""
        delay = min(60.0, 2.0 ** attempt)
        time.sleep(delay)
