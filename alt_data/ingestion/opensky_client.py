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

from alt_data.config.settings import alt_settings
from alt_data.ingestion.opensky_auth import TokenManager
from alt_data.utils.logging import get_logger

logger = get_logger(__name__)


class OpenSkyQuotaError(RuntimeError):
    """Raised when OpenSky reports the API quota is exhausted (429 with a
    Retry-After beyond our waiting cap).  Retrying other windows in the
    same run is pointless -- callers should abort and resume once the
    advertised wait has passed."""


class OpenSkyClient:
    """Rate-limited, retrying HTTP client for OpenSky.

    Public methods mirror the two endpoints we use:

    * :meth:`get_arrivals_by_airport`
    * :meth:`get_departures_by_airport`

    Authentication uses OAuth2 client-credentials via
    :class:`TokenManager`.  When ``OPENSKY_CLIENT_ID`` and
    ``OPENSKY_CLIENT_SECRET`` are unset the client falls back to
    anonymous mode (much lower quota -- ~100 req/day).

    The client is safe to share between threads -- the rate-limit
    window is protected by a lock and the token manager serializes
    refreshes.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token_manager: TokenManager | None = None,
        max_requests_per_minute: int | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        max_retry_after: int | None = None,
    ) -> None:
        cfg = alt_settings.opensky
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.max_rpm = max_requests_per_minute or cfg.max_requests_per_minute
        self.timeout = timeout or cfg.request_timeout_seconds
        self.max_retries = max_retries or alt_settings.pipeline.max_retries
        self.max_retry_after = max_retry_after or cfg.max_retry_after_seconds

        if token_manager is not None:
            self._token_manager: TokenManager | None = token_manager
        elif cfg.client_id and cfg.client_secret:
            self._token_manager = TokenManager(
                token_url=cfg.token_url,
                client_id=cfg.client_id,
                client_secret=cfg.client_secret,
                refresh_margin_seconds=cfg.token_refresh_margin_seconds,
                timeout=self.timeout,
            )
        else:
            logger.warning(
                "OpenSky client running anonymously: "
                "OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET are not set"
            )
            self._token_manager = None

        self._session = requests.Session()
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
        retried_after_401 = False

        for attempt in range(1, self.max_retries + 1):
            self._rate_limit()
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    headers=self._auth_headers(),
                )
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

            # 401 = token rejected. Force a refresh and retry once
            # without backoff (clock-skew / server-side revocation).
            if (
                response.status_code == 401
                and self._token_manager is not None
                and not retried_after_401
            ):
                logger.warning("OpenSky 401 -- invalidating token and retrying once")
                self._token_manager.invalidate()
                retried_after_401 = True
                continue

            # 429 = rate limit hit; 5xx = transient -> retry with backoff.
            if response.status_code in (429, 500, 502, 503, 504):
                last_exc = RuntimeError(f"HTTP {response.status_code}")
                retry_after = self._retry_after_seconds(response)
                if (
                    response.status_code == 429
                    and retry_after is not None
                    and retry_after > self.max_retry_after
                ):
                    raise OpenSkyQuotaError(
                        f"OpenSky asks to retry {path} after {retry_after:.0f}s "
                        f"(~{retry_after / 3600:.1f}h; cap: {self.max_retry_after}s) "
                        "-- API quota is exhausted. Resume once that wait has passed"
                        + (
                            ""
                            if self._token_manager is not None
                            else " or set OPENSKY_CLIENT_ID / "
                            "OPENSKY_CLIENT_SECRET for a much higher quota"
                        )
                    )
                logger.warning(
                    "OpenSky {} on attempt {}/{} for {} {} (retry-after: {})",
                    response.status_code,
                    attempt,
                    self.max_retries,
                    path,
                    params,
                    retry_after if retry_after is not None else "n/a",
                )
                if retry_after is not None:
                    # Obey the server, plus a small margin for clock skew.
                    time.sleep(retry_after + 1)
                else:
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

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float | None:
        """Extract the server-advertised wait from a 429/5xx response.

        OpenSky uses ``X-Rate-Limit-Retry-After-Seconds``; the standard
        ``Retry-After`` is checked as a fallback.  Returns ``None`` when
        neither header is present or parseable.
        """
        for header in ("X-Rate-Limit-Retry-After-Seconds", "Retry-After"):
            raw = response.headers.get(header)
            if raw is None:
                continue
            try:
                return max(0.0, float(raw))
            except ValueError:
                continue
        return None

    def _auth_headers(self) -> dict[str, str]:
        """Return per-request auth headers (empty when anonymous)."""
        if self._token_manager is None:
            return {}
        return self._token_manager.headers()

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
