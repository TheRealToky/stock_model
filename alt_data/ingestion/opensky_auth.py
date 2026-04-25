"""OAuth2 client-credentials token manager for OpenSky.

OpenSky has migrated authentication onto a Keycloak-backed OAuth2
client-credentials flow.  This module owns the lifecycle of the
bearer token: fetch on demand, cache until shortly before expiry,
refresh transparently on the next call.

The manager is safe to share between threads -- a single lock guards
the cached token so concurrent callers do not trigger a thundering
herd of refresh requests.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from threading import Lock

import requests

from alt_data.utils.logging import get_logger

logger = get_logger(__name__)


class OpenSkyAuthError(RuntimeError):
    """Raised when token acquisition fails (bad creds, server error, ...)."""


class TokenManager:
    """Thread-safe OAuth2 client-credentials token cache.

    Parameters
    ----------
    token_url:
        Full URL of the OAuth2 token endpoint (the Keycloak
        ``/protocol/openid-connect/token`` URL for OpenSky).
    client_id, client_secret:
        OAuth2 client credentials issued by OpenSky.
    refresh_margin_seconds:
        Refresh this many seconds before the advertised expiry to
        avoid edge-case 401s from clock skew.
    timeout:
        HTTP timeout for the token endpoint, in seconds.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        refresh_margin_seconds: int = 30,
        timeout: int = 30,
    ) -> None:
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_margin_seconds = refresh_margin_seconds
        self.timeout = timeout

        self._token: str | None = None
        self._expires_at: datetime | None = None
        self._lock = Lock()

    def get_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        with self._lock:
            if self._is_valid():
                assert self._token is not None  # for type-checkers
                return self._token
            return self._refresh_locked()

    def invalidate(self) -> None:
        """Drop the cached token so the next call refreshes."""
        with self._lock:
            self._token = None
            self._expires_at = None

    def headers(self) -> dict[str, str]:
        """Return ``Authorization: Bearer <token>`` (refreshing if needed)."""
        return {"Authorization": f"Bearer {self.get_token()}"}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _is_valid(self) -> bool:
        return (
            self._token is not None
            and self._expires_at is not None
            and datetime.now() < self._expires_at
        )

    def _refresh_locked(self) -> str:
        """Fetch a new access token; caller must hold ``self._lock``."""
        try:
            response = requests.post(
                self.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OpenSkyAuthError(
                f"Failed to fetch OpenSky access token: {exc}"
            ) from exc

        try:
            data = response.json()
            token = data["access_token"]
        except (ValueError, KeyError) as exc:
            raise OpenSkyAuthError(
                f"Malformed token response from OpenSky: {exc}"
            ) from exc

        expires_in = int(data.get("expires_in", 1800))
        self._token = token
        self._expires_at = datetime.now() + timedelta(
            seconds=max(1, expires_in - self.refresh_margin_seconds)
        )
        logger.debug(
            "Refreshed OpenSky token; expires at {} (in {}s)",
            self._expires_at.isoformat(),
            expires_in,
        )
        return token
