"""Mapping layer: airports / routes <-> tradable assets.

The raw dictionaries live in ``config/mappings.yaml`` so analysts can
edit them without touching code.  :class:`AssetMapper` wraps that YAML
and exposes sensible query methods for the DataHub.
"""

from __future__ import annotations

from alt_data.config.settings import alt_settings


class AssetMapper:
    """Lookup helper between airport codes and tradable tickers."""

    def __init__(
        self,
        airport_to_tickers: dict[str, list[str]] | None = None,
        airport_to_region: dict[str, str] | None = None,
        route_to_tickers: dict[str, list[str]] | None = None,
    ) -> None:
        # Defer to settings (YAML) when no explicit dicts are provided,
        # so callers can inject overrides in tests.
        self._airport_to_tickers = airport_to_tickers or alt_settings.airport_to_tickers()
        self._airport_to_region = airport_to_region or self._load_region_map()
        self._route_to_tickers = route_to_tickers or self._load_route_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tickers_for_airport(self, airport_icao: str) -> list[str]:
        """Return all tickers associated with an airport (uppercased)."""
        return list(self._airport_to_tickers.get(airport_icao.upper(), []))

    def airports_for_ticker(self, ticker: str) -> list[str]:
        """Return all airports that list *ticker* in the mapping."""
        ticker = ticker.upper()
        return sorted(
            airport
            for airport, tickers in self._airport_to_tickers.items()
            if ticker in [t.upper() for t in tickers]
        )

    def region_for_airport(self, airport_icao: str) -> str | None:
        return self._airport_to_region.get(airport_icao.upper())

    def tickers_for_route(self, origin: str, destination: str) -> list[str]:
        key = f"{origin.upper()}->{destination.upper()}"
        return list(self._route_to_tickers.get(key, []))

    def all_airports(self) -> list[str]:
        return sorted(self._airport_to_tickers.keys())

    def all_tickers(self) -> list[str]:
        seen: set[str] = set()
        for tickers in self._airport_to_tickers.values():
            seen.update(t.upper() for t in tickers)
        return sorted(seen)

    # ------------------------------------------------------------------
    # YAML loaders (fallback path if not injected)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_region_map() -> dict[str, str]:
        from pathlib import Path
        import yaml

        path = Path(alt_settings.pipeline.mapping_config_path)
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return dict(data.get("airport_to_region", {}))

    @staticmethod
    def _load_route_map() -> dict[str, list[str]]:
        from pathlib import Path
        import yaml

        path = Path(alt_settings.pipeline.mapping_config_path)
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return dict(data.get("route_to_tickers", {}))
