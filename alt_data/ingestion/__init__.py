"""Flight-data ingestion layer."""

from alt_data.ingestion.flight_data import FlightData
from alt_data.ingestion.fetcher import DataFetcher
from alt_data.ingestion.opensky_client import OpenSkyClient

__all__ = ["FlightData", "DataFetcher", "OpenSkyClient"]
