"""Flight-data ingestion layer."""

from alt_data_pipeline.src.ingestion.flight_data import FlightData
from alt_data_pipeline.src.ingestion.fetcher import DataFetcher
from alt_data_pipeline.src.ingestion.opensky_client import OpenSkyClient

__all__ = ["FlightData", "DataFetcher", "OpenSkyClient"]
