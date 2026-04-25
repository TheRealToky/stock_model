"""End-to-end scripted example of the alt-data workflow.

Run this after the pipeline has been populated for KJFK::

    python alt_data_pipeline/notebooks/example_cross_dataset.py

The output mirrors the notebook -- useful for smoke-testing the
DataHub without a Jupyter kernel.
"""

from __future__ import annotations

import pandas as pd

from alt_data_pipeline.src.query.datahub import DataHub
from alt_data_pipeline.src.query.mappings import AssetMapper
from alt_data_pipeline.src.utils.logging import configure_logging


def main() -> None:
    configure_logging("INFO")
    hub = DataHub()
    mapper = AssetMapper()

    print("Airports mapped:", mapper.all_airports())
    print("KJFK -> tickers:", mapper.tickers_for_airport("KJFK"))

    flights = hub.get_flight_data(
        airport="KJFK", start="2024-01-01", end="2024-02-01"
    )
    print("\nDaily aggregated flights:")
    print(flights.head())

    features = hub.get_flight_data(
        airport="KJFK",
        start="2024-01-01",
        end="2024-02-01",
        features=["net_flow", "hub_connectivity", "supply_demand_z_7d"],
    )
    print("\nStored features (wide):")
    print(features.head())

    joined = hub.get_joined_data(
        airport="KJFK",
        ticker="DAL",
        start="2024-01-01",
        end="2024-02-01",
        features=["net_flow", "arrivals", "departures"],
        shift_days=1,
    )
    if not joined.empty:
        joined["close_ret"] = joined["close"].pct_change()
        print("\nJoined (flights(t) vs return(t+1)):")
        print(joined[["net_flow", "close", "close_ret"]].head(10))
        corr = joined[["net_flow", "close_ret"]].dropna().corr().iloc[0, 1]
        print(f"\ncorr(net_flow(t), return(t+1)) = {corr:.4f}")


if __name__ == "__main__":
    main()
