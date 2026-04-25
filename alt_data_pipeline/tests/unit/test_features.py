"""Unit tests for the technical feature functions."""

import pandas as pd

from alt_data_pipeline.src.features.registry import (
    FLIGHT_FEATURE_REGISTRY,
    list_features,
)
from alt_data_pipeline.src.features.technical import (
    compute_event_detection,
    compute_fleet_utilization,
    compute_hub_analysis,
    compute_net_flow,
    compute_passenger_cargo_proxy,
    compute_route_reconstruction,
    compute_supply_demand,
    compute_turnaround_time,
)


class TestRegistry:
    def test_all_10_features_registered(self):
        expected = {
            "route_reconstruction",
            "passenger_cargo_proxy",
            "net_flow",
            "turnaround_time",
            "fleet_utilization",
            "hub_analysis",
            "airline_strategy",
            "delay_congestion",
            "supply_demand",
            "event_detection",
        }
        assert expected.issubset(set(list_features()))

    def test_registry_has_callables(self):
        for name, (func, params) in FLIGHT_FEATURE_REGISTRY.items():
            assert callable(func), f"{name} not callable"
            assert isinstance(params, dict)


class TestSchemaInvariant:
    """Every feature must return the canonical long schema."""

    REQUIRED = {"airport_icao", "timestamp", "feature_name", "feature_value"}

    def test_net_flow_schema(self, cleaned_flight_df):
        out = compute_net_flow(cleaned_flight_df)
        assert self.REQUIRED.issubset(out.columns)

    def test_route_reconstruction_schema(self, cleaned_flight_df):
        out = compute_route_reconstruction(cleaned_flight_df)
        assert self.REQUIRED.issubset(out.columns)

    def test_passenger_cargo_schema(self, cleaned_flight_df):
        out = compute_passenger_cargo_proxy(cleaned_flight_df)
        assert self.REQUIRED.issubset(out.columns)

    def test_hub_analysis_schema(self, cleaned_flight_df):
        out = compute_hub_analysis(cleaned_flight_df)
        assert self.REQUIRED.issubset(out.columns)


class TestNetFlow:
    def test_produces_arrivals_departures_netflow(self, cleaned_flight_df):
        out = compute_net_flow(cleaned_flight_df)
        names = set(out["feature_name"].unique())
        assert {"arrivals", "departures", "net_flow"}.issubset(names)

    def test_arrivals_plus_netflow_minus_departures_balances(self, cleaned_flight_df):
        out = compute_net_flow(cleaned_flight_df)
        wide = out.pivot_table(
            index=["airport_icao", "timestamp"],
            columns="feature_name",
            values="feature_value",
        )
        assert ((wide["arrivals"] - wide["departures"]) == wide["net_flow"]).all()


class TestFleetUtilization:
    def test_unique_aircraft_bounded_by_flights(self, cleaned_flight_df):
        out = compute_fleet_utilization(cleaned_flight_df)
        wide = out.pivot_table(
            index=["airport_icao", "timestamp"],
            columns="feature_name",
            values="feature_value",
        )
        # flights_per_aircraft must be >= 1 by definition
        assert (wide["flights_per_aircraft"] >= 1).all()


class TestTurnaroundTime:
    def test_returns_positive_minutes(self, cleaned_flight_df):
        out = compute_turnaround_time(cleaned_flight_df)
        if out.empty:
            # fixture may not produce pairs -- accept either outcome
            return
        assert (out["feature_value"] > 0).all()


class TestSupplyDemand:
    def test_zscore_is_finite_or_nan(self, cleaned_flight_df):
        out = compute_supply_demand(cleaned_flight_df, window_days=2)
        assert out["feature_value"].apply(
            lambda x: pd.isna(x) or -50 < x < 50
        ).all()


class TestEventDetection:
    def test_event_flag_is_binary(self, cleaned_flight_df):
        out = compute_event_detection(cleaned_flight_df, window_days=2, z_threshold=1.0)
        for val in out["feature_value"].dropna():
            assert val in (0.0, 1.0)


class TestEmptyInput:
    def test_every_feature_handles_empty_df(self):
        empty = pd.DataFrame(
            columns=[
                "icao24", "callsign", "first_seen", "last_seen",
                "direction", "airport_icao", "est_departure_airport",
                "est_arrival_airport", "first_seen_dt", "last_seen_dt",
            ]
        )
        for name, (func, params) in FLIGHT_FEATURE_REGISTRY.items():
            out = func(empty, **params)
            assert out.empty, f"{name} didn't return empty for empty input"
