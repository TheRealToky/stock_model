"""Technical features computed from cleaned flight data.

Every function signature is::

    def compute_<name>(df: pd.DataFrame, **params) -> pd.DataFrame

The input DataFrame is the output of :class:`FlightDataCleaner`
(columns: ``icao24``, ``callsign``, ``first_seen_dt``, ``last_seen_dt``,
``direction``, ``airport_icao``, ``est_departure_airport``,
``est_arrival_airport``, ...).

The output DataFrame is always in **long format** suitable for
persistence into ``flight_features``::

    airport_icao | timestamp (daily) | feature_name | feature_value

That uniformity means the :class:`FlightFeatureEngine` can persist
every feature the same way and notebooks can pivot trivially.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alt_data.features.registry import register_feature


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ensure_day(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with a ``day`` column (tz-aware UTC midnight)."""
    out = df.copy()
    out["day"] = pd.to_datetime(out["first_seen_dt"], utc=True).dt.floor("D")
    return out


def _long(
    grouped: pd.DataFrame | pd.Series,
    feature_name: str,
    value_col: str | None = None,
) -> pd.DataFrame:
    """Convert a grouped (airport, day) result to the long feature schema."""
    if isinstance(grouped, pd.Series):
        out = grouped.rename("feature_value").reset_index()
    else:
        if value_col is None:
            raise ValueError("value_col required when passing a DataFrame")
        out = grouped.rename(columns={value_col: "feature_value"})[
            ["airport_icao", "day", "feature_value"]
        ]
    out = out.rename(columns={"day": "timestamp"})
    out["feature_name"] = feature_name
    return out[["airport_icao", "timestamp", "feature_name", "feature_value"]]


# ---------------------------------------------------------------------------
# 1. Route reconstruction
# ---------------------------------------------------------------------------


@register_feature("route_reconstruction")
def compute_route_reconstruction(df: pd.DataFrame) -> pd.DataFrame:
    """Count unique routes (orig -> dest) observed per airport per day.

    A route is the ``(est_departure_airport, est_arrival_airport)``
    tuple.  Rows missing either endpoint are skipped.
    """
    if df.empty:
        return _empty_long_result()

    day_df = _ensure_day(df).dropna(
        subset=["est_departure_airport", "est_arrival_airport"]
    )
    grouped = (
        day_df.groupby(["airport_icao", "day"])
        .apply(
            lambda g: g[
                ["est_departure_airport", "est_arrival_airport"]
            ].drop_duplicates().shape[0],
            include_groups=False,
        )
        .reset_index(name="value")
    )
    return _long(grouped, "unique_routes_per_day", value_col="value")


# ---------------------------------------------------------------------------
# 2. Passenger & cargo flow proxies
# ---------------------------------------------------------------------------


# Callsign prefix -> rough airline category.  This is a proxy mapping;
# expand it freely from a richer airline-type database.
_CARGO_PREFIXES = {
    "GTI", "FDX", "UPS", "CLX", "GEC", "ABW", "CKS", "MPH", "CAO",
}


@register_feature("passenger_cargo_proxy")
def compute_passenger_cargo_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy daily passenger vs cargo volumes by callsign prefix.

    Returns three features per airport/day:

    * ``passenger_flights`` - count of non-cargo callsigns
    * ``cargo_flights``     - count of cargo-prefix callsigns
    * ``cargo_ratio``       - cargo / (passenger + cargo)
    """
    if df.empty:
        return _empty_long_result()

    d = _ensure_day(df).copy()
    prefix = d["callsign"].astype("string").str[:3].fillna("")
    d["is_cargo"] = prefix.isin(_CARGO_PREFIXES)

    cargo = (
        d.groupby(["airport_icao", "day"])["is_cargo"].sum().rename("cargo_flights")
    )
    passenger = (
        d.groupby(["airport_icao", "day"])["is_cargo"]
        .apply(lambda s: (~s).sum())
        .rename("passenger_flights")
    )
    total = (cargo + passenger).replace(0, np.nan)
    ratio = (cargo / total).rename("cargo_ratio")

    return pd.concat(
        [
            _long(passenger, "passenger_flights"),
            _long(cargo, "cargo_flights"),
            _long(ratio, "cargo_ratio"),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# 3. Net flow (arrivals - departures)
# ---------------------------------------------------------------------------


@register_feature("net_flow")
def compute_net_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Per-airport daily arrivals, departures, and net flow."""
    if df.empty:
        return _empty_long_result()

    d = _ensure_day(df)
    pivot = (
        d.groupby(["airport_icao", "day", "direction"])
        .size()
        .unstack(fill_value=0)
    )
    arrivals = pivot.get("arrival", pd.Series(0, index=pivot.index)).rename("arrivals")
    departures = pivot.get("departure", pd.Series(0, index=pivot.index)).rename(
        "departures"
    )
    net = (arrivals - departures).rename("net_flow")

    return pd.concat(
        [
            _long(arrivals, "arrivals"),
            _long(departures, "departures"),
            _long(net, "net_flow"),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# 4. Aircraft turnaround time
# ---------------------------------------------------------------------------


@register_feature("turnaround_time")
def compute_turnaround_time(df: pd.DataFrame) -> pd.DataFrame:
    """Mean turnaround (arrival -> next departure) in minutes, per airport/day.

    For each aircraft (``icao24``), we pair an arrival at the airport
    with its next departure from the same airport.  Turnaround is
    ``next_departure.first_seen - arrival.last_seen``.  Rows that can't
    be paired are ignored.
    """
    if df.empty:
        return _empty_long_result()

    work = df.sort_values(["icao24", "first_seen_dt"]).copy()
    results: list[dict] = []
    for (icao, airport), g in work.groupby(["icao24", "airport_icao"]):
        arrivals = g[g["direction"] == "arrival"]
        departures = g[g["direction"] == "departure"].set_index("first_seen_dt")
        for _, arr in arrivals.iterrows():
            later = departures[departures.index > arr["last_seen_dt"]]
            if later.empty:
                continue
            dep_time = later.index[0]
            turnaround_minutes = (dep_time - arr["last_seen_dt"]).total_seconds() / 60.0
            results.append(
                {
                    "airport_icao": airport,
                    "day": pd.Timestamp(arr["last_seen_dt"]).floor("D"),
                    "turnaround_minutes": turnaround_minutes,
                }
            )

    if not results:
        return _empty_long_result()

    tr = pd.DataFrame(results)
    grouped = (
        tr.groupby(["airport_icao", "day"])["turnaround_minutes"]
        .mean()
        .rename("mean_turnaround_minutes")
    )
    return _long(grouped, "mean_turnaround_minutes")


# ---------------------------------------------------------------------------
# 5. Fleet utilization
# ---------------------------------------------------------------------------


@register_feature("fleet_utilization")
def compute_fleet_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """Unique aircraft (``icao24``) active at the airport, per day.

    Higher unique-aircraft counts indicate broader fleet engagement;
    combined with total flights this proxies utilization intensity.
    """
    if df.empty:
        return _empty_long_result()

    d = _ensure_day(df)
    unique_fleet = (
        d.groupby(["airport_icao", "day"])["icao24"]
        .nunique()
        .rename("unique_aircraft")
    )
    total = (
        d.groupby(["airport_icao", "day"]).size().rename("total_flights")
    )
    intensity = (total / unique_fleet.replace(0, np.nan)).rename(
        "flights_per_aircraft"
    )
    return pd.concat(
        [
            _long(unique_fleet, "unique_aircraft"),
            _long(intensity, "flights_per_aircraft"),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# 6. Hub analysis
# ---------------------------------------------------------------------------


@register_feature("hub_analysis")
def compute_hub_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Hub-score proxies per airport per day.

    * ``hub_connectivity``   - unique non-null counterpart airports
    * ``hub_concentration``  - Herfindahl index of counterpart shares
    """
    if df.empty:
        return _empty_long_result()

    d = _ensure_day(df).copy()
    d["counterpart"] = np.where(
        d["direction"] == "arrival",
        d["est_departure_airport"],
        d["est_arrival_airport"],
    )
    d = d.dropna(subset=["counterpart"])

    connectivity = (
        d.groupby(["airport_icao", "day"])["counterpart"]
        .nunique()
        .rename("hub_connectivity")
    )

    shares = (
        d.groupby(["airport_icao", "day", "counterpart"])
        .size()
        .groupby(level=[0, 1])
        .apply(lambda s: ((s / s.sum()) ** 2).sum())
        .rename("hub_concentration")
    )

    return pd.concat(
        [
            _long(connectivity, "hub_connectivity"),
            _long(shares, "hub_concentration")
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# 7. Airline route strategy detection
# ---------------------------------------------------------------------------


@register_feature("airline_strategy")
def compute_airline_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy airline route-strategy metrics per airport per day.

    Groups flights by airline callsign prefix (first 3 chars) and
    computes the number of distinct airlines plus the share of the
    top-1 airline (``top1_airline_share``).
    """
    if df.empty:
        return _empty_long_result()

    d = _ensure_day(df).copy()
    d["airline"] = d["callsign"].astype("string").str[:3]
    d = d.dropna(subset=["airline"])

    n_airlines = (
        d.groupby(["airport_icao", "day"])["airline"]
        .nunique()
        .rename("unique_airlines")
    )
    top_share = (
        d.groupby(["airport_icao", "day", "airline"])
        .size()
        .groupby(level=[0, 1])
        .apply(lambda s: s.max() / s.sum())
        .rename("top1_airline_share")
    )

    return pd.concat(
        [
            _long(n_airlines, "unique_airlines"),
            _long(top_share, "top1_airline_share")
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# 8. Delay & congestion inference
# ---------------------------------------------------------------------------


@register_feature("delay_congestion")
def compute_delay_congestion(
    df: pd.DataFrame,
    congestion_window_minutes: int = 15,
) -> pd.DataFrame:
    """Infer congestion via short-window flight density.

    For each airport/day we compute the max number of flights whose
    ``first_seen_dt`` falls inside any rolling *congestion_window_minutes*
    window, and the mean flight *duration* (last_seen - first_seen) --
    excessive durations correlate with holds and taxi delays.
    """
    if df.empty:
        return _empty_long_result()

    d = df.copy()
    d["first_seen_dt"] = pd.to_datetime(d["first_seen_dt"], utc=True)
    d["last_seen_dt"] = pd.to_datetime(d["last_seen_dt"], utc=True)
    d["duration_min"] = (d["last_seen_dt"] - d["first_seen_dt"]).dt.total_seconds() / 60.0
    d["day"] = d["first_seen_dt"].dt.floor("D")

    def _peak(group: pd.DataFrame) -> float:
        s = group.set_index("first_seen_dt").sort_index()
        if s.empty:
            return 0.0
        counts = s.assign(one=1)["one"].rolling(
            f"{congestion_window_minutes}min"
        ).sum()
        return float(counts.max())

    peak = (
        d.groupby(["airport_icao", "day"])
        .apply(_peak, include_groups=False)
        .rename("peak_flights_per_window")
    )
    mean_dur = (
        d.groupby(["airport_icao", "day"])["duration_min"]
        .mean()
        .rename("mean_flight_duration_min")
    )

    return pd.concat(
        [
            _long(peak, "peak_flights_per_window"),
            _long(mean_dur, "mean_flight_duration_min"),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# 9. Supply-demand imbalance
# ---------------------------------------------------------------------------


@register_feature("supply_demand")
def compute_supply_demand(
    df: pd.DataFrame,
    window_days: int = 7,
) -> pd.DataFrame:
    """Rolling supply/demand imbalance.

    ``supply_demand_z`` is the z-score of daily flight volume against
    a trailing *window_days* rolling mean / std per airport.
    """
    if df.empty:
        return _empty_long_result()

    d = _ensure_day(df)
    daily = d.groupby(["airport_icao", "day"]).size().rename("flights")

    def _z(g: pd.Series) -> pd.Series:
        mean = g.rolling(window_days, min_periods=1).mean()
        std = g.rolling(window_days, min_periods=2).std().replace(0, np.nan)
        return (g - mean) / std

    z = daily.groupby(level=0).transform(_z).rename("supply_demand_z")
    return _long(z, f"supply_demand_z_{window_days}d")


# ---------------------------------------------------------------------------
# 10. Event detection
# ---------------------------------------------------------------------------


@register_feature("event_detection")
def compute_event_detection(
    df: pd.DataFrame,
    window_days: int = 14,
    z_threshold: float = 2.5,
) -> pd.DataFrame:
    """Boolean event flag when daily flow deviates > *z_threshold* sigma.

    ``event_flag`` is 1.0 on days where flight volume z-score (over the
    trailing *window_days* window) exceeds *z_threshold* in absolute
    value; otherwise 0.0.  Useful for proxying storms, strikes,
    holidays, or route disruptions.
    """
    if df.empty:
        return _empty_long_result()

    d = _ensure_day(df)
    daily = d.groupby(["airport_icao", "day"]).size().rename("flights")

    def _z(g: pd.Series) -> pd.Series:
        mean = g.rolling(window_days, min_periods=2).mean()
        std = g.rolling(window_days, min_periods=2).std().replace(0, np.nan)
        return (g - mean) / std

    z = daily.groupby(level=0).transform(_z)
    flag = (z.abs() >= z_threshold).astype(float).rename("event_flag")
    return _long(flag, "event_flag")


# ---------------------------------------------------------------------------
# Empty-result helper
# ---------------------------------------------------------------------------


def _empty_long_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["airport_icao", "timestamp", "feature_name", "feature_value"]
    )
