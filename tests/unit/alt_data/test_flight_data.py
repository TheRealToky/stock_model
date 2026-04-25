"""Unit tests for the FlightData dataclass."""

from alt_data.ingestion.flight_data import FlightData


class TestFlightDataFactory:
    def test_from_api_payload_happy_path(self, sample_api_payload):
        raw = sample_api_payload[0]
        fd = FlightData.from_api_payload(raw, "arrival", "KJFK")
        assert fd.icao24 == "abc123"
        assert fd.callsign == "  UAL100 "   # not yet stripped -- cleaner does that
        assert fd.direction == "arrival"
        assert fd.airport_icao == "KJFK"
        assert fd.est_departure_airport == "EGLL"
        assert fd.first_seen == 1_704_067_200
        assert fd.last_seen == 1_704_078_000

    def test_from_api_payload_with_missing_optional_fields(self, sample_api_payload):
        fd = FlightData.from_api_payload(sample_api_payload[1], "arrival", "KJFK")
        assert fd.departure_airport_candidates_count is None
        assert fd.est_arrival_airport == "KJFK"

    def test_datetimes_derived_from_unix(self, sample_api_payload):
        fd = FlightData.from_api_payload(sample_api_payload[0], "arrival", "KJFK")
        assert fd.first_seen_dt.year == 2024
        assert fd.last_seen_dt > fd.first_seen_dt

    def test_to_dict_round_trip(self, sample_api_payload):
        fd = FlightData.from_api_payload(sample_api_payload[0], "arrival", "KJFK")
        d = fd.to_dict()
        assert d["direction"] == "arrival"
        assert d["icao24"] == "abc123"

    def test_airport_icao_is_uppercased(self, sample_api_payload):
        fd = FlightData.from_api_payload(sample_api_payload[0], "arrival", "kjfk")
        assert fd.airport_icao == "KJFK"
