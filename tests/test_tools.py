from datetime import date

import pytest

from agent.utils import (
    match_sensor,
    narrow_current,
    narrow_daily,
    parse_date_range,
    payload_sensors,
)

TODAY = date(2026, 8, 17)


def test_parse_date_range_accepts_and_clamps_valid_dates() -> None:
    assert parse_date_range("2026-08-10", "2026-08-16", TODAY) == (
        date(2026, 8, 10),
        date(2026, 8, 16),
    )
    assert parse_date_range("2026-08-01", "2026-12-31", TODAY) == (
        date(2026, 8, 1),
        TODAY,
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2026-08-16", "2026-08-10"),
        ("20260816", "2026-08-16"),
        ("امبارح", "2026-08-16"),
        ("2026-08-18", "2026-08-20"),
    ],
)
def test_parse_date_range_rejects_invalid_dates(start: str, end: str) -> None:
    assert parse_date_range(start, end, TODAY) is None


LATEST = {
    "count": 2,
    "readings": [
        {"sensor_type": "PH", "sensor_type_ar": "درجة الحموضة", "measurement_unit": "pH"},
        {"sensor_type": "Flow", "sensor_type_ar": "معدل التدفق", "measurement_unit": "m³/h"},
    ],
}
DAILY = {
    "device_name": "جهاز 2",
    "sensors": [
        {"name": "PH", "name_ar": "درجة الحموضة", "unit": "pH", "daily": [{"avg": 7.4}]},
        {"name": "Flow", "name_ar": "معدل التدفق", "unit": "m³/h", "daily": []},
    ],
}


def test_payload_sensors_reads_the_latest_readings_shape() -> None:
    assert payload_sensors(LATEST) == [
        {"type": "PH", "type_ar": "درجة الحموضة", "unit": "pH"},
        {"type": "Flow", "type_ar": "معدل التدفق", "unit": "m³/h"},
    ]


def test_payload_sensors_reads_the_daily_averages_shape() -> None:
    assert payload_sensors(DAILY) == [
        {"type": "PH", "type_ar": "درجة الحموضة", "unit": "pH"},
        {"type": "Flow", "type_ar": "معدل التدفق", "unit": "m³/h"},
    ]


@pytest.mark.parametrize(
    "payload",
    [{"count": 0, "readings": []}, {"device_name": "جهاز 1", "sensors": []}, {}],
)
def test_payload_sensors_is_empty_when_nothing_is_reported(payload: dict) -> None:
    assert payload_sensors(payload) == []


def test_payload_sensors_skips_entries_without_a_name() -> None:
    assert payload_sensors({"readings": [{"value": 1.0}, "junk"]}) == []


def test_match_sensor_finds_a_reported_measurement() -> None:
    assert match_sensor("Flow", payload_sensors(LATEST))["type_ar"] == "معدل التدفق"


@pytest.mark.parametrize("sensor_type", ["ph", "temperature", "", None])
def test_match_sensor_rejects_anything_not_reported(sensor_type: object) -> None:
    assert match_sensor(sensor_type, payload_sensors(LATEST)) is None


def test_narrow_current_keeps_one_sensor_and_recomputes_the_count() -> None:
    narrowed = narrow_current(LATEST, "PH")

    assert narrowed["count"] == 1
    assert [entry["sensor_type"] for entry in narrowed["readings"]] == ["PH"]
    assert LATEST["count"] == 2  # the input is never mutated


def test_narrow_current_keeps_every_row_of_the_same_sensor() -> None:
    payload = {"count": 2, "readings": [{"sensor_type": "PH"}, {"sensor_type": "PH"}]}

    assert narrow_current(payload, "PH")["count"] == 2


def test_narrow_daily_keeps_one_series() -> None:
    narrowed = narrow_daily(DAILY, "PH")

    assert [entry["name"] for entry in narrowed["sensors"]] == ["PH"]
    assert narrowed["device_name"] == "جهاز 2"
    assert len(DAILY["sensors"]) == 2  # the input is never mutated


def test_narrowing_an_unreported_sensor_empties_the_payload() -> None:
    assert narrow_current(LATEST, "temperature")["readings"] == []
    assert narrow_daily(DAILY, "temperature")["sensors"] == []
