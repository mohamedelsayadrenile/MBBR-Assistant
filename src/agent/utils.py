from datetime import date, datetime
from typing import Any


def parse_date_range(
    raw_from: str, raw_to: str, today: date
) -> tuple[date, date] | None:
    """Parse a strict ISO date range, rejecting future or reversed windows."""

    def parse(value: str) -> date | None:
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()  # noqa: DTZ007
        except ValueError:
            return None

    start = parse(raw_from)
    end = parse(raw_to)
    if start is None or end is None or start > end or start > today:
        return None
    return start, min(end, today)


def match_by_id(
    device_id: Any, devices: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return the live-list entry whose id equals `device_id`, if any."""
    for device in devices:
        if device.get("id") == device_id:
            return device
    return None


def real_candidates(candidates: Any, devices: list[dict[str, Any]]) -> list[str]:
    """Floor resolver candidate names to exact live-list names, taking two max."""
    names = [device["name"] for device in devices]
    seen: list[str] = []
    for candidate in candidates or []:
        if candidate in names and candidate not in seen:
            seen.append(candidate)
        if len(seen) == 2:
            break
    return seen


def payload_sensors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """List the sensors present in a readings payload, whichever shape it has.

    The two readings endpoints name the same fields differently — latest uses
    `sensor_type` / `sensor_type_ar` / `measurement_unit` per reading, while
    daily-averages uses `name` / `name_ar` / `unit` per sensor. Normalising both
    here is what keeps the support check and the narrowing helpers shape-blind.
    """
    sensors: list[dict[str, Any]] = []
    for reading in payload.get("readings") or []:
        if isinstance(reading, dict) and reading.get("sensor_type"):
            sensors.append(
                {
                    "type": str(reading["sensor_type"]),
                    "type_ar": str(reading.get("sensor_type_ar") or ""),
                    "unit": reading.get("measurement_unit"),
                }
            )
    for sensor in payload.get("sensors") or []:
        if isinstance(sensor, dict) and sensor.get("name"):
            sensors.append(
                {
                    "type": str(sensor["name"]),
                    "type_ar": str(sensor.get("name_ar") or ""),
                    "unit": sensor.get("unit"),
                }
            )
    return sensors


def match_sensor(
    sensor_type: Any, sensors: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return the payload entry whose type equals `sensor_type`, if any.

    The deterministic floor for measurement resolution, exactly as
    `match_by_id` is for devices: a name the payload does not carry never
    reaches the reply.
    """
    for sensor in sensors:
        if sensor["type"] == sensor_type:
            return sensor
    return None


def narrow_current(payload: dict[str, Any], sensor_type: str) -> dict[str, Any]:
    """Keep only the latest readings for `sensor_type`, count included."""
    readings = [
        reading
        for reading in payload.get("readings") or []
        if isinstance(reading, dict) and reading.get("sensor_type") == sensor_type
    ]
    return {**payload, "count": len(readings), "readings": readings}


def narrow_daily(payload: dict[str, Any], sensor_type: str) -> dict[str, Any]:
    """Keep only the daily-average series for `sensor_type`."""
    sensors = [
        sensor
        for sensor in payload.get("sensors") or []
        if isinstance(sensor, dict) and sensor.get("name") == sensor_type
    ]
    return {**payload, "sensors": sensors}
