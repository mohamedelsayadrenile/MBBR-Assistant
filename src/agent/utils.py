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
