from datetime import date

import pytest

from agent.agent import parse_date_range, resolve_device_id

DEVICES = [
    {"id": "b9eaf606-536b-4f38-a58e-d741cd96155b", "name": "جهاز 1"},
    {"id": "c4ca7915-78e2-46f8-81df-31848d8c1b6c", "name": "device 1"},
    {"id": "a1000000-0000-4000-8000-00000000000d", "name": "MBBR Tank A"},
]
TODAY = date(2026, 8, 17)


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("c4ca7915-78e2-46f8-81df-31848d8c1b6c", DEVICES[1]["id"]),
        ("2", DEVICES[1]["id"]),
        ("جهاز 1", DEVICES[0]["id"]),
        ("mbbr tank a", DEVICES[2]["id"]),
    ],
)
def test_resolve_device_id_accepts_verified_references(
    reference: str, expected: str
) -> None:
    assert resolve_device_id(reference, DEVICES) == expected


@pytest.mark.parametrize(
    "reference",
    ["00000000-dead-4000-8000-000000000000", "جهاز أحمد", "جهاز", "9", "   "],
)
def test_resolve_device_id_rejects_unverified_references(reference: str) -> None:
    assert resolve_device_id(reference, DEVICES) is None


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
