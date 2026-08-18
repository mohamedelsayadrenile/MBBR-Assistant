from datetime import date

import pytest

from agent.agent import parse_date_range

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
