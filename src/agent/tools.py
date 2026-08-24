"""The three things the assistant can do.

Each tool calls one API and hands the data back to the model untouched. No
matching, no filtering, no reshaping — reading a figure and a measurement out of
a payload is the model's job.
"""

import json
from datetime import date, datetime
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from core.config import Settings
from services.figures import get_figures as fetch_figures
from services.history import get_historical_readings as fetch_historical_readings
from services.readings import get_current_readings as fetch_current_readings

MAX_RANGE_DAYS = 31
BAD_PERIOD = (
    "Invalid period. Both dates must be real YYYY-MM-DD dates, the first not "
    "after the second, and not in the future. Ask the operator for a clearer period."
)
RANGE_TOO_LONG = (
    f"Period too long. Ask the operator for a period of at most {MAX_RANGE_DAYS} days."
)


def build_tools(jwt: str, settings: Settings, today: date) -> list[BaseTool]:
    """Build this turn's tools. The JWT rides in the closure, never as a tool argument."""

    async def get_figures() -> str:
        """List every figure in the plant with its id and name."""
        return _dump(await fetch_figures(jwt, settings))

    async def get_current_readings() -> str:
        """Read the plant's latest readings.

        Returns every figure with every sensor it reports and that sensor's
        current value. A null value means the figure has no reading right now.
        """
        return _dump(await fetch_current_readings(jwt, settings))

    async def get_historical_readings(
        figure_id: str, from_date: str, to_date: str
    ) -> str:
        """Read the daily average readings of one figure over a past period.

        `figure_id` must come from `get_figures`. `from_date` and `to_date` are
        YYYY-MM-DD and at most one month apart.
        """
        period = _parse_period(from_date, to_date, today)
        if isinstance(period, str):
            return period
        start, end = period
        return _dump(
            await fetch_historical_readings(jwt, figure_id, start, end, settings)
        )

    return [
        StructuredTool.from_function(coroutine=function)
        for function in (get_figures, get_current_readings, get_historical_readings)
    ]


def _parse_period(raw_from: Any, raw_to: Any, today: date) -> tuple[date, date] | str:
    """Return the period, or say what is wrong with it so the model can ask again.

    The only check left in Python: a malformed or year-long range would otherwise
    reach the upstream API as-is.
    """
    start, end = _parse_date(raw_from), _parse_date(raw_to)
    if start is None or end is None or start > end or start > today:
        return BAD_PERIOD
    end = min(end, today)
    if (end - start).days >= MAX_RANGE_DAYS:
        return RANGE_TOO_LONG
    return start, end


def _parse_date(value: Any) -> date | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()  # noqa: DTZ007
    except (TypeError, ValueError):
        return None


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)
