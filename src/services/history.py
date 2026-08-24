import logging
from datetime import date
from typing import Any

from core.config import Settings
from services.mbbr_api import MBBRAPIError, get_json_data

logger = logging.getLogger(__name__)


async def get_historical_readings(
    jwt: str, figure_id: str, start: date, end: date, settings: Settings
) -> dict[str, Any]:
    """Return daily-average sensor readings for one figure over a past period.

    Like `get_current_readings`, the `data` object is passed through untouched.
    The response is `{figures: [{figure_id, figure_name, sensors, run_status}]}`.
    A listed figure with `sensors: []` has no averages in the period, and
    `run_status` may be present.
    """
    from_iso = start.isoformat()
    to_iso = end.isoformat()
    logger.info(
        "history_fetch_started figure_id=%s from=%s to=%s", figure_id, from_iso, to_iso
    )
    data = await get_json_data(
        settings.historical_readings_api_path,
        jwt,
        {"figure_id": figure_id, "from": from_iso, "to": to_iso},
        settings,
    )

    if not isinstance(data, dict) or not isinstance(data.get("figures"), list):
        logger.warning("history_fetch_invalid_shape data_type=%s", type(data).__name__)
        raise MBBRAPIError("Historical readings API did not return a figure list")

    logger.info(
        "history_fetch_completed figure_id=%s from=%s to=%s figures=%s",
        figure_id,
        from_iso,
        to_iso,
        len(data["figures"]),
    )
    return data
