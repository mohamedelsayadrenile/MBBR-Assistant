import logging
from datetime import date
from typing import Any

from core.config import Settings
from services.mbbr_api import MBBRAPIError, get_json_data

logger = logging.getLogger(__name__)


async def get_historical_readings(
    jwt: str, device_id: str, start: date, end: date, settings: Settings
) -> dict[str, Any]:
    """Return daily-average sensor readings for one device over a past period.

    Like `get_current_readings`, the `data` object is passed through untouched:
    the sensor/daily shape is read by the LLM, not pinned in code, so an upstream
    change remains a prompt concern rather than a code change.
    """
    from_iso = start.isoformat()
    to_iso = end.isoformat()
    logger.info(
        "history_fetch_started device_id=%s from=%s to=%s", device_id, from_iso, to_iso
    )
    data = await get_json_data(
        settings.historical_readings_api_path,
        jwt,
        {"device_id": device_id, "from": from_iso, "to": to_iso},
        settings,
    )

    if not isinstance(data, dict):
        logger.warning("history_fetch_invalid_shape data_type=%s", type(data).__name__)
        raise MBBRAPIError("Historical readings API did not return an object")

    sensors = data.get("sensors")
    sensor_count = len(sensors) if isinstance(sensors, list) else 0
    logger.info(
        "history_fetch_completed device_id=%s from=%s to=%s sensors=%s",
        device_id,
        from_iso,
        to_iso,
        sensor_count,
    )
    return data