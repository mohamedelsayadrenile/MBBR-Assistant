import logging
from typing import Any

from core.config import Settings
from services.mbbr_api import MBBRAPIError, get_json_data

logger = logging.getLogger(__name__)


async def get_current_readings(jwt: str, device_id: str, settings: Settings) -> dict[str, Any]:
    """Return the latest readings for one device.

    The `data` object is passed through untouched; only the agent reads inside
    it, through `payload_sensors` in `agent/utils.py`. Note the endpoint lists
    only sensors that have a stored reading, so an absent measurement means "no
    reading", not "the device is not wired for it".
    """
    logger.info("readings_fetch_started device_id=%s", device_id)
    data = await get_json_data(
        settings.current_readings_api_path,
        jwt,
        {"device_id": device_id},
        settings,
    )

    if not isinstance(data, dict):
        logger.warning("readings_fetch_invalid_shape data_type=%s", type(data).__name__)
        raise MBBRAPIError("Readings API did not return an object")

    logger.info(
        "readings_fetch_completed device_id=%s count=%s",
        device_id,
        data.get("count"),
    )
    return data
