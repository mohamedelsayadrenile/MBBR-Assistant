import logging
from typing import Any

from core.config import Settings
from services.mbbr_api import MBBRAPIError, get_json_data

logger = logging.getLogger(__name__)


async def get_current_readings(jwt: str, settings: Settings) -> dict[str, Any]:
    """Return the latest readings for every device in the plant.

    The endpoint takes no parameters and answers with the whole plant at once:
    `{generated_at, count, devices: [{device_id, device_name, sensors}]}`, every
    sensor a device is wired for listed with a null `value` when it has no stored
    reading. API in, raw data out: the `data` object goes to the composer
    untouched, and reading a device and a measurement out of it is the model's
    job — this function only fetches and checks the envelope.
    """
    logger.info("readings_fetch_started")
    data = await get_json_data(settings.current_readings_api_path, jwt, {}, settings)

    if not isinstance(data, dict) or not isinstance(data.get("devices"), list):
        logger.warning("readings_fetch_invalid_shape data_type=%s", type(data).__name__)
        raise MBBRAPIError("Readings API did not return a device list")

    logger.info("readings_fetch_completed devices=%s", len(data["devices"]))
    return data
