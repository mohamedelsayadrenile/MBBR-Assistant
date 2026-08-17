import logging
from typing import Any

from core.config import Settings
from services.mbbr_api import MBBRAPIError, get_json_data

logger = logging.getLogger(__name__)


async def get_devices(jwt: str, settings: Settings) -> list[dict[str, Any]]:
    """Return the plant's devices as `[{"id": ..., "name": ...}, ...]`.

    Only the two fields the agent needs are kept — the API also returns
    timestamps and status fields that would only be noise in the LLM context.
    """
    logger.info("devices_fetch_started")
    data = await get_json_data(settings.devices_api_path, jwt, {}, settings)

    if not isinstance(data, list):
        logger.warning("devices_fetch_invalid_shape data_type=%s", type(data).__name__)
        raise MBBRAPIError("Devices API did not return a list")

    devices = [
        {"id": str(device["id"]), "name": str(device.get("name", "")).strip()}
        for device in data
        if isinstance(device, dict) and device.get("id")
    ]
    logger.info("devices_fetch_completed devices=%s", len(devices))
    return devices
