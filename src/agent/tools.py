"""The two tools exposed to the LLM, and their dispatch.

The JWT is deliberately absent from every schema below. It is supplied as a
Python keyword argument at dispatch time, so the model can neither see it nor
be tricked into echoing it.
"""

import logging
from typing import Any

from core.config import Settings
from services.devices import get_devices
from services.readings import get_current_readings

logger = logging.getLogger(__name__)

GET_DEVICES = "get_devices"
GET_CURRENT_READINGS = "get_current_readings"

OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": GET_DEVICES,
            "description": (
                "List the plant's devices with their real ids and names. "
                "Call this before get_current_readings, every time."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": GET_CURRENT_READINGS,
            "description": (
                "Get the latest sensor readings for one device. Use only after "
                "get_devices returned the real device id. Never pass a device name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": (
                            "The real device id copied from the get_devices result. "
                            "This must be an id, not the device name."
                        ),
                    },
                },
                "required": ["device_id"],
                "additionalProperties": False,
            },
        },
    },
]


async def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    jwt: str,
    settings: Settings,
) -> Any:
    """Run one tool by name, injecting the JWT that the LLM never sees."""
    if name == GET_DEVICES:
        return await get_devices(jwt, settings)
    if name == GET_CURRENT_READINGS:
        return await get_current_readings(jwt, str(arguments["device_id"]), settings)
    raise ValueError(f"Unknown tool: {name}")


def resolve_device_id(raw_device_id: str, devices: list[dict[str, Any]]) -> str | None:
    """Map whatever the model produced onto a real device id, or None.

    The model reliably produces one of three things: the id itself, the device
    name the operator said, or a bare position ("2" for "جهاز 2"). Anything else
    is a hallucination and must not reach the API.
    """
    candidate = raw_device_id.strip()
    if not candidate:
        return None

    for device in devices:
        if candidate == device["id"]:
            return device["id"]

    folded = candidate.casefold()
    # Enumerate the full list in the order the tool returned it, so a bare
    # position lines up with the list the model was given.
    for index, device in enumerate(devices, start=1):
        if candidate == str(index) or folded == device["name"].casefold():
            return device["id"]
    return None
