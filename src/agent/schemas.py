from dataclasses import dataclass
from datetime import date
from typing import Any, TypedDict

from services.memory import MemoryMessage


class AgentState(TypedDict, total=False):
    history: list[MemoryMessage]
    user_message: str
    request: dict[str, Any]
    resolution: dict[str, Any]
    result: Any
    reply: str


@dataclass(frozen=True)
class RunContext:
    conversation_id: str
    jwt: str
    today: date


INTERPRET_SCHEMA: dict[str, Any] = {
    "title": "interpret_turn",
    "description": "Interpret the operator's current turn into structured data.",
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["reply", "current", "historical", "station", "unsupported"],
        },
        "sensor": {
            "type": "string",
            "description": 'The measurement exactly as the operator worded it, or "all".',
        },
        "device_name": {
            "type": "string",
            "description": "The device wording the operator used, if any.",
        },
        "from_date": {
            "type": "string",
            "description": "YYYY-MM-DD for historical intents",
        },
        "to_date": {
            "type": "string",
            "description": "YYYY-MM-DD for historical intents",
        },
        "reply": {
            "type": "string",
            "description": "Egyptian Arabic reply for reply intent.",
        },
    },
    "required": ["intent"],
}

RESOLVER_SCHEMA: dict[str, Any] = {
    "title": "resolve_device",
    "description": "Resolve the operator's device wording against the live device list.",
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["matched", "ambiguous", "not_found"],
        },
        "device_id": {
            "type": "string",
            "description": "Exact id copied from the supplied device list (matched only).",
        },
        "device_name": {
            "type": "string",
            "description": "Exact name copied from the supplied device list (matched only).",
        },
        "candidates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Top plausible device names from the supplied list (ambiguous only).",
        },
    },
    "required": ["status"],
}

SENSOR_SCHEMA: dict[str, Any] = {
    "title": "resolve_sensor",
    "description": "Match the operator's measurement wording to a sensor the device reports.",
    "type": "object",
    "properties": {
        "sensor_type": {
            "type": "string",
            "description": "Exact sensor name copied from the supplied list; empty if none matches.",
        },
    },
    "required": ["sensor_type"],
}
