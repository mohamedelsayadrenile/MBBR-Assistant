"""The two tools exposed to the LLM, built fresh for each request.

The JWT is deliberately absent from every schema below. It is held on the request
context rather than on a pydantic field, so the model can neither see it nor be
tricked into echoing it: CrewAI derives the tool schema from `args_schema` alone,
and its tool-failure message re-injects that schema verbatim.

This is also why the tools are constructed per request rather than once at
startup -- CrewAI cannot bind tools at kickoff time, and mutates `Agent.tools`
in place while running, so a shared instance would not be safe anyway.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from core.config import Settings
from core.logging import json_preview
from services.devices import get_devices
from services.mbbr_api import MBBRAPIError
from services.readings import get_current_readings

logger = logging.getLogger(__name__)

GET_DEVICES = "get_devices"
GET_CURRENT_READINGS = "get_current_readings"

TOOL_FAILED_RESULT = "Tool failed temporarily."
DEVICE_NOT_FOUND_RESULT = "Device not found."


def resolve_device_id(raw_device_id: str, devices: list[dict[str, Any]]) -> str | None:
    """Map whatever the model produced onto a real device id, or None.

    The model reliably produces one of three things: the id itself, the device
    name the operator said, or a bare position ("2" for "جهاز 2"). Anything else
    is a hallucination and must not reach the API.

    Strict on purpose. This is the plant's list having the final word on which
    devices exist: the prompt requires an id copied out of the get_devices
    result, and reading the operator's own wording is the model's job, not this
    function's.
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


@dataclass
class ToolContext:
    """Per-request state the two tools share: the credentials and this turn's
    device list, fetched once so get_current_readings need not fetch it again."""

    jwt: str
    settings: Settings
    devices: list[dict[str, Any]] | None = None


class _ContextBoundTool(BaseTool):
    """Shared plumbing: the request context, and how a coroutine is run.

    `_run` has to be synchronous. CrewAI registers the sync wrapper as the
    callable for native tool calls and invokes it without awaiting, and its
    `_arun` hook is never reached because `to_structured_tool` binds `_run`.
    Running the coroutine here is safe because `MBBRAgent` executes the whole
    kickoff on a worker thread with no event loop of its own, and every MBBR
    call builds its own short-lived httpx client.
    """

    _context: ToolContext = PrivateAttr()

    def __init__(self, context: ToolContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._context = context

    def _call(self, coroutine: Any) -> Any:
        """Run one MBBR call, or raise nothing: failure is a value here.

        Letting the error propagate makes CrewAI retry the tool and then feed the
        model `Error executing tool: ...`, which both leaks internals into the
        context and bypasses the fixed Arabic sentence the prompt defines for a
        failed tool.
        """
        try:
            return asyncio.run(coroutine)
        except (MBBRAPIError, KeyError, ValueError):
            logger.exception("tool_call_failed tool_name=%s", self.name)
            return None

    def _fetch_devices(self) -> list[dict[str, Any]] | None:
        devices = self._call(get_devices(self._context.jwt, self._context.settings))
        if devices is not None:
            self._context.devices = devices
        return devices


class _NoArguments(BaseModel):
    pass


class GetDevicesTool(_ContextBoundTool):
    name: str = GET_DEVICES
    description: str = (
        "List the plant's devices with their real ids and names. "
        "Call this before get_current_readings, every time."
    )
    args_schema: type[BaseModel] = _NoArguments

    def _run(self) -> str:
        logger.info("tool_call_started tool_name=%s", self.name)
        devices = self._fetch_devices()
        if devices is None:
            return TOOL_FAILED_RESULT

        logger.info(
            "tool_call_completed tool_name=%s result=%s", self.name, json_preview(devices)
        )
        return json.dumps(devices, ensure_ascii=False)


class _ReadingsArguments(BaseModel):
    device_id: str = Field(
        ...,
        description=(
            "The real device id copied from the get_devices result. "
            "This must be an id, not the device name."
        ),
    )


class GetCurrentReadingsTool(_ContextBoundTool):
    name: str = GET_CURRENT_READINGS
    description: str = (
        "Get the latest sensor readings for one device. Use only after "
        "get_devices returned the real device id. Never pass a device name."
    )
    args_schema: type[BaseModel] = _ReadingsArguments

    def _run(self, device_id: str) -> str:
        logger.info("tool_call_started tool_name=%s", self.name)

        # The prompt tells the model to call get_devices first. If it skipped
        # that, fetch the list here rather than trusting the argument.
        devices = self._context.devices
        if devices is None:
            devices = self._fetch_devices()
            if devices is None:
                return TOOL_FAILED_RESULT

        real_device_id = resolve_device_id(device_id, devices)
        if real_device_id is None:
            # Refusing here is what makes "this device does not exist" a fact
            # rather than something the model has to remember to say.
            logger.warning("device_not_found devices=%s", len(devices))
            return DEVICE_NOT_FOUND_RESULT

        readings = self._call(
            get_current_readings(self._context.jwt, real_device_id, self._context.settings)
        )
        if readings is None:
            return TOOL_FAILED_RESULT

        logger.info(
            "tool_call_completed tool_name=%s result=%s", self.name, json_preview(readings)
        )
        return json.dumps(readings, ensure_ascii=False)


def build_tools(context: ToolContext) -> list[BaseTool]:
    """One fresh pair of tools sharing this request's context."""
    return [GetDevicesTool(context), GetCurrentReadingsTool(context)]
