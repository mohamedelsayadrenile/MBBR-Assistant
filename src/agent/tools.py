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
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from agent.device_match import DeviceMatch, match_device
from agent.prompts import confirm_device_question
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
DEVICE_UNCLEAR_RESULT = "Device unclear. Reply with exactly: {question}"


def resolve_device_id(raw_device_id: str, devices: list[dict[str, Any]]) -> str | None:
    """Map whatever the model produced onto a real device id, or None.

    The model reliably produces one of three things: the id itself, the device
    name the operator said, or a bare position ("2" for "جهاز 2"). Anything else
    is a hallucination and must not reach the API.

    Strict on purpose. This reads a machine-written argument, and the prompt
    requires it to be an id copied out of the get_devices result, so there is
    nothing here to be tolerant of. Tolerance belongs on the operator's own
    words, which is what `match_spoken_device` reads.
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


def match_spoken_device(spoken: str, devices: list[dict[str, Any]]) -> DeviceMatch | None:
    """Resolve what the operator said onto one device, or None.

    This reads the operator's own words rather than the id the model produced,
    which is the only way to catch a name the plant does not have: a model that
    decides "جهاز النفخ" means "جهاز 1" passes a perfectly valid id, and no
    amount of id validation downstream can tell that it was the wrong device.

    Far more forgiving than `resolve_device_id`, because this is speech: the
    spelling, the alphabet and the filler words are all the operator's to choose.
    See `agent.device_match` for what that tolerance is and where it stops.
    """
    return match_device(spoken, devices)


class ToolContext:
    """Per-request state the two tools share.

    Holds the credentials, the device list fetched during this turn, whether the
    turn ended on a device the plant does not have, and the device the operator
    still has to confirm. `MBBRAgent` reads those last two to answer
    deterministically instead of trusting the model to.
    """

    def __init__(self, jwt: str, settings: Settings) -> None:
        self.jwt = jwt
        self.settings = settings
        self.devices: list[dict[str, Any]] | None = None
        self.device_not_found = False
        self.device_to_confirm: str | None = None


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

        devices = self._context.devices
        if devices is None:
            # The prompt tells the model to call get_devices first. If it skipped
            # that, fetch the list here rather than trusting the argument.
            devices = self._fetch_devices()
            if devices is None:
                return TOOL_FAILED_RESULT

        real_device_id = resolve_device_id(device_id, devices)
        if real_device_id is None:
            # Not an id, so it is the operator's own wording arriving here
            # unresolved. Read it the tolerant way before refusing.
            match = match_device(device_id, devices)
            if match is None:
                # Whatever the operator named is not in the plant's list.
                # Refusing here is what makes "this device does not exist" a fact
                # rather than something the model has to remember to say.
                logger.warning("device_not_found devices=%s", len(devices))
                self._context.device_not_found = True
                return DEVICE_NOT_FOUND_RESULT
            if not match.confident:
                # Close to an entry, but not clearly it. Asking costs one short
                # question; reading out the wrong device's numbers does not.
                logger.info("device_confirmation_needed devices=%s", len(devices))
                self._context.device_to_confirm = match.name
                return DEVICE_UNCLEAR_RESULT.format(
                    question=confirm_device_question(match.name)
                )
            real_device_id = match.id

        readings = self._call(
            get_current_readings(self._context.jwt, real_device_id, self._context.settings)
        )
        if readings is None:
            return TOOL_FAILED_RESULT

        # The turn recovered onto a device that does exist.
        self._context.device_not_found = False
        self._context.device_to_confirm = None
        logger.info(
            "tool_call_completed tool_name=%s result=%s", self.name, json_preview(readings)
        )
        return json.dumps(readings, ensure_ascii=False)


def build_tools(context: ToolContext) -> list[BaseTool]:
    """One fresh pair of tools sharing this request's context."""
    return [GetDevicesTool(context), GetCurrentReadingsTool(context)]
