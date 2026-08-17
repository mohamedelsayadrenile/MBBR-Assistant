import json
import logging
from typing import Any

from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    GET_CURRENT_READINGS,
    GET_DEVICES,
    OPENAI_TOOLS,
    execute_tool,
    resolve_device_id,
)
from core.config import Settings
from core.logging import json_preview
from services.llm.interface import LLMProvider
from services.mbbr_api import MBBRAPIError
from services.memory import MemoryMessage

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3
FALLBACK_RESPONSE = "معلش، مش قادر أوصل لإجابة واضحة دلوقتي."
TOOL_FAILED_RESULT = "Tool failed temporarily."


class MBBRAgent:
    """Orchestrates one turn: LLM, tools, and the device-selection guard."""

    def __init__(self, llm: LLMProvider, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    async def run(
        self,
        *,
        conversation_id: str,
        jwt: str,
        history: list[MemoryMessage],
        user_message: str,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *({"role": message["role"], "content": message["content"]} for message in history),
            {"role": "user", "content": user_message},
        ]
        # Devices fetched during this turn, used to validate whatever device_id
        # the model produces. Not persisted: the next turn fetches them again.
        devices: list[dict[str, Any]] | None = None

        for tool_round in range(MAX_TOOL_ROUNDS + 1):
            assistant_message = await self._llm.chat(messages, tools=OPENAI_TOOLS)
            tool_calls = getattr(assistant_message, "tool_calls", None) or []

            if not tool_calls:
                return (assistant_message.content or "").strip() or FALLBACK_RESPONSE

            if tool_round == MAX_TOOL_ROUNDS:
                logger.warning(
                    "agent_tool_round_limit_reached conversation_id=%s max_rounds=%s",
                    conversation_id,
                    MAX_TOOL_ROUNDS,
                )
                break

            messages.append(assistant_message.model_dump(exclude_none=True))
            for tool_call in tool_calls:
                result_message, devices = await self._run_tool_call(tool_call, jwt, devices)
                messages.append(result_message)

        return FALLBACK_RESPONSE

    async def _run_tool_call(
        self,
        tool_call: Any,
        jwt: str,
        devices: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
        name = tool_call.function.name
        logger.info("tool_call_started tool_name=%s", name)

        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
            if name == GET_CURRENT_READINGS:
                result, devices = await self._readings_with_verified_device(
                    arguments, jwt, devices
                )
            else:
                result = await execute_tool(name, arguments, jwt=jwt, settings=self._settings)
                if name == GET_DEVICES:
                    devices = result
        except (json.JSONDecodeError, KeyError, ValueError, MBBRAPIError):
            logger.exception("tool_call_failed tool_name=%s", name)
            return self._tool_message(tool_call, TOOL_FAILED_RESULT), devices

        logger.info("tool_call_completed tool_name=%s result=%s", name, json_preview(result))
        return self._tool_message(tool_call, json.dumps(result, ensure_ascii=False)), devices

    async def _readings_with_verified_device(
        self,
        arguments: dict[str, Any],
        jwt: str,
        devices: list[dict[str, Any]] | None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Reject a device_id that is not in the real device list.

        This is what keeps "the API is the source of truth" enforceable in code
        rather than only in the prompt: a model that invents or misremembers a
        UUID gets the device list back instead of a reading.
        """
        if devices is None:
            # The prompt tells the model to call get_devices first. If it skipped
            # that, fetch the list ourselves rather than trusting the argument.
            devices = await execute_tool(GET_DEVICES, {}, jwt=jwt, settings=self._settings)

        raw_device_id = str(arguments.get("device_id", ""))
        device_id = resolve_device_id(raw_device_id, devices)
        if device_id is None:
            logger.warning("device_resolution_failed devices=%s", len(devices))
            return {
                "error": "unknown device_id; ask the user which device they mean",
                "devices": devices,
            }, devices

        readings = await execute_tool(
            GET_CURRENT_READINGS, {"device_id": device_id}, jwt=jwt, settings=self._settings
        )
        return readings, devices

    @staticmethod
    def _tool_message(tool_call: Any, content: str) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": content,
        }
