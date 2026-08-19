import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, TypedDict
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agent.llm import LLMError, build_llm, strip_thinking
from agent.prompts import COMPOSE_PROMPT, DEVICE_RESOLVER_PROMPT, SYSTEM_PROMPT
from core.config import Settings
from services.devices import get_devices
from services.history import get_historical_readings
from services.mbbr_api import MBBRAPIError
from services.memory import MemoryMessage
from services.readings import get_current_readings

logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = "معلش، مش قادر أوصل لإجابة واضحة دلوقتي."
MAX_RANGE_DAYS = 31
ASK_DEVICE = "تقصد أي جهاز؟"
ASK_MEASUREMENT = "تقصد أنهي قراءة؟"


class AgentState(TypedDict, total=False):
    history: list[MemoryMessage]
    user_message: str
    request: dict[str, Any]
    device_required: bool
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
            "enum": ["reply", "current", "historical"],
        },
        "sensor": {
            "type": "string",
            "description": "The measurement the operator asked about.",
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


def parse_date_range(
    raw_from: str, raw_to: str, today: date
) -> tuple[date, date] | None:
    """Parse a strict ISO date range, rejecting future or reversed windows."""

    def parse(value: str) -> date | None:
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None

    start = parse(raw_from)
    end = parse(raw_to)
    if start is None or end is None or start > end or start > today:
        return None
    return start, min(end, today)


class MBBRAgent:
    """Runs one turn through a small, bounded LangGraph workflow."""

    def __init__(self, settings: Settings, llm: BaseChatModel | None = None) -> None:
        self._settings = settings
        self._llm = llm if llm is not None else build_llm(settings)
        self._interpreter = self._llm.with_structured_output(
            INTERPRET_SCHEMA, method="function_calling"
        )
        self._resolver = self._llm.with_structured_output(
            RESOLVER_SCHEMA, method="function_calling"
        )

        graph = StateGraph(AgentState, context_schema=RunContext)
        graph.add_node("interpret", self._interpret)
        graph.add_node("validate", self._validate)
        graph.add_node("execute", self._execute)
        graph.add_node("compose", self._compose)
        graph.add_edge(START, "interpret")
        graph.add_edge("interpret", "validate")
        graph.add_edge("validate", "execute")
        graph.add_edge("execute", "compose")
        graph.add_edge("compose", END)
        self._graph = graph.compile()

    async def run(
        self,
        *,
        conversation_id: str,
        jwt: str,
        history: list[MemoryMessage],
        user_message: str,
    ) -> str:
        today = datetime.now(ZoneInfo(self._settings.plant_timezone)).date()
        try:
            result = await self._graph.ainvoke(
                {"history": history, "user_message": user_message},
                context=RunContext(
                    conversation_id=conversation_id,
                    jwt=jwt,
                    today=today,
                ),
            )
        except Exception as exc:
            logger.exception("agent_run_failed conversation_id=%s", conversation_id)
            raise LLMError("LangGraph agent run failed") from exc

        return result.get("reply") or FALLBACK_RESPONSE

    async def _interpret(
        self, state: AgentState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        conversation_id = runtime.context.conversation_id
        logger.info("agent_interpret_started conversation_id=%s", conversation_id)

        messages: list[SystemMessage | HumanMessage | AIMessage] = [
            SystemMessage(
                SYSTEM_PROMPT.format(
                    today=runtime.context.today.isoformat(),
                )
            )
        ]
        messages.extend(
            HumanMessage(message["content"])
            if message["role"] == "user"
            else AIMessage(message["content"])
            for message in state["history"]
        )
        messages.append(HumanMessage(state["user_message"]))

        interpretation = await self._interpreter.ainvoke(messages)
        if not isinstance(interpretation, dict) or "intent" not in interpretation:
            raise TypeError("LLM did not return a structured interpretation")

        logger.info(
            "agent_interpreted conversation_id=%s intent=%s sensor=%s device_name=%s",
            conversation_id,
            interpretation.get("intent"),
            interpretation.get("sensor"),
            interpretation.get("device_name"),
        )
        return {"request": interpretation}

    async def _validate(
        self, state: AgentState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        request = state["request"]
        intent = request.get("intent")
        sensor = request.get("sensor")
        device_name = request.get("device_name")
        device_required = (
            intent in {"current", "historical"} and bool(sensor) and bool(device_name)
        )
        logger.info(
            "agent_validation_completed conversation_id=%s intent=%s sensor_present=%s device_required=%s",
            runtime.context.conversation_id,
            intent,
            bool(sensor),
            device_required,
        )
        return {"device_required": device_required}

    async def _execute(
        self, state: AgentState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        if "result" in state:
            return {}
        request = state["request"]
        conversation_id = runtime.context.conversation_id
        intent = request.get("intent")

        if intent == "reply":
            return {"result": request.get("reply")}

        if not request.get("sensor"):
            return {"result": {"error": "ask_measurement"}}

        if not state.get("device_required"):
            logger.info(
                "agent_clarification_required conversation_id=%s", conversation_id
            )
            return {"result": {"error": "ask_device"}}

        try:
            devices = await get_devices(runtime.context.jwt, self._settings)
        except MBBRAPIError:
            logger.exception("agent_devices_failed conversation_id=%s", conversation_id)
            return {"result": {"error": "api_failure"}}

        logger.info(
            "agent_device_resolution_started conversation_id=%s",
            conversation_id,
        )
        resolution = await self._resolve_device(request.get("device_name", ""), devices)
        if resolution["status"] == "ambiguous":
            candidates = self._real_candidates(resolution.get("candidates"), devices)
            logger.info(
                "agent_clarification_required conversation_id=%s", conversation_id
            )
            return {
                "result": {
                    "error": "ask_device",
                    "candidates": candidates,
                }
            }
        if resolution["status"] != "matched":
            return {"result": {"error": "device_not_found"}}

        device = self._match_by_id(resolution.get("device_id"), devices)
        if device is None:
            logger.warning(
                "agent_device_resolution_untrusted conversation_id=%s device_id=%s",
                conversation_id,
                resolution.get("device_id"),
            )
            return {"result": {"error": "device_not_found"}}

        resolved = {"device_id": device["id"], "device_name": device["name"]}
        logger.info(
            "agent_device_resolved conversation_id=%s status=matched device_id=%s device_name=%s",
            conversation_id,
            device["id"],
            device["name"],
        )

        if intent == "historical":
            if not request.get("from_date") or not request.get("to_date"):
                return {"result": {"error": "invalid_period"}, "resolution": resolved}
            date_range = parse_date_range(
                request["from_date"], request["to_date"], runtime.context.today
            )
            if date_range is None:
                return {"result": {"error": "invalid_period"}, "resolution": resolved}
            if (date_range[1] - date_range[0]).days >= MAX_RANGE_DAYS:
                return {"result": {"error": "range_too_long"}, "resolution": resolved}

        try:
            if intent == "current":
                payload = await get_current_readings(
                    runtime.context.jwt, resolved["device_id"], self._settings
                )
                if payload.get("count") == 0 or payload.get("readings") == []:
                    return {"result": {"error": "no_readings"}, "resolution": resolved}
            else:
                start, end = date_range
                payload = await get_historical_readings(
                    runtime.context.jwt,
                    resolved["device_id"],
                    start,
                    end,
                    self._settings,
                )
        except MBBRAPIError:
            logger.exception(
                "agent_execute_failed conversation_id=%s",
                conversation_id,
            )
            return {"result": {"error": "api_failure"}, "resolution": resolved}

        return {
            "resolution": resolved,
            "result": {"device_name": resolved["device_name"], "data": payload},
        }

    async def _resolve_device(
        self, device_name: str, devices: list[dict[str, Any]]
    ) -> dict[str, Any]:
        response = await self._resolver.ainvoke(
            [
                SystemMessage(DEVICE_RESOLVER_PROMPT),
                HumanMessage(
                    json.dumps(
                        {
                            "user_device": device_name,
                            "available_devices": devices,
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        if not isinstance(response, dict) or "status" not in response:
            raise TypeError("LLM did not return a structured device resolution")
        return response

    @staticmethod
    def _match_by_id(
        device_id: Any, devices: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Return the live-list entry whose id equals `device_id`, if any."""
        for device in devices:
            if device.get("id") == device_id:
                return device
        return None

    @staticmethod
    def _real_candidates(candidates: Any, devices: list[dict[str, Any]]) -> list[str]:
        """Floor resolver candidate names to exact live-list names, taking two max."""
        names = [device["name"] for device in devices]
        seen: list[str] = []
        for candidate in candidates or []:
            if candidate in names and candidate not in seen:
                seen.append(candidate)
            if len(seen) == 2:
                break
        return seen

    async def _compose(
        self, state: AgentState, runtime: Runtime[RunContext]
    ) -> dict[str, str]:
        result = state["result"]
        if isinstance(result, str):
            return {"reply": self._clean_reply(result)}
        if isinstance(result, dict) and result.get("error") in {
            "ask_device",
            "ask_measurement",
        }:
            return {"reply": self._clarify_reply(result)}

        prompt_input: dict[str, Any] = {
            "user_message": state["user_message"],
            "request": state["request"],
            "result": result,
        }
        if state.get("resolution"):
            prompt_input["resolution"] = state["resolution"]
        response = await self._llm.ainvoke(
            [
                SystemMessage(
                    COMPOSE_PROMPT.format(today=runtime.context.today.isoformat())
                ),
                HumanMessage(json.dumps(prompt_input, ensure_ascii=False, default=str)),
            ]
        )
        logger.info(
            "agent_reply_composed conversation_id=%s",
            runtime.context.conversation_id,
        )
        return {"reply": self._clean_reply(response.text)}

    @staticmethod
    def _clarify_reply(result: dict[str, Any]) -> str:
        if result.get("error") == "ask_measurement":
            return ASK_MEASUREMENT
        candidates = result.get("candidates") or []
        if len(candidates) >= 2:
            return f"تقصد {candidates[0]} ولا {candidates[1]}؟"
        return ASK_DEVICE

    @staticmethod
    def _clean_reply(content: str | None) -> str:
        return (strip_thinking(content) or "").strip() or FALLBACK_RESPONSE
