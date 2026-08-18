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
from agent.prompts import SYSTEM_PROMPT
from core.config import Settings
from services.devices import get_devices
from services.history import get_historical_readings
from services.mbbr_api import MBBRAPIError
from services.memory import MemoryMessage
from services.readings import get_current_readings

logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = "معلش، مش قادر أوصل لإجابة واضحة دلوقتي."
MAX_RANGE_DAYS = 31

class AgentState(TypedDict, total=False):
    history: list[MemoryMessage]
    user_message: str
    request: dict[str, Any]
    devices: list[dict[str, Any]]
    result: Any
    reply: str


@dataclass(frozen=True)
class RunContext:
    conversation_id: str
    jwt: str
    today: date


def parse_date_range(raw_from: str, raw_to: str, today: date) -> tuple[date, date] | None:
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
            {
                "title": "interpret_turn",
                "description": "Interpret the operator's current turn.",
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["reply", "devices", "current", "historical"],
                    },
                    "device": {
                        "type": "string",
                        "description": "Device wording understood from the operator.",
                    },
                    "device_id": {
                        "type": "string",
                        "description": "Exact id from the supplied device list.",
                    },
                    "device_name": {
                        "type": "string",
                        "description": "Exact name from the supplied device list.",
                    },
                    "measurement": {"type": "string"},
                    "from_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "to_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "reply": {
                        "type": "string",
                        "description": "Egyptian Arabic reply for reply intent.",
                    },
                },
                "required": ["intent"],
            },
            method="function_calling",
        )

        graph = StateGraph(AgentState, context_schema=RunContext)
        graph.add_node("interpret", self._interpret)
        graph.add_node("execute", self._execute)
        graph.add_node("compose", self._compose)
        graph.add_edge(START, "interpret")
        graph.add_edge("interpret", "execute")
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

        update: dict[str, Any] = {"request": interpretation}
        if interpretation["intent"] != "reply":
            try:
                devices = await get_devices(runtime.context.jwt, self._settings)
            except MBBRAPIError:
                logger.exception(
                    "agent_devices_failed conversation_id=%s",
                    runtime.context.conversation_id,
                )
                update["result"] = {"error": "api_failure"}
            else:
                update["devices"] = devices
                if interpretation["intent"] in {"current", "historical"}:
                    resolution = await self._interpreter.ainvoke(
                        [
                            SystemMessage(
                                SYSTEM_PROMPT.format(
                                    today=runtime.context.today.isoformat()
                                )
                            ),
                            HumanMessage(
                                json.dumps(
                                    {
                                        "interpretation": interpretation,
                                        "available_devices": devices,
                                    },
                                    ensure_ascii=False,
                                )
                            ),
                        ]
                    )
                    if not isinstance(resolution, dict) or "intent" not in resolution:
                        raise TypeError("LLM did not return a device resolution")
                    update["request"] = resolution
        logger.info(
            "agent_turn_interpreted conversation_id=%s intent=%s",
            runtime.context.conversation_id,
            update["request"]["intent"],
        )
        return update

    async def _execute(
        self, state: AgentState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        if "result" in state:
            return {}
        request = state["request"]
        if request["intent"] == "reply":
            return {"result": request.get("reply")}
        if request["intent"] == "devices":
            return {"result": [device["name"] for device in state["devices"]]}

        device = next(
            (
                device
                for device in state["devices"]
                if device["id"] == request.get("device_id")
            ),
            None,
        )
        if device is None:
            return {"result": {"error": "device_not_found"}}

        if request["intent"] == "historical":
            if not request.get("from_date") or not request.get("to_date"):
                return {"result": {"error": "invalid_period"}}
            date_range = parse_date_range(
                request["from_date"], request["to_date"], runtime.context.today
            )
            if date_range is None:
                return {"result": {"error": "invalid_period"}}
            if (date_range[1] - date_range[0]).days >= MAX_RANGE_DAYS:
                return {"result": {"error": "range_too_long"}}

        try:
            if request["intent"] == "current":
                payload = await get_current_readings(
                    runtime.context.jwt, device["id"], self._settings
                )
                if payload.get("count") == 0 or payload.get("readings") == []:
                    return {"result": {"error": "no_readings"}}
            else:
                start, end = date_range
                payload = await get_historical_readings(
                    runtime.context.jwt, device["id"], start, end, self._settings
                )
        except MBBRAPIError:
            logger.exception(
                "agent_execute_failed conversation_id=%s",
                runtime.context.conversation_id,
            )
            return {"result": {"error": "api_failure"}}

        return {"result": {"device_name": device["name"], "data": payload}}

    async def _compose(
        self, state: AgentState, runtime: Runtime[RunContext]
    ) -> dict[str, str]:
        if state["request"]["intent"] == "reply":
            return {"reply": self._clean_reply(state["result"])}

        prompt_input = {
            "user_message": state["user_message"],
            "request": state["request"],
            "result": state["result"],
        }
        response = await self._llm.ainvoke(
            [
                SystemMessage(
                    SYSTEM_PROMPT.format(today=runtime.context.today.isoformat())
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
    def _clean_reply(content: str | None) -> str:
        return (strip_thinking(content) or "").strip() or FALLBACK_RESPONSE
