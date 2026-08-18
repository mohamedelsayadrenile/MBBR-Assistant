import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, TypedDict
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

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

Intent = Literal["reply", "devices", "current", "historical"]
Outcome = Literal[
    "ask_device",
    "device_not_found",
    "invalid_period",
    "range_too_long",
    "no_readings",
    "api_failure",
]


class TurnInterpretation(BaseModel):
    """The only decision the LLM makes before the graph executes the request."""

    intent: Intent
    device: str | None = Field(
        default=None,
        description="Intended device name or one-based position, normalized from the conversation.",
    )
    measurement: str | None = None
    from_date: str | None = Field(default=None, description="YYYY-MM-DD")
    to_date: str | None = Field(default=None, description="YYYY-MM-DD")
    reply: str | None = Field(
        default=None,
        description="Egyptian Arabic reply, used only when intent is reply.",
    )


class AgentState(TypedDict, total=False):
    history: list[MemoryMessage]
    user_message: str
    turn: TurnInterpretation
    date_range: tuple[date, date]
    device: dict[str, Any]
    payload: Any
    outcome: Outcome
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
            TurnInterpretation, method="function_calling"
        )

        graph = StateGraph(AgentState, context_schema=RunContext)
        graph.add_node("interpret", self._interpret)
        graph.add_node("validate", self._validate)
        graph.add_node("fetch", self._fetch)
        graph.add_node("compose", self._compose)
        graph.add_edge(START, "interpret")
        graph.add_conditional_edges(
            "interpret", self._after_interpret, {"end": END, "validate": "validate"}
        )
        graph.add_conditional_edges(
            "validate", self._after_validate, {"fetch": "fetch", "compose": "compose"}
        )
        graph.add_edge("fetch", "compose")
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

        turn = await self._interpreter.ainvoke(messages)
        if not isinstance(turn, TurnInterpretation):
            raise TypeError("LLM did not return a TurnInterpretation")

        update: dict[str, Any] = {"turn": turn}
        if turn.intent == "reply":
            update["reply"] = self._clean_reply(turn.reply)
        logger.info(
            "agent_turn_interpreted conversation_id=%s intent=%s",
            runtime.context.conversation_id,
            turn.intent,
        )
        return update

    @staticmethod
    def _after_interpret(state: AgentState) -> Literal["end", "validate"]:
        return "end" if state["turn"].intent == "reply" else "validate"

    @staticmethod
    def _validate(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
        turn = state["turn"]
        if turn.intent in {"current", "historical"} and not turn.device:
            return {"outcome": "ask_device"}

        if turn.intent != "historical":
            return {}
        if not turn.from_date or not turn.to_date:
            return {"outcome": "invalid_period"}

        date_range = parse_date_range(turn.from_date, turn.to_date, runtime.context.today)
        if date_range is None:
            return {"outcome": "invalid_period"}
        if (date_range[1] - date_range[0]).days >= MAX_RANGE_DAYS:
            return {"outcome": "range_too_long"}
        return {"date_range": date_range}

    @staticmethod
    def _after_validate(state: AgentState) -> Literal["fetch", "compose"]:
        return "compose" if state.get("outcome") else "fetch"

    async def _fetch(
        self, state: AgentState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        turn = state["turn"]
        try:
            devices = await get_devices(runtime.context.jwt, self._settings)
            if turn.intent == "devices":
                return {"payload": [device["name"] for device in devices]}

            reference = (turn.device or "").strip()
            device = next(
                (
                    device
                    for index, device in enumerate(devices, start=1)
                    if reference == device["id"]
                    or reference == str(index)
                    or reference.casefold() == device["name"].casefold()
                ),
                None,
            )
            if device is None:
                return {"outcome": "device_not_found"}
            device_id = device["id"]

            if turn.intent == "current":
                payload = await get_current_readings(
                    runtime.context.jwt, device_id, self._settings
                )
                if payload.get("count") == 0 or payload.get("readings") == []:
                    return {
                        "device": device,
                        "outcome": "no_readings",
                    }
            else:
                start, end = state["date_range"]
                payload = await get_historical_readings(
                    runtime.context.jwt, device_id, start, end, self._settings
                )
        except MBBRAPIError:
            logger.exception(
                "agent_fetch_failed conversation_id=%s", runtime.context.conversation_id
            )
            return {"outcome": "api_failure"}

        return {"device": device, "payload": payload}

    async def _compose(
        self, state: AgentState, runtime: Runtime[RunContext]
    ) -> dict[str, str]:
        device = state.get("device")
        prompt_input = {
            "request": state["turn"].model_dump(),
            "device_name": device.get("name") if device else None,
            "outcome": state.get("outcome"),
            "data": state.get("payload"),
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
            "agent_reply_composed conversation_id=%s outcome=%s",
            runtime.context.conversation_id,
            state.get("outcome", "success"),
        )
        return {"reply": self._clean_reply(response.text)}

    @staticmethod
    def _clean_reply(content: str | None) -> str:
        return (strip_thinking(content) or "").strip() or FALLBACK_RESPONSE
