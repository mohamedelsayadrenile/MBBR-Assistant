import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from agent.llm import LLMError, build_llm, strip_thinking
from agent.prompts import SYSTEM_PROMPT
from agent.tools import build_tools
from core.config import Settings
from services.mbbr_api import MBBRAPIError
from services.memory import MemoryMessage

logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = "معلش، مش قادر أوصل لإجابة واضحة دلوقتي."
API_FAILURE = (
    "The plant API is not answering right now. Tell the operator there is a "
    "temporary problem and to try again shortly."
)
MAX_STEPS = 4


class MBBRAgent:
    """One turn: the model reads the operator, calls tools if it needs data, then answers."""

    def __init__(self, settings: Settings, llm: BaseChatModel | None = None) -> None:
        self._settings = settings
        self._llm = llm if llm is not None else build_llm(settings)

    async def run(
        self,
        *,
        conversation_id: str,
        jwt: str,
        history: list[MemoryMessage],
        user_message: str,
    ) -> str:
        today = datetime.now(ZoneInfo(self._settings.plant_timezone)).date()
        tools = {tool.name: tool for tool in build_tools(jwt, self._settings, today)}
        model = self._llm.bind_tools(list(tools.values()))

        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT.format(today=today.isoformat()))
        ]
        messages.extend(_as_chat(history))
        messages.append(HumanMessage(user_message))

        logger.info("agent_run_started conversation_id=%s", conversation_id)
        try:
            for _ in range(MAX_STEPS):
                response = await model.ainvoke(messages)
                messages.append(response)
                if not response.tool_calls:
                    logger.info("agent_replied conversation_id=%s", conversation_id)
                    return _clean(response.text)
                for call in response.tool_calls:
                    messages.append(await _run_tool(tools, call, conversation_id))
        except Exception as exc:
            logger.exception("agent_run_failed conversation_id=%s", conversation_id)
            raise LLMError("Agent run failed") from exc

        logger.warning("agent_step_limit_reached conversation_id=%s", conversation_id)
        return FALLBACK_RESPONSE


async def _run_tool(
    tools: dict[str, BaseTool], call: dict[str, Any], conversation_id: str
) -> ToolMessage:
    """Run one tool call and wrap whatever comes back for the model to read."""
    name = call.get("name", "")
    logger.info(
        "agent_tool_called conversation_id=%s tool=%s", conversation_id, name
    )
    tool = tools.get(name)
    if tool is None:
        content = f"There is no tool called {name}."
    else:
        try:
            content = await tool.ainvoke(call.get("args") or {})
        except MBBRAPIError:
            logger.exception(
                "agent_tool_failed conversation_id=%s tool=%s", conversation_id, name
            )
            content = API_FAILURE
    return ToolMessage(content=content, tool_call_id=call.get("id") or "")


def _as_chat(history: list[MemoryMessage]) -> list[BaseMessage]:
    return [
        HumanMessage(message["content"])
        if message["role"] == "user"
        else AIMessage(message["content"])
        for message in history
    ]


def _clean(content: Any) -> str:
    """Drop an inline reasoning block, and never hand an empty reply to TTS."""
    return (strip_thinking(str(content)) or "").strip() or FALLBACK_RESPONSE
