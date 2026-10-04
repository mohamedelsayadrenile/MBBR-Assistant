import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRequest,
    ToolCallRequest,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agent.llm import LLMError, build_llm, strip_thinking
from agent.prompts import SYSTEM_PROMPT
from agent.tools import TurnContext, build_tools
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
        self._graph = create_agent(
            model=llm if llm is not None else build_llm(settings),
            tools=build_tools(settings),
            context_schema=TurnContext,
            middleware=[
                _system_prompt,
                _tool_errors,
                ModelCallLimitMiddleware(run_limit=MAX_STEPS, exit_behavior="error"),
            ],
        )

    async def run(
        self,
        *,
        conversation_id: str,
        jwt: str,
        history: list[MemoryMessage],
        user_message: str,
    ) -> str:
        today = datetime.now(ZoneInfo(self._settings.plant_timezone)).date()
        messages = [*_as_chat(history), HumanMessage(user_message)]

        logger.info("agent_run_started conversation_id=%s", conversation_id)
        try:
            result = await self._graph.ainvoke(
                {"messages": messages},
                context=TurnContext(
                    conversation_id=conversation_id, jwt=jwt, today=today
                ),
            )
        except ModelCallLimitExceededError:
            logger.warning(
                "agent_step_limit_reached conversation_id=%s", conversation_id
            )
            return FALLBACK_RESPONSE
        except Exception as exc:
            logger.exception("agent_run_failed conversation_id=%s", conversation_id)
            raise LLMError("Agent run failed") from exc

        logger.info("agent_replied conversation_id=%s", conversation_id)
        return _clean(result["messages"][-1].text)


@dynamic_prompt
def _system_prompt(request: ModelRequest) -> str:
    return SYSTEM_PROMPT.format(today=request.runtime.context.today.isoformat())


@wrap_tool_call
async def _tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    """Turn an unknown tool or a down API into a message the model can explain."""
    call = request.tool_call
    name = call.get("name", "")
    conversation_id = request.runtime.context.conversation_id
    logger.info("agent_tool_called conversation_id=%s tool=%s", conversation_id, name)
    if request.tool is None:
        return _tool_message(f"There is no tool called {name}.", call)
    try:
        return await handler(request)
    except MBBRAPIError:
        logger.exception(
            "agent_tool_failed conversation_id=%s tool=%s", conversation_id, name
        )
        return _tool_message(API_FAILURE, call)


def _tool_message(content: str, call: dict[str, Any]) -> ToolMessage:
    return ToolMessage(
        content=content, tool_call_id=call.get("id") or "", name=call.get("name")
    )


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
