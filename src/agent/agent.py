import logging
from datetime import datetime
from functools import partial
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agent.llm import LLMError, build_llm
from agent.nodes import FALLBACK_RESPONSE, compose, execute, interpret
from agent.schemas import INTERPRET_SCHEMA, RESOLVER_SCHEMA, AgentState, RunContext
from core.config import Settings
from services.memory import MemoryMessage

logger = logging.getLogger(__name__)


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
        graph.add_node("interpret", partial(interpret, self))
        graph.add_node("execute", partial(execute, self))
        graph.add_node("compose", partial(compose, self))
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