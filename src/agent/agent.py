import asyncio
import logging

from crewai import LLM, Agent
from crewai.events.event_listener import EventListener

from agent.llm import LLMError, build_llm, strip_thinking
from agent.prompts import AGENT_GOAL, AGENT_ROLE, SYSTEM_PROMPT, build_input
from agent.tools import ToolContext, build_tools
from core.config import Settings
from services.memory import MemoryMessage

logger = logging.getLogger(__name__)

MAX_ITER = 3
FALLBACK_RESPONSE = "معلش، مش قادر أوصل لإجابة واضحة دلوقتي."


def silence_crewai_console() -> None:
    """Stop CrewAI printing its banners to stdout.

    `Agent(verbose=False)` does not reach them: the event listener is a
    singleton that builds its console formatter with `verbose=True` hardcoded,
    so every turn would otherwise dump the whole system prompt twice.
    """
    EventListener().formatter.verbose = False


class MBBRAgent:
    """Runs one turn as a single CrewAI agent over the two MBBR tools.

    Which device the operator means, how they spelled it, and whether it carries
    over from the previous question are all the model's to work out from the
    prompt and the transcript. The tools stay the source of truth for whether a
    device exists at all.
    """

    def __init__(self, settings: Settings, llm: LLM | None = None) -> None:
        silence_crewai_console()
        self._settings = settings
        # Built once and shared. Handing CrewAI a model string instead would make
        # it construct a fresh pair of HTTP clients for every agent, and agents
        # are built per request.
        self._llm = llm if llm is not None else build_llm(settings)

    async def run(
        self,
        *,
        conversation_id: str,
        jwt: str,
        history: list[MemoryMessage],
        user_message: str,
    ) -> str:
        context = ToolContext(jwt=jwt, settings=self._settings)
        try:
            raw = await asyncio.to_thread(
                self._kickoff, build_input(history, user_message), context
            )
        except Exception as exc:
            # Everything below is infrastructure: the model, the transport, or a
            # CrewAI executor that ran out of iterations without an answer. Same
            # class the caller already treats as a temporary outage.
            logger.exception("agent_run_failed conversation_id=%s", conversation_id)
            raise LLMError("CrewAI agent run failed") from exc

        return (strip_thinking(raw) or "").strip() or FALLBACK_RESPONSE

    def _kickoff(self, user_input: str, context: ToolContext) -> str:
        """Run the agent to completion. Called on a worker thread.

        CrewAI's kickoff is synchronous, and hands back a coroutine instead of a
        result if it finds a running event loop. Keeping it off the loop entirely
        avoids that, leaves the API free to serve other requests, and lets the
        tools block on their HTTP calls.
        """
        agent = Agent(
            role=AGENT_ROLE,
            goal=AGENT_GOAL,
            backstory=SYSTEM_PROMPT,
            llm=self._llm,
            tools=build_tools(context),
            max_iter=MAX_ITER,
            verbose=False,
        )
        return agent.kickoff(user_input).raw
