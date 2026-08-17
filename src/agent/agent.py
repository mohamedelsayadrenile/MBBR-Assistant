import asyncio
import logging

from crewai import LLM, Agent
from crewai.events.event_listener import EventListener

from agent.llm import build_llm, strip_thinking
from agent.prompts import (
    AGENT_GOAL,
    AGENT_ROLE,
    ASK_WHICH_DEVICE,
    DEVICE_NOT_FOUND,
    SYSTEM_PROMPT,
    build_input,
)
from agent.tools import ToolContext, build_tools, match_spoken_device
from core.config import Settings
from services.devices import get_devices
from services.llm.interface import LLMError
from services.mbbr_api import MBBRAPIError
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
    """Orchestrates one turn as a single CrewAI agent over the two MBBR tools."""

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
        if await self._answered_with_an_unknown_device(history, user_message, context):
            logger.info("device_not_found_reply conversation_id=%s", conversation_id)
            return DEVICE_NOT_FOUND

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

        if context.device_not_found:
            # The operator named something the plant does not have. Say so here
            # rather than leaving it to the model, which reliably prefers to
            # answer about a device that does exist.
            logger.info("device_not_found_reply conversation_id=%s", conversation_id)
            return DEVICE_NOT_FOUND

        return (strip_thinking(raw) or "").strip() or FALLBACK_RESPONSE

    async def _answered_with_an_unknown_device(
        self, history: list[MemoryMessage], user_message: str, context: ToolContext
    ) -> bool:
        """True when the operator has just named a device the plant does not have.

        Only applies to the turn straight after "أنهي جهاز؟", where the whole
        message is the operator's answer and can be matched against the real list
        without guessing. Left to the model, this comes out wrong most of the
        time: it maps an unknown name onto some device that does exist and
        reports that device's readings instead.
        """
        if not history:
            return False

        last = history[-1]
        if last["role"] != "assistant" or last["content"].strip() != ASK_WHICH_DEVICE:
            return False

        try:
            devices = await get_devices(jwt=context.jwt, settings=context.settings)
        except MBBRAPIError:
            # Cannot prove it is missing, so let the normal flow report the
            # outage rather than accusing the operator of a bad device name.
            logger.warning("device_precheck_skipped reason=devices_unavailable")
            return False

        # The list is good for the rest of the turn either way.
        context.devices = devices
        return match_spoken_device(user_message, devices) is None

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
