import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from crewai import LLM, Agent
from crewai.events.event_listener import EventListener

from agent.device_match import DeviceMatch, find_device_mention, normalise_words
from agent.llm import LLMError, build_llm, strip_thinking
from agent.prompts import (
    AGENT_GOAL,
    AGENT_ROLE,
    ASK_WHICH_DEVICE,
    DEVICE_NOT_FOUND,
    SYSTEM_PROMPT,
    build_input,
    confirm_device_question,
    confirmed_device_name,
)
from agent.tools import ToolContext, build_tools, match_spoken_device
from core.config import Settings
from services.devices import get_devices
from services.mbbr_api import MBBRAPIError
from services.memory import MemoryMessage

logger = logging.getLogger(__name__)

MAX_ITER = 3
FALLBACK_RESPONSE = "معلش، مش قادر أوصل لإجابة واضحة دلوقتي."

# Answers to "هل تقصد جهاز 1؟", in the normalised form `normalise_words` returns.
AGREEMENTS = frozenset({"ايوه", "ايوا", "اه", "اها", "نعم", "صح", "تمام", "ماشي",
                        "اكيد", "بالظبط", "مظبوط", "yes", "yeah", "yep", "ok", "okay"})
REFUSALS = frozenset({"لا", "لاء", "لاا", "مش", "غلط", "no", "nope", "nah"})


@dataclass(frozen=True)
class _DeviceDecision:
    """What the pre-model device check concluded about this turn.

    Either the turn is already answered (`reply`), or it goes to the model with
    the operator's message, possibly rewritten to the device's real name.
    """

    reply: str | None = None
    user_message: str | None = None


def silence_crewai_console() -> None:
    """Stop CrewAI printing its banners to stdout.

    `Agent(verbose=False)` does not reach them: the event listener is a
    singleton that builds its console formatter with `verbose=True` hardcoded,
    so every turn would otherwise dump the whole system prompt twice.
    """
    EventListener().formatter.verbose = False


def _restate_with_device(message: str, device_name: str) -> str:
    """The same question with the device said out loud.

    Only used to ask the model again after it ignored the active device and
    asked which one. Naming the device inside the question is the path it does
    follow.
    """
    return f"بخصوص {device_name}: {message}"


def _carried_device(
    history: list[MemoryMessage], devices: list[dict[str, Any]]
) -> DeviceMatch | None:
    """The device this conversation is already about, or None if none is.

    Read back out of the transcript, because Redis holds the words and nothing
    else. The most recent naming wins, so the operator changes device simply by
    saying another one.
    """
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if message["role"] == "user":
            mention = find_device_mention(message["content"], devices)
            if mention is not None and mention.confident:
                return mention
            continue

        # "هل تقصد جهاز 1؟" followed by "أيوه" names a device as surely as the
        # operator saying it, and is the one case where their own words do not.
        offered = confirmed_device_name(message["content"])
        if offered is None or index + 1 >= len(history):
            continue
        answer = history[index + 1]
        if answer["role"] == "user" and _agreement(answer["content"]) is True:
            agreed = find_device_mention(offered, devices)
            if agreed is not None:
                return agreed
    return None


def _agreement(message: str) -> bool | None:
    """True for yes, False for no, None when the message says something else.

    Only a message made entirely of yes-words or entirely of no-words counts, so
    "لا، جهاز 2" falls through to be read as the device it names.
    """
    words = set(normalise_words(message))
    if not words:
        return None
    if words <= AGREEMENTS:
        return True
    if words <= REFUSALS:
        return False
    return None


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
        decision = await self._read_device_answer(history, user_message, context)
        if decision.reply is not None:
            logger.info("device_reply_decided conversation_id=%s", conversation_id)
            return decision.reply

        message = decision.user_message or user_message
        active = None
        if decision.user_message is None:
            # No device settled this turn, so an earlier one may still stand.
            active = await self._active_device(history, user_message, context)

        raw = await self._ask_model(
            build_input(history, message, active.name if active else None),
            context,
            conversation_id,
        )
        reply = self._settle(raw, context, conversation_id)

        if reply == ASK_WHICH_DEVICE and active is not None:
            # The operator already answered this question, on an earlier turn.
            # Asking it again is the one reply that is certainly wrong here, so
            # put the device inside the question and ask the model once more.
            logger.warning(
                "active_device_ignored conversation_id=%s", conversation_id
            )
            context.device_not_found = False
            context.device_to_confirm = None
            raw = await self._ask_model(
                build_input(
                    history, _restate_with_device(user_message, active.name), active.name
                ),
                context,
                conversation_id,
            )
            reply = self._settle(raw, context, conversation_id)

        return reply

    async def _ask_model(
        self, user_input: str, context: ToolContext, conversation_id: str
    ) -> str:
        try:
            return await asyncio.to_thread(self._kickoff, user_input, context)
        except Exception as exc:
            # Everything below is infrastructure: the model, the transport, or a
            # CrewAI executor that ran out of iterations without an answer. Same
            # class the caller already treats as a temporary outage.
            logger.exception("agent_run_failed conversation_id=%s", conversation_id)
            raise LLMError("CrewAI agent run failed") from exc

    def _settle(self, raw: str, context: ToolContext, conversation_id: str) -> str:
        """The reply, with the device outcomes decided here rather than by the model."""
        if context.device_not_found:
            # The operator named something the plant does not have. Say so here
            # rather than leaving it to the model, which reliably prefers to
            # answer about a device that does exist.
            logger.info("device_not_found_reply conversation_id=%s", conversation_id)
            return DEVICE_NOT_FOUND

        if context.device_to_confirm is not None:
            logger.info("device_confirm_reply conversation_id=%s", conversation_id)
            return confirm_device_question(context.device_to_confirm)

        return (strip_thinking(raw) or "").strip() or FALLBACK_RESPONSE

    async def _active_device(
        self, history: list[MemoryMessage], user_message: str, context: ToolContext
    ) -> DeviceMatch | None:
        """The device to carry into this turn, or None to let the model ask.

        Nothing is carried when the operator has named a device in this message,
        even an unclear one: they are choosing, and their choice replaces the
        old one whatever it turns out to be.
        """
        if not history:
            return None
        devices = await self._device_list(context)
        if devices is None:
            return None
        if match_spoken_device(user_message, devices) is not None:
            return None
        return _carried_device(history, devices)

    async def _device_list(self, context: ToolContext) -> list[dict[str, Any]] | None:
        """This turn's device list, fetched once, or None if it cannot be had."""
        if context.devices is not None:
            return context.devices
        try:
            context.devices = await get_devices(
                jwt=context.jwt, settings=context.settings
            )
        except MBBRAPIError:
            # Cannot prove anything about a device name, so let the normal flow
            # report the outage rather than accusing the operator of a bad name.
            logger.warning("device_lookup_skipped reason=devices_unavailable")
            return None
        return context.devices

    async def _read_device_answer(
        self, history: list[MemoryMessage], user_message: str, context: ToolContext
    ) -> _DeviceDecision:
        """Decide the turn in code when the operator is answering about a device.

        Applies to the turn straight after "أنهي جهاز؟" or after a confirmation
        question, where the whole message is the operator's answer and can be
        read against the real list without guessing. Left to the model, this
        comes out wrong most of the time: it maps a name it does not recognise
        onto some device that does exist and reports that device's readings.

        The device the operator settles on is passed to the model under its real
        name, so that a turn decided here and a turn decided by the model both
        reach get_current_readings by the same route.
        """
        proceed = _DeviceDecision()
        if not history:
            return proceed

        last = history[-1]
        if last["role"] != "assistant":
            return proceed
        question = last["content"].strip()
        offered = confirmed_device_name(question)
        if offered is None and question != ASK_WHICH_DEVICE:
            return proceed

        devices = await self._device_list(context)
        if devices is None:
            return proceed

        if offered is not None:
            answer = _agreement(user_message)
            if answer is True:
                # The question may have come from the model rather than from
                # here, so the name it offered still has to be a real one.
                agreed = match_spoken_device(offered, devices)
                if agreed is None:
                    return _DeviceDecision(reply=DEVICE_NOT_FOUND)
                return _DeviceDecision(user_message=agreed.name)
            if answer is False:
                # They rejected the only device worth offering and named no
                # other, so we are back to the question that started this.
                return _DeviceDecision(reply=ASK_WHICH_DEVICE)
            # Anything else is them naming a different device: read it below.

        match = match_spoken_device(user_message, devices)
        if match is None:
            return _DeviceDecision(reply=DEVICE_NOT_FOUND)
        if not match.confident:
            if offered is not None and match.name == offered:
                # Asking the same question again would loop. They have not agreed
                # to it, so start over rather than press.
                return _DeviceDecision(reply=ASK_WHICH_DEVICE)
            return _DeviceDecision(reply=confirm_device_question(match.name))
        return _DeviceDecision(user_message=match.name)

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
