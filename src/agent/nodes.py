import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agent.llm import strip_thinking
from agent.prompts import (
    COMPOSE_PROMPT,
    DEVICE_RESOLVER_PROMPT,
    SENSOR_RESOLVER_PROMPT,
    SYSTEM_PROMPT,
)
from agent.schemas import AgentState, RunContext
from agent.utils import (
    match_by_id,
    match_sensor,
    narrow_current,
    narrow_daily,
    parse_date_range,
    payload_sensors,
    real_candidates,
)
from services.devices import get_devices
from services.history import get_historical_readings
from services.mbbr_api import MBBRAPIError
from services.readings import get_current_readings

if TYPE_CHECKING:
    from agent.agent import MBBRAgent

logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = "معلش، مش قادر أوصل لإجابة واضحة دلوقتي."
MAX_RANGE_DAYS = 31
ASK_DEVICE = "تقصد أي جهاز؟"
ASK_MEASUREMENT = "تقصد أنهي قراءة؟"
UNSUPPORTED_RESPONSE = "معلش، أنا بساعدك في قراءات المحطة بس."


async def interpret(
    agent: "MBBRAgent", state: AgentState, runtime: Runtime[RunContext]
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

    interpretation = await agent._interpreter.ainvoke(messages)
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


async def execute(
    agent: "MBBRAgent", state: AgentState, runtime: Runtime[RunContext]
) -> dict[str, Any]:
    if "result" in state:
        return {}
    request = state["request"]
    conversation_id = runtime.context.conversation_id
    intent = request.get("intent")

    device_required = (
        intent in {"current", "historical"}
        and bool(request.get("sensor"))
        and bool(request.get("device_name"))
    )
    logger.info(
        "agent_validation_completed conversation_id=%s intent=%s sensor_present=%s device_required=%s",
        conversation_id,
        intent,
        bool(request.get("sensor")),
        device_required,
    )

    if intent == "reply":
        return {"result": request.get("reply")}

    if intent == "unsupported":
        return {"result": UNSUPPORTED_RESPONSE}

    if intent == "station":
        try:
            devices = await get_devices(runtime.context.jwt, agent._settings)
        except MBBRAPIError:
            logger.exception(
                "agent_devices_failed conversation_id=%s", conversation_id
            )
            return {"result": {"error": "api_failure"}}
        # Counted here, not in the prompt: asked to tally a list of this length
        # the model quietly misses one (33 devices were reported as 32).
        return {"result": {"devices": devices, "count": len(devices)}}

    if not request.get("sensor"):
        return {"result": {"error": "ask_measurement"}}

    if not device_required:
        logger.info(
            "agent_clarification_required conversation_id=%s", conversation_id
        )
        return {"result": {"error": "ask_device"}}

    try:
        devices = await get_devices(runtime.context.jwt, agent._settings)
    except MBBRAPIError:
        logger.exception("agent_devices_failed conversation_id=%s", conversation_id)
        return {"result": {"error": "api_failure"}}

    logger.info(
        "agent_device_resolution_started conversation_id=%s",
        conversation_id,
    )
    resolution = await _resolve_device(agent, request.get("device_name", ""), devices)
    if resolution["status"] == "ambiguous":
        candidates = real_candidates(resolution.get("candidates"), devices)
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

    device = match_by_id(resolution.get("device_id"), devices)
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
                runtime.context.jwt, resolved["device_id"], agent._settings
            )
        else:
            start, end = date_range
            payload = await get_historical_readings(
                runtime.context.jwt,
                resolved["device_id"],
                start,
                end,
                agent._settings,
            )
    except MBBRAPIError:
        logger.exception(
            "agent_execute_failed conversation_id=%s",
            conversation_id,
        )
        return {"result": {"error": "api_failure"}, "resolution": resolved}

    # What the device reports IS the payload: a measurement in it is available,
    # one absent from it is not. An empty payload means the device sent nothing
    # at all, which is a missing reading rather than a missing sensor.
    sensors = payload_sensors(payload)
    if not sensors:
        return {"result": {"error": "no_readings"}, "resolution": resolved}

    requested = request.get("sensor")
    if requested != "all":
        resolution = await _resolve_sensor(agent, requested, sensors)
        sensor = match_sensor(resolution.get("sensor_type"), sensors)
        if sensor is None:
            if resolution.get("sensor_type"):
                logger.warning(
                    "agent_sensor_resolution_untrusted conversation_id=%s sensor_type=%s",
                    conversation_id,
                    resolution.get("sensor_type"),
                )
            logger.info(
                "agent_sensor_not_supported conversation_id=%s device_id=%s",
                conversation_id,
                resolved["device_id"],
            )
            return {
                "result": {
                    "error": "sensor_not_supported",
                    "device_name": resolved["device_name"],
                    "device_sensors": [
                        entry["type_ar"] or entry["type"] for entry in sensors
                    ],
                },
                "resolution": resolved,
            }

        logger.info(
            "agent_sensor_resolved conversation_id=%s device_id=%s sensor_type=%s",
            conversation_id,
            resolved["device_id"],
            sensor["type"],
        )
        narrow = narrow_current if intent == "current" else narrow_daily
        payload = narrow(payload, sensor["type"])
        # A daily-averages sensor is listed even with an empty `daily` series, so
        # the narrowed payload can still hold no actual values.
        if intent == "historical" and not any(
            entry.get("daily") for entry in payload.get("sensors") or []
        ):
            return {"result": {"error": "no_readings"}, "resolution": resolved}

    return {
        "resolution": resolved,
        "result": {"device_name": resolved["device_name"], "data": payload},
    }


async def _resolve_device(
    agent: "MBBRAgent", device_name: str, devices: list[dict[str, Any]]
) -> dict[str, Any]:
    response = await agent._resolver.ainvoke(
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


async def _resolve_sensor(
    agent: "MBBRAgent", user_sensor: str, sensors: list[dict[str, Any]]
) -> dict[str, Any]:
    response = await agent._sensor_resolver.ainvoke(
        [
            SystemMessage(SENSOR_RESOLVER_PROMPT),
            HumanMessage(
                json.dumps(
                    {"user_sensor": user_sensor, "sensors": sensors},
                    ensure_ascii=False,
                )
            ),
        ]
    )
    if not isinstance(response, dict):
        raise TypeError("LLM did not return a structured sensor resolution")
    return response


async def compose(
    agent: "MBBRAgent", state: AgentState, runtime: Runtime[RunContext]
) -> dict[str, str]:
    result = state["result"]
    if isinstance(result, str):
        return {"reply": _clean_reply(result)}
    if isinstance(result, dict) and result.get("error") in {
        "ask_device",
        "ask_measurement",
    }:
        return {"reply": _clarify_reply(result)}

    prompt_input: dict[str, Any] = {
        "user_message": state["user_message"],
        "request": state["request"],
        "result": result,
    }
    if state.get("resolution"):
        prompt_input["resolution"] = state["resolution"]
    response = await agent._llm.ainvoke(
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
    return {"reply": _clean_reply(response.text)}


def _clarify_reply(result: dict[str, Any]) -> str:
    if result.get("error") == "ask_measurement":
        return ASK_MEASUREMENT
    candidates = result.get("candidates") or []
    if len(candidates) >= 2:
        return f"تقصد {candidates[0]} ولا {candidates[1]}؟"
    return ASK_DEVICE


def _clean_reply(content: str | None) -> str:
    return (strip_thinking(content) or "").strip() or FALLBACK_RESPONSE
