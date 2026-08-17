import json
from typing import Any

import pytest

from agent import tools as tools_module
from agent.agent import FALLBACK_RESPONSE, TOOL_FAILED_RESULT, MBBRAgent
from services.mbbr_api import MBBRAPIError
from tests.llm_fake import FakeAssistantMessage, ScriptedLLM, tool_call
from tests.settings_factory import build_settings

DEVICE_1 = "b9eaf606-536b-4f38-a58e-d741cd96155b"
DEVICE_2 = "c4ca7915-78e2-46f8-81df-31848d8c1b6c"

DEVICES = [
    {"id": DEVICE_1, "name": "جهاز 1"},
    {"id": DEVICE_2, "name": "جهاز 2"},
]

WATER_TEMPERATURE = {
    "generated_at": "2026-08-16T09:16:53.011Z",
    "count": 1,
    "readings": [{"sensor": "water_temperature", "value": 24.7, "unit": "C"}],
}


class RecordedServices:
    def __init__(self) -> None:
        self.device_ids: list[str] = []
        self.devices_calls = 0


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch) -> RecordedServices:
    """Replace the two MBBR services and assert the JWT reaches both."""
    recorded = RecordedServices()

    async def fake_get_devices(jwt: str, settings: Any) -> list[dict[str, Any]]:
        assert jwt == "runtime-jwt"
        recorded.devices_calls += 1
        return DEVICES

    async def fake_get_current_readings(
        jwt: str, device_id: str, settings: Any
    ) -> dict[str, Any]:
        assert jwt == "runtime-jwt"
        recorded.device_ids.append(device_id)
        return WATER_TEMPERATURE

    monkeypatch.setattr(tools_module, "get_devices", fake_get_devices)
    monkeypatch.setattr(tools_module, "get_current_readings", fake_get_current_readings)
    return recorded


def build_agent(responses: list[FakeAssistantMessage]) -> tuple[MBBRAgent, ScriptedLLM]:
    llm = ScriptedLLM(responses)
    return MBBRAgent(llm, build_settings()), llm  # type: ignore[arg-type]


async def test_full_device_selection_flow(services: RecordedServices) -> None:
    """User asks for temperature, agent asks which device, user picks, agent answers."""
    # Turn 1: the model looks up devices, then asks which one.
    agent, _ = build_agent(
        [
            FakeAssistantMessage(tool_calls=[tool_call("call-1", "get_devices")]),
            FakeAssistantMessage(content="أنهي جهاز؟"),
        ]
    )
    first_reply = await agent.run(
        conversation_id="conversation-1",
        jwt="runtime-jwt",
        history=[],
        user_message="عايز درجة حرارة الماية دلوقتي",
    )

    assert first_reply == "أنهي جهاز؟"
    assert services.device_ids == []

    # Turn 2: the user says "جهاز 2"; the agent re-fetches devices, resolves the
    # selection to a real id, and reads the temperature back.
    history = [
        {"role": "user", "content": "عايز درجة حرارة الماية دلوقتي"},
        {"role": "assistant", "content": first_reply},
    ]
    agent, llm = build_agent(
        [
            FakeAssistantMessage(tool_calls=[tool_call("call-2", "get_devices")]),
            FakeAssistantMessage(
                tool_calls=[tool_call("call-3", "get_current_readings", device_id="2")]
            ),
            FakeAssistantMessage(content="درجة حرارة الماية دلوقتي 24.7 درجة مئوية."),
        ]
    )
    second_reply = await agent.run(
        conversation_id="conversation-1",
        jwt="runtime-jwt",
        history=history,  # type: ignore[arg-type]
        user_message="جهاز 2",
    )

    assert second_reply == "درجة حرارة الماية دلوقتي 24.7 درجة مئوية."
    assert services.device_ids == [DEVICE_2]
    # The prior turn was carried into the prompt so "جهاز 2" made sense.
    assert llm.calls[0][1]["content"] == "عايز درجة حرارة الماية دلوقتي"
    assert llm.calls[0][3]["content"] == "جهاز 2"


async def test_device_name_resolves_to_its_real_id(services: RecordedServices) -> None:
    agent, _ = build_agent(
        [
            FakeAssistantMessage(tool_calls=[tool_call("call-1", "get_devices")]),
            FakeAssistantMessage(
                tool_calls=[tool_call("call-2", "get_current_readings", device_id="جهاز 1")]
            ),
            FakeAssistantMessage(content="درجة حرارة الماية 24.7 درجة."),
        ]
    )

    await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="جهاز 1"
    )

    assert services.device_ids == [DEVICE_1]


async def test_invented_device_id_never_reaches_the_readings_api(
    services: RecordedServices,
) -> None:
    agent, llm = build_agent(
        [
            FakeAssistantMessage(tool_calls=[tool_call("call-1", "get_devices")]),
            FakeAssistantMessage(
                tool_calls=[
                    tool_call(
                        "call-2",
                        "get_current_readings",
                        device_id="00000000-dead-4000-8000-000000000000",
                    )
                ]
            ),
            FakeAssistantMessage(content="أنهي جهاز؟"),
        ]
    )

    reply = await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert services.device_ids == []
    assert reply == "أنهي جهاز؟"
    # The model is handed the real list back so it re-asks instead of guessing.
    tool_result = json.loads(llm.calls[-1][-1]["content"])
    assert tool_result["devices"] == DEVICES
    assert "unknown device_id" in tool_result["error"]


async def test_devices_are_fetched_even_when_the_model_skips_the_lookup(
    services: RecordedServices,
) -> None:
    agent, _ = build_agent(
        [
            FakeAssistantMessage(
                tool_calls=[tool_call("call-1", "get_current_readings", device_id="2")]
            ),
            FakeAssistantMessage(content="درجة حرارة الماية 24.7 درجة."),
        ]
    )

    await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert services.devices_calls == 1
    assert services.device_ids == [DEVICE_2]


async def test_empty_reading_set_is_passed_to_the_model(
    monkeypatch: pytest.MonkeyPatch, services: RecordedServices
) -> None:
    # This is what the live API returns today for every device.
    empty = {"generated_at": "2026-08-16T09:16:53.011Z", "count": 0, "readings": []}

    async def fake_readings(jwt: str, device_id: str, settings: Any) -> dict[str, Any]:
        return empty

    monkeypatch.setattr(tools_module, "get_current_readings", fake_readings)
    agent, llm = build_agent(
        [
            FakeAssistantMessage(tool_calls=[tool_call("call-1", "get_devices")]),
            FakeAssistantMessage(
                tool_calls=[tool_call("call-2", "get_current_readings", device_id="1")]
            ),
            FakeAssistantMessage(content="مفيش قراءات متاحة للجهاز ده دلوقتي."),
        ]
    )

    reply = await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert reply == "مفيش قراءات متاحة للجهاز ده دلوقتي."
    assert json.loads(llm.calls[-1][-1]["content"]) == empty


async def test_api_failure_becomes_a_tool_error_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_devices(jwt: str, settings: Any) -> list[dict[str, Any]]:
        raise MBBRAPIError("upstream is down")

    monkeypatch.setattr(tools_module, "get_devices", failing_devices)
    agent, llm = build_agent(
        [
            FakeAssistantMessage(tool_calls=[tool_call("call-1", "get_devices")]),
            FakeAssistantMessage(content="معلش، مش قادر أجيب البيانات دلوقتي."),
        ]
    )

    reply = await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert reply == "معلش، مش قادر أجيب البيانات دلوقتي."
    assert llm.calls[-1][-1]["content"] == TOOL_FAILED_RESULT


async def test_agent_gives_up_after_the_tool_round_limit(services: RecordedServices) -> None:
    looping = [
        FakeAssistantMessage(tool_calls=[tool_call(f"call-{index}", "get_devices")])
        for index in range(6)
    ]
    agent, _ = build_agent(looping)

    reply = await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert reply == FALLBACK_RESPONSE


async def test_jwt_never_enters_the_message_array(services: RecordedServices) -> None:
    agent, llm = build_agent(
        [
            FakeAssistantMessage(tool_calls=[tool_call("call-1", "get_devices")]),
            FakeAssistantMessage(
                tool_calls=[tool_call("call-2", "get_current_readings", device_id="1")]
            ),
            FakeAssistantMessage(content="درجة حرارة الماية 24.7 درجة."),
        ]
    )

    await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert "runtime-jwt" not in json.dumps(llm.calls, ensure_ascii=False)
