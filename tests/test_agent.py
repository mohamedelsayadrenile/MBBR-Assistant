import json
from typing import Any

import pytest

from agent import agent as agent_module
from agent import tools as tools_module
from agent.agent import FALLBACK_RESPONSE, MBBRAgent
from agent.prompts import ASK_WHICH_DEVICE, DEVICE_NOT_FOUND, SYSTEM_PROMPT
from agent.tools import DEVICE_NOT_FOUND_RESULT, TOOL_FAILED_RESULT
from services.llm.interface import LLMError
from services.mbbr_api import MBBRAPIError
from services.memory import MemoryMessage
from tests.crewai_llm_fake import ScriptedLLM, tool_call
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
    # The agent looks devices up directly for the "أنهي جهاز؟" follow-up check,
    # so that import needs replacing too or the check reaches the network.
    monkeypatch.setattr(agent_module, "get_devices", fake_get_devices)
    return recorded


def build_agent(responses: list[Any]) -> tuple[MBBRAgent, ScriptedLLM]:
    llm = ScriptedLLM(responses)
    return MBBRAgent(build_settings(), llm=llm), llm


async def test_full_device_selection_flow(services: RecordedServices) -> None:
    """User asks for temperature, agent asks which device, user picks, agent answers."""
    # Turn 1: the model looks up devices, then asks which one.
    agent, _ = build_agent(
        [
            [tool_call("call-1", "get_devices")],
            "أنهي جهاز؟",
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

    # Turn 2: the user says "جهاز 2"; the agent re-fetches devices and reads the
    # temperature back off the real id.
    history: list[MemoryMessage] = [
        {"role": "user", "content": "عايز درجة حرارة الماية دلوقتي"},
        {"role": "assistant", "content": first_reply},
    ]
    agent, llm = build_agent(
        [
            [tool_call("call-2", "get_devices")],
            [tool_call("call-3", "get_current_readings", device_id=DEVICE_2)],
            "درجة حرارة الماية دلوقتي 24.7 درجة مئوية.",
        ]
    )
    second_reply = await agent.run(
        conversation_id="conversation-1",
        jwt="runtime-jwt",
        history=history,
        user_message="جهاز 2",
    )

    assert second_reply == "درجة حرارة الماية دلوقتي 24.7 درجة مئوية."
    assert services.device_ids == [DEVICE_2]


async def test_history_reaches_the_model_with_the_speakers_labelled(
    services: RecordedServices,
) -> None:
    """CrewAI drops message roles, so build_input has to name who said what."""
    agent, llm = build_agent(["أنهي جهاز؟"])
    history: list[MemoryMessage] = [
        {"role": "user", "content": "عايز درجة حرارة الماية دلوقتي"},
        {"role": "assistant", "content": "أنهي جهاز؟"},
    ]

    await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=history, user_message="جهاز 2"
    )

    prompt = llm.calls[0][-1]["content"]
    assert "Operator: عايز درجة حرارة الماية دلوقتي" in prompt
    assert "Assistant: أنهي جهاز؟" in prompt
    # The current turn is separated from the transcript, not appended to it.
    assert prompt.rstrip().endswith("جهاز 2")


async def test_system_prompt_reaches_the_model_verbatim(
    services: RecordedServices,
) -> None:
    agent, llm = build_agent(["أنهي جهاز؟"])

    await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    system_message = llm.calls[0][0]
    assert system_message["role"] == "system"
    assert SYSTEM_PROMPT in system_message["content"]


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
            [tool_call("call-1", "get_devices")],
            [tool_call("call-2", "get_current_readings", device_id=DEVICE_1)],
            "مفيش قراءات متاحة للجهاز ده دلوقتي.",
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
            [tool_call("call-1", "get_devices")],
            "معلش، مش قادر أجيب البيانات دلوقتي.",
        ]
    )

    reply = await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert reply == "معلش، مش قادر أجيب البيانات دلوقتي."
    assert llm.calls[-1][-1]["content"] == TOOL_FAILED_RESULT


async def test_blank_answer_falls_back(services: RecordedServices) -> None:
    agent, _ = build_agent(["   "])

    reply = await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert reply == FALLBACK_RESPONSE


async def test_a_thinking_block_is_stripped_from_the_reply(
    services: RecordedServices,
) -> None:
    agent, _ = build_agent(["<think>the operator wants a device</think>أنهي جهاز؟"])

    reply = await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert reply == "أنهي جهاز؟"


async def test_an_llm_failure_surfaces_as_llm_error(services: RecordedServices) -> None:
    """ChatService turns this into the temporary-outage reply, not a 500."""
    agent, _ = build_agent([])  # empty script: the fake raises on first call

    with pytest.raises(LLMError):
        await agent.run(
            conversation_id="c1",
            jwt="runtime-jwt",
            history=[],
            user_message="الحرارة كام؟",
        )


ASKED_WHICH: list[MemoryMessage] = [
    {"role": "user", "content": "عايز درجة حرارة الماية دلوقتي"},
    {"role": "assistant", "content": ASK_WHICH_DEVICE},
]


async def test_an_unknown_device_name_is_reported_as_not_found(
    services: RecordedServices,
) -> None:
    """The operator answers "أنهي جهاز؟" with a device the plant does not have."""
    # The model is given a script that would answer about some real device. The
    # reply is decided before it ever runs, so none of it is used.
    agent, llm = build_agent(
        [
            [tool_call("call-1", "get_devices")],
            [tool_call("call-2", "get_current_readings", device_id=DEVICE_1)],
            "درجة حرارة الماية 24.7 درجة.",
        ]
    )

    reply = await agent.run(
        conversation_id="c1",
        jwt="runtime-jwt",
        history=ASKED_WHICH,
        user_message="جهاز أحمد",
    )

    assert reply == DEVICE_NOT_FOUND
    assert services.device_ids == []
    assert llm.calls == []


@pytest.mark.parametrize("answer", ["جهاز أحمد", "جهاز النفخ", "المضخة الكبيرة", "جهاز 9"])
async def test_unknown_device_answers_never_reach_the_readings_api(
    services: RecordedServices, answer: str
) -> None:
    agent, _ = build_agent(["درجة حرارة الماية 24.7 درجة."])

    reply = await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=ASKED_WHICH, user_message=answer
    )

    assert reply == DEVICE_NOT_FOUND
    assert services.device_ids == []


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("جهاز 2", DEVICE_2),
        ("جهاز واحد", DEVICE_1),
        ("واحد", DEVICE_1),
        ("جهاز 2 من فضلك", DEVICE_2),
    ],
)
async def test_a_real_device_answer_is_handed_to_the_model(
    services: RecordedServices, answer: str, expected: str
) -> None:
    agent, _ = build_agent(
        [
            [tool_call("call-1", "get_devices")],
            [tool_call("call-2", "get_current_readings", device_id=expected)],
            "درجة حرارة الماية 24.7 درجة.",
        ]
    )

    reply = await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=ASKED_WHICH, user_message=answer
    )

    assert reply == "درجة حرارة الماية 24.7 درجة."
    assert services.device_ids == [expected]


async def test_the_check_only_applies_right_after_the_device_question(
    services: RecordedServices,
) -> None:
    """An ordinary question is not an answer to "أنهي جهاز؟" and must not trip it."""
    history: list[MemoryMessage] = [
        {"role": "user", "content": "جهاز 1"},
        {"role": "assistant", "content": "درجة حرارة الماية 24.7 درجة."},
    ]
    agent, _ = build_agent(["أنهي جهاز؟"])

    reply = await agent.run(
        conversation_id="c1",
        jwt="runtime-jwt",
        history=history,
        user_message="والضغط كام؟",
    )

    assert reply == "أنهي جهاز؟"


async def test_the_check_defers_when_the_device_list_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, services: RecordedServices
) -> None:
    """An outage must not be reported to the operator as a bad device name."""

    async def failing_devices(jwt: str, settings: Any) -> list[dict[str, Any]]:
        raise MBBRAPIError("upstream is down")

    monkeypatch.setattr(agent_module, "get_devices", failing_devices)
    monkeypatch.setattr(tools_module, "get_devices", failing_devices)
    agent, _ = build_agent(
        [
            [tool_call("call-1", "get_devices")],
            "معلش، مش قادر أجيب البيانات دلوقتي. جرّب تاني بعد شوية.",
        ]
    )

    reply = await agent.run(
        conversation_id="c1",
        jwt="runtime-jwt",
        history=ASKED_WHICH,
        user_message="جهاز أحمد",
    )

    assert reply == "معلش، مش قادر أجيب البيانات دلوقتي. جرّب تاني بعد شوية."


async def test_an_invented_device_id_never_reaches_the_readings_api(
    services: RecordedServices,
) -> None:
    agent, llm = build_agent(
        [
            [tool_call("call-1", "get_devices")],
            [
                tool_call(
                    "call-2",
                    "get_current_readings",
                    device_id="00000000-dead-4000-8000-000000000000",
                )
            ],
            "أنهي جهاز؟",
        ]
    )

    reply = await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert services.device_ids == []
    assert reply == DEVICE_NOT_FOUND
    assert llm.calls[-1][-1]["content"] == DEVICE_NOT_FOUND_RESULT


async def test_devices_are_fetched_even_when_the_model_skips_the_lookup(
    services: RecordedServices,
) -> None:
    agent, _ = build_agent(
        [
            [tool_call("call-1", "get_current_readings", device_id="2")],
            "درجة حرارة الماية 24.7 درجة.",
        ]
    )

    await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert services.devices_calls == 1
    assert services.device_ids == [DEVICE_2]


async def test_recovering_onto_a_real_device_clears_the_not_found_reply(
    services: RecordedServices,
) -> None:
    """A bad id followed by a good one must answer normally, not "not found"."""
    agent, _ = build_agent(
        [
            [tool_call("call-1", "get_devices")],
            [tool_call("call-2", "get_current_readings", device_id="جهاز أحمد")],
            [tool_call("call-3", "get_current_readings", device_id=DEVICE_1)],
            "درجة حرارة الماية 24.7 درجة.",
        ]
    )

    reply = await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert reply == "درجة حرارة الماية 24.7 درجة."
    assert services.device_ids == [DEVICE_1]


async def test_jwt_never_enters_the_message_array(services: RecordedServices) -> None:
    agent, llm = build_agent(
        [
            [tool_call("call-1", "get_devices")],
            [tool_call("call-2", "get_current_readings", device_id=DEVICE_1)],
            "درجة حرارة الماية 24.7 درجة.",
        ]
    )

    await agent.run(
        conversation_id="c1", jwt="runtime-jwt", history=[], user_message="الحرارة كام؟"
    )

    assert "runtime-jwt" not in json.dumps(llm.calls, ensure_ascii=False)
