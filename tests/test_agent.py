import json
from datetime import date
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

import agent.agent as agent_module
from agent.agent import FALLBACK_RESPONSE, MBBRAgent
from agent.llm import LLMError
from services.mbbr_api import MBBRAPIError
from services.memory import MemoryMessage
from tests.settings_factory import build_settings

DEVICE_1 = "b9eaf606-536b-4f38-a58e-d741cd96155b"
DEVICE_2 = "c4ca7915-78e2-46f8-81df-31848d8c1b6c"
DEVICES = [
    {"id": DEVICE_1, "name": "جهاز 1"},
    {"id": DEVICE_2, "name": "جهاز 2"},
]
READINGS = {
    "count": 1,
    "readings": [{"sensor": "water_temperature", "value": 24.7, "unit": "C"}],
}


class StructuredModel:
    def __init__(self, parent: "ScriptedModel") -> None:
        self.parent = parent

    async def ainvoke(self, messages: list[BaseMessage]) -> dict[str, Any]:
        self.parent.calls.append(("interpret", messages))
        result = self.parent.pop()
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, dict)
        return result


class ScriptedModel:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, list[BaseMessage]]] = []

    def with_structured_output(self, *_: Any, **__: Any) -> StructuredModel:
        return StructuredModel(self)

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(("compose", messages))
        result = self.pop()
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, AIMessage) else AIMessage(content=str(result))

    def pop(self) -> Any:
        if not self.responses:
            raise AssertionError("Scripted model ran out of responses")
        return self.responses.pop(0)


class RecordedServices:
    def __init__(self) -> None:
        self.devices_calls = 0
        self.current_ids: list[str] = []
        self.history_calls: list[tuple[str, date, date]] = []


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch) -> RecordedServices:
    recorded = RecordedServices()

    async def fake_devices(jwt: str, settings: Any) -> list[dict[str, str]]:
        assert jwt == "runtime-jwt"
        recorded.devices_calls += 1
        return DEVICES

    async def fake_current(jwt: str, device_id: str, settings: Any) -> dict[str, Any]:
        assert jwt == "runtime-jwt"
        recorded.current_ids.append(device_id)
        return READINGS

    async def fake_history(
        jwt: str, device_id: str, start: date, end: date, settings: Any
    ) -> dict[str, Any]:
        assert jwt == "runtime-jwt"
        recorded.history_calls.append((device_id, start, end))
        return {"device_name": "جهاز 2", "sensors": []}

    monkeypatch.setattr(agent_module, "get_devices", fake_devices)
    monkeypatch.setattr(agent_module, "get_current_readings", fake_current)
    monkeypatch.setattr(agent_module, "get_historical_readings", fake_history)
    return recorded


def build_agent(responses: list[Any]) -> tuple[MBBRAgent, ScriptedModel]:
    model = ScriptedModel(responses)
    return MBBRAgent(build_settings(), llm=model), model  # type: ignore[arg-type]


def interpretation(**fields: Any) -> dict[str, Any]:
    return fields


async def run(agent: MBBRAgent, history: list[MemoryMessage] | None = None) -> str:
    return await agent.run(
        conversation_id="conversation-1",
        jwt="runtime-jwt",
        history=history or [],
        user_message="الحرارة كام؟",
    )


async def test_direct_reply_ends_after_interpretation(services: RecordedServices) -> None:
    agent, model = build_agent(
        [interpretation(intent="reply", reply="أهلاً، أقدر أساعدك في قراءات المحطة.")]
    )

    reply = await run(agent)

    assert reply == "أهلاً، أقدر أساعدك في قراءات المحطة."
    assert [kind for kind, _ in model.calls] == ["interpret"]
    assert services.devices_calls == 0


async def test_current_reading_follows_the_linear_graph(services: RecordedServices) -> None:
    agent, model = build_agent(
        [
            interpretation(
                intent="current", device="2", measurement="درجة حرارة الماية"
            ),
            interpretation(
                intent="current",
                device_id=DEVICE_2,
                device_name="جهاز 2",
                measurement="درجة حرارة الماية",
            ),
            "درجة حرارة الماية 24.7 درجة مئوية.",
        ]
    )

    reply = await run(agent)

    assert reply == "درجة حرارة الماية 24.7 درجة مئوية."
    assert services.devices_calls == 1
    assert services.current_ids == [DEVICE_2]
    assert [kind for kind, _ in model.calls] == ["interpret", "interpret", "compose"]
    resolution_input = json.loads(model.calls[1][1][-1].content)
    assert resolution_input["available_devices"] == DEVICES


async def test_history_is_passed_as_real_chat_roles(services: RecordedServices) -> None:
    history: list[MemoryMessage] = [
        {"role": "user", "content": "مستوى المياه في جهاز 2 كام؟"},
        {"role": "assistant", "content": "مستوى المياه 1.4 متر."},
    ]
    agent, model = build_agent(
        [
            interpretation(intent="current", device="2", measurement="الضغط"),
            interpretation(
                intent="current",
                device_id=DEVICE_2,
                device_name="جهاز 2",
                measurement="الضغط",
            ),
            "الضغط 2.1 بار.",
        ]
    )

    await run(agent, history)

    messages = model.calls[0][1]
    assert [message.type for message in messages] == ["system", "human", "ai", "human"]
    assert messages[1].content == history[0]["content"]
    assert messages[2].content == history[1]["content"]


async def test_missing_device_skips_all_apis_and_is_composed(
    services: RecordedServices,
) -> None:
    agent, _ = build_agent(
        [
            interpretation(intent="reply", reply="تقصد أنهي جهاز؟"),
        ]
    )

    reply = await run(agent)

    assert reply == "تقصد أنهي جهاز؟"
    assert services.devices_calls == 0


async def test_unknown_device_never_reaches_readings_api(
    services: RecordedServices,
) -> None:
    agent, _ = build_agent(
        [
            interpretation(intent="current", device="جهاز أحمد", measurement="الضغط"),
            interpretation(intent="reply", reply="الجهاز ده مش موجود في المحطة."),
        ]
    )

    reply = await run(agent)

    assert reply == "الجهاز ده مش موجود في المحطة."
    assert services.devices_calls == 1
    assert services.current_ids == []


async def test_historical_request_uses_validated_dates(services: RecordedServices) -> None:
    agent, _ = build_agent(
        [
            interpretation(
                intent="historical",
                device="جهاز 2",
                measurement="التدفق",
                from_date="2026-07-01",
                to_date="2026-07-07",
            ),
            interpretation(
                intent="historical",
                device_id=DEVICE_2,
                device_name="جهاز 2",
                measurement="التدفق",
                from_date="2026-07-01",
                to_date="2026-07-07",
            ),
            "مفيش قراءات مسجلة للفترة دي.",
        ]
    )

    await run(agent)

    assert services.history_calls == [
        (DEVICE_2, date(2026, 7, 1), date(2026, 7, 7))
    ]


async def test_invalid_period_does_not_call_an_api(services: RecordedServices) -> None:
    agent, _ = build_agent(
        [
            interpretation(
                intent="historical",
                device="1",
                measurement="التدفق",
                from_date="امبارح",
                to_date="2026-07-01",
            ),
            interpretation(
                intent="historical",
                device_id=DEVICE_1,
                device_name="جهاز 1",
                measurement="التدفق",
                from_date="امبارح",
                to_date="2026-07-01",
            ),
            "قولّي الفترة بالتاريخ بشكل أوضح.",
        ]
    )

    reply = await run(agent)

    assert reply == "قولّي الفترة بالتاريخ بشكل أوضح."
    assert services.devices_calls == 1
    assert services.history_calls == []


async def test_empty_readings_are_composed_without_exposing_payload(
    monkeypatch: pytest.MonkeyPatch, services: RecordedServices
) -> None:
    async def empty_current(*_: Any) -> dict[str, Any]:
        return {"count": 0, "readings": []}

    monkeypatch.setattr(agent_module, "get_current_readings", empty_current)
    agent, model = build_agent(
        [
            interpretation(intent="current", device="1", measurement="الحرارة"),
            interpretation(
                intent="current",
                device_id=DEVICE_1,
                device_name="جهاز 1",
                measurement="الحرارة",
            ),
            "مفيش قراءات متاحة للجهاز ده دلوقتي.",
        ]
    )

    await run(agent)

    compose_input = json.loads(model.calls[-1][1][-1].content)
    assert compose_input["result"] == {"error": "no_readings"}


async def test_api_failure_becomes_a_composable_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_devices(*_: Any) -> list[dict[str, Any]]:
        raise MBBRAPIError("upstream is down")

    monkeypatch.setattr(agent_module, "get_devices", failing_devices)
    agent, model = build_agent(
        [
            interpretation(intent="devices"),
            "مش قادر أجيب بيانات الأجهزة دلوقتي، جرّب كمان شوية.",
        ]
    )

    reply = await run(agent)

    assert reply == "مش قادر أجيب بيانات الأجهزة دلوقتي، جرّب كمان شوية."
    assert json.loads(model.calls[-1][1][-1].content)["result"] == {
        "error": "api_failure"
    }


@pytest.mark.parametrize(
    ("content", "expected"),
    [("   ", FALLBACK_RESPONSE), ("<think>reasoning</think>الرد النهائي", "الرد النهائي")],
)
async def test_composed_reply_is_cleaned(
    services: RecordedServices, content: str, expected: str
) -> None:
    agent, _ = build_agent([interpretation(intent="devices"), content])

    assert await run(agent) == expected


async def test_model_failure_surfaces_as_llm_error() -> None:
    agent, _ = build_agent([RuntimeError("model down")])

    with pytest.raises(LLMError):
        await run(agent)


async def test_jwt_never_enters_model_messages(services: RecordedServices) -> None:
    agent, model = build_agent(
        [interpretation(intent="devices"), "في المحطة جهازين."]
    )

    await run(agent)

    serialized = json.dumps(
        [[message.content for message in messages] for _, messages in model.calls],
        ensure_ascii=False,
    )
    assert "runtime-jwt" not in serialized
