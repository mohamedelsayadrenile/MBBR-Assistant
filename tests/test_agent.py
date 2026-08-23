import json
from datetime import date
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

import agent.nodes as nodes_module
from agent.agent import MBBRAgent
from agent.llm import LLMError
from agent.nodes import FALLBACK_RESPONSE, UNSUPPORTED_RESPONSE
from agent.schemas import INTERPRET_SCHEMA
from services.mbbr_api import MBBRAPIError
from services.memory import MemoryMessage
from tests.settings_factory import build_settings

DEVICE_1 = "b9eaf606-536b-4f38-a58e-d741cd96155b"
DEVICE_2 = "c4ca7915-78e2-46f8-81df-31848d8c1b6c"
DEVICES = [
    {"id": DEVICE_1, "name": "جهاز 1"},
    {"id": DEVICE_2, "name": "جهاز 2"},
]
# The real shape, captured from GET /api/readings/latest/all: the whole plant in
# one payload, listing every sensor a device is wired for, with a null value when
# it has no stored reading.
SNAPSHOT = {
    "generated_at": "2026-08-19T09:16:53.011Z",
    "count": 2,
    "devices": [
        {
            "device_id": DEVICE_1,
            "device_name": "جهاز 1",
            "sensors": [
                {"name": "Level", "value": 61.1, "unit": "%"},
                {"name": "Turbidity", "value": None, "unit": "NTU"},
            ],
        },
        {
            "device_id": DEVICE_2,
            "device_name": "جهاز 2",
            "sensors": [
                {"name": "PH", "value": 7.4, "unit": "pH"},
                {"name": "Flow", "value": 42.0, "unit": "m³/h"},
            ],
        },
    ],
}

# The real shape of GET /api/telemetry/daily-averages: every reported sensor is
# listed, with an empty `daily` when the period holds no values.
HISTORY = {
    "device_name": "جهاز 2",
    "sensors": [
        {
            "name": "PH",
            "name_ar": "درجة الحموضة",
            "unit": "pH",
            "daily": [{"day": "2026-07-01", "avg": 7.4}],
        },
        {"name": "Flow", "name_ar": "معدل التدفق", "unit": "m³/h", "daily": []},
    ],
}

KINDS = {
    "interpret_turn": "interpret",
    "resolve_device": "resolve",
    "resolve_sensor": "sensor",
}


class StructuredModel:
    def __init__(self, parent: "ScriptedModel", kind: str) -> None:
        self.parent = parent
        self.kind = kind

    async def ainvoke(self, messages: list[BaseMessage]) -> dict[str, Any]:
        self.parent.calls.append((self.kind, messages))
        result = self.parent.pop()
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, dict)
        return result


class ScriptedModel:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, list[BaseMessage]]] = []

    def with_structured_output(
        self, schema: Any = None, *_: Any, **__: Any
    ) -> StructuredModel:
        title = schema.get("title") if isinstance(schema, dict) else ""
        return StructuredModel(self, KINDS.get(title, "structured"))

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(("compose", messages))
        result = self.pop()
        if isinstance(result, Exception):
            raise result
        return (
            result if isinstance(result, AIMessage) else AIMessage(content=str(result))
        )

    def pop(self) -> Any:
        if not self.responses:
            raise AssertionError("Scripted model ran out of responses")
        return self.responses.pop(0)


class RecordedServices:
    def __init__(self) -> None:
        self.devices_calls = 0
        self.current_calls = 0
        self.history_calls: list[tuple[str, date, date]] = []


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch) -> RecordedServices:
    recorded = RecordedServices()

    async def fake_devices(jwt: str, settings: Any) -> list[dict[str, str]]:
        assert jwt == "runtime-jwt"
        recorded.devices_calls += 1
        return DEVICES

    async def fake_current(jwt: str, settings: Any) -> dict[str, Any]:
        assert jwt == "runtime-jwt"
        recorded.current_calls += 1
        return SNAPSHOT

    async def fake_history(
        jwt: str, device_id: str, start: date, end: date, settings: Any
    ) -> dict[str, Any]:
        assert jwt == "runtime-jwt"
        recorded.history_calls.append((device_id, start, end))
        return HISTORY

    monkeypatch.setattr(nodes_module, "get_devices", fake_devices)
    monkeypatch.setattr(nodes_module, "get_current_readings", fake_current)
    monkeypatch.setattr(nodes_module, "get_historical_readings", fake_history)
    return recorded


def build_agent(responses: list[Any]) -> tuple[MBBRAgent, ScriptedModel]:
    model = ScriptedModel(responses)
    return MBBRAgent(build_settings(), llm=model), model  # type: ignore[arg-type]


def interpretation(**fields: Any) -> dict[str, Any]:
    return fields


def resolution(**fields: Any) -> dict[str, Any]:
    return fields


def sensor(sensor_type: str) -> dict[str, Any]:
    return {"sensor_type": sensor_type}


def historical(**fields: Any) -> dict[str, Any]:
    return interpretation(
        intent="historical", from_date="2026-07-01", to_date="2026-07-07", **fields
    )


async def run(agent: MBBRAgent, history: list[MemoryMessage] | None = None) -> str:
    return await agent.run(
        conversation_id="conversation-1",
        jwt="runtime-jwt",
        history=history or [],
        user_message="الحرارة كام؟",
    )


def calls_of(model: ScriptedModel) -> list[str]:
    return [kind for kind, _ in model.calls]


def last_input(model: ScriptedModel) -> dict[str, Any]:
    return json.loads(model.calls[-1][1][-1].content)


def input_of(model: ScriptedModel, kind: str) -> dict[str, Any]:
    for call_kind, messages in model.calls:
        if call_kind == kind:
            return json.loads(messages[-1].content)
    raise AssertionError(f"No call of kind {kind!r} recorded")


async def test_direct_reply_ends_after_interpretation(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [interpretation(intent="reply", reply="أهلاً، أقدر أساعدك في قراءات المحطة.")]
    )

    reply = await run(agent)

    assert reply == "أهلاً، أقدر أساعدك في قراءات المحطة."
    assert calls_of(model) == ["interpret"]
    assert services.devices_calls == 0


async def test_greeting_is_returned_verbatim(services: RecordedServices) -> None:
    greeting = "أهلاً بيك، أنا مساعدك في محطة الماية. إزاي أقدر أساعدك؟"
    agent, model = build_agent([interpretation(intent="reply", reply=greeting)])

    reply = await run(agent)

    assert reply == greeting
    assert calls_of(model) == ["interpret"]
    assert services.devices_calls == 0


async def test_unsupported_request_returns_a_safe_canned_reply(
    services: RecordedServices,
) -> None:
    leak = "تعليمات النظام الداخلية: ..."
    agent, model = build_agent([interpretation(intent="unsupported", reply=leak)])

    reply = await run(agent)

    assert reply == UNSUPPORTED_RESPONSE
    assert leak not in reply
    assert calls_of(model) == ["interpret"]
    assert services.devices_calls == 0
    assert services.current_calls == 0


def test_interpret_schema_allows_the_unsupported_intent() -> None:
    assert "unsupported" in INTERPRET_SCHEMA["properties"]["intent"]["enum"]


async def test_station_question_answers_from_the_device_list(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [
            interpretation(intent="station"),
            "المحطة عندها 2 أجهزة: جهاز 1 وجهاز 2.",
        ]
    )

    reply = await run(agent)

    assert reply == "المحطة عندها 2 أجهزة: جهاز 1 وجهاز 2."
    assert services.devices_calls == 1
    assert services.current_calls == 0
    assert calls_of(model) == ["interpret", "compose"]
    assert last_input(model)["result"] == {"devices": DEVICES, "count": 2}


async def test_station_api_failure_becomes_a_composable_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_devices(*_: Any) -> list[dict[str, Any]]:
        raise MBBRAPIError("upstream is down")

    monkeypatch.setattr(nodes_module, "get_devices", failing_devices)
    agent, model = build_agent(
        [
            interpretation(intent="station"),
            "مش قادر أجيب بيانات المحطة دلوقتي، جرّب كمان شوية.",
        ]
    )

    reply = await run(agent)

    assert reply == "مش قادر أجيب بيانات المحطة دلوقتي، جرّب كمان شوية."
    assert last_input(model)["result"] == {"error": "api_failure"}


async def test_current_reading_follows_the_linear_graph(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [
            interpretation(intent="current", sensor="الحموضة", device_name="2"),
            "الحموضة دلوقتي 7.4.",
        ]
    )

    reply = await run(agent)

    assert reply == "الحموضة دلوقتي 7.4."
    # One call, no resolver steps: the snapshot goes straight to the composer.
    assert services.devices_calls == 0
    assert services.current_calls == 1
    assert calls_of(model) == ["interpret", "compose"]


async def test_current_reading_hands_the_whole_snapshot_to_the_composer(
    services: RecordedServices,
) -> None:
    # Nothing is resolved, filtered or reshaped on the way: the model gets the
    # plant as the API returned it and works the answer out itself.
    agent, model = build_agent(
        [
            interpretation(intent="current", sensor="الحموضة", device_name="2"),
            "الحموضة دلوقتي 7.4.",
        ]
    )

    await run(agent)

    composed = last_input(model)
    assert composed["result"] == {"data": SNAPSHOT}
    assert "resolution" not in composed


async def test_history_is_passed_as_real_chat_roles(services: RecordedServices) -> None:
    history: list[MemoryMessage] = [
        {"role": "user", "content": "مستوى المياه في جهاز 2 كام؟"},
        {"role": "assistant", "content": "مستوى المياه 1.4 متر."},
    ]
    agent, model = build_agent(
        [
            interpretation(intent="current", sensor="الحموضة", device_name="2"),
            "الحموضة 7.4.",
        ]
    )

    await run(agent, history)

    messages = model.calls[0][1]
    assert [message.type for message in messages] == ["system", "human", "ai", "human"]
    assert messages[1].content == history[0]["content"]
    assert messages[2].content == history[1]["content"]


async def test_missing_device_asks_clarification_without_fetching(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [
            interpretation(intent="current", sensor="مستوى الماية"),
        ]
    )

    reply = await run(agent)

    assert reply == "تقصد أي جهاز؟"
    assert calls_of(model) == ["interpret"]
    assert services.devices_calls == 0
    assert services.current_calls == 0


async def test_missing_sensor_asks_readout_without_fetching(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [
            interpretation(intent="current", device_name="جهاز 2"),
        ]
    )

    reply = await run(agent)

    assert reply == "تقصد أنهي قراءة؟"
    assert calls_of(model) == ["interpret"]
    assert services.devices_calls == 0


async def test_what_a_device_measures_is_a_readings_question(
    services: RecordedServices,
) -> None:
    # «جهاز تست وتر ستيشن بيقيس إيه؟» — one device, no measurement named. It is a
    # `current` turn with sensor "all", never a `station` one, so it reads the
    # plant snapshot and never asks the devices API for a count.
    agent, model = build_agent(
        [
            interpretation(
                intent="current", sensor="all", device_name="تست وتر ستيشن"
            ),
            "الجهاز بيقيس التدفق والضغط ومستوى الماية.",
        ]
    )

    reply = await run(agent)

    assert reply == "الجهاز بيقيس التدفق والضغط ومستوى الماية."
    assert services.current_calls == 1
    assert services.devices_calls == 0
    assert last_input(model)["result"] == {"data": SNAPSHOT}


async def test_follow_up_about_the_same_device_reads_it_again(
    services: RecordedServices,
) -> None:
    # «الجهاز بيقيس إيه تاني؟» — the device comes from the conversation, and the
    # composer gets both the fresh snapshot and what was already said.
    history: list[MemoryMessage] = [
        {"role": "user", "content": "تست وتر ستيشن بيقيس إيه؟"},
        {"role": "assistant", "content": "بيقيس الحموضة والتدفق."},
    ]
    agent, model = build_agent(
        [
            interpretation(
                intent="current", sensor="all", device_name="تست وتر ستيشن"
            ),
            "دول كل اللي الجهاز بيقيسه.",
        ]
    )

    reply = await run(agent, history)

    assert reply == "دول كل اللي الجهاز بيقيسه."
    assert services.current_calls == 1
    composed = last_input(model)
    assert composed["history"] == history
    assert composed["result"] == {"data": SNAPSHOT}


async def test_compose_sees_no_history_on_a_first_turn(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [
            interpretation(intent="current", sensor="الحموضة", device_name="2"),
            "الحموضة 7.4.",
        ]
    )

    await run(agent)

    assert "history" not in last_input(model)


async def test_device_follow_up_recovers_the_pending_sensor(
    services: RecordedServices,
) -> None:
    history: list[MemoryMessage] = [
        {"role": "user", "content": "عايز مستوى الماية"},
        {"role": "assistant", "content": "في أنهي جهاز؟"},
    ]
    agent, _ = build_agent(
        [
            interpretation(intent="current", sensor="الحموضة", device_name="جهاز 2"),
            "الحموضة 7.4.",
        ]
    )

    reply = await run(agent, history)

    assert reply == "الحموضة 7.4."
    assert services.current_calls == 1


async def test_not_found_device_never_reaches_readings_api(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [
            historical(sensor="الضغط", device_name="جهاز أحمد"),
            resolution(status="not_found"),
            "الجهاز ده مش موجود في المحطة.",
        ]
    )

    reply = await run(agent)

    assert reply == "الجهاز ده مش موجود في المحطة."
    assert services.devices_calls == 1
    assert services.history_calls == []
    assert calls_of(model) == ["interpret", "resolve", "compose"]
    assert last_input(model)["result"] == {"error": "device_not_found"}


async def test_ambiguous_device_asks_named_clarification(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [
            historical(sensor="الضغط", device_name="جهاز"),
            resolution(
                status="ambiguous",
                candidates=["جهاز 1", "جهاز 2", "invented name"],
            ),
        ]
    )

    reply = await run(agent)

    assert reply == "تقصد جهاز 1 ولا جهاز 2؟"
    assert calls_of(model) == ["interpret", "resolve"]
    assert services.devices_calls == 1
    assert services.history_calls == []


async def test_untrusted_resolver_id_is_treated_as_not_found(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [
            historical(sensor="الضغط", device_name="جهاز 8"),
            resolution(status="matched", device_id="bogus-id", device_name="جهاز 8"),
            "الجهاز ده مش موجود في المحطة.",
        ]
    )

    reply = await run(agent)

    assert reply == "الجهاز ده مش موجود في المحطة."
    assert services.devices_calls == 1
    assert services.history_calls == []
    assert last_input(model)["result"] == {"error": "device_not_found"}


async def test_historical_request_uses_validated_dates(
    services: RecordedServices,
) -> None:
    agent, _ = build_agent(
        [
            interpretation(
                intent="historical",
                sensor="التدفق",
                device_name="جهاز 2",
                from_date="2026-07-01",
                to_date="2026-07-07",
            ),
            resolution(status="matched", device_id=DEVICE_2, device_name="جهاز 2"),
            sensor("Flow"),
            "مفيش قراءات مسجلة للفترة دي.",
        ]
    )

    await run(agent)

    assert services.history_calls == [(DEVICE_2, date(2026, 7, 1), date(2026, 7, 7))]


async def test_all_readings_request_fetches_history(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [
            interpretation(
                intent="historical",
                sensor="all",
                device_name="تست وتر",
                from_date="2026-08-18",
                to_date="2026-08-18",
            ),
            resolution(status="matched", device_id=DEVICE_2, device_name="جهاز 2"),
            "امبارح المتوسطات: حرارة 24.7، pH 7.4، عكارة 3.2.",
        ]
    )

    reply = await run(agent)

    assert reply == "امبارح المتوسطات: حرارة 24.7، pH 7.4، عكارة 3.2."
    assert services.history_calls == [(DEVICE_2, date(2026, 8, 18), date(2026, 8, 18))]
    assert last_input(model)["request"]["sensor"] == "all"


async def test_invalid_period_does_not_call_an_api(services: RecordedServices) -> None:
    agent, _ = build_agent(
        [
            interpretation(
                intent="historical",
                sensor="التدفق",
                device_name="1",
                from_date="امبارح",
                to_date="2026-07-01",
            ),
            resolution(status="matched", device_id=DEVICE_1, device_name="جهاز 1"),
            "قولّي الفترة بالتاريخ بشكل أوضح.",
        ]
    )

    reply = await run(agent)

    assert reply == "قولّي الفترة بالتاريخ بشكل أوضح."
    assert services.devices_calls == 1
    assert services.history_calls == []


async def test_api_failure_becomes_a_composable_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_readings(*_: Any) -> dict[str, Any]:
        raise MBBRAPIError("upstream is down")

    monkeypatch.setattr(nodes_module, "get_current_readings", failing_readings)
    agent, model = build_agent(
        [
            interpretation(intent="current", sensor="الضغط", device_name="جهاز 1"),
            "مش قادر أجيب بيانات الأجهزة دلوقتي، جرّب كمان شوية.",
        ]
    )

    reply = await run(agent)

    assert reply == "مش قادر أجيب بيانات الأجهزة دلوقتي، جرّب كمان شوية."
    assert last_input(model)["result"] == {"error": "api_failure"}


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("   ", FALLBACK_RESPONSE),
        ("<think>reasoning</think>الرد النهائي", "الرد النهائي"),
    ],
)
async def test_composed_reply_is_cleaned(
    services: RecordedServices, content: str, expected: str
) -> None:
    agent, _ = build_agent(
        [
            interpretation(intent="current", sensor="مستوى الماية", device_name="جهاز 1"),
            content,
        ]
    )

    assert await run(agent) == expected


async def test_model_failure_surfaces_as_llm_error() -> None:
    agent, _ = build_agent([RuntimeError("model down")])

    with pytest.raises(LLMError):
        await run(agent)


async def test_jwt_never_enters_model_messages(services: RecordedServices) -> None:
    agent, model = build_agent(
        [
            interpretation(intent="current", sensor="مستوى الماية", device_name="جهاز 1"),
            "الحموضة 7.4.",
        ]
    )

    await run(agent)

    serialized = json.dumps(
        [[message.content for message in messages] for _, messages in model.calls],
        ensure_ascii=False,
    )
    assert "runtime-jwt" not in serialized


async def test_unsupported_sensor_never_reaches_a_value(
    services: RecordedServices,
) -> None:
    # جهاز 2 reports pH and flow; the operator asks for temperature.
    agent, model = build_agent(
        [
            historical(sensor="الحرارة", device_name="جهاز 2"),
            resolution(status="matched", device_id=DEVICE_2, device_name="جهاز 2"),
            sensor(""),
            "جهاز 2 مش بيقيس الحرارة، بيقيس الحموضة والتدفق.",
        ]
    )

    reply = await run(agent)

    assert reply == "جهاز 2 مش بيقيس الحرارة، بيقيس الحموضة والتدفق."
    assert calls_of(model) == ["interpret", "resolve", "sensor", "compose"]
    assert last_input(model)["result"] == {
        "error": "sensor_not_supported",
        "device_name": "جهاز 2",
        "device_sensors": ["درجة الحموضة", "معدل التدفق"],
    }


async def test_untrusted_sensor_type_is_treated_as_unsupported(
    services: RecordedServices,
) -> None:
    # A name the payload does not carry must not survive into the reply.
    agent, model = build_agent(
        [
            historical(sensor="الحرارة", device_name="جهاز 2"),
            resolution(status="matched", device_id=DEVICE_2, device_name="جهاز 2"),
            sensor("temperature"),
            "جهاز 2 مش بيقيس الحرارة.",
        ]
    )

    await run(agent)

    assert last_input(model)["result"]["error"] == "sensor_not_supported"


async def test_historical_sensor_is_narrowed_to_its_own_series(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [
            interpretation(
                intent="historical",
                sensor="الحموضة",
                device_name="جهاز 2",
                from_date="2026-07-01",
                to_date="2026-07-07",
            ),
            resolution(status="matched", device_id=DEVICE_2, device_name="جهاز 2"),
            sensor("PH"),
            "متوسط الحموضة كان 7.4.",
        ]
    )

    await run(agent)

    sensors = last_input(model)["result"]["data"]["sensors"]
    assert [entry["name"] for entry in sensors] == ["PH"]


async def test_reported_sensor_without_daily_values_is_no_readings(
    services: RecordedServices,
) -> None:
    # Flow is reported by the device but holds no averages for the period.
    agent, model = build_agent(
        [
            interpretation(
                intent="historical",
                sensor="التدفق",
                device_name="جهاز 2",
                from_date="2026-07-01",
                to_date="2026-07-07",
            ),
            resolution(status="matched", device_id=DEVICE_2, device_name="جهاز 2"),
            sensor("Flow"),
            "مفيش قراءات في الفترة دي.",
        ]
    )

    await run(agent)

    assert last_input(model)["result"] == {"error": "no_readings"}


async def test_sensor_resolver_sees_only_the_reported_measurements(
    services: RecordedServices,
) -> None:
    agent, model = build_agent(
        [
            historical(sensor="الحموضة", device_name="جهاز 2"),
            resolution(status="matched", device_id=DEVICE_2, device_name="جهاز 2"),
            sensor("PH"),
            "الحموضة 7.4.",
        ]
    )

    await run(agent)

    resolver_input = input_of(model, "sensor")
    assert resolver_input["user_sensor"] == "الحموضة"
    assert resolver_input["sensors"] == [
        {"type": "PH", "type_ar": "درجة الحموضة", "unit": "pH"},
        {"type": "Flow", "type_ar": "معدل التدفق", "unit": "m³/h"},
    ]
