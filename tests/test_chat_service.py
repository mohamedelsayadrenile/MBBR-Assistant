from redis.exceptions import RedisError

from agent.agent import MBBRAgent
from agent.llm import LLMError
from services.chat_service import TEMPORARY_FAILURE_REPLY, ChatService
from services.memory import MemoryMessage
from services.tts.interface import TTSError


class FakeMemory:
    def __init__(self, history: list[MemoryMessage] | None = None, fail: bool = False) -> None:
        self.history: list[MemoryMessage] = history or []
        self.appended: list[tuple[str, str, str]] = []
        self._fail = fail

    async def load(self, conversation_id: str) -> list[MemoryMessage]:
        if self._fail:
            raise RedisError("redis is down")
        return self.history

    async def append(self, conversation_id: str, role: str, content: str) -> None:
        self.appended.append((conversation_id, role, content))


class FakeAgent:
    def __init__(self, reply: str = "درجة حرارة الماية 24.7 درجة.", error: Exception | None = None):
        self.reply = reply
        self._error = error
        self.calls: list[dict[str, object]] = []

    async def run(self, *, conversation_id, jwt, history, user_message) -> str:  # noqa: ANN001
        self.calls.append(
            {
                "conversation_id": conversation_id,
                "jwt": jwt,
                "history": history,
                "user_message": user_message,
            }
        )
        if self._error:
            raise self._error
        return self.reply


class FakeTTS:
    def __init__(self, wav_bytes: bytes = b"fake-wav", should_fail: bool = False) -> None:
        self.spoken: list[str] = []
        self._wav_bytes = wav_bytes
        self._should_fail = should_fail

    async def synthesize(self, text: str) -> bytes:
        self.spoken.append(text)
        if self._should_fail:
            raise TTSError("failed")
        return self._wav_bytes


def build_service(
    memory: FakeMemory | None = None,
    agent: FakeAgent | None = None,
    tts: FakeTTS | None = None,
) -> tuple[ChatService, FakeMemory, FakeAgent, FakeTTS]:
    memory = memory or FakeMemory()
    agent = agent or FakeAgent()
    tts = tts or FakeTTS()
    service = ChatService(memory=memory, agent=agent, tts=tts)  # type: ignore[arg-type]
    return service, memory, agent, tts


async def test_turn_persists_user_and_assistant_messages() -> None:
    service, memory, agent, tts = build_service()

    response = await service.chat(
        conversation_id="c1", jwt="runtime-jwt", transcript="عايز درجة حرارة الماية"
    )

    assert response.reply == "درجة حرارة الماية 24.7 درجة."
    assert response.transcript == "عايز درجة حرارة الماية"
    assert memory.appended == [
        ("c1", "user", "عايز درجة حرارة الماية"),
        ("c1", "assistant", "درجة حرارة الماية 24.7 درجة."),
    ]
    assert tts.spoken == ["درجة حرارة الماية 24.7 درجة."]


async def test_prior_turns_are_handed_to_the_agent() -> None:
    history: list[MemoryMessage] = [
        {"role": "user", "content": "عايز درجة حرارة الماية"},
        {"role": "assistant", "content": "عايز أنهي جهاز؟"},
    ]
    service, _, agent, _ = build_service(memory=FakeMemory(history=history))

    await service.chat(conversation_id="c1", jwt="runtime-jwt", transcript="جهاز 2")

    assert agent.calls[0]["history"] == history
    assert agent.calls[0]["user_message"] == "جهاز 2"


async def test_reply_is_returned_as_base64_wav() -> None:
    service, _, _, _ = build_service(tts=FakeTTS(wav_bytes=b"fake-wav"))

    response = await service.chat(conversation_id="c1", jwt="runtime-jwt", transcript="مرحبا")

    assert response.audio_base64 == "ZmFrZS13YXY="
    assert response.audio_content_type == "audio/wav"


async def test_speak_false_skips_the_tts_entirely() -> None:
    service, _, _, tts = build_service(tts=FakeTTS(wav_bytes=b"fake-wav"))

    response = await service.chat(
        conversation_id="c1", jwt="runtime-jwt", transcript="مرحبا", speak=False
    )

    assert response.reply == "درجة حرارة الماية 24.7 درجة."
    assert response.audio_base64 is None
    assert response.audio_content_type is None
    assert tts.spoken == []


async def test_tts_failure_degrades_to_a_text_only_reply() -> None:
    service, _, _, _ = build_service(tts=FakeTTS(should_fail=True))

    response = await service.chat(conversation_id="c1", jwt="runtime-jwt", transcript="مرحبا")

    assert response.reply == "درجة حرارة الماية 24.7 درجة."
    assert response.audio_base64 is None
    assert response.audio_content_type is None


async def test_llm_failure_becomes_a_spoken_arabic_apology() -> None:
    service, _, _, tts = build_service(agent=FakeAgent(error=LLMError("llm is down")))

    response = await service.chat(conversation_id="c1", jwt="runtime-jwt", transcript="مرحبا")

    assert response.reply == TEMPORARY_FAILURE_REPLY
    # The operator still hears the apology rather than getting silence.
    assert tts.spoken == [TEMPORARY_FAILURE_REPLY]
    assert response.audio_base64 is not None


async def test_redis_failure_becomes_a_spoken_arabic_apology() -> None:
    service, _, _, _ = build_service(memory=FakeMemory(fail=True))

    response = await service.chat(conversation_id="c1", jwt="runtime-jwt", transcript="مرحبا")

    assert response.reply == TEMPORARY_FAILURE_REPLY


async def test_service_never_returns_the_jwt() -> None:
    service, _, _, _ = build_service()

    response = await service.chat(
        conversation_id="c1", jwt="runtime-jwt", transcript="عايز درجة حرارة الماية"
    )

    assert "runtime-jwt" not in response.model_dump_json()


def test_agent_signature_matches_what_the_service_calls() -> None:
    # FakeAgent stands in for MBBRAgent; keep the keyword contract in step.
    import inspect

    assert set(inspect.signature(MBBRAgent.run).parameters) == {
        "self",
        "conversation_id",
        "jwt",
        "history",
        "user_message",
    }
