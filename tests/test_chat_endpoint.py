from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.v1.endpoints.chat import router
from models.schemas.chat import ChatResponse
from services.asr.interface import ASRError


class FakeASR:
    def __init__(self, text: str = "عايز درجة حرارة الماية دلوقتي", should_fail: bool = False):
        self._text = text
        self._should_fail = should_fail

    async def transcribe(self, audio_bytes: bytes) -> str:
        if self._should_fail:
            raise ASRError("failed")
        return self._text


class FakeChatService:
    def __init__(self, audio_base64: str | None = "ZmFrZS13YXY=") -> None:
        self.calls: list[dict[str, object]] = []
        self._audio_base64 = audio_base64

    async def chat(
        self, *, conversation_id: str, jwt: str, transcript: str, speak: bool = True
    ) -> ChatResponse:
        self.calls.append(
            {
                "conversation_id": conversation_id,
                "jwt": jwt,
                "transcript": transcript,
                "speak": speak,
            }
        )
        audio_base64 = self._audio_base64 if speak else None
        return ChatResponse(
            conversation_id=conversation_id,
            transcript=transcript,
            reply="عندك جهاز 1 وجهاز 2، عايز أنهي جهاز؟",
            audio_base64=audio_base64,
            audio_content_type="audio/wav" if audio_base64 else None,
        )


def make_client(
    asr: FakeASR | None = None,
    chat_service: FakeChatService | None = None,
    max_audio_bytes: int = 1024,
) -> tuple[TestClient, FakeChatService]:
    app = FastAPI()
    app.include_router(router)
    service = chat_service or FakeChatService()
    app.state.asr = asr or FakeASR()
    app.state.chat_service = service
    app.state.asr_max_audio_bytes = max_audio_bytes
    return TestClient(app), service


def post_audio(client: TestClient, **overrides: object) -> object:
    data = {"conversation_id": "conversation-1", "jwt": "runtime-jwt"}
    data.update({k: v for k, v in overrides.items() if k != "files"})
    files = overrides.get("files", {"audio": ("voice.wav", b"fake-wav", "audio/wav")})
    return client.post("/api/v1/chat", data=data, files=files)


def post_text(client: TestClient, **overrides: object) -> object:
    data = {
        "conversation_id": "conversation-1",
        "jwt": "runtime-jwt",
        "text": "عايز درجة حرارة الماية دلوقتي",
    }
    data.update(overrides)
    return client.post("/api/v1/chat", data=data)


def test_chat_endpoint_returns_transcript_reply_and_audio() -> None:
    client, service = make_client()

    response = post_audio(client)

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": "conversation-1",
        "transcript": "عايز درجة حرارة الماية دلوقتي",
        "reply": "عندك جهاز 1 وجهاز 2، عايز أنهي جهاز؟",
        "audio_base64": "ZmFrZS13YXY=",
        "audio_content_type": "audio/wav",
    }
    assert service.calls == [
        {
            "conversation_id": "conversation-1",
            "jwt": "runtime-jwt",
            "transcript": "عايز درجة حرارة الماية دلوقتي",
            "speak": True,
        }
    ]


def test_chat_endpoint_answers_text_with_text_only() -> None:
    client, service = make_client()

    response = post_text(client)

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": "conversation-1",
        "transcript": "عايز درجة حرارة الماية دلوقتي",
        "reply": "عندك جهاز 1 وجهاز 2، عايز أنهي جهاز؟",
    }
    assert service.calls == [
        {
            "conversation_id": "conversation-1",
            "jwt": "runtime-jwt",
            "transcript": "عايز درجة حرارة الماية دلوقتي",
            "speak": False,
        }
    ]


def test_chat_endpoint_trims_text_and_skips_the_asr() -> None:
    client, service = make_client(asr=FakeASR(should_fail=True))

    response = post_text(client, text="  عايز أعرف الأجهزة  ")

    assert response.status_code == 200
    assert service.calls[0]["transcript"] == "عايز أعرف الأجهزة"


def test_chat_endpoint_treats_blank_text_alongside_audio_as_voice_only() -> None:
    client, service = make_client()

    response = post_audio(client, text="   ")

    assert response.status_code == 200
    assert service.calls[0]["speak"] is True
    assert service.calls[0]["transcript"] == "عايز درجة حرارة الماية دلوقتي"


def test_chat_endpoint_rejects_a_request_with_neither_audio_nor_text() -> None:
    client, service = make_client()

    response = client.post(
        "/api/v1/chat", data={"conversation_id": "conversation-1", "jwt": "runtime-jwt"}
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Provide either audio or text."
    assert service.calls == []


def test_chat_endpoint_rejects_a_request_with_both_audio_and_text() -> None:
    client, service = make_client()

    response = post_audio(client, text="عايز أعرف الأجهزة")

    assert response.status_code == 422
    assert response.json()["detail"] == "Provide either audio or text, not both."
    assert service.calls == []


def test_chat_endpoint_omits_audio_fields_when_tts_failed() -> None:
    client, _ = make_client(chat_service=FakeChatService(audio_base64=None))

    response = post_audio(client)

    assert response.status_code == 200
    assert "audio_base64" not in response.json()
    assert "audio_content_type" not in response.json()


def test_chat_endpoint_rejects_a_blank_jwt() -> None:
    client, service = make_client()

    response = post_audio(client, jwt="   ")

    assert response.status_code == 422
    assert response.json()["detail"] == "jwt must not be empty."
    assert service.calls == []


def test_chat_endpoint_rejects_a_missing_jwt() -> None:
    client, _ = make_client()

    response = client.post(
        "/api/v1/chat",
        data={"conversation_id": "conversation-1"},
        files={"audio": ("voice.wav", b"fake-wav", "audio/wav")},
    )

    assert response.status_code == 422


def test_chat_endpoint_rejects_a_blank_conversation_id() -> None:
    client, _ = make_client()

    response = post_audio(client, conversation_id="  ")

    assert response.status_code == 422
    assert response.json()["detail"] == "conversation_id must not be empty."


def test_chat_endpoint_rejects_non_wav_audio() -> None:
    client, _ = make_client()

    response = post_audio(client, files={"audio": ("voice.mp3", b"fake", "audio/mpeg")})

    assert response.status_code == 422
    assert response.json()["detail"] == "audio must be a WAV file."


def test_chat_endpoint_rejects_empty_audio() -> None:
    client, _ = make_client()

    response = post_audio(client, files={"audio": ("voice.wav", b"", "audio/wav")})

    assert response.status_code == 422
    assert response.json()["detail"] == "audio must not be empty."


def test_chat_endpoint_rejects_oversized_audio() -> None:
    client, _ = make_client(max_audio_bytes=4)

    response = post_audio(client, files={"audio": ("voice.wav", b"much too large", "audio/wav")})

    assert response.status_code == 413
    assert response.json()["detail"] == "audio is too large."


def test_chat_endpoint_returns_503_when_transcription_fails() -> None:
    client, service = make_client(asr=FakeASR(should_fail=True))

    response = post_audio(client)

    assert response.status_code == 503
    assert response.json()["detail"] == "معلش، مش قادر أسمعك كويس. جرّب تاني."
    assert service.calls == []


def test_chat_endpoint_rejects_silent_audio() -> None:
    client, service = make_client(asr=FakeASR(text="   "))

    response = post_audio(client)

    assert response.status_code == 422
    assert service.calls == []
