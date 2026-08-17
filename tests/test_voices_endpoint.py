from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.v1.endpoints.voices import router


def make_client(voices: tuple[str, ...] = ("Asmaa", "Mohamed", "Elsayad")) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.tts_voices = voices
    app.state.tts_default_voice = "Asmaa"
    return TestClient(app)


def test_voices_endpoint_lists_what_chat_accepts() -> None:
    response = make_client().get("/api/v1/voices")

    assert response.status_code == 200
    assert response.json() == {
        "voices": ["Asmaa", "Mohamed", "Elsayad"],
        "default": "Asmaa",
    }
