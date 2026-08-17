from typing import Any

import pytest
from fastapi import FastAPI

import main as main_module
from tests.settings_factory import build_settings


class FakeRedis:
    def __init__(self) -> None:
        self.closed = False

    @classmethod
    def from_url(cls, url: str, decode_responses: bool = False) -> "FakeRedis":
        return cls()

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        self.closed = True


class FakeModelProvider:
    def __init__(self) -> None:
        self.loads = 0

    async def load_model(self) -> None:
        self.loads += 1

    def voices(self) -> tuple[str, ...]:
        return ("Asmaa", "Elsayad")


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeModelProvider]:
    """Stand in for everything the lifespan reaches out to."""
    asr, tts = FakeModelProvider(), FakeModelProvider()

    monkeypatch.setattr(main_module, "Redis", FakeRedis)
    monkeypatch.setattr(main_module, "get_settings", build_settings)
    monkeypatch.setattr(main_module, "create_asr_provider", lambda settings: asr)
    monkeypatch.setattr(main_module, "create_tts_provider", lambda settings: tts)
    monkeypatch.setattr(main_module, "MBBRAgent", lambda settings: object())
    monkeypatch.setattr(
        main_module, "ChatService", lambda **kwargs: kwargs  # type: ignore[arg-type]
    )
    return {"asr": asr, "tts": tts}


async def test_both_models_are_loaded_before_the_app_serves(
    providers: dict[str, FakeModelProvider],
) -> None:
    """No operator should pay a model load, on the first turn or any other."""
    app: Any = FastAPI()

    async with main_module.lifespan(app):
        assert providers["asr"].loads == 1
        assert providers["tts"].loads == 1


async def test_the_voice_list_is_published_for_request_validation(
    providers: dict[str, FakeModelProvider],
) -> None:
    """The endpoint rejects unknown voices against this, so it must be there."""
    app: Any = FastAPI()

    async with main_module.lifespan(app):
        assert app.state.tts_voices == ("Asmaa", "Elsayad")
        assert app.state.tts_default_voice == "Asmaa"
