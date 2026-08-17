"""The voice registry, exercised without loading a model.

`VoiceTutTTSProvider` imports torch and voicetut_tts inside `_get_model`, so the
provider itself can be built and driven here with the model stubbed out.
"""

from pathlib import Path
from typing import Any

import pytest

from services.tts import voices as voices_module
from services.tts.interface import TTSError
from services.tts.providers.voicetut import VoiceTutTTSProvider
from tests.settings_factory import build_settings


class FakeSpeaker:
    """One entry of the model's built-in speaker registry."""

    def __init__(self, name: str) -> None:
        self.speaker_name = name
        self.audio_path = f"/models/{name}.wav"
        self.reference_text = f"{name} reference"


class FakeVoiceTutModel:
    def __init__(self, speakers: list[str] | None = None) -> None:
        self._speakers = [FakeSpeaker(name) for name in speakers or ["Asmaa", "Mohamed"]]
        self.calls: list[dict[str, Any]] = []

    def list_speakers(self) -> list[FakeSpeaker]:
        return self._speakers

    def synthesize(self, text: str, **kwargs: Any) -> None:
        self.calls.append({"text": text, **kwargs})
        Path(kwargs["output"]).write_bytes(b"fake-wav")


@pytest.fixture
def voices_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in assets/voices/ holding one complete custom voice."""
    (tmp_path / "elsayad.wav").write_bytes(b"fake-reference")
    (tmp_path / "elsayad.txt").write_text("أنا محمد الصياد", encoding="utf-8")
    monkeypatch.setattr(voices_module, "VOICES_DIR", tmp_path)
    return tmp_path


def build_provider(
    voices_dir: Path, model: FakeVoiceTutModel | None = None, **overrides: Any
) -> tuple[VoiceTutTTSProvider, FakeVoiceTutModel]:
    provider = VoiceTutTTSProvider(build_settings(**overrides))
    model = model or FakeVoiceTutModel()
    # Stand in for from_pretrained, but keep the registry the real code builds.
    provider._model = model
    provider._voices = provider._build_voices(model)
    return provider, model


def test_a_missing_clip_fails_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Providers are built before the models load, so this is the cheap moment to
    # find out the voice cannot work.
    monkeypatch.setattr(voices_module, "VOICES_DIR", tmp_path)

    with pytest.raises(ValueError, match="elsayad.wav"):
        VoiceTutTTSProvider(build_settings())


def test_a_missing_transcript_fails_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "elsayad.wav").write_bytes(b"fake-reference")
    monkeypatch.setattr(voices_module, "VOICES_DIR", tmp_path)

    with pytest.raises(ValueError, match="elsayad.txt"):
        VoiceTutTTSProvider(build_settings())


def test_a_blank_transcript_fails_at_construction(voices_dir: Path) -> None:
    # Cloning needs the words as well as the clip; an empty file is not a voice.
    (voices_dir / "elsayad.txt").write_text("   ", encoding="utf-8")

    with pytest.raises(ValueError, match="empty transcript"):
        VoiceTutTTSProvider(build_settings())


def test_voices_are_unavailable_until_the_model_loads(voices_dir: Path) -> None:
    provider = VoiceTutTTSProvider(build_settings())

    with pytest.raises(TTSError):
        provider.voices()


def test_the_built_in_speakers_come_from_the_model(voices_dir: Path) -> None:
    provider, _ = build_provider(voices_dir)

    assert provider.voices() == ("Asmaa", "Mohamed", "Elsayad")


async def test_a_built_in_voice_is_spoken_by_name(voices_dir: Path) -> None:
    provider, model = build_provider(voices_dir)

    await provider.synthesize("مرحبا", voice="Mohamed")

    assert model.calls[0]["speaker"] == "Mohamed"
    assert "ref_audio" not in model.calls[0]


async def test_a_cloned_voice_is_spoken_from_its_clip_and_transcript(
    voices_dir: Path,
) -> None:
    provider, model = build_provider(voices_dir)

    await provider.synthesize("مرحبا", voice="Elsayad")

    assert model.calls[0]["ref_audio"] == str(voices_dir / "elsayad.wav")
    assert model.calls[0]["ref_text"] == "أنا محمد الصياد"
    assert "speaker" not in model.calls[0]


async def test_no_voice_falls_back_to_the_configured_default(voices_dir: Path) -> None:
    provider, model = build_provider(voices_dir, TTS_DEFAULT_VOICE="Mohamed")

    await provider.synthesize("مرحبا")

    assert model.calls[0]["speaker"] == "Mohamed"


async def test_a_voice_name_is_matched_ignoring_case(voices_dir: Path) -> None:
    provider, model = build_provider(voices_dir)

    await provider.synthesize("مرحبا", voice="ELSAYAD")

    assert model.calls[0]["ref_text"] == "أنا محمد الصياد"


async def test_an_unknown_voice_never_reaches_the_model(voices_dir: Path) -> None:
    provider, model = build_provider(voices_dir)

    with pytest.raises(TTSError, match="Unknown voice"):
        await provider.synthesize("مرحبا", voice="Bob")

    assert model.calls == []


def test_a_default_voice_nobody_has_fails_while_loading(voices_dir: Path) -> None:
    """Better to refuse to start than to answer every request without audio."""
    provider = VoiceTutTTSProvider(build_settings(TTS_DEFAULT_VOICE="Nobody"))

    with pytest.raises(ValueError, match="TTS_DEFAULT_VOICE=Nobody"):
        provider._build_voices(FakeVoiceTutModel())


def test_a_custom_voice_may_not_shadow_a_built_in(voices_dir: Path) -> None:
    provider = VoiceTutTTSProvider(build_settings())

    with pytest.raises(ValueError, match="collides"):
        provider._build_voices(FakeVoiceTutModel(speakers=["Asmaa", "elsayad"]))
