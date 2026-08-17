import asyncio
import logging
import os
import tempfile
from time import perf_counter
from typing import Any

from core.config import Settings
from services.tts.interface import TTSError
from services.tts.voices import Voice, load_custom_voices

logger = logging.getLogger(__name__)


class VoiceTutTTSProvider:
    """Local Egyptian Arabic speech synthesis via mohammedaly22/VoiceTut-TTS."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.tts_model
        self._device = settings.tts_device
        self._dtype = settings.tts_dtype
        self._default_voice = settings.tts_default_voice
        self._num_step = settings.tts_num_step
        self._guidance_scale = settings.tts_guidance_scale
        self._speed = settings.tts_speed
        # Read the clips now: a missing one is worth failing on before the model
        # download rather than minutes later.
        self._custom_voices = load_custom_voices()
        self._voices: dict[str, Voice] | None = None
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def load_model(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._get_model)

    def voices(self) -> tuple[str, ...]:
        if self._voices is None:
            raise TTSError("Voices are unavailable until the TTS model is loaded.")
        return tuple(self._voices)

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        # Resolved outside the lock so a bad name fails now instead of queueing
        # behind whatever is being synthesized.
        resolved = self._resolve(voice)
        async with self._lock:
            return await asyncio.to_thread(self._synthesize_sync, text, resolved)

    def _resolve(self, voice: str | None) -> Voice:
        if self._voices is None:
            raise TTSError("Voices are unavailable until the TTS model is loaded.")

        name = voice or self._default_voice
        resolved = self._voices.get(name) or _match_ignoring_case(name, self._voices)
        if resolved is None:
            raise TTSError(f"Unknown voice: {name}")
        return resolved

    def _get_model(self) -> Any:
        if self._model is None:
            logger.info("tts_model_loading model=%s device=%s", self._model_name, self._device)
            from voicetut_tts import VoiceTutTTS

            self._model = VoiceTutTTS.from_pretrained(
                self._model_name,
                device=self._device,
                dtype=self._dtype,
            )
            self._voices = self._build_voices(self._model)
            logger.info(
                "tts_model_loaded model=%s voices=%s", self._model_name, len(self._voices)
            )
        return self._model

    def _build_voices(self, model: Any) -> dict[str, Voice]:
        """The speakers shipped with the model, then the cloned voices on top.

        The built-in list comes from the model repo rather than from a list kept
        here, so it stays right when the checkpoint ships a new speaker.
        """
        voices = {
            speaker.speaker_name: Voice(
                name=speaker.speaker_name, speaker=speaker.speaker_name
            )
            for speaker in model.list_speakers()
        }
        for name, voice in self._custom_voices.items():
            if _match_ignoring_case(name, voices) is not None:
                raise ValueError(f"Voice {name} collides with a built-in speaker.")
            voices[name] = voice

        if _match_ignoring_case(self._default_voice, voices) is None:
            raise ValueError(
                f"TTS_DEFAULT_VOICE={self._default_voice} is not one of: "
                f"{', '.join(voices)}"
            )
        return voices

    def _synthesize_sync(self, text: str, voice: Voice) -> bytes:
        started_at = perf_counter()
        path = ""
        try:
            # The library writes a .wav to a path, so hand it a temp file and read it back.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                path = audio_file.name

            self._get_model().synthesize(
                text,
                **voice.voice_kwargs(),
                num_step=self._num_step,
                guidance_scale=self._guidance_scale,
                speed=self._speed,
                output=path,
            )
            with open(path, "rb") as audio_file:
                wav_bytes = audio_file.read()
        except Exception as exc:
            logger.warning(
                "tts_failed voice=%s text_chars=%s error=%s",
                voice.name,
                len(text),
                type(exc).__name__,
            )
            raise TTSError("Speech synthesis failed") from exc
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        logger.info(
            "tts_completed voice=%s text_chars=%s audio_bytes=%s latency_ms=%s",
            voice.name,
            len(text),
            len(wav_bytes),
            int((perf_counter() - started_at) * 1000),
        )
        return wav_bytes


def _match_ignoring_case(name: str, voices: dict[str, Voice]) -> Voice | None:
    """Voice names are spoken and typed by hand, so casing should not matter."""
    folded = name.casefold()
    for known, voice in voices.items():
        if known.casefold() == folded:
            return voice
    return None
