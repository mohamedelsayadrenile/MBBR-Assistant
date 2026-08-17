import asyncio
import logging
import os
import tempfile
from time import perf_counter
from typing import Any

from core.config import Settings
from services.tts.interface import TTSError

logger = logging.getLogger(__name__)


class VoiceTutTTSProvider:
    """Local Egyptian Arabic speech synthesis via mohammedaly22/VoiceTut-TTS."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.tts_model
        self._device = settings.tts_device
        self._dtype = settings.tts_dtype
        self._speaker = settings.tts_speaker
        self._num_step = settings.tts_num_step
        self._guidance_scale = settings.tts_guidance_scale
        self._speed = settings.tts_speed
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def load_model(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._get_model)

    async def synthesize(self, text: str) -> bytes:
        async with self._lock:
            return await asyncio.to_thread(self._synthesize_sync, text)

    def _get_model(self) -> Any:
        if self._model is None:
            logger.info("tts_model_loading model=%s device=%s", self._model_name, self._device)
            from voicetut_tts import VoiceTutTTS

            self._model = VoiceTutTTS.from_pretrained(
                self._model_name,
                device=self._device,
                dtype=self._dtype,
            )
            logger.info("tts_model_loaded model=%s", self._model_name)
        return self._model

    def _synthesize_sync(self, text: str) -> bytes:
        started_at = perf_counter()
        path = ""
        try:
            # The library writes a .wav to a path, so hand it a temp file and read it back.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                path = audio_file.name

            self._get_model().synthesize(
                text,
                speaker=self._speaker,
                num_step=self._num_step,
                guidance_scale=self._guidance_scale,
                speed=self._speed,
                output=path,
            )
            with open(path, "rb") as audio_file:
                wav_bytes = audio_file.read()
        except Exception as exc:
            logger.warning("tts_failed text_chars=%s error=%s", len(text), type(exc).__name__)
            raise TTSError("Speech synthesis failed") from exc
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        logger.info(
            "tts_completed text_chars=%s audio_bytes=%s latency_ms=%s",
            len(text),
            len(wav_bytes),
            int((perf_counter() - started_at) * 1000),
        )
        return wav_bytes
