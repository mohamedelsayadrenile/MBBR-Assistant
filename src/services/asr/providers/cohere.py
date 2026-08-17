import asyncio
import logging
import os
import tempfile
from time import perf_counter
from typing import Any

from core.config import Settings
from services.asr.interface import ASRError

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000

SUPPORTED_DTYPES = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
}


def resolve_torch_dtype(dtype_name: str) -> Any:
    """Map an ASR_DTYPE value onto a torch dtype, importing torch lazily."""
    key = dtype_name.lower().strip()
    if key not in SUPPORTED_DTYPES:
        supported = ", ".join(sorted(set(SUPPORTED_DTYPES)))
        raise ValueError(f"Unsupported ASR_DTYPE: {dtype_name}. Supported values: {supported}")
    import torch

    return getattr(torch, SUPPORTED_DTYPES[key])


class CohereASRProvider:
    """Local Arabic speech-to-text via CohereLabs/cohere-transcribe-arabic-07-2026."""

    def __init__(self, settings: Settings) -> None:
        if not settings.asr_language.strip():
            raise ValueError("ASR_LANGUAGE is required for the cohere provider.")
        # Validate the dtype now so a typo fails at startup, not on the first request.
        if settings.asr_dtype.lower().strip() not in SUPPORTED_DTYPES:
            supported = ", ".join(sorted(set(SUPPORTED_DTYPES)))
            raise ValueError(
                f"Unsupported ASR_DTYPE: {settings.asr_dtype}. Supported values: {supported}"
            )

        self._model_name = settings.asr_model
        self._device = settings.asr_device
        self._dtype = settings.asr_dtype
        self._language = settings.asr_language.strip()
        self._max_new_tokens = settings.asr_max_new_tokens
        self._processor: Any = None
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def load_model(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._get_model)

    async def transcribe(self, audio_bytes: bytes) -> str:
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, audio_bytes)

    def _get_model(self) -> tuple[Any, Any]:
        if self._model is None:
            logger.info("asr_model_loading model=%s device=%s", self._model_name, self._device)
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(self._model_name)
            model = CohereAsrForConditionalGeneration.from_pretrained(
                self._model_name,
                dtype=resolve_torch_dtype(self._dtype),
                device_map=self._device,
            )
            model.eval()
            self._model = model
            logger.info("asr_model_loaded model=%s", self._model_name)
        return self._processor, self._model

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        import torch
        from transformers.audio_utils import load_audio

        started_at = perf_counter()
        path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                audio_file.write(audio_bytes)
                path = audio_file.name

            # load_audio takes a path, resamples to 16 kHz, and downmixes to mono.
            audio = load_audio(path, sampling_rate=SAMPLING_RATE)
            processor, model = self._get_model()
            inputs = processor(
                audio,
                language=self._language,
                sampling_rate=SAMPLING_RATE,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=self._max_new_tokens)
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        except Exception as exc:
            logger.warning("asr_failed audio_bytes=%s error=%s", len(audio_bytes), type(exc).__name__)
            raise ASRError("Audio transcription failed") from exc
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        logger.info(
            "asr_completed audio_bytes=%s text_chars=%s latency_ms=%s",
            len(audio_bytes),
            len(text),
            int((perf_counter() - started_at) * 1000),
        )
        return text
