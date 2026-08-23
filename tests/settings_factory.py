"""Build a Settings object from a literal dict.

Init values outrank the .env file, so tests never depend on a local .env.
"""

from typing import Any

from core.config import Settings

BASE_ENV: dict[str, Any] = {
    "APP_NAME": "MBBR Assistant",
    "APP_ENV": "test",
    "LOG_LEVEL": "INFO",
    "LLM_BASE_URL": "http://llm.test/v1",
    "LLM_API_KEY": "test-key",
    "LLM_MODEL": "test-model",
    "LLM_TEMPERATURE": 0.2,
    "LLM_MAX_TOKENS": 1024,
    "LLM_TOP_P": 0.8,
    "LLM_TOP_K": "",
    "LLM_ENABLE_THINKING": "",
    "ASR_PROVIDER": "cohere",
    "ASR_MODEL": "CohereLabs/cohere-transcribe-arabic-07-2026",
    "ASR_DEVICE": "cpu",
    "ASR_DTYPE": "float32",
    "ASR_LANGUAGE": "ar",
    "ASR_MAX_NEW_TOKENS": 256,
    "ASR_MAX_AUDIO_BYTES": 1024,
    "TTS_PROVIDER": "voicetut",
    "TTS_MODEL": "mohammedaly22/VoiceTut-TTS",
    "TTS_DEVICE": "cpu",
    "TTS_DTYPE": "float32",
    "TTS_DEFAULT_VOICE": "Asmaa",
    "TTS_NUM_STEP": 48,
    "TTS_GUIDANCE_SCALE": 2.5,
    "TTS_SPEED": 1.05,
    "REDIS_URL": "redis://localhost:6379/0",
    "REDIS_TTL_SECONDS": 1800,
    "MEMORY_MAX_MESSAGES": 12,
    "MBBR_API_BASE_URL": "http://mbbr.test",
    "DEVICES_API_PATH": "/api/devices",
    "CURRENT_READINGS_API_PATH": "/api/readings/latest/all",
    "HISTORICAL_READINGS_API_PATH": "/api/telemetry/daily-averages",
    "PLANT_TIMEZONE": "Africa/Cairo",
    "HTTP_TIMEOUT_SECONDS": 30,
}


def build_settings(**overrides: Any) -> Settings:
    return Settings(**{**BASE_ENV, **overrides})
