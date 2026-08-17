from functools import lru_cache
from typing import Annotated, Any

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _blank_to_none(value: Any) -> Any:
    """Treat an empty env value as unset so optional keys can stay in .env."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


OptionalInt = Annotated[int | None, BeforeValidator(_blank_to_none)]
OptionalBool = Annotated[bool | None, BeforeValidator(_blank_to_none)]


class Settings(BaseSettings):
    """Application settings.

    `.env` is the single source of truth: every field below is required and has
    no in-code default, so a missing key fails fast at startup instead of
    silently falling back to a value hidden in this file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(alias="APP_NAME")
    app_env: str = Field(alias="APP_ENV")
    log_level: str = Field(alias="LOG_LEVEL")

    llm_provider: str = Field(alias="LLM_PROVIDER")
    llm_base_url: str = Field(alias="LLM_BASE_URL")
    # Must be non-empty: the OpenAI client rejects a blank key at construction
    # with an error that does not name the setting. Self-hosted vLLM ignores the
    # value, so use a placeholder such as EMPTY there.
    llm_api_key: str = Field(alias="LLM_API_KEY", min_length=1)
    llm_model: str = Field(alias="LLM_MODEL")
    llm_temperature: float = Field(alias="LLM_TEMPERATURE", ge=0, le=2)
    llm_max_tokens: int = Field(alias="LLM_MAX_TOKENS", gt=0)
    llm_top_p: float = Field(alias="LLM_TOP_P", gt=0, le=1)
    # vLLM-server extensions. Blank on hosted endpoints that reject unknown fields.
    llm_top_k: OptionalInt = Field(alias="LLM_TOP_K")
    llm_enable_thinking: OptionalBool = Field(alias="LLM_ENABLE_THINKING")

    asr_provider: str = Field(alias="ASR_PROVIDER")
    asr_model: str = Field(alias="ASR_MODEL")
    asr_device: str = Field(alias="ASR_DEVICE")
    asr_dtype: str = Field(alias="ASR_DTYPE")
    asr_language: str = Field(alias="ASR_LANGUAGE")
    asr_max_new_tokens: int = Field(alias="ASR_MAX_NEW_TOKENS", gt=0)
    asr_max_audio_bytes: int = Field(alias="ASR_MAX_AUDIO_BYTES", gt=0)

    tts_provider: str = Field(alias="TTS_PROVIDER")
    tts_model: str = Field(alias="TTS_MODEL")
    tts_device: str = Field(alias="TTS_DEVICE")
    tts_dtype: str = Field(alias="TTS_DTYPE")
    tts_speaker: str = Field(alias="TTS_SPEAKER")
    tts_num_step: int = Field(alias="TTS_NUM_STEP", gt=0)
    tts_guidance_scale: float = Field(alias="TTS_GUIDANCE_SCALE", gt=0)
    tts_speed: float = Field(alias="TTS_SPEED", gt=0)

    redis_url: str = Field(alias="REDIS_URL")
    redis_ttl_seconds: int = Field(alias="REDIS_TTL_SECONDS", gt=0)
    memory_max_messages: int = Field(alias="MEMORY_MAX_MESSAGES", gt=0)

    mbbr_api_base_url: str = Field(alias="MBBR_API_BASE_URL")
    devices_api_path: str = Field(alias="DEVICES_API_PATH")
    current_readings_api_path: str = Field(alias="CURRENT_READINGS_API_PATH")
    http_timeout_seconds: float = Field(alias="HTTP_TIMEOUT_SECONDS", gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
