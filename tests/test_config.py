import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from core.config import Settings
from tests.settings_factory import build_settings


class ExampleSettings(Settings):
    """Loads .env.example so a drifted example file fails the suite."""

    model_config = SettingsConfigDict(
        env_file=".env.example", env_file_encoding="utf-8", extra="ignore"
    )


def test_env_example_satisfies_every_required_setting() -> None:
    settings = ExampleSettings()

    assert settings.redis_ttl_seconds == 1800
    assert settings.devices_api_path == "/api/devices"
    assert settings.current_readings_api_path == "/api/readings/latest"


def test_env_example_leaves_the_vllm_only_knobs_unset() -> None:
    settings = ExampleSettings()

    # Blank in .env means "do not send this field" — see the LLM provider.
    assert settings.llm_top_k is None
    assert settings.llm_enable_thinking is None


def test_blank_llm_api_key_is_rejected_naming_the_env_key() -> None:
    # The OpenAI client raises an error that does not name the setting, so catch
    # it here instead of at startup. Pydantic reports the env alias.
    with pytest.raises(ValidationError, match="LLM_API_KEY"):
        build_settings(LLM_API_KEY="")


def test_missing_key_fails_fast_rather_than_defaulting() -> None:
    from tests.settings_factory import BASE_ENV

    env = {key: value for key, value in BASE_ENV.items() if key != "REDIS_URL"}
    with pytest.raises(ValidationError, match="REDIS_URL"):
        Settings(_env_file=None, **env)  # type: ignore[arg-type]
