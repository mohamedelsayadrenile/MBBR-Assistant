import pytest

from agent.prompts import COMPOSE_PROMPT, DEVICE_RESOLVER_PROMPT, SYSTEM_PROMPT


def test_system_prompt_accepts_the_runtime_date() -> None:
    prompt = SYSTEM_PROMPT.format(today="2026-08-17")

    assert "Today is 2026-08-17" in prompt


@pytest.mark.parametrize(
    "rule",
    [
        "`current`",
        "`historical`",
        "`reply`",
        "conversational",
        "conversation history",
        "Arabic-Indic",
        "device ids",
        "never",
    ],
)
def test_system_prompt_contains_its_interpretation_rules(rule: str) -> None:
    assert rule in SYSTEM_PROMPT


def test_system_prompt_keeps_reply_for_conversational_only() -> None:
    assert "A reading request never becomes `reply`" in SYSTEM_PROMPT
    assert "السلام عليكم" in SYSTEM_PROMPT


def test_system_prompt_does_not_let_the_model_resolve_devices() -> None:
    assert "Never resolve or match device wording" in SYSTEM_PROMPT
    assert "تقصد أي جهاز؟" not in SYSTEM_PROMPT


def test_resolver_prompt_handles_spoken_wording_variants() -> None:
    assert "typos" in DEVICE_RESOLVER_PROMPT
    assert "partial names" in DEVICE_RESOLVER_PROMPT
    assert "Arabic pronunciations" in DEVICE_RESOLVER_PROMPT
    assert "gehaz 1" in DEVICE_RESOLVER_PROMPT


@pytest.mark.parametrize(
    "status",
    ["matched", "ambiguous", "not_found"],
)
def test_resolver_prompt_defines_every_status(status: str) -> None:
    assert f"`{status}`" in DEVICE_RESOLVER_PROMPT


def test_resolver_prompt_requires_verbatim_list_values() -> None:
    assert "VERBATIM" in DEVICE_RESOLVER_PROMPT
    assert "Never invent" in DEVICE_RESOLVER_PROMPT
    assert "Do NOT guess" in DEVICE_RESOLVER_PROMPT


def test_compose_prompt_defines_every_result_error() -> None:
    for outcome in [
        "device_not_found",
        "invalid_period",
        "range_too_long",
        "no_readings",
        "api_failure",
    ]:
        assert f"`{outcome}`" in COMPOSE_PROMPT


def test_compose_prompt_requires_grounded_spoken_arabic() -> None:
    assert "Egyptian Arabic" in COMPOSE_PROMPT
    assert "Use only what is in the supplied input" in COMPOSE_PROMPT
    assert "device ids" in COMPOSE_PROMPT
