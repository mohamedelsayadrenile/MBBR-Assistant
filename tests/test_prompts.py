import pytest

from agent.prompts import SYSTEM_PROMPT


def test_system_prompt_accepts_the_runtime_date() -> None:
    prompt = SYSTEM_PROMPT.format(today="2026-08-17")

    assert "Today is 2026-08-17" in prompt


@pytest.mark.parametrize(
    "rule",
    [
        '"current"',
        '"historical"',
        '"reply"',
        "conversation history",
        "Arabic-Indic",
        "exact id",
        "invent an API",
    ],
)
def test_system_prompt_contains_its_interpretation_rules(rule: str) -> None:
    assert rule in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "outcome",
    [
        "ask_measurement",
        "ask_device",
        "device_not_found",
        "invalid_period",
        "range_too_long",
        "no_readings",
        "api_failure",
    ],
)
def test_system_prompt_defines_every_result_error(outcome: str) -> None:
    assert outcome in SYSTEM_PROMPT


def test_system_prompt_requires_grounded_spoken_arabic() -> None:
    assert "Egyptian Arabic" in SYSTEM_PROMPT
    assert "Use only that input" in SYSTEM_PROMPT
    assert "device ids" in SYSTEM_PROMPT
