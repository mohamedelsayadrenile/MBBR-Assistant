import pytest

from agent.prompts import COMPOSE_PROMPT, INTERPRET_PROMPT


def test_interpret_prompt_accepts_the_runtime_date() -> None:
    prompt = INTERPRET_PROMPT.format(today="2026-08-17")

    assert "Today is 2026-08-17" in prompt


@pytest.mark.parametrize(
    "rule",
    [
        'intent "devices"',
        'intent "current"',
        'intent "historical"',
        'intent "reply"',
        "conversation history",
        "Arabic-Indic",
        "one-based position",
        "invent an API id",
    ],
)
def test_interpret_prompt_contains_its_decision_rules(rule: str) -> None:
    assert rule in INTERPRET_PROMPT


@pytest.mark.parametrize(
    "outcome",
    [
        "ask_device",
        "device_not_found",
        "invalid_period",
        "range_too_long",
        "no_readings",
        "api_failure",
    ],
)
def test_compose_prompt_defines_every_graph_outcome(outcome: str) -> None:
    assert outcome in COMPOSE_PROMPT


def test_compose_prompt_requires_grounded_spoken_arabic() -> None:
    assert "Egyptian Arabic" in COMPOSE_PROMPT
    assert "Use only that input" in COMPOSE_PROMPT
    assert "device ids" in COMPOSE_PROMPT
