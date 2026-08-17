from agent.prompts import (
    ASK_WHICH_DEVICE,
    CURRENT_HEADER,
    DEVICE_NOT_FOUND,
    HISTORY_HEADER,
    NO_READINGS,
    OUT_OF_SCOPE,
    SYSTEM_PROMPT,
    TOOL_FAILURE_REPLY,
    build_input,
)
from services.memory import MemoryMessage

FIXED_REPLIES = [
    ASK_WHICH_DEVICE,
    DEVICE_NOT_FOUND,
    NO_READINGS,
    OUT_OF_SCOPE,
    TOOL_FAILURE_REPLY,
]


def test_every_fixed_reply_appears_in_the_prompt_verbatim() -> None:
    """The operator hears these verbatim, so the prompt must carry them verbatim."""
    for reply in FIXED_REPLIES:
        assert reply in SYSTEM_PROMPT


def test_fixed_replies_are_not_wrapped_in_quotes_in_the_prompt() -> None:
    """A quoted sentence gets copied with its quotes and then spoken aloud."""
    for reply in FIXED_REPLIES:
        assert f'"{reply}"' not in SYSTEM_PROMPT


def test_the_fixed_replies_are_all_distinct() -> None:
    assert len(set(FIXED_REPLIES)) == len(FIXED_REPLIES)


def test_an_unmatched_device_has_its_own_reply() -> None:
    # Missing device and unknown device are different situations with different
    # answers; conflating them is what the prompt rule exists to prevent.
    assert DEVICE_NOT_FOUND != ASK_WHICH_DEVICE


def test_build_input_returns_a_first_turn_unchanged() -> None:
    assert build_input([], "الحرارة كام؟") == "الحرارة كام؟"


def test_build_input_labels_each_speaker_and_separates_the_current_turn() -> None:
    history: list[MemoryMessage] = [
        {"role": "user", "content": "عايز درجة حرارة الماية"},
        {"role": "assistant", "content": ASK_WHICH_DEVICE},
    ]

    rendered = build_input(history, "جهاز 2")

    assert rendered == (
        f"{HISTORY_HEADER}\n"
        f"Operator: عايز درجة حرارة الماية\n"
        f"Assistant: {ASK_WHICH_DEVICE}\n\n"
        f"{CURRENT_HEADER}\n"
        f"جهاز 2"
    )
