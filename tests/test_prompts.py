import pytest

from agent.prompts import (
    ACTIVE_DEVICE_HEADER,
    ACTIVE_DEVICE_NOTE,
    ASK_WHICH_DEVICE,
    CURRENT_HEADER,
    DEVICE_NOT_FOUND,
    HISTORY_HEADER,
    NO_READINGS,
    OUT_OF_SCOPE,
    SYSTEM_PROMPT,
    TOOL_FAILURE_REPLY,
    build_input,
    confirm_device_question,
    confirmed_device_name,
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


def test_the_refusal_sentence_is_stated_once_and_in_its_own_section() -> None:
    """Refusing is a last resort, so the prompt may offer it in one place only.

    It used to sit inside the source-of-truth rules as well, which is how a
    perfectly ordinary readings question ("ضغط المياه عامل ايه؟") came back
    refused instead of answered.
    """
    assert SYSTEM_PROMPT.count(OUT_OF_SCOPE) == 1


def test_the_fixed_replies_are_all_distinct() -> None:
    assert len(set(FIXED_REPLIES)) == len(FIXED_REPLIES)


def test_an_unmatched_device_has_its_own_reply() -> None:
    # Missing device and unknown device are different situations with different
    # answers; conflating them is what the prompt rule exists to prevent.
    assert DEVICE_NOT_FOUND != ASK_WHICH_DEVICE


def test_the_confirmation_question_names_one_device_and_asks_nothing_else() -> None:
    assert confirm_device_question("جهاز 1") == "هل تقصد جهاز 1؟"


def test_the_confirmation_question_does_not_repeat_the_word_the_name_has() -> None:
    """The names already start with جهاز; "هل تقصد جهاز جهاز 1؟" is read aloud."""
    assert confirm_device_question("جهاز 1").count("جهاز") == 1


def test_the_confirmation_question_is_shown_to_the_model_in_full() -> None:
    # With a placeholder it would be paraphrased; the model copies what it sees.
    assert confirm_device_question("جهاز 1") in SYSTEM_PROMPT


@pytest.mark.parametrize("name", ["جهاز 1", "MBBR Tank A", "جهاز الطرد المركزي"])
def test_the_device_offered_is_read_back_out_of_the_question(name: str) -> None:
    """Redis holds only the words, so the question is the record of the offer."""
    assert confirmed_device_name(confirm_device_question(name)) == name


@pytest.mark.parametrize(
    "reply",
    [ASK_WHICH_DEVICE, DEVICE_NOT_FOUND, NO_READINGS, "هل تقصد ؟", "درجة الحرارة 24.7."],
)
def test_a_reply_that_offered_no_device_reads_back_as_nothing(reply: str) -> None:
    assert confirmed_device_name(reply) is None


def test_build_input_states_the_active_device_beside_the_message_not_inside_it() -> None:
    """Their words are what the scope rules are judged on, so they stay theirs."""
    history: list[MemoryMessage] = [
        {"role": "user", "content": "مستوى المياه في جهاز 2 كام؟"},
        {"role": "assistant", "content": "مستوى المياه 1.4 متر."},
    ]

    rendered = build_input(history, "والضغط كام؟", active_device="جهاز 2")

    assert f"{ACTIVE_DEVICE_HEADER}\nجهاز 2\n{ACTIVE_DEVICE_NOTE}" in rendered
    assert rendered.rstrip().endswith(f"{CURRENT_HEADER}\nوالضغط كام؟")


def test_build_input_leaves_the_section_out_when_no_device_is_active() -> None:
    history: list[MemoryMessage] = [{"role": "user", "content": "الحرارة كام؟"}]

    assert ACTIVE_DEVICE_HEADER not in build_input(history, "أنهي جهاز؟")


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
