import pytest

from agent.device_match import (
    DeviceMatch,
    find_device_mention,
    match_device,
    normalise_words,
)

DEVICE_1 = "b9eaf606-536b-4f38-a58e-d741cd96155b"
DEVICE_2 = "c4ca7915-78e2-46f8-81df-31848d8c1b6c"

# The plant as it really is: every device named "جهاز <n>".
PLANT = [
    {"id": DEVICE_1, "name": "جهاز 1"},
    {"id": DEVICE_2, "name": "جهاز 2"},
]
TANKS = [
    {"id": "t1", "name": "MBBR Tank A"},
    {"id": "t2", "name": "MBBR Tank B"},
]
SOLE_DEVICE = [{"id": DEVICE_1, "name": "جهاز 1"}]


def matched(spoken: str, devices: list[dict[str, str]] = PLANT) -> DeviceMatch:
    match = match_device(spoken, devices)
    assert match is not None, spoken
    return match


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("جهاز 2", DEVICE_2),
        ("جهاز 1", DEVICE_1),
        # Arabic-Indic digits, with and without the space the transcriber loses.
        ("جهاز ٢", DEVICE_2),
        ("جهاز٢", DEVICE_2),
        ("جهاز1", DEVICE_1),
        # Written in latin letters, spelled however the operator hears it.
        ("gehaz 2", DEVICE_2),
        ("jihaz 2", DEVICE_2),
        ("gehaz1", DEVICE_1),
        # Number words and ordinals.
        ("جهاز واحد", DEVICE_1),
        ("الجهاز التاني", DEVICE_2),
        ("الجهاز رقم اتنين", DEVICE_2),
        ("الجهاز الأول", DEVICE_1),
        # The article, the filler words, and the politeness around the name.
        ("الجهاز 2", DEVICE_2),
        ("جهاز رقم 1", DEVICE_1),
        ("جهاز 2 من فضلك", DEVICE_2),
        ("جهاز2 لو سمحت", DEVICE_2),
        # A bare position, as an answer to "أنهي جهاز؟".
        ("2", DEVICE_2),
        ("واحد", DEVICE_1),
        # Letters the two spellings disagree about.
        ("جهاز ١", DEVICE_1),
    ],
)
def test_a_device_the_plant_has_is_matched_however_it_was_said(
    spoken: str, expected: str
) -> None:
    match = matched(spoken)

    assert match.id == expected
    assert match.confident is True


@pytest.mark.parametrize(
    "spoken",
    [
        "جهاز أحمد",
        "جهاز النفخ",
        "المضخة الكبيرة",
        "جهاز الطرد المركزي",
        # A number the plant does not go up to is not a near miss for one it does.
        "جهاز 9",
        "جهاز 12",
        # جهاز is on every device, so on its own it names none of them.
        "جهاز",
        "الجهاز",
        "",
        "   ",
    ],
)
def test_a_device_the_plant_does_not_have_is_no_match(spoken: str) -> None:
    assert match_device(spoken, PLANT) is None


def test_an_invented_id_is_no_match() -> None:
    assert match_device("00000000-dead-4000-8000-000000000000", PLANT) is None


def test_a_real_id_is_matched_outright() -> None:
    match = matched(DEVICE_2)

    assert (match.id, match.confident) == (DEVICE_2, True)


def test_an_empty_device_list_matches_nothing() -> None:
    assert match_device("جهاز 1", []) is None


def test_two_devices_fitting_equally_well_are_not_picked_between() -> None:
    """The operator said what the tanks have in common and nothing else."""
    match = matched("MBBR Tank", TANKS)

    assert match.confident is False
    assert match.name in {"MBBR Tank A", "MBBR Tank B"}


def test_the_distinguishing_word_settles_it() -> None:
    match = matched("tank b", TANKS)

    assert (match.id, match.confident) == ("t2", True)


def test_a_typo_in_the_distinguishing_part_is_still_the_device() -> None:
    match = matched("MBR Tank B", TANKS)

    assert (match.id, match.confident) == ("t2", True)


def test_a_position_said_beside_words_that_fit_nothing_is_only_a_suggestion() -> None:
    """"ديفايس 1" is the operator naming something, not counting down the list."""
    match = matched("ديفايس 1", PLANT)

    assert match.confident is False


def test_a_position_answers_a_list_whose_names_are_not_numbered() -> None:
    match = matched("الجهاز رقم 2", TANKS)

    assert match.id == "t2"


def test_an_unknown_name_is_not_forced_onto_the_only_device() -> None:
    """One device in the list is not a reason to answer about it unasked."""
    match = match_device("جهاز النفخ", SOLE_DEVICE)

    assert match is None or match.confident is False


def test_the_name_is_returned_as_the_api_spells_it() -> None:
    """The confirmation question reads this out, so it cannot be the operator's."""
    assert matched("gehaz 2").name == "جهاز 2"


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("مستوى المياه في جهاز 2 كام؟", DEVICE_2),
        ("عايز درجة حرارة الماية في الجهاز التاني", DEVICE_2),
        ("ممكن قراءات gehaz1 لو سمحت", DEVICE_1),
        ("جهاز 1 بيقيس إيه؟", DEVICE_1),
    ],
)
def test_a_device_named_inside_a_sentence_is_found(sentence: str, expected: str) -> None:
    """The words around the name are not held against it, unlike in a match."""
    mention = find_device_mention(sentence, PLANT)

    assert mention is not None
    assert (mention.id, mention.confident) == (expected, True)


@pytest.mark.parametrize(
    "sentence",
    [
        "والضغط كام؟",
        "الحرارة كام دلوقتي؟",
        "إيه الأجهزة اللي عندي؟",
        "إنت بتعمل إيه؟",
        "جهاز أحمد بيقيس إيه؟",
        # جهاز is on every device, so a sentence with only that names none.
        "الجهاز بيقيس إيه؟",
        "",
    ],
)
def test_a_sentence_that_names_no_device_finds_none(sentence: str) -> None:
    assert find_device_mention(sentence, PLANT) is None


def test_a_sentence_naming_two_devices_settles_nothing() -> None:
    mention = find_device_mention("إيه الفرق بين جهاز 1 و جهاز 2؟", PLANT)

    assert mention is not None
    assert mention.confident is False


def test_a_mention_needs_the_part_of_the_name_that_distinguishes_it() -> None:
    assert find_device_mention("الضغط في MBBR Tank كام؟", TANKS) is None


def test_a_mention_is_found_for_the_device_the_rest_of_the_name_picks() -> None:
    mention = find_device_mention("الضغط في MBBR Tank B كام؟", TANKS)

    assert mention is not None
    assert (mention.id, mention.confident) == ("t2", True)


@pytest.mark.parametrize(
    ("written", "expected"),
    [("أيوة", ["ايوه"]), ("لأ", ["لا"]), ("آه", ["اه"]), ("  نعم  ", ["نعم"])],
)
def test_normalise_words_folds_one_answer_onto_one_spelling(
    written: str, expected: list[str]
) -> None:
    assert normalise_words(written) == expected
