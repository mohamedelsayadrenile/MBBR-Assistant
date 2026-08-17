"""Reading what the operator said onto one of the plant's real devices.

The operator speaks, and a transcriber writes it down. Neither is spelling
accurate, so the name that arrives here is often not the name the API holds:
"جهاز ١" for "جهاز 1", "gehaz 1" typed on a latin keyboard, "الجهاز رقم اتنين",
a shortened name, or a plain typo. Comparing those by equality reports a device
the plant obviously has as missing.

The opposite failure is worse: quietly answering about a device the operator did
not ask for. So a match carries a confidence. A confident match is acted on; a
weaker one is handed back for the operator to confirm ("هل تقصد جهاز 1؟"); a name
that resembles nothing in the list is no match at all.

The scoring is deliberately word-based rather than one whole-string distance.
Almost every device here is called "جهاز ...", so a whole-string distance is
dominated by the part that distinguishes nothing -- which is exactly how
"جهاز الطرد المركزي" ends up looking like a typo for "جهاز 1".
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

# A word every device shares still counts for something, but far less than the
# word that tells them apart.
SHARED_TOKEN_WEIGHT = 0.25

# Two words are the same word above this. Below it they are different words.
TOKEN_MATCH = 0.8

# Act on the match without asking.
CONFIDENT_SCORE = 0.85
# ...but only if the runner-up is not this close behind it.
CONFIDENT_MARGIN = 0.08
# Worth putting to the operator as a question. Below this, nothing was named.
PLAUSIBLE_SCORE = 0.45

# How much of a device's name a sentence has to carry to be naming that device,
# and how far clear of the next device it has to be.
MENTION_COVERAGE = 0.85
MENTION_MARGIN = 0.2


@dataclass(frozen=True)
class DeviceMatch:
    """One device from the list, and whether we may act on it unasked."""

    id: str
    name: str
    confident: bool


# Spoken digits and ordinals, as an operator answering "أنهي جهاز؟" says them.
# Written in their normalised form: `_normalise` has already folded ة to ه and
# the hamza forms to ا by the time these are looked up.
_NUMBER_WORDS = {
    "واحد": "1",
    "اول": "1",
    "اتنين": "2",
    "اثنين": "2",
    "تاني": "2",
    "ثاني": "2",
    "تلاته": "3",
    "ثلاثه": "3",
    "تالت": "3",
    "ثالث": "3",
    "اربعه": "4",
    "رابع": "4",
    "خمسه": "5",
    "خامس": "5",
    "سته": "6",
    "سادس": "6",
    "سبعه": "7",
    "سابع": "7",
    "تمانيه": "8",
    "ثمانيه": "8",
    "تامن": "8",
    "ثامن": "8",
    "تسعه": "9",
    "تاسع": "9",
    "عشره": "10",
    "عاشر": "10",
}

# Words that carry no device in them. "لا" and "مش" are here so that correcting a
# confirmation question ("لا، جهاز 2") reads as the device it names.
_FILLERS = frozenset(
    {
        "من",
        "فضلك",
        "لو",
        "سمحت",
        "رقم",
        "نمره",
        "بتاع",
        "بتاعه",
        "ده",
        "دي",
        "يا",
        "لا",
        "مش",
        "please",
        "the",
    }
)

_ARABIC_INDIC_DIGITS = {
    **{0x0660 + digit: str(digit) for digit in range(10)},
    **{0x06F0 + digit: str(digit) for digit in range(10)},
}
_LETTER_FOLDING = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
    }
)
_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
# The space before a device's number is the operator's to leave out, and the
# transcriber's to lose: جهاز١ and جهاز 1 are the same two words.
_DIGIT_BOUNDARY = re.compile(r"(?<=\d)(?=\D)|(?<=\D)(?=\d)")

# A consonant skeleton, which is what survives being said out loud and written
# down in the other alphabet: جهاز and gehaz and jihaz all reduce to "ghz".
# Long vowels go, because they are what the two alphabets disagree about most.
_ARABIC_SKELETON = {
    "ا": "",
    "و": "",
    "ي": "",
    "ع": "",
    "ء": "",
    "ب": "b",
    "ت": "t",
    "ث": "t",
    "ج": "g",
    "ح": "h",
    "خ": "k",
    "د": "d",
    "ذ": "d",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "s",
    "ص": "s",
    "ض": "d",
    "ط": "t",
    "ظ": "z",
    "غ": "g",
    "ف": "f",
    "ق": "k",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
}
_LATIN_SKELETON = {"c": "k", "q": "k", "j": "g", "p": "b", "v": "f", "x": "ks"}
_LATIN_VOWELS = frozenset("aeiouwy")


def _normalise(text: str) -> str:
    """Fold away everything two spellings of the same name may disagree about."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ARABIC_INDIC_DIGITS)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_LETTER_FOLDING)
    text = _PUNCTUATION.sub(" ", text)
    text = _DIGIT_BOUNDARY.sub(" ", text)
    return " ".join(text.casefold().split())


def normalise_words(text: str) -> list[str]:
    """The words of `text`, folded so two spellings of one word compare equal.

    "أيوة" and "ايوه" are the same answer; only the transcriber chose between
    them.
    """
    return _normalise(text).split()


def _strip_article(token: str) -> str:
    """Drop a leading ال, which the operator adds and drops freely.

    Only on a word long enough to survive it, so ال itself and two-letter words
    are left alone.
    """
    if len(token) >= 5 and token.startswith("ال"):
        return token[2:]
    return token


def _tokens(text: str) -> list[str]:
    """The words of `text` that could name a device, in the order they were said."""
    tokens = []
    for word in _normalise(text).split():
        token = _strip_article(word)
        token = _NUMBER_WORDS.get(token, token)
        if token and token not in _FILLERS:
            tokens.append(token)
    return tokens


def _skeleton(token: str) -> str:
    letters: list[str] = []
    for character in token:
        if character.isdigit():
            letters.append(character)
            continue
        if character in _ARABIC_SKELETON:
            mapped = _ARABIC_SKELETON[character]
        elif character in _LATIN_VOWELS:
            mapped = ""
        else:
            mapped = _LATIN_SKELETON.get(character, character)
        if not mapped:
            continue
        # A doubled letter is one sound, and one of the two is usually a typo.
        if letters and letters[-1] == mapped:
            continue
        letters.append(mapped)
    return "".join(letters)


def _similarity(spoken: str, listed: str) -> float:
    """How far one word is from another, 0 to 1, across spellings and alphabets."""
    if spoken == listed:
        return 1.0
    # A number is never a near miss for another number: جهاز 1 and جهاز 2 differ
    # by exactly the character that matters.
    if spoken.isdigit() or listed.isdigit():
        return 0.0

    score = SequenceMatcher(None, spoken, listed).ratio()
    spoken_skeleton, listed_skeleton = _skeleton(spoken), _skeleton(listed)
    if spoken_skeleton and listed_skeleton:
        score = max(score, SequenceMatcher(None, spoken_skeleton, listed_skeleton).ratio())
    return score


def _device_tokens(device: dict[str, Any]) -> list[str]:
    """The words of one device's name, never empty for a device that has a name."""
    name = str(device.get("name", ""))
    tokens = _tokens(name)
    if tokens:
        return tokens
    # A name made entirely of filler words is still that device's name.
    normalised = _normalise(name)
    return [normalised] if normalised else []


def _token_weights(device_tokens: list[list[str]]) -> dict[str, float]:
    """Weight each word by how much it tells the devices apart.

    A word carried by more than one device -- جهاز, tank -- cannot on its own
    select one of them.
    """
    counts = Counter(token for tokens in device_tokens for token in set(tokens))
    return {
        token: SHARED_TOKEN_WEIGHT if count > 1 else 1.0 for token, count in counts.items()
    }


def _score(spoken: list[str], listed: list[str], weights: dict[str, float]) -> float:
    """Weighted overlap of two word lists, 0 to 1.

    Symmetric on purpose: a word the operator said that the device does not have
    counts against the match just as a word of the name they left out does. That
    is what keeps "جهاز أحمد" away from "جهاز 1" while letting "جهاز رقم ١" reach
    it.
    """
    if not listed:
        return 0.0

    unmatched = list(listed)
    matched = 0.0
    spoken_weight = 0.0
    for token in spoken:
        best_similarity, best_index = 0.0, None
        for index, candidate in enumerate(unmatched):
            similarity = _similarity(token, candidate)
            if similarity > best_similarity:
                best_similarity, best_index = similarity, index
        if best_index is None or best_similarity < TOKEN_MATCH:
            # An unrecognised word is fully informative: it is the operator
            # naming something this device is not.
            spoken_weight += 1.0
            continue
        weight = weights.get(unmatched.pop(best_index), 1.0)
        matched += weight * best_similarity
        spoken_weight += weight

    listed_weight = sum(weights.get(token, 1.0) for token in listed)
    return 2 * matched / (spoken_weight + listed_weight)


def _coverage(spoken: list[str], listed: list[str], weights: dict[str, float]) -> float:
    """How much of a device's name is present in what was said, 0 to 1.

    Deliberately one-sided, unlike `_score`. This answers a different question:
    not "which device is this name?" but "is this device named in this sentence?"
    -- and a sentence is mostly words that are not a device name. Penalising
    those, as `_score` rightly does when the whole message is the name, would put
    "الحرارة في جهاز 2 كام؟" below "جهاز 2".
    """
    total = sum(weights.get(token, 1.0) for token in listed)
    if not total:
        return 0.0

    unmatched = list(spoken)
    matched = 0.0
    for token in listed:
        best_similarity, best_index = 0.0, None
        for index, candidate in enumerate(unmatched):
            similarity = _similarity(candidate, token)
            if similarity > best_similarity:
                best_similarity, best_index = similarity, index
        if best_index is not None and best_similarity >= TOKEN_MATCH:
            unmatched.pop(best_index)
            matched += weights.get(token, 1.0) * best_similarity
    return matched / total


def find_device_mention(text: str, devices: list[dict[str, Any]]) -> DeviceMatch | None:
    """The device named somewhere inside `text`, or None if none is.

    Used to carry a device across turns: the operator names it once, in the
    middle of a sentence, and every follow-up is about that device until they
    name another. A sentence that names two devices settles nothing, and comes
    back with `confident` False.
    """
    if not devices:
        return None
    tokens = _tokens(text)
    if not tokens:
        return None

    listed = [_device_tokens(device) for device in devices]
    weights = _token_weights(listed)
    scores = sorted(
        (
            (_coverage(tokens, device_tokens, weights), index)
            for index, device_tokens in enumerate(listed)
        ),
        key=lambda scored: (-scored[0], scored[1]),
    )

    best, best_index = scores[0]
    if best < MENTION_COVERAGE:
        return None
    runner_up = scores[1][0] if len(scores) > 1 else 0.0
    device = devices[best_index]
    return DeviceMatch(
        device["id"], device["name"], confident=best - runner_up >= MENTION_MARGIN
    )


def _position(tokens: list[str], total: int) -> int | None:
    """The list position a bare number refers to, as the operator counts them."""
    for token in tokens:
        if token.isdigit() and 1 <= int(token) <= total:
            return int(token) - 1
    return None


def match_device(spoken: str, devices: list[dict[str, Any]]) -> DeviceMatch | None:
    """Resolve what the operator said onto one device, or None if nothing fits.

    Tolerant of spelling, alphabet, number words, the definite article and filler
    words, and intolerant of everything else. `confident` is False when the name
    is close to one entry but not clearly it, or close to two entries at once --
    the caller asks the operator rather than picking.
    """
    if not devices:
        return None

    exact_id = spoken.strip()
    for device in devices:
        if exact_id == device["id"]:
            return DeviceMatch(device["id"], device["name"], confident=True)

    tokens = _tokens(spoken)
    if not tokens:
        return None

    listed = [_device_tokens(device) for device in devices]
    weights = _token_weights(listed)
    scores = sorted(
        (
            (_score(tokens, device_tokens, weights), index)
            for index, device_tokens in enumerate(listed)
        ),
        key=lambda scored: (-scored[0], scored[1]),
    )

    best, best_index = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0.0
    if best >= PLAUSIBLE_SCORE:
        device = devices[best_index]
        confident = best >= CONFIDENT_SCORE and best - runner_up >= CONFIDENT_MARGIN
        return DeviceMatch(device["id"], device["name"], confident=confident)

    # Nothing in the list was named, but a number in range still points at one
    # ("الجهاز رقم 2" where the names are not numbered at all).
    position = _position(tokens, len(devices))
    if position is not None:
        device = devices[position]
        # A bare number is an unambiguous answer to "أنهي جهاز؟". A number said
        # alongside words that fit no entry is not: the operator was naming
        # something, and counting down the list instead would answer about a
        # device they never mentioned.
        counted = all(token.isdigit() for token in tokens)
        return DeviceMatch(device["id"], device["name"], confident=counted)
    return None
