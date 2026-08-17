"""The voices a caller may ask for, and how each one is spoken.

Two kinds sit behind one name. A built-in is one of the speakers shipped inside
the model repo, selected by name. A custom voice is cloned zero-shot from a clip
in `assets/voices/` and its transcript -- the model needs both, and a transcript
that does not match the clip degrades every reply in that voice.
"""

from dataclasses import dataclass
from pathlib import Path

VOICES_DIR = Path(__file__).resolve().parents[3] / "assets" / "voices"

# name -> (clip, transcript), both relative to VOICES_DIR.
CUSTOM_VOICES: dict[str, tuple[str, str]] = {
    "Elsayad": ("elsayad.wav", "elsayad.txt"),
}


@dataclass(frozen=True)
class Voice:
    """One selectable voice, as the arguments the TTS model needs for it."""

    name: str
    speaker: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None

    def voice_kwargs(self) -> dict[str, str]:
        """The only place the built-in and the cloned path differ."""
        if self.speaker is not None:
            return {"speaker": self.speaker}
        return {"ref_audio": str(self.ref_audio), "ref_text": str(self.ref_text)}


def load_custom_voices() -> dict[str, Voice]:
    """Read every custom voice off disk, or say which file is missing.

    Called while the provider is constructed, so a missing clip fails in
    milliseconds at startup rather than after the model has finished loading.
    """
    voices: dict[str, Voice] = {}
    for name, (audio_file, text_file) in CUSTOM_VOICES.items():
        audio_path = VOICES_DIR / audio_file
        text_path = VOICES_DIR / text_file
        if not audio_path.is_file():
            raise ValueError(f"Voice {name} is missing its reference clip: {audio_path}")
        if not text_path.is_file():
            raise ValueError(f"Voice {name} is missing its transcript: {text_path}")

        ref_text = text_path.read_text(encoding="utf-8").strip()
        if not ref_text:
            raise ValueError(f"Voice {name} has an empty transcript: {text_path}")

        voices[name] = Voice(name=name, ref_audio=str(audio_path), ref_text=ref_text)
    return voices
