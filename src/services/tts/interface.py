from typing import Protocol


class TTSError(Exception):
    """Raised when speech synthesis fails."""


class TTSProvider(Protocol):
    async def load_model(self) -> None:
        """Load the underlying TTS model."""

    def voices(self) -> tuple[str, ...]:
        """The selectable voice names. Only known once the model is loaded."""

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        """Synthesize text into WAV audio bytes. None means the default voice."""
