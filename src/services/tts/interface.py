from typing import Protocol


class TTSError(Exception):
    """Raised when speech synthesis fails."""


class TTSProvider(Protocol):
    async def load_model(self) -> None:
        """Load the underlying TTS model."""

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text into WAV audio bytes."""
