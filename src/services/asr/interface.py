from typing import Protocol


class ASRError(Exception):
    """Raised when audio transcription fails."""


class ASRProvider(Protocol):
    async def load_model(self) -> None:
        """Load the underlying ASR model."""

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe WAV audio bytes into text."""
