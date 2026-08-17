from typing import Self

from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, model_validator


class ChatRequest(BaseModel):
    """The multipart body of POST /api/v1/chat, normalised and checked on arrival."""

    conversation_id: str
    jwt: str
    audio: UploadFile | None = None
    text: str | None = None
    voice: str | None = None

    @property
    def speak(self) -> bool:
        """Voice in, voice out; text in, text out."""
        return self.audio is not None

    @model_validator(mode="after")
    def _check(self) -> Self:
        # An HTTPException raised here travels out of pydantic untouched, so the
        # endpoint answers with the same status codes and messages it always has.
        self.conversation_id = self.conversation_id.strip()
        if not self.conversation_id:
            raise HTTPException(
                status_code=422, detail="conversation_id must not be empty."
            )

        self.jwt = self.jwt.strip()
        if not self.jwt:
            raise HTTPException(status_code=422, detail="jwt must not be empty.")

        # A blank text field alongside audio is treated as "no text", so clients
        # that always send both parts do not have to omit one.
        self.text = (self.text or "").strip() or None
        if self.audio is None and self.text is None:
            raise HTTPException(status_code=422, detail="Provide either audio or text.")
        if self.audio is not None and self.text is not None:
            raise HTTPException(
                status_code=422, detail="Provide either audio or text, not both."
            )

        # Blank means "whichever voice is configured as the default". Which names
        # are valid is only known once the TTS model is loaded, so the endpoint
        # checks that against app state rather than this validator.
        self.voice = (self.voice or "").strip() or None
        return self


class ChatResponse(BaseModel):
    conversation_id: str
    transcript: str
    reply: str
    audio_base64: str | None = None
    audio_content_type: str | None = None
