from pydantic import BaseModel


class ChatResponse(BaseModel):
    conversation_id: str
    transcript: str
    reply: str
    audio_base64: str | None = None
    audio_content_type: str | None = None
