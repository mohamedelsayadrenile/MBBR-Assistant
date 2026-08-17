import logging
from typing import Annotated

from fastapi import APIRouter, Form, Request

from core.helpers import resolve_voice, transcribe
from models.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(
    request: Request, payload: Annotated[ChatRequest, Form()]
) -> ChatResponse:
    voice = resolve_voice(request, payload.voice)
    transcript = (
        payload.text
        if payload.audio is None
        else await transcribe(request, payload.audio)
    )
    return await request.app.state.chat_service.chat(
        conversation_id=payload.conversation_id,
        jwt=payload.jwt,
        transcript=transcript,
        speak=payload.speak,
        voice=voice,
    )
