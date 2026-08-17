from fastapi import APIRouter, Request

from models.schemas.voices import VoicesResponse

router = APIRouter(prefix="/api/v1", tags=["voices"])


@router.get("/voices", response_model=VoicesResponse)
async def voices(request: Request) -> VoicesResponse:
    """What may be sent as the chat endpoint's `voice` field.

    Read from app state rather than hardcoded: the built-in speakers come out of
    the model repo when it loads, so a checkpoint that ships a new one is
    reflected here without a code change.
    """
    return VoicesResponse(
        voices=list(request.app.state.tts_voices),
        default=request.app.state.tts_default_voice,
    )
