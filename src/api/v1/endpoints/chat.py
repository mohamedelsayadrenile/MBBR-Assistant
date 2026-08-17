import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from core.helpers import transcribe
from models.schemas.chat import ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(
    request: Request,
    conversation_id: str = Form(...),
    jwt: str = Form(...),
    audio: UploadFile | None = File(None),
    text: str | None = Form(None),
) -> ChatResponse:
    if not conversation_id.strip():
        raise HTTPException(status_code=422, detail="conversation_id must not be empty.")
    if not jwt.strip():
        raise HTTPException(status_code=422, detail="jwt must not be empty.")

    # A blank text field alongside audio is treated as "no text", so clients that
    # always send both parts do not have to omit one.
    written = (text or "").strip()
    if audio is None and not written:
        raise HTTPException(status_code=422, detail="Provide either audio or text.")
    if audio is not None and written:
        raise HTTPException(
            status_code=422, detail="Provide either audio or text, not both."
        )

    # Voice in, voice out; text in, text out.
    transcript = written if audio is None else await transcribe(request, audio)
    return await request.app.state.chat_service.chat(
        conversation_id=conversation_id.strip(),
        jwt=jwt.strip(),
        transcript=transcript,
        speak=audio is not None,
    )
