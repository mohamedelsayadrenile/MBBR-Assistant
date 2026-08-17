from fastapi import HTTPException, Request, UploadFile

from services.asr.interface import ASRError

WAV_CONTENT_TYPES = {"audio/wav", "audio/wave", "audio/x-wav"}


async def transcribe(request: Request, audio: UploadFile) -> str:
    if not is_wav(audio):
        raise HTTPException(status_code=422, detail="audio must be a WAV file.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="audio must not be empty.")
    if len(audio_bytes) > request.app.state.asr_max_audio_bytes:
        raise HTTPException(status_code=413, detail="audio is too large.")

    try:
        transcript = await request.app.state.asr.transcribe(audio_bytes)
    except ASRError as exc:
        raise HTTPException(
            status_code=503, detail="معلش، مش قادر أسمعك كويس. جرّب تاني."
        ) from exc

    if not transcript.strip():
        raise HTTPException(status_code=422, detail="لم يتم التعرف على أي كلام في الصوت.")
    return transcript.strip()


def is_wav(audio: UploadFile) -> bool:
    filename = (audio.filename or "").lower()
    content_type = (audio.content_type or "").lower()
    return filename.endswith(".wav") or content_type in WAV_CONTENT_TYPES
