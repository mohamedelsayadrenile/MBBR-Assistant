from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from redis.asyncio import Redis

from agent.agent import MBBRAgent
from api.v1.endpoints.chat import router as chat_router
from core.config import get_settings
from core.logging import configure_logging
from services.asr.factory import create_asr_provider
from services.chat_service import ChatService
from services.llm.factory import create_llm_provider
from services.memory import RedisMemory
from services.tts.factory import create_tts_provider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    await redis.ping()

    memory = RedisMemory(
        redis=redis,
        ttl_seconds=settings.redis_ttl_seconds,
        max_messages=settings.memory_max_messages,
    )
    llm = create_llm_provider(settings)
    asr = create_asr_provider(settings)
    tts = create_tts_provider(settings)
    # ASR runs on every request, so pay its load cost at startup. TTS loads
    # lazily on first synthesis to keep startup off the critical path.
    await asr.load_model()

    app.state.redis = redis
    app.state.asr = asr
    app.state.asr_max_audio_bytes = settings.asr_max_audio_bytes
    app.state.chat_service = ChatService(
        memory=memory,
        agent=MBBRAgent(llm, settings),
        tts=tts,
    )

    try:
        yield
    finally:
        await redis.aclose()


app = FastAPI(title="MBBR Assistant", lifespan=lifespan)
app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
