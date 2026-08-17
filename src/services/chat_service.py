import base64
import logging
from time import perf_counter

from redis.exceptions import RedisError

from agent.agent import MBBRAgent
from agent.llm import LLMError
from models.schemas.chat import ChatResponse
from services.mbbr_api import MBBRAPIError
from services.memory import RedisMemory
from services.tts.interface import TTSError, TTSProvider

logger = logging.getLogger(__name__)

TEMPORARY_FAILURE_REPLY = "معلش، حصلت مشكلة مؤقتة. جرّب تاني بعد شوية."


class ChatService:
    """Wires one turn together: memory, agent, memory again, then speech if asked."""

    def __init__(self, memory: RedisMemory, agent: MBBRAgent, tts: TTSProvider) -> None:
        self._memory = memory
        self._agent = agent
        self._tts = tts

    async def chat(
        self, *, conversation_id: str, jwt: str, transcript: str, speak: bool = True
    ) -> ChatResponse:
        started_at = perf_counter()
        logger.info(
            "chat_request_started conversation_id=%s transcript_chars=%s speak=%s",
            conversation_id,
            len(transcript),
            speak,
        )

        try:
            history = await self._memory.load(conversation_id)
            reply = await self._agent.run(
                conversation_id=conversation_id,
                jwt=jwt,
                history=history,
                user_message=transcript,
            )
            await self._memory.append(conversation_id, "user", transcript)
            await self._memory.append(conversation_id, "assistant", reply)
        except (LLMError, MBBRAPIError, RedisError):
            # Infrastructure is down. Apologise in Arabic and still speak it,
            # rather than surfacing an error the operator cannot act on.
            logger.exception("chat_request_failed conversation_id=%s", conversation_id)
            reply = TEMPORARY_FAILURE_REPLY

        response = ChatResponse(
            conversation_id=conversation_id,
            transcript=transcript,
            reply=reply,
        )
        if speak:
            await self._add_voice(response)

        logger.info(
            "chat_request_completed conversation_id=%s reply_chars=%s has_audio=%s latency_ms=%s",
            conversation_id,
            len(reply),
            response.audio_base64 is not None,
            int((perf_counter() - started_at) * 1000),
        )
        return response

    async def _add_voice(self, response: ChatResponse) -> None:
        """Attach spoken audio. A TTS failure degrades to a text-only reply."""
        try:
            wav_bytes = await self._tts.synthesize(response.reply)
        except TTSError:
            logger.warning(
                "voice_response_failed conversation_id=%s", response.conversation_id
            )
            return

        response.audio_base64 = base64.b64encode(wav_bytes).decode("ascii")
        response.audio_content_type = "audio/wav"
