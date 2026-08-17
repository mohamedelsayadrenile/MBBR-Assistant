import logging
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from core.config import Settings
from services.llm.interface import LLMError

logger = logging.getLogger(__name__)


class OpenAICompatibleLLMProvider:
    """Talks to any OpenAI-compatible Chat Completions endpoint.

    The same class serves the Qwen API in development and self-hosted vLLM in
    production; only base URL, key, and model differ.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self._top_p = settings.llm_top_p
        self._extra_body = self._build_extra_body(settings)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        started_at = perf_counter()
        logger.info("llm_chat_started model=%s messages=%s", self._model, len(messages))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
        }
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            logger.warning("llm_chat_failed model=%s error=%s", self._model, type(exc).__name__)
            raise LLMError("LLM request failed") from exc

        assistant_message = response.choices[0].message
        assistant_message.content = self._strip_thinking(assistant_message.content)
        tool_calls = getattr(assistant_message, "tool_calls", None) or []
        logger.info(
            "llm_chat_completed model=%s latency_ms=%s finish_reason=%s tool_calls=%s response_chars=%s",
            self._model,
            int((perf_counter() - started_at) * 1000),
            response.choices[0].finish_reason,
            [tool_call.function.name for tool_call in tool_calls],
            len(assistant_message.content or ""),
        )
        return assistant_message

    @staticmethod
    def _build_extra_body(settings: Settings) -> dict[str, Any]:
        """Assemble vLLM-only sampling extensions, omitting anything unset.

        A hosted endpoint such as the Qwen API can reject unknown fields with a
        400, so these keys are only sent when explicitly configured.
        """
        extra_body: dict[str, Any] = {}
        if settings.llm_top_k is not None:
            extra_body["top_k"] = settings.llm_top_k
        if settings.llm_enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": settings.llm_enable_thinking
            }
        return extra_body

    @staticmethod
    def _strip_thinking(content: str | None) -> str | None:
        """Drop a reasoning block that a thinking-enabled model leaves in the text."""
        if not content or "</think>" not in content:
            return content
        return content.rsplit("</think>", maxsplit=1)[-1].strip()
