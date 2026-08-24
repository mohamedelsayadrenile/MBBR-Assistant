from typing import Any

from langchain_openai import ChatOpenAI

from core.config import Settings


class LLMError(Exception):
    """Raised when an agent model call fails."""


def build_llm(settings: Settings) -> ChatOpenAI:
    """Build the shared asynchronous OpenAI-compatible chat model."""
    model = settings.llm_model.removeprefix("openai/")
    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": settings.llm_base_url,
        "api_key": settings.llm_api_key,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "top_p": settings.llm_top_p,
        "max_retries": 0,
    }
    extra_body = _build_extra_body(settings)
    if extra_body:
        kwargs["extra_body"] = extra_body
    return ChatOpenAI(**kwargs)


def _build_extra_body(settings: Settings) -> dict[str, Any]:
    if _is_google_openai_endpoint(settings.llm_base_url):
        return {}

    extra_body: dict[str, Any] = {}
    if settings.llm_top_k is not None:
        extra_body["top_k"] = settings.llm_top_k
    if settings.llm_enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": settings.llm_enable_thinking
        }
    return extra_body


def _is_google_openai_endpoint(base_url: str) -> bool:
    return "generativelanguage.googleapis.com" in base_url.lower()


def strip_thinking(content: str | None) -> str | None:
    """Drop an inline reasoning block before text reaches TTS.

    Some models close the block a second time after the answer
    (`<thought>…</thought>الرد</thought>`); splitting on the last tag would
    hand TTS an empty string, so a trailing tag is dropped first.
    """
    if not content or "</thought>" not in content:
        return content
    trimmed = content.strip().removesuffix("</thought>")
    if "</thought>" not in trimmed:
        return ""
    return trimmed.rsplit("</thought>", maxsplit=1)[-1].strip()
