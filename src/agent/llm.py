"""The CrewAI LLM handle, configured from the same settings as before.

CrewAI drives the model itself, so the sampling extensions and the reasoning
block stripping that used to live in the OpenAI-compatible provider are kept
here instead. Neither has an equivalent inside CrewAI.
"""

from typing import Any

from crewai import LLM

from core.config import Settings


def build_llm(settings: Settings) -> LLM:
    """Build the shared LLM handle. Construct once and reuse across requests."""
    model = settings.llm_model
    kwargs: dict[str, Any] = {
        # A bare model id is ambiguous to CrewAI's provider inference. The
        # `openai/` prefix pins it to the native OpenAI-compatible client, which
        # is what both the Qwen API and self-hosted vLLM speak.
        "model": model if "/" in model else f"openai/{model}",
        "base_url": settings.llm_base_url,
        "api_key": settings.llm_api_key,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "top_p": settings.llm_top_p,
    }

    extra_body = _build_extra_body(settings)
    if extra_body:
        kwargs["extra_body"] = extra_body

    return LLM(**kwargs)


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


def strip_thinking(content: str | None) -> str | None:
    """Drop a reasoning block that a thinking-enabled model leaves in the text.

    CrewAI returns the completion verbatim, so an inline `<think>` block would
    otherwise be read out loud by the TTS stage.
    """
    if not content or "</think>" not in content:
        return content
    return content.rsplit("</think>", maxsplit=1)[-1].strip()
