from types import SimpleNamespace
from typing import Any

import pytest

from services.llm.factory import create_llm_provider
from services.llm.providers import openai_compatible as llm_module
from tests.settings_factory import build_settings


class FakeCompletions:
    def __init__(self, recorder: dict[str, Any]) -> None:
        self._recorder = recorder

    async def create(self, **kwargs: Any) -> Any:
        self._recorder.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="تمام", tool_calls=None),
                    finish_reason="stop",
                )
            ]
        )


class FakeAsyncOpenAI:
    last_kwargs: dict[str, Any] = {}

    def __init__(self, api_key: str, base_url: str) -> None:
        FakeAsyncOpenAI.last_kwargs = {}
        self.chat = SimpleNamespace(completions=FakeCompletions(FakeAsyncOpenAI.last_kwargs))


@pytest.fixture(autouse=True)
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)


async def test_extra_body_is_omitted_when_vllm_knobs_are_unset() -> None:
    # The Qwen API can reject unknown fields, so an unset knob must not be sent.
    provider = create_llm_provider(build_settings(LLM_TOP_K="", LLM_ENABLE_THINKING=""))

    await provider.chat([{"role": "user", "content": "hi"}])

    assert "extra_body" not in FakeAsyncOpenAI.last_kwargs


async def test_extra_body_carries_vllm_knobs_when_configured() -> None:
    provider = create_llm_provider(build_settings(LLM_TOP_K=20, LLM_ENABLE_THINKING=False))

    await provider.chat([{"role": "user", "content": "hi"}])

    assert FakeAsyncOpenAI.last_kwargs["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def test_tools_are_sent_with_auto_tool_choice() -> None:
    provider = create_llm_provider(build_settings())
    tools = [{"type": "function", "function": {"name": "get_devices"}}]

    await provider.chat([{"role": "user", "content": "hi"}], tools=tools)

    assert FakeAsyncOpenAI.last_kwargs["tools"] == tools
    assert FakeAsyncOpenAI.last_kwargs["tool_choice"] == "auto"


async def test_tool_fields_are_absent_when_no_tools_are_passed() -> None:
    provider = create_llm_provider(build_settings())

    await provider.chat([{"role": "user", "content": "hi"}])

    assert "tools" not in FakeAsyncOpenAI.last_kwargs
    assert "tool_choice" not in FakeAsyncOpenAI.last_kwargs


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_provider(build_settings(LLM_PROVIDER="nope"))


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("<think>reasoning here</think>الرد النهائي", "الرد النهائي"),
        ("مفيش تفكير", "مفيش تفكير"),
        (None, None),
    ],
)
def test_thinking_block_is_stripped_from_the_reply(content: str | None, expected: str | None) -> None:
    assert llm_module.OpenAICompatibleLLMProvider._strip_thinking(content) == expected
