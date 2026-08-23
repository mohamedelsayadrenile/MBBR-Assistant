import pytest

from agent.llm import build_llm, strip_thinking
from tests.settings_factory import build_settings


def test_extra_body_is_omitted_when_vllm_knobs_are_unset() -> None:
    llm = build_llm(build_settings(LLM_TOP_K="", LLM_ENABLE_THINKING=""))

    assert llm.extra_body is None


def test_extra_body_carries_configured_vllm_knobs() -> None:
    llm = build_llm(build_settings(LLM_TOP_K=20, LLM_ENABLE_THINKING=False))

    assert llm.extra_body == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_settings_reach_chat_openai() -> None:
    llm = build_llm(build_settings())

    assert llm.openai_api_base == "http://llm.test/v1"
    assert llm.openai_api_key.get_secret_value() == "test-key"
    assert llm.temperature == 0.2
    assert llm.max_tokens == 1024
    assert llm.top_p == 0.8
    assert llm.max_retries == 0


@pytest.mark.parametrize(
    "model", ["qwen3.6-35b-a3b", "openai/qwen3.6-35b-a3b"]
)
def test_model_id_is_normalized_for_chat_openai(model: str) -> None:
    assert build_llm(build_settings(LLM_MODEL=model)).model_name == "qwen3.6-35b-a3b"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("<thought>reasoning here</thought>الرد النهائي", "الرد النهائي"),
        ("مفيش تفكير", "مفيش تفكير"),
        (None, None),
    ],
)
def test_thinking_block_is_stripped(content: str | None, expected: str | None) -> None:
    assert strip_thinking(content) == expected
