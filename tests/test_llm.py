import pytest

from agent.llm import build_llm, strip_thinking
from tests.settings_factory import build_settings


def test_extra_body_is_omitted_when_the_vllm_knobs_are_unset() -> None:
    # The Qwen API can reject unknown fields, so an unset knob must not be sent.
    llm = build_llm(build_settings(LLM_TOP_K="", LLM_ENABLE_THINKING=""))

    assert llm.additional_params.get("extra_body") is None


def test_extra_body_carries_the_vllm_knobs_when_configured() -> None:
    llm = build_llm(build_settings(LLM_TOP_K=20, LLM_ENABLE_THINKING=False))

    assert llm.additional_params["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_settings_reach_the_llm_handle() -> None:
    llm = build_llm(build_settings())

    assert llm.base_url == "http://llm.test/v1"
    assert llm.api_key == "test-key"
    assert llm.temperature == 0.2
    assert llm.max_tokens == 1024
    assert llm.top_p == 0.8


@pytest.mark.parametrize(
    "model", ["qwen3.6-35b-a3b", "openai/qwen3.6-35b-a3b"]
)
def test_a_bare_model_id_is_pinned_to_the_openai_compatible_client(model: str) -> None:
    # CrewAI cannot infer a provider from a bare id, and rejects the model
    # outright unless one is named.
    llm = build_llm(build_settings(LLM_MODEL=model))

    assert llm.provider == "openai"
    assert llm.model == "qwen3.6-35b-a3b"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("<think>reasoning here</think>الرد النهائي", "الرد النهائي"),
        ("مفيش تفكير", "مفيش تفكير"),
        (None, None),
    ],
)
def test_a_thinking_block_is_stripped(content: str | None, expected: str | None) -> None:
    assert strip_thinking(content) == expected
