"""A scripted stand-in for the CrewAI LLM, plus the native tool-call shape.

CrewAI recognises a response as tool calls when the first element carries a
`function` key, and otherwise treats it as the final answer.
"""

import json
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import PrivateAttr


def tool_call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ScriptedLLM(BaseLLM):
    """Returns queued responses in order and records what it was sent.

    A queued entry is either a list of `tool_call(...)` dicts or a string, which
    CrewAI reads as the final answer. `BaseLLM` is a pydantic model that folds
    undeclared constructor arguments into `additional_params`, so the queue and
    the log have to be private attributes.
    """

    _responses: list[Any] = PrivateAttr(default_factory=list)
    _calls: list[list[dict[str, Any]]] = PrivateAttr(default_factory=list)

    def __init__(self, responses: list[Any], **kwargs: Any) -> None:
        super().__init__(model="scripted-model", **kwargs)
        self._responses = list(responses)
        self._calls = []

    @property
    def calls(self) -> list[list[dict[str, Any]]]:
        """Every message array handed to the model, in order."""
        return self._calls

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(messages, str):
            self._calls.append([{"role": "user", "content": messages}])
        else:
            self._calls.append([dict(message) for message in messages])

        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._responses.pop(0)

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192
