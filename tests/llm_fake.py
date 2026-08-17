"""A scripted stand-in for the LLM provider, plus the OpenAI message shapes."""

import json
from types import SimpleNamespace
from typing import Any


def tool_call(call_id: str, name: str, **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class FakeAssistantMessage:
    """Mimics the OpenAI SDK assistant message, including model_dump()."""

    def __init__(
        self, content: str | None = None, tool_calls: list[SimpleNamespace] | None = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        if exclude_none:
            payload = {key: value for key, value in payload.items() if value is not None}
        return payload


class ScriptedLLM:
    """Returns queued assistant messages in order and records what it was sent."""

    def __init__(self, responses: list[FakeAssistantMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> FakeAssistantMessage:
        self.calls.append([dict(message) for message in messages])
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._responses.pop(0)
