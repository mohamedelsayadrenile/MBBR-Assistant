from typing import Any, Protocol


class LLMError(Exception):
    """Raised when the LLM call fails."""


class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Send a chat completion request and return the assistant message."""
