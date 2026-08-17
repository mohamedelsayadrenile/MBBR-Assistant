"""A minimal httpx.AsyncClient stand-in that records the request it received."""

from typing import Any

import httpx


class RecordedRequest:
    def __init__(self) -> None:
        self.path: str = ""
        self.params: dict[str, Any] = {}
        self.headers: dict[str, str] = {}
        self.base_url: str = ""


def install_fake_client(
    monkeypatch: Any,
    module: Any,
    *,
    json_payload: Any = None,
    status_code: int = 200,
) -> RecordedRequest:
    """Patch `module.httpx.AsyncClient` and return the recorder."""
    recorded = RecordedRequest()

    class FakeAsyncClient:
        def __init__(self, base_url: str = "", timeout: float = 0) -> None:
            recorded.base_url = base_url

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get(
            self,
            path: str,
            params: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> httpx.Response:
            recorded.path = path
            recorded.params = params or {}
            recorded.headers = headers or {}
            return httpx.Response(
                status_code=status_code,
                json=json_payload,
                request=httpx.Request("GET", f"http://mbbr.test{path}"),
            )

    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)
    return recorded
