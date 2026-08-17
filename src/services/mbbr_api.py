"""Shared HTTP plumbing for the two MBBR endpoints.

Both endpoints are GET, both authenticate with `Authorization: Bearer <jwt>`,
and both wrap their payload in a `{success, message, data}` envelope. That is
the whole of what is shared, so it lives here rather than in a client class.
"""

import logging
from time import perf_counter
from typing import Any

import httpx

from core.config import Settings

logger = logging.getLogger(__name__)


class MBBRAPIError(Exception):
    """Raised when an MBBR API call fails or returns an unexpected shape."""


async def get_json_data(path: str, jwt: str, params: dict[str, Any], settings: Settings) -> Any:
    """GET `path` with the JWT attached and return the unwrapped `data` field."""
    started_at = perf_counter()
    # The JWT is built into the header here and never logged or returned.
    headers = {"Authorization": f"Bearer {jwt}"}
    try:
        async with httpx.AsyncClient(
            base_url=settings.mbbr_api_base_url,
            timeout=settings.http_timeout_seconds,
        ) as client:
            response = await client.get(path, params=params, headers=headers)
            elapsed_ms = int((perf_counter() - started_at) * 1000)
            logger.info(
                "mbbr_http_get_completed path=%s status_code=%s latency_ms=%s response_bytes=%s",
                path,
                response.status_code,
                elapsed_ms,
                len(response.content),
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning("mbbr_http_get_failed path=%s error=%s", path, type(exc).__name__)
        raise MBBRAPIError(f"MBBR API request failed: {path}") from exc
    except ValueError as exc:
        logger.warning("mbbr_http_get_invalid_json path=%s", path)
        raise MBBRAPIError(f"MBBR API returned invalid JSON: {path}") from exc

    if not isinstance(payload, dict) or "data" not in payload:
        logger.warning("mbbr_http_get_invalid_envelope path=%s", path)
        raise MBBRAPIError(f"MBBR API returned an unexpected envelope: {path}")

    return payload["data"]
