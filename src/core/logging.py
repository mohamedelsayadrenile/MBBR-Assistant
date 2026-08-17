import json
import logging
from typing import Any


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def json_preview(payload: Any, max_chars: int = 2000) -> str:
    """Serialize a payload for a log line, truncated so large API responses stay readable.

    Only ever called on downstream API payloads, never on anything carrying the JWT.
    """
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    if len(serialized) <= max_chars:
        return serialized
    return f"{serialized[:max_chars]}...<truncated>"
