import logging
from typing import Any

from core.config import Settings
from services.mbbr_api import MBBRAPIError, get_json_data

logger = logging.getLogger(__name__)


async def get_figures(jwt: str, settings: Settings) -> list[dict[str, Any]]:
    """Return the plant's figures as `[{"id": ..., "name": ...}, ...]`.

    Only the two fields the agent needs are kept; any extra API fields would only
    be noise in the LLM context.
    """
    logger.info("figures_fetch_started")
    data = await get_json_data(settings.figures_api_path, jwt, {}, settings)

    if not isinstance(data, dict) or not isinstance(data.get("figures"), list):
        logger.warning("figures_fetch_invalid_shape data_type=%s", type(data).__name__)
        raise MBBRAPIError("Figures API did not return a figure list")

    figures = [
        {"id": str(figure["id"]), "name": str(figure.get("name", "")).strip()}
        for figure in data["figures"]
        if isinstance(figure, dict) and figure.get("id")
    ]
    logger.info("figures_fetch_completed figures=%s", len(figures))
    return figures
