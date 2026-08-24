import pytest

from services import mbbr_api
from services.figures import get_figures
from services.mbbr_api import MBBRAPIError
from tests.http_fake import install_fake_client
from tests.settings_factory import build_settings

FIGURES = [
    {
        "id": "b9eaf606-536b-4f38-a58e-d741cd96155b",
        "name": "جهاز 1",
        "created_at": "2026-08-12T06:36:44.844Z",
    },
    {
        "id": "c4ca7915-78e2-46f8-81df-31848d8c1b6c",
        "name": " device 1 ",
        "rtu_communication_status": "offline",
    },
]
PAYLOAD = {"success": True, "message": "نجاح", "data": {"figures": FIGURES}}


async def test_figures_request_forwards_jwt_as_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = install_fake_client(monkeypatch, mbbr_api, json_payload=PAYLOAD)

    await get_figures("runtime-jwt", build_settings())

    assert recorded.headers["Authorization"] == "Bearer runtime-jwt"
    assert recorded.path == "/api/telemetry/figures"
    assert recorded.params == {}


async def test_figures_unwraps_envelope_and_keeps_id_and_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_client(monkeypatch, mbbr_api, json_payload=PAYLOAD)

    figures = await get_figures("runtime-jwt", build_settings())

    assert figures == [
        {"id": "b9eaf606-536b-4f38-a58e-d741cd96155b", "name": "جهاز 1"},
        {"id": "c4ca7915-78e2-46f8-81df-31848d8c1b6c", "name": "device 1"},
    ]


async def test_figures_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_client(monkeypatch, mbbr_api, json_payload={"detail": "nope"}, status_code=500)

    with pytest.raises(MBBRAPIError):
        await get_figures("runtime-jwt", build_settings())


@pytest.mark.parametrize("data", [[], {}, {"figures": "none"}])
async def test_figures_raises_without_a_figure_list(
    monkeypatch: pytest.MonkeyPatch, data: object
) -> None:
    install_fake_client(
        monkeypatch, mbbr_api, json_payload={"success": True, "message": "ok", "data": data}
    )

    with pytest.raises(MBBRAPIError):
        await get_figures("runtime-jwt", build_settings())
