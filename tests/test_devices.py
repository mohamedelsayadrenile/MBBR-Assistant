import pytest

from services import mbbr_api
from services.devices import get_devices
from services.mbbr_api import MBBRAPIError
from tests.http_fake import install_fake_client
from tests.settings_factory import build_settings

# Trimmed from the real response captured in APIs.ipynb.
DEVICES_PAYLOAD = {
    "success": True,
    "message": "نجاح",
    "data": [
        {
            "id": "b9eaf606-536b-4f38-a58e-d741cd96155b",
            "name": "جهاز 1",
            "created_at": "2026-08-12T06:36:44.844Z",
            "rtu_communication_status": "offline",
            "last_read": None,
            "operational_status": "UNKNOWN",
        },
        {
            "id": "c4ca7915-78e2-46f8-81df-31848d8c1b6c",
            "name": "device 1",
            "created_at": "2026-08-05T11:02:29.898Z",
            "rtu_communication_status": "offline",
            "last_read": None,
            "operational_status": "UNKNOWN",
        },
    ],
}


async def test_devices_request_forwards_jwt_as_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = install_fake_client(monkeypatch, mbbr_api, json_payload=DEVICES_PAYLOAD)

    await get_devices("runtime-jwt", build_settings())

    assert recorded.headers["Authorization"] == "Bearer runtime-jwt"
    assert recorded.path == "/api/devices"
    assert recorded.params == {}


async def test_devices_unwraps_envelope_and_keeps_id_and_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_client(monkeypatch, mbbr_api, json_payload=DEVICES_PAYLOAD)

    devices = await get_devices("runtime-jwt", build_settings())

    assert devices == [
        {"id": "b9eaf606-536b-4f38-a58e-d741cd96155b", "name": "جهاز 1"},
        {"id": "c4ca7915-78e2-46f8-81df-31848d8c1b6c", "name": "device 1"},
    ]


async def test_devices_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_client(monkeypatch, mbbr_api, json_payload={"detail": "nope"}, status_code=500)

    with pytest.raises(MBBRAPIError):
        await get_devices("runtime-jwt", build_settings())


async def test_devices_raises_when_envelope_has_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_client(monkeypatch, mbbr_api, json_payload={"success": False})

    with pytest.raises(MBBRAPIError):
        await get_devices("runtime-jwt", build_settings())
