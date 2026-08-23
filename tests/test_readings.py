import pytest

from services import mbbr_api
from services.mbbr_api import MBBRAPIError
from services.readings import get_current_readings
from tests.http_fake import install_fake_client
from tests.settings_factory import build_settings

# The real response captured in test.ipynb, trimmed to two devices: every sensor
# the device is wired for is listed, with a null value when it has no reading.
SNAPSHOT = {
    "generated_at": "2026-08-23T05:04:40.249Z",
    "count": 2,
    "devices": [
        {
            "device_id": "11111111-1111-4111-8111-111111111111",
            "device_name": "Test Water Station",
            "sensors": [
                {"name": "flow_rate", "value": 217.4, "unit": "L/min"},
                {"name": "ph", "value": None, "unit": "pH"},
            ],
        },
        {
            "device_id": "a1000000-0000-4000-8000-000000000009",
            "device_name": "MBBR Tank A",
            "sensors": [
                {"name": "DO", "value": None, "unit": "mg/L"},
                {"name": "Temperature", "value": None, "unit": "°C"},
            ],
        },
    ],
}
PAYLOAD = {"success": True, "message": "أحدث قراءات جميع الأجهزة", "data": SNAPSHOT}


async def test_readings_request_forwards_jwt_as_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = install_fake_client(monkeypatch, mbbr_api, json_payload=PAYLOAD)

    await get_current_readings("runtime-jwt", build_settings())

    assert recorded.headers["Authorization"] == "Bearer runtime-jwt"


async def test_readings_asks_the_all_devices_endpoint_without_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One call answers for the whole plant, so there is no device to select.
    recorded = install_fake_client(monkeypatch, mbbr_api, json_payload=PAYLOAD)

    await get_current_readings("runtime-jwt", build_settings())

    assert recorded.path == "/api/readings/latest/all"
    assert recorded.params == {}


async def test_readings_passes_the_snapshot_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The per-sensor shape is unverified upstream, so the service must not
    # reformat it — whatever the plant sends reaches the agent as-is.
    install_fake_client(monkeypatch, mbbr_api, json_payload=PAYLOAD)

    readings = await get_current_readings("runtime-jwt", build_settings())

    assert readings == SNAPSHOT


async def test_readings_accepts_an_empty_device_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = {"generated_at": "2026-08-23T05:04:40.249Z", "count": 0, "devices": []}
    install_fake_client(
        monkeypatch, mbbr_api, json_payload={"success": True, "message": "ok", "data": data}
    )

    assert await get_current_readings("runtime-jwt", build_settings()) == data


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"generated_at": "2026-08-23T05:04:40.249Z", "count": 0},
        {"devices": "none"},
    ],
)
async def test_readings_raises_without_a_device_list(
    monkeypatch: pytest.MonkeyPatch, data: object
) -> None:
    install_fake_client(
        monkeypatch, mbbr_api, json_payload={"success": True, "message": "ok", "data": data}
    )

    with pytest.raises(MBBRAPIError):
        await get_current_readings("runtime-jwt", build_settings())


async def test_readings_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_client(monkeypatch, mbbr_api, json_payload={"detail": "nope"}, status_code=404)

    with pytest.raises(MBBRAPIError):
        await get_current_readings("runtime-jwt", build_settings())
