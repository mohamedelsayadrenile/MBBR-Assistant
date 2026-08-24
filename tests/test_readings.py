import pytest

from services import mbbr_api
from services.mbbr_api import MBBRAPIError
from services.readings import get_current_readings
from tests.http_fake import install_fake_client
from tests.settings_factory import build_settings

SNAPSHOT = [
    {
        "figureId": "11111111-1111-4111-8111-111111111111",
        "figureName": "Test Water Station",
        "sensors": [
            {"sensorId": "flow", "name": "flow_rate", "value": 217.4, "unit": "L/min"},
            {"sensorId": "ph", "name": "ph", "value": None, "unit": "pH"},
        ],
    },
    {
        "figureId": "a1000000-0000-4000-8000-000000000009",
        "figureName": "MBBR Tank A",
        "sensors": [
            {"sensorId": "do", "name": "DO", "value": None, "unit": "mg/L"},
            {"sensorId": "temp", "name": "Temperature", "value": None, "unit": "°C"},
        ],
    },
]
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

    assert recorded.path == "/api/figures/readings/latest/"
    assert recorded.params == {}


async def test_readings_passes_the_snapshot_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The per-sensor shape is unverified upstream, so the service must not
    # reformat it — whatever the plant sends reaches the agent as-is.
    install_fake_client(monkeypatch, mbbr_api, json_payload=PAYLOAD)

    readings = await get_current_readings("runtime-jwt", build_settings())

    assert readings == SNAPSHOT


async def test_readings_accepts_an_empty_figure_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_client(
        monkeypatch, mbbr_api, json_payload={"success": True, "message": "ok", "data": []}
    )

    assert await get_current_readings("runtime-jwt", build_settings()) == []


@pytest.mark.parametrize(
    "data",
    [
        {"generated_at": "2026-08-23T05:04:40.249Z", "count": 0},
        {"figures": []},
    ],
)
async def test_readings_raises_when_data_is_not_a_list(
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
