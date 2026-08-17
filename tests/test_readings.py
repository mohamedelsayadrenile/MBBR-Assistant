import pytest

from services import mbbr_api
from services.mbbr_api import MBBRAPIError
from services.readings import get_current_readings
from tests.http_fake import install_fake_client
from tests.settings_factory import build_settings

DEVICE_ID = "b9eaf606-536b-4f38-a58e-d741cd96155b"

# The real response captured in APIs.ipynb: every device is offline, so the
# live API returns an empty reading set. This is the common path today.
EMPTY_READINGS_PAYLOAD = {
    "success": True,
    "message": "أحدث القراءات",
    "data": {"generated_at": "2026-08-16T09:16:53.011Z", "count": 0, "readings": []},
}


async def test_readings_request_forwards_jwt_as_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = install_fake_client(monkeypatch, mbbr_api, json_payload=EMPTY_READINGS_PAYLOAD)

    await get_current_readings("runtime-jwt", DEVICE_ID, build_settings())

    assert recorded.headers["Authorization"] == "Bearer runtime-jwt"


async def test_readings_passes_selected_device_id_as_query_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = install_fake_client(monkeypatch, mbbr_api, json_payload=EMPTY_READINGS_PAYLOAD)

    await get_current_readings("runtime-jwt", DEVICE_ID, build_settings())

    assert recorded.path == "/api/readings/latest"
    assert recorded.params == {"device_id": DEVICE_ID}


async def test_readings_returns_empty_reading_set_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_client(monkeypatch, mbbr_api, json_payload=EMPTY_READINGS_PAYLOAD)

    readings = await get_current_readings("runtime-jwt", DEVICE_ID, build_settings())

    assert readings == {
        "generated_at": "2026-08-16T09:16:53.011Z",
        "count": 0,
        "readings": [],
    }


async def test_readings_passes_populated_reading_set_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The per-reading shape is unverified upstream, so the service must not
    # reformat it — whatever the sensors send reaches the LLM as-is.
    data = {
        "generated_at": "2026-08-16T09:16:53.011Z",
        "count": 1,
        "readings": [{"sensor": "water_temperature", "value": 24.7, "unit": "C"}],
    }
    install_fake_client(
        monkeypatch, mbbr_api, json_payload={"success": True, "message": "ok", "data": data}
    )

    readings = await get_current_readings("runtime-jwt", DEVICE_ID, build_settings())

    assert readings == data


async def test_readings_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_client(monkeypatch, mbbr_api, json_payload={"detail": "nope"}, status_code=404)

    with pytest.raises(MBBRAPIError):
        await get_current_readings("runtime-jwt", DEVICE_ID, build_settings())
