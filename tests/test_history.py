from datetime import date

import pytest

from services import mbbr_api
from services.history import get_historical_readings
from services.mbbr_api import MBBRAPIError
from tests.http_fake import install_fake_client
from tests.settings_factory import build_settings

DEVICE_ID = "11111111-1111-4111-8111-111111111111"
START = date(2026, 8, 15)
END = date(2026, 8, 17)

# The real response captured in test.ipynb for the seeded telemetry device.
HISTORICAL_PAYLOAD = {
    "success": True,
    "message": "تم جلب المتوسطات اليومية بنجاح",
    "data": {
        "device_name": "Test Water Station",
        "sensors": [
            {
                "name": "flow_rate",
                "name_ar": "معدل التدفق",
                "unit": "L/min",
                "daily": [
                    {"day": "2026-08-15", "avg": 130.2955},
                    {"day": "2026-08-16", "avg": 155.1122},
                ],
            },
            {"name": "ph", "name_ar": "درجة الحموضة", "unit": "pH", "daily": []},
            {"name": "pressure", "name_ar": "الضغط", "unit": "bar", "daily": []},
            {"name": "residual_chlorine", "name_ar": "الكلور المتبقي", "unit": "mg/L", "daily": []},
            {"name": "tds", "name_ar": "المواد الصلبة الذائبة", "unit": "ppm", "daily": []},
            {"name": "temperature", "name_ar": "درجة الحرارة", "unit": "C", "daily": []},
            {"name": "turbidity", "name_ar": "العكارة", "unit": "NTU", "daily": []},
            {"name": "water_level", "name_ar": "مستوى الماء", "unit": "%", "daily": []},
        ],
    },
}


async def test_history_request_forwards_jwt_as_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = install_fake_client(monkeypatch, mbbr_api, json_payload=HISTORICAL_PAYLOAD)

    await get_historical_readings("runtime-jwt", DEVICE_ID, START, END, build_settings())

    assert recorded.headers["Authorization"] == "Bearer runtime-jwt"


async def test_history_passes_device_id_and_dates_as_query_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = install_fake_client(monkeypatch, mbbr_api, json_payload=HISTORICAL_PAYLOAD)

    await get_historical_readings("runtime-jwt", DEVICE_ID, START, END, build_settings())

    assert recorded.path == "/api/telemetry/daily-averages"
    assert recorded.params == {
        "device_id": DEVICE_ID,
        "from": "2026-08-15",
        "to": "2026-08-17",
    }


async def test_history_returns_captured_payload_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_client(monkeypatch, mbbr_api, json_payload=HISTORICAL_PAYLOAD)

    data = await get_historical_readings("runtime-jwt", DEVICE_ID, START, END, build_settings())

    assert data == HISTORICAL_PAYLOAD["data"]


async def test_history_raises_on_a_non_object_data_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_client(
        monkeypatch, mbbr_api, json_payload={"success": True, "data": [1, 2]}
    )

    with pytest.raises(MBBRAPIError):
        await get_historical_readings("runtime-jwt", DEVICE_ID, START, END, build_settings())


async def test_history_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_client(
        monkeypatch, mbbr_api, json_payload={"detail": "نطاق غير صالح"}, status_code=400
    )

    with pytest.raises(MBBRAPIError):
        await get_historical_readings("runtime-jwt", DEVICE_ID, START, END, build_settings())