from agent.tools import GET_CURRENT_READINGS, GET_DEVICES, OPENAI_TOOLS, resolve_device_id

DEVICES = [
    {"id": "b9eaf606-536b-4f38-a58e-d741cd96155b", "name": "جهاز 1"},
    {"id": "c4ca7915-78e2-46f8-81df-31848d8c1b6c", "name": "device 1"},
    {"id": "a1000000-0000-4000-8000-00000000000d", "name": "MBBR Tank A"},
]


def test_only_two_tools_are_exposed() -> None:
    assert [tool["function"]["name"] for tool in OPENAI_TOOLS] == [
        GET_DEVICES,
        GET_CURRENT_READINGS,
    ]


def test_tool_schemas_do_not_expose_the_jwt() -> None:
    payload = str(OPENAI_TOOLS).lower()

    assert "jwt" not in payload
    assert "authorization" not in payload
    assert "bearer" not in payload


def test_get_devices_takes_no_arguments() -> None:
    devices_tool = next(t for t in OPENAI_TOOLS if t["function"]["name"] == GET_DEVICES)

    assert devices_tool["function"]["parameters"]["properties"] == {}
    assert devices_tool["function"]["parameters"]["required"] == []


def test_get_current_readings_takes_only_device_id() -> None:
    readings_tool = next(
        t for t in OPENAI_TOOLS if t["function"]["name"] == GET_CURRENT_READINGS
    )
    parameters = readings_tool["function"]["parameters"]

    assert list(parameters["properties"]) == ["device_id"]
    assert parameters["required"] == ["device_id"]
    assert parameters["additionalProperties"] is False


def test_resolve_device_id_accepts_a_real_id() -> None:
    assert resolve_device_id("c4ca7915-78e2-46f8-81df-31848d8c1b6c", DEVICES) == (
        "c4ca7915-78e2-46f8-81df-31848d8c1b6c"
    )


def test_resolve_device_id_accepts_the_position_read_out_to_the_user() -> None:
    # The model numbers the list exactly as returned, so "2" is the second entry.
    assert resolve_device_id("2", DEVICES) == "c4ca7915-78e2-46f8-81df-31848d8c1b6c"


def test_resolve_device_id_accepts_a_device_name() -> None:
    assert resolve_device_id("جهاز 1", DEVICES) == "b9eaf606-536b-4f38-a58e-d741cd96155b"


def test_resolve_device_id_ignores_name_casing() -> None:
    assert resolve_device_id("mbbr tank a", DEVICES) == "a1000000-0000-4000-8000-00000000000d"


def test_resolve_device_id_rejects_an_invented_id() -> None:
    assert resolve_device_id("00000000-dead-4000-8000-000000000000", DEVICES) is None


def test_resolve_device_id_rejects_an_out_of_range_position() -> None:
    assert resolve_device_id("9", DEVICES) is None


def test_resolve_device_id_rejects_blank_input() -> None:
    assert resolve_device_id("   ", DEVICES) is None
