import json

from agent.tools import (
    GET_CURRENT_READINGS,
    GET_DEVICES,
    ToolContext,
    build_tools,
    resolve_device_id,
)
from tests.settings_factory import build_settings

DEVICES = [
    {"id": "b9eaf606-536b-4f38-a58e-d741cd96155b", "name": "جهاز 1"},
    {"id": "c4ca7915-78e2-46f8-81df-31848d8c1b6c", "name": "device 1"},
    {"id": "a1000000-0000-4000-8000-00000000000d", "name": "MBBR Tank A"},
]


def build_context(jwt: str = "runtime-jwt") -> ToolContext:
    return ToolContext(jwt=jwt, settings=build_settings())


TOOLS = build_tools(build_context())


def test_only_two_tools_are_exposed() -> None:
    assert [tool.name for tool in TOOLS] == [GET_DEVICES, GET_CURRENT_READINGS]


def test_tool_schemas_do_not_expose_the_jwt() -> None:
    payload = json.dumps(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.args_schema.model_json_schema(),
            }
            for tool in TOOLS
        ]
    ).lower()

    assert "jwt" not in payload
    assert "authorization" not in payload
    assert "bearer" not in payload
    assert "runtime-jwt" not in payload


def test_get_devices_takes_no_arguments() -> None:
    devices_tool = next(tool for tool in TOOLS if tool.name == GET_DEVICES)
    schema = devices_tool.args_schema.model_json_schema()

    assert schema.get("properties", {}) == {}
    assert schema.get("required", []) == []


def test_get_current_readings_takes_only_device_id() -> None:
    readings_tool = next(tool for tool in TOOLS if tool.name == GET_CURRENT_READINGS)
    schema = readings_tool.args_schema.model_json_schema()

    assert list(schema["properties"]) == ["device_id"]
    assert schema["required"] == ["device_id"]


def test_each_request_gets_its_own_tool_instances() -> None:
    """A shared instance would carry one request's context into the next."""
    other = build_tools(build_context("other-jwt"))

    assert all(mine is not theirs for mine, theirs in zip(TOOLS, other))
    assert other[0]._context.jwt == "other-jwt"
    assert TOOLS[0]._context.jwt == "runtime-jwt"


def test_a_fresh_context_has_no_device_list_yet() -> None:
    assert build_context().devices is None


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


def test_resolve_device_id_rejects_an_unknown_name() -> None:
    # The operator answering "أنهي جهاز؟" with a device the plant does not have.
    assert resolve_device_id("جهاز أحمد", DEVICES) is None


def test_resolve_device_id_rejects_a_partial_name_match() -> None:
    # "جهاز" alone is a prefix on most names and must never select one.
    assert resolve_device_id("جهاز", DEVICES) is None


def test_resolve_device_id_rejects_an_out_of_range_position() -> None:
    assert resolve_device_id("9", DEVICES) is None


def test_resolve_device_id_rejects_blank_input() -> None:
    assert resolve_device_id("   ", DEVICES) is None
