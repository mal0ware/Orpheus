"""Task 11.1 (§8 step 3): bridge-contract + tool tests for the FX apply verbs
(add_fx_by_name / set_fx_param / get_fx_params) and the dB/Hz/ms <-> normalized
translation layer against data/fx_param_maps.json."""
from __future__ import annotations

import math

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, FakeTrack, make_handlers
from orpheus_mcp.bridge.client import BridgeClient
from orpheus_mcp.registry import register_tools
from orpheus_mcp.tools import mix


def _make_track(project, name):
    return FakeTrack(guid=project._next_guid(), name=name)


@pytest.fixture
def project():
    return FakeReaperProject(tempo=120.0)


@pytest.fixture
def client(project, tmp_path):
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield BridgeClient(bridge_dir=tmp_path)


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    mcp = FastMCP(name="OrpheusFxTest")
    register_tools(mcp, profile="default")
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield Client(mcp)


# --------------------------------------------------------------------------- #
# Bridge-contract level (raw normalized 0..1 wire protocol)
# --------------------------------------------------------------------------- #


def test_add_fx_by_name_contract(client, project):
    project.tracks.append(_make_track(project, "bus"))
    res = client.call("add_fx_by_name", track="bus", name="ReaEQ")
    assert res["fx_index"] == 0
    assert res["already_present"] is False


def test_add_fx_by_name_is_idempotent(client, project):
    project.tracks.append(_make_track(project, "bus"))
    first = client.call("add_fx_by_name", track="bus", name="ReaEQ")
    second = client.call("add_fx_by_name", track="bus", name="ReaEQ")
    assert second["already_present"] is True
    assert second["fx_index"] == first["fx_index"]
    tr = project.resolve_track("bus")
    assert len(tr.fx_chain) == 1  # no duplicate instance stacked


def test_set_then_get_fx_param_round_trip_at_wire_level(client, project):
    project.tracks.append(_make_track(project, "bus"))
    added = client.call("add_fx_by_name", track="bus", name="ReaEQ")
    client.call(
        "set_fx_param", track="bus", fx_index=added["fx_index"], param_index=1, value=0.625
    )
    got = client.call("get_fx_params", track="bus", fx_index=added["fx_index"])
    assert got["fx_name"] == "ReaEQ"
    row = next(p for p in got["params"] if p["index"] == 1)
    assert row["value"] == pytest.approx(0.625)


# --------------------------------------------------------------------------- #
# Pure translation functions (no bridge)
# --------------------------------------------------------------------------- #


def test_to_normalized_linear():
    entry = {"min": -24.0, "max": 24.0, "to_normalized": "linear"}
    assert mix.to_normalized(6.0, entry) == pytest.approx(0.625)
    assert mix.to_normalized(-24.0, entry) == pytest.approx(0.0)
    assert mix.to_normalized(24.0, entry) == pytest.approx(1.0)


def test_to_normalized_log_matches_the_documented_formula():
    entry = {"min": 20.0, "max": 20000.0, "to_normalized": "log"}
    value = 3500.0
    expected = math.log(value / entry["min"]) / math.log(entry["max"] / entry["min"])
    assert mix.to_normalized(value, entry) == pytest.approx(expected)


def test_from_normalized_inverts_to_normalized_linear():
    entry = {"min": -24.0, "max": 24.0, "to_normalized": "linear"}
    for value in (-24.0, -6.0, 0.0, 6.0, 24.0):
        norm = mix.to_normalized(value, entry)
        assert mix.from_normalized(norm, entry) == pytest.approx(value)


def test_from_normalized_inverts_to_normalized_log():
    entry = {"min": 20.0, "max": 20000.0, "to_normalized": "log"}
    for value in (20.0, 100.0, 3500.0, 20000.0):
        norm = mix.to_normalized(value, entry)
        assert mix.from_normalized(norm, entry) == pytest.approx(value, rel=1e-6)


def test_to_normalized_clamps_out_of_range_value():
    entry = {"min": -24.0, "max": 24.0, "to_normalized": "linear"}
    assert mix.to_normalized(999.0, entry) == pytest.approx(1.0)
    assert mix.to_normalized(-999.0, entry) == pytest.approx(0.0)


def test_unknown_to_normalized_kind_raises():
    entry = {"min": 0.0, "max": 1.0, "to_normalized": "exponential"}
    with pytest.raises(mix.FxParamMapError):
        mix.to_normalized(0.5, entry)
    with pytest.raises(mix.FxParamMapError):
        mix.from_normalized(0.5, entry)


def test_log_kind_rejects_non_positive_min():
    entry = {"min": 0.0, "max": 100.0, "to_normalized": "log"}
    with pytest.raises(mix.FxParamMapError):
        mix.to_normalized(50.0, entry)


# --------------------------------------------------------------------------- #
# The shipped data/fx_param_maps.json — ReaEQ dB<->normalized against the real map
# --------------------------------------------------------------------------- #


def test_shipped_param_maps_load_and_cover_the_required_plugins():
    maps = mix.load_param_maps()
    for plugin in ("ReaEQ", "ReaComp", "ReaDelay", "ReaVerb", "ReaXcomp"):
        assert plugin in maps, f"missing required plugin: {plugin}"
    for plugin, entry in maps.items():
        if plugin.startswith("_"):
            continue
        for param_name, param_entry in entry["params"].items():
            # Live-verified 2026-07-25 against REAPER 7.78 (the <PIN_BEFORE_SHIP> smoke):
            # every entry must carry the flag as an explicit bool — and having done the
            # verification, shipping an UNverified entry again would be a regression.
            assert param_entry["live_verified"] is True, (
                f"{plugin}.{param_name} lost its live_verified=true status "
                "(indices were read back from a real REAPER 7.78 instance)"
            )


def test_reaeq_gain_db_to_normalized_against_the_shipped_map():
    maps = mix.load_param_maps()
    gain_entry = maps["ReaEQ"]["params"]["gain"]
    assert gain_entry["unit"] == "db"
    assert gain_entry["to_normalized"] == "linear"
    lo, hi = gain_entry["min"], gain_entry["max"]

    normalized = mix.to_normalized(6.0, gain_entry)
    assert normalized == pytest.approx((6.0 - lo) / (hi - lo))
    assert mix.from_normalized(normalized, gain_entry) == pytest.approx(6.0)


def test_reaeq_freq_hz_to_normalized_is_logarithmic():
    maps = mix.load_param_maps()
    freq_entry = maps["ReaEQ"]["params"]["freq"]
    assert freq_entry["unit"] == "hz"
    assert freq_entry["to_normalized"] == "log"
    lo, hi = freq_entry["min"], freq_entry["max"]

    normalized = mix.to_normalized(3500.0, freq_entry)
    assert normalized == pytest.approx(math.log(3500.0 / lo) / math.log(hi / lo))
    assert mix.from_normalized(normalized, freq_entry) == pytest.approx(3500.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Tool-level: mix.py wires the bridge verbs through the translation layer
# --------------------------------------------------------------------------- #


def test_set_fx_param_translates_db_to_normalized_and_writes_the_bridge(client, project):
    project.tracks.append(_make_track(project, "bus"))
    added = mix.add_fx_by_name("bus", "ReaEQ", client=client)

    result = mix.set_fx_param("bus", added["fx_index"], "gain", 6.0, client=client)
    assert result["applied"] is True
    assert result["unit"] == "db"
    maps = mix.load_param_maps()
    gain_entry = maps["ReaEQ"]["params"]["gain"]
    assert result["normalized"] == pytest.approx(mix.to_normalized(6.0, gain_entry))

    tr = project.resolve_track("bus")
    stored = tr.fx_chain[added["fx_index"]].params[gain_entry["index"]]
    assert stored == pytest.approx(mix.to_normalized(6.0, gain_entry))


def test_get_fx_params_reports_native_value_for_mapped_params(client, project):
    project.tracks.append(_make_track(project, "bus"))
    added = mix.add_fx_by_name("bus", "ReaEQ", client=client)
    mix.set_fx_param("bus", added["fx_index"], "gain", 6.0, client=client)

    result = mix.get_fx_params("bus", added["fx_index"], client=client)
    assert result["fx_name"] == "ReaEQ"
    maps = mix.load_param_maps()
    gain_index = maps["ReaEQ"]["params"]["gain"]["index"]
    row = next(p for p in result["params"] if p["index"] == gain_index)
    assert row["mapped_name"] == "gain"
    assert row["unit"] == "db"
    assert row["value_native"] == pytest.approx(6.0)


def test_get_fx_params_leaves_unmapped_params_raw(client, project):
    project.tracks.append(_make_track(project, "bus"))
    added = mix.add_fx_by_name("bus", "SomeThirdPartySynth", client=client)
    result = mix.get_fx_params("bus", added["fx_index"], client=client)
    for row in result["params"]:
        assert "value_native" not in row
        assert "unit" not in row


def test_set_fx_param_on_unmapped_plugin_returns_human_instructions(client, project):
    project.tracks.append(_make_track(project, "bus"))
    added = mix.add_fx_by_name("bus", "SomeThirdPartySynth", client=client)

    result = mix.set_fx_param("bus", added["fx_index"], "cutoff", 800.0, client=client)
    assert result["applied"] is False
    instr = result["human_instructions"]
    assert instr["plugin"] == "SomeThirdPartySynth"
    assert instr["param"] == "cutoff"
    assert instr["requested_value"] == 800.0
    assert instr["reason"] == "unmapped_plugin"
    assert "manually" in instr["message"]

    # No bridge write happened — the fake track's FX carries no params at all.
    tr = project.resolve_track("bus")
    assert tr.fx_chain[added["fx_index"]].params == {}


def test_set_fx_param_on_unmapped_param_of_a_mapped_plugin_returns_human_instructions(
    client, project
):
    project.tracks.append(_make_track(project, "bus"))
    added = mix.add_fx_by_name("bus", "ReaEQ", client=client)

    result = mix.set_fx_param("bus", added["fx_index"], "not_a_real_param", 1.0, client=client)
    assert result["applied"] is False
    instr = result["human_instructions"]
    assert instr["plugin"] == "ReaEQ"
    assert instr["param"] == "not_a_real_param"
    assert instr["reason"] == "unmapped_param"

    tr = project.resolve_track("bus")
    assert tr.fx_chain[added["fx_index"]].params == {}


def test_plugin_name_resolves_via_substring_against_a_vendor_prefixed_report(client, project):
    """A live REAPER-reported fx_name may carry a vendor/category prefix
    (e.g. 'VST: ReaEQ (Cockos)') — the map still resolves via substring, never guessing
    across plugins."""
    project.tracks.append(_make_track(project, "bus"))
    added = mix.add_fx_by_name("bus", "VST: ReaEQ (Cockos)", client=client)
    result = mix.set_fx_param("bus", added["fx_index"], "gain", 3.0, client=client)
    assert result["applied"] is True


# --------------------------------------------------------------------------- #
# MCP tool wiring (default profile — the registry graduation)
# --------------------------------------------------------------------------- #


async def test_mix_tools_are_wired_end_to_end_through_the_mcp_client(mcp_client, project):
    project.tracks.append(_make_track(project, "bus"))
    async with mcp_client as c:
        added = await c.call_tool("add_fx_by_name", {"track": "bus", "name": "ReaEQ"})
        assert added.data["already_present"] is False
        fx_index = added.data["fx_index"]

        set_res = await c.call_tool(
            "set_fx_param",
            {"track": "bus", "fx_index": fx_index, "param": "gain", "value": 6.0},
        )
        assert set_res.data["applied"] is True

        got = await c.call_tool("get_fx_params", {"track": "bus", "fx_index": fx_index})
        assert got.data["fx_name"] == "ReaEQ"
