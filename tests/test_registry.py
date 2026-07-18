"""Registry honesty: the advertised tool surface must only contain tools that work.

Stubs raising NotImplementedError at CALL time used to be advertised live to MCP
clients (the registry's except-NotImplementedError branch never fired, because import
succeeds and only the call raises). Now: explain/default expose implemented tools only;
the full profile may keep stubs but must label them "[NOT IMPLEMENTED]" up front.
"""

from __future__ import annotations

from fastmcp import Client, FastMCP

from orpheus_mcp.registry import PROFILES, register_tools

# The complete implemented surface as of M2. Adding a tool here requires it to actually
# work end-to-end — this list is the honesty contract the profiles are checked against.
IMPLEMENTED = {
    # bridge / project / transport / tracks / midi (M1)
    "get_connection_status",
    "get_project_info",
    "list_tracks",
    "get_track_midi",
    "set_tempo",
    "set_time_signature",
    "play_stop_record",
    "create_track",
    "insert_midi_notes",
    "create_midi_item",
    "transpose_notes",
    # instruments (pulled forward from Slice 2)
    "list_installed_fx",
    "add_instrument",
    # theory (M2)
    "get_scale_notes",
    "suggest_chord_progression",
    "constrain_to_key",
    "get_genre_profile",
    # analyze + style (M2)
    "analyze_harmony",
    "analyze_groove",
    "analyze_audio_character",
    "build_project_spec",
    "list_style_fingerprints",
    "explain_style",
    # compose (M4) — graduated tools, category not yet in `default`
    "create_chord_progression",
    "create_bassline",
    "create_drum_pattern",
    "humanize_pass",
    "compose_section",
}


async def _tool_list(profile: str):
    mcp = FastMCP(name=f"RegistryTest-{profile}")
    register_tools(mcp, profile=profile)
    async with Client(mcp) as client:
        return await client.list_tools()


async def test_default_profile_contains_no_stub():
    names = {t.name for t in await _tool_list("default")}
    assert names <= IMPLEMENTED, f"stubs advertised in default: {sorted(names - IMPLEMENTED)}"
    assert "insert_midi_notes" in names  # sanity: the profile is not accidentally empty


async def test_explain_profile_contains_no_stub_and_is_read_shaped():
    names = {t.name for t in await _tool_list("explain")}
    assert names <= IMPLEMENTED, f"stubs advertised in explain: {sorted(names - IMPLEMENTED)}"
    assert {"analyze_harmony", "explain_style", "build_project_spec"} <= names
    assert "insert_midi_notes" not in names  # midi writes are not part of 'explain'


async def test_full_profile_labels_every_stub():
    tools = await _tool_list("full")
    stubs = [t for t in tools if t.name not in IMPLEMENTED]
    assert stubs, "full profile should still expose the labeled stubs"
    for tool in stubs:
        assert (tool.description or "").startswith("[NOT IMPLEMENTED]"), tool.name


async def test_full_profile_still_superset_of_default():
    full = {t.name for t in await _tool_list("full")}
    default = {t.name for t in await _tool_list("default")}
    assert default <= full


def test_register_tools_reports_plain_categories():
    mcp = FastMCP(name="RegistryReturnTest")
    registered = register_tools(mcp, profile="default")
    assert registered == list(PROFILES["default"])
    assert not any("(stub)" in c for c in registered)  # the dead branch is gone
