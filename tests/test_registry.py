"""Registry honesty: the advertised tool surface must only contain tools that work.

Stubs raising NotImplementedError at CALL time used to be advertised live to MCP
clients (the registry's except-NotImplementedError branch never fired, because import
succeeds and only the call raises). Now: explain/default expose implemented tools only;
the full profile may keep stubs but must label them "[NOT IMPLEMENTED]" up front.
"""

from __future__ import annotations

import importlib.util

import pytest
from fastmcp import Client, FastMCP

from orpheus_mcp.registry import PROFILES, register_tools

# The complete implemented surface as of M2. Adding a tool here requires it to actually
# work end-to-end — this list is the honesty contract the profiles are checked against.
IMPLEMENTED = {
    # bridge / project / transport / tracks / midi (M1)
    "get_connection_status",
    "get_project_info",
    "get_project_timeline",
    "parse_chord_chart",
    "parse_riff_tab",
    "preview_lead_sheet",
    "export_yue2_score",
    "arrange_lead_sheet",
    "project_changes",
    "list_tracks",
    "get_track_midi",
    "set_tempo",
    "set_time_signature",
    "play_stop_record",
    "create_track",
    "insert_midi_notes",
    "create_midi_item",
    "transpose_notes",
    # tracks — the opt-in local-audio import verb (drag-and-drop, programmatically)
    "import_audio_files",
    # tracks — artificial-double-tracking primitives (clone a take onto a new track;
    # set/read a track's fader volume + pan)
    "clone_track_items",
    "set_track_params",
    # instruments (pulled forward from Slice 2)
    "list_installed_fx",
    "add_instrument",
    "install_sound_pack",
    "load_sound_pack_instrument",
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
    "create_melody",
    # arrange (Slice 2)
    "add_marker",
    "build_section",
    "arrange_song",
    "place_lyric_markers",
    # render (Task 7.1 — §15.2 bridge verb)
    "render_track_region",
    # reference (Task 1.3 — T1 recon; Task 2.1 — T2 instrument timeline; Task 3.2 — §13
    # chords; Task 4.2 — §5 verification gates 1-3 + separate_region; Task 5.3 — verify_stem;
    # Task 6.1 — §7 SoundCharacter descriptors + characterize; Task 8.2 — §14.2
    # transcribe_lyrics + locate_lyric; Task 12.1 — §9 analyze_reference orchestrator)
    "recon_reference",
    "instrument_timeline",
    "get_chords",
    "separate_region",
    "verify_stem",
    "characterize",
    "compare_character",
    # section-diff — the minute-details differential pass (analysis/sectiondiff.py)
    "compare_sections",
    "transcribe_lyrics",
    "locate_lyric",
    "analyze_reference",
    # Task 12.2 — §15.4 curated free-VST catalog lookup + §9 cache admin
    "suggest_free_vst",
    "clear_analysis_cache",
    # mix (Task 11.1 — §8 step 3 FX apply verbs)
    "add_fx_by_name",
    "set_fx_param",
    "get_fx_params",
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


def test_compose_and_instruments_in_default_profile():
    from fastmcp import FastMCP

    from orpheus_mcp.registry import register_tools

    mcp = FastMCP(name="t")
    cats = register_tools(mcp, profile="default")
    assert "compose" in cats
    assert "instruments" in cats


def test_compose_not_in_explain_profile():
    from fastmcp import FastMCP

    from orpheus_mcp.registry import register_tools

    mcp = FastMCP(name="t")
    cats = register_tools(mcp, profile="explain")
    assert "compose" not in cats


def test_arrange_in_default_not_explain():
    from fastmcp import FastMCP

    from orpheus_mcp.registry import register_tools

    m1 = FastMCP(name="t")
    assert "arrange" in register_tools(m1, profile="default")
    m2 = FastMCP(name="t")
    assert "arrange" not in register_tools(m2, profile="explain")


def test_arrange_in_full_profile():
    from fastmcp import FastMCP

    from orpheus_mcp.registry import register_tools

    mcp = FastMCP(name="t")
    assert "arrange" in register_tools(mcp, profile="full")


def test_reference_in_default_and_full_not_explain():
    from fastmcp import FastMCP

    from orpheus_mcp.registry import register_tools

    m_default = FastMCP(name="t")
    assert "reference" in register_tools(m_default, profile="default")
    m_full = FastMCP(name="t")
    assert "reference" in register_tools(m_full, profile="full")
    m_explain = FastMCP(name="t")
    assert "reference" not in register_tools(m_explain, profile="explain")


def test_mix_in_default_profile():
    from fastmcp import FastMCP

    from orpheus_mcp.registry import register_tools

    mcp = FastMCP(name="t")
    cats = register_tools(mcp, profile="default")
    assert "mix" in cats


async def test_mix_tools_registered_and_implemented():
    names = {t.name for t in await _tool_list("default")}
    assert {"add_fx_by_name", "set_fx_param", "get_fx_params"} <= names


def test_reference_skipped_with_warning_when_librosa_missing(monkeypatch):
    """Registry-honesty probe (review finding): ``tools.reference.register`` must raise
    ImportError when librosa isn't importable, so ``register_tools``'s existing
    ``except ImportError`` skips the whole ``reference`` category with a warning instead
    of registering tools that would raise at call time."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "librosa":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    mcp = FastMCP(name="t")
    with pytest.warns(UserWarning, match="reference"):
        registered = register_tools(mcp, profile="default")

    assert "reference" not in registered
