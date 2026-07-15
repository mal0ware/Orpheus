"""M2 theory tools: the read-only knowledge oracle, end-to-end through an in-memory
MCP client. Pure functions over the theory data layer — no bridge, no REAPER."""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from orpheus_mcp.tools.theory import register


@pytest.fixture
def theory_client():
    mcp = FastMCP(name="TheoryTest")
    register(mcp)
    return Client(mcp)


# --------------------------------------------------------------------------- #
# get_scale_notes
# --------------------------------------------------------------------------- #


async def test_get_scale_notes_c_major(theory_client):
    async with theory_client as c:
        res = (await c.call_tool("get_scale_notes", {"key": "C", "mode": "major"})).data
    assert res["midi_notes"] == [60, 62, 64, 65, 67, 69, 71]
    assert res["pitch_classes"] == [0, 2, 4, 5, 7, 9, 11]
    assert res["key"] == "C"
    assert res["mode"] == "major"


async def test_get_scale_notes_dorian_works(theory_client):
    async with theory_client as c:
        res = (await c.call_tool("get_scale_notes", {"key": "D", "mode": "dorian"})).data
    assert res["midi_notes"] == [62, 64, 65, 67, 69, 71, 72]


async def test_get_scale_notes_unknown_mode_errors(theory_client):
    async with theory_client as c:
        with pytest.raises(ToolError, match="Unknown mode"):
            await c.call_tool("get_scale_notes", {"key": "C", "mode": "klingon"})


# --------------------------------------------------------------------------- #
# suggest_chord_progression
# --------------------------------------------------------------------------- #


async def test_suggest_progression_c_major_default(theory_client):
    async with theory_client as c:
        res = (await c.call_tool("suggest_chord_progression", {"key": "C"})).data
    assert res["roman_numerals"][0].isupper()  # major progressions start on I
    first = res["chords"][0]
    assert first["roman"] == res["roman_numerals"][0]
    # Every chord tone must be diatonic to C major.
    major_pcs = {0, 2, 4, 5, 7, 9, 11}
    for chord in res["chords"]:
        assert all(p % 12 in major_pcs for p in chord["midi"]), chord


async def test_suggest_progression_minor_uses_minor_numerals(theory_client):
    async with theory_client as c:
        res = (
            await c.call_tool("suggest_chord_progression", {"key": "A", "mode": "minor"})
        ).data
    assert res["roman_numerals"][0] == "i"
    # i in A minor = A-C-E
    assert [p % 12 for p in res["chords"][0]["midi"]] == [9, 0, 4]


async def test_suggest_progression_respects_genre_profile(theory_client):
    from orpheus_mcp.theory.genre_profiles import GENRE_PROFILES

    async with theory_client as c:
        res = (
            await c.call_tool(
                "suggest_chord_progression",
                {"key": "C", "mode": "minor", "genre": "hiphop"},
            )
        ).data
    assert res["progression"] in GENRE_PROFILES["hiphop"]["progressions"]


async def test_suggest_progression_unknown_genre_errors(theory_client):
    async with theory_client as c:
        with pytest.raises(ToolError, match="No profile"):
            await c.call_tool(
                "suggest_chord_progression", {"key": "C", "genre": "polka-core"}
            )


# --------------------------------------------------------------------------- #
# constrain_to_key
# --------------------------------------------------------------------------- #


async def test_constrain_to_key_snaps_out_of_key_notes(theory_client):
    async with theory_client as c:
        res = (
            await c.call_tool(
                "constrain_to_key", {"notes": [60, 61, 62], "key": "C", "mode": "major"}
            )
        ).data
    # 61 (C#) is out of key; the tie between C (60) and D (62) resolves DOWN.
    assert res["notes"] == [60, 60, 62]
    assert res["changed"] == 1


async def test_constrain_to_key_leaves_diatonic_notes_alone(theory_client):
    async with theory_client as c:
        res = (
            await c.call_tool(
                "constrain_to_key",
                {"notes": [69, 71, 72], "key": "A", "mode": "minor"},
            )
        ).data
    assert res["notes"] == [69, 71, 72]
    assert res["changed"] == 0


# --------------------------------------------------------------------------- #
# get_genre_profile
# --------------------------------------------------------------------------- #


async def test_get_genre_profile_returns_profile_dict(theory_client):
    async with theory_client as c:
        res = (await c.call_tool("get_genre_profile", {"genre": "Hip-Hop"})).data
    assert res["bpm_range"] == [80, 100]
    assert "sub_bass" in res["instruments"]


async def test_get_genre_profile_unknown_errors(theory_client):
    async with theory_client as c:
        with pytest.raises(ToolError, match="No profile"):
            await c.call_tool("get_genre_profile", {"genre": "polka-core"})
