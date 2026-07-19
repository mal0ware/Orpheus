# src/orpheus_mcp/tools/arrange.py
"""Song arrangement: section markers + section/song builders (Slice 2)."""
from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient
from orpheus_mcp.drumkit import load_drumkit
from orpheus_mcp.instruments import select_instrument
from orpheus_mcp.theory.chords import resolve_progression, voice_lead
from orpheus_mcp.theory.melody import parse_melody
from orpheus_mcp.theory.patterns import DRUM_PATTERNS, bassline_notes, parse_drum_grid
from orpheus_mcp.tools.compose import _chord_notes, _write_notes

_DESTRUCTIVE = {"destructiveHint": True}


def _build_section(
    bridge, key: str, mode: str, progression: str, bars: int, at_bar: int,
    drums: str = "backbeat", melody: str | None = None,
    inventory: list[str] | None = None,
    drum_track: str = "drums", chord_track: str = "chords",
    bass_track: str = "bass", lead_track: str = "lead",
) -> dict:
    """Lay one section (chords+bass+drums, optional melody) at `at_bar` on shared tracks.
    Creates missing tracks (idempotent by name) and loads instruments. Returns placement."""
    inv = inventory if inventory is not None else bridge.call("list_installed_fx").get("fx", [])

    # tracks (create is idempotent-enough for the fake/live; name lookup drives writes)
    existing = {t["name"] for t in bridge.call("list_tracks")}
    needed = [drum_track, chord_track, bass_track] + ([lead_track] if melody else [])
    for name in needed:
        if name not in existing:
            bridge.call("create_track", name=name)

    # chords: 1 bar/chord, progression repeated to fill `bars`
    voiced = voice_lead(resolve_progression(progression, key=key, mode=mode, octave=4))
    filled = [voiced[i % len(voiced)] for i in range(bars)]
    chord_notes = _chord_notes(filled, 4.0)
    _write_notes(bridge, chord_track, chord_notes, at_bar=at_bar)
    ci = select_instrument("keys", inv)
    bridge.call("add_instrument", track=chord_track, kind="named",
                name=ci["name"] if ci["kind"] == "named" else "ReaSynth")

    # bass: roots from a low-octave resolution
    bass_chords = resolve_progression(progression, key=key, mode=mode, octave=2)
    bass_filled = [bass_chords[i % len(bass_chords)] for i in range(bars)]
    _write_notes(bridge, bass_track, bassline_notes(bass_filled, style="root",
                                                    bars_per_chord=1), at_bar=at_bar)
    bi = select_instrument("bass", inv)
    bridge.call("add_instrument", track=bass_track, kind="named",
                name=bi["name"] if bi["kind"] == "named" else "ReaSynth")

    # drums: named pattern (or raw grid), tiled across `bars`
    grid = DRUM_PATTERNS.get(drums, drums)
    one_bar = parse_drum_grid(grid)
    drum_notes = [
        {**hit, "start_beat": hit["start_beat"] + b * 4.0}
        for b in range(bars) for hit in one_bar
    ]
    _write_notes(bridge, drum_track, drum_notes, at_bar=at_bar)
    di = select_instrument("drums", inv)
    if di["kind"] == "drumkit":
        load_drumkit(bridge, drum_track)
    else:
        bridge.call("add_instrument", track=drum_track, kind="named", name=di["name"])

    # optional melody
    if melody:
        mel = parse_melody(melody, key=key, mode=mode)
        _write_notes(bridge, lead_track, mel, at_bar=at_bar)
        li = select_instrument("keys", inv)
        bridge.call("add_instrument", track=lead_track, kind="named",
                    name=li["name"] if li["kind"] == "named" else "ReaSynth")

    return {"at_bar": at_bar, "bars": bars, "tracks": needed}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(annotations=_DESTRUCTIVE)
    def add_marker(name: str, bar: int = 1) -> dict:
        """Place a named marker at the start of `bar` (1-based) on the REAPER timeline."""
        result = BridgeClient().call("add_marker", name=name, bar=bar)
        return {"name": result.get("name"), "bar": result.get("bar"),
                "index": result.get("index")}

    @mcp.tool(annotations=_DESTRUCTIVE)
    def build_section(
        key: str, mode: str, progression: str, bars: int = 4, at_bar: int = 1,
        drums: str = "backbeat", melody: str | None = None,
    ) -> dict:
        """Lay one section — chords, bass, drums, and an optional in-key melody — at `at_bar`
        on the shared drums/chords/bass/lead tracks. `drums` is a named pattern
        (backbeat/halftime/fourfloor) or a raw step grid."""
        return _build_section(BridgeClient(), key=key, mode=mode, progression=progression,
                              bars=bars, at_bar=at_bar, drums=drums, melody=melody)
