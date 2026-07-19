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
    instruments_loaded: set[str] | None = None,
) -> dict:
    """Lay one section (chords+bass+drums, optional melody) at `at_bar` on shared tracks.
    Creates missing tracks (idempotent by name) and loads instruments. Returns placement.

    `instruments_loaded`, if given, is a caller-owned set of track names whose instrument
    has already been loaded; a track's instrument is loaded at most once per set (needed
    because `add_instrument(kind="drumkit")` stacks new FX on every call instead of
    deduping like the named-instrument path does). Defaults to a fresh set per call, so a
    standalone `build_section` invocation always (re)loads, matching prior behavior."""
    inv = inventory if inventory is not None else bridge.call("list_installed_fx").get("fx", [])
    loaded = instruments_loaded if instruments_loaded is not None else set()

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
    if chord_track not in loaded:
        ci = select_instrument("keys", inv)
        bridge.call("add_instrument", track=chord_track, kind="named",
                    name=ci["name"] if ci["kind"] == "named" else "ReaSynth")
        loaded.add(chord_track)

    # bass: roots from a low-octave resolution
    bass_chords = resolve_progression(progression, key=key, mode=mode, octave=2)
    bass_filled = [bass_chords[i % len(bass_chords)] for i in range(bars)]
    _write_notes(bridge, bass_track, bassline_notes(bass_filled, style="root",
                                                    bars_per_chord=1), at_bar=at_bar)
    if bass_track not in loaded:
        bi = select_instrument("bass", inv)
        bridge.call("add_instrument", track=bass_track, kind="named",
                    name=bi["name"] if bi["kind"] == "named" else "ReaSynth")
        loaded.add(bass_track)

    # drums: named pattern (or raw grid), tiled across `bars`
    grid = DRUM_PATTERNS.get(drums, drums)
    one_bar = parse_drum_grid(grid)
    drum_notes = [
        {**hit, "start_beat": hit["start_beat"] + b * 4.0}
        for b in range(bars) for hit in one_bar
    ]
    _write_notes(bridge, drum_track, drum_notes, at_bar=at_bar)
    if drum_track not in loaded:
        di = select_instrument("drums", inv)
        if di["kind"] == "drumkit":
            load_drumkit(bridge, drum_track)
        else:
            bridge.call("add_instrument", track=drum_track, kind="named", name=di["name"])
        loaded.add(drum_track)

    # optional melody
    if melody:
        mel = parse_melody(melody, key=key, mode=mode)
        _write_notes(bridge, lead_track, mel, at_bar=at_bar)
        if lead_track not in loaded:
            li = select_instrument("keys", inv)
            bridge.call("add_instrument", track=lead_track, kind="named",
                        name=li["name"] if li["kind"] == "named" else "ReaSynth")
            loaded.add(lead_track)

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

    @mcp.tool(annotations=_DESTRUCTIVE)
    def arrange_song(
        tempo: float, key: str, mode: str, sections: list[dict],
    ) -> dict:
        """Build a full song: set tempo, then place each section end-to-end on shared tracks
        with a marker at its start. `sections` is a list of
        {name, bars, progression, drums?, melody?}. The model composes this list + the
        melodies/lyrics from the user's description."""
        bridge = BridgeClient()
        bridge.call("set_tempo", bpm=float(tempo))
        inv = bridge.call("list_installed_fx").get("fx", [])
        bar = 1
        placed: list[dict] = []
        instruments_loaded: set[str] = set()
        for s in sections:
            bridge.call("add_marker", name=s["name"], bar=bar)
            _build_section(bridge, key=key, mode=mode, progression=s["progression"],
                           bars=int(s["bars"]), at_bar=bar,
                           drums=s.get("drums", "backbeat"), melody=s.get("melody"),
                           inventory=inv, instruments_loaded=instruments_loaded)
            placed.append({"name": s["name"], "at_bar": bar, "bars": int(s["bars"])})
            bar += int(s["bars"])
        return {"tempo": float(tempo), "key": key, "sections": placed,
                "markers": [{"name": p["name"], "bar": p["at_bar"]} for p in placed]}

    @mcp.tool(annotations=_DESTRUCTIVE)
    def place_lyric_markers(lines: list[str], at_bars: list[int]) -> dict:
        """Place model-authored lyric lines as timeline markers. `lines[i]` goes at
        `at_bars[i]`. Lyrics must be original text, never a copyrighted song's lyrics."""
        if len(lines) != len(at_bars):
            raise ValueError("lines and at_bars must be the same length")
        bridge = BridgeClient()
        for line, bar in zip(lines, at_bars, strict=True):
            bridge.call("add_marker", name=line, bar=int(bar))
        return {"placed": len(lines)}
