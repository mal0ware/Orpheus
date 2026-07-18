# src/orpheus_mcp/tools/compose.py
"""Generate-from-scratch composers — thin orchestrators over the theory layer + the M1
MIDI writer. Composers never get private superpowers the model couldn't call directly."""
from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient
from orpheus_mcp.theory.chords import resolve_progression, voice_lead

_DESTRUCTIVE = {"destructiveHint": True}


def _write_notes(bridge: BridgeClient, track: str, notes: list[dict], at_bar: int = 1) -> int:
    """Batch notes through insert_midi_notes (<=512/call)."""
    written = 0
    for i in range(0, len(notes), 512):
        chunk = notes[i : i + 512]
        bridge.call("insert_midi_notes", track=track, notes=chunk, at_bar=at_bar)
        written += len(chunk)
    return written


def _chord_notes(voiced: list[list[int]], beats_per_chord: float, velocity: int = 90) -> list[dict]:
    """Flatten voice-led chords into note dicts, one block per chord. Shared by
    create_chord_progression and compose_section."""
    notes: list[dict] = []
    for i, chord in enumerate(voiced):
        start = i * beats_per_chord
        for pitch in chord:
            notes.append({"pitch": pitch, "start_beat": start,
                          "duration_beats": beats_per_chord, "velocity": velocity})
    return notes


def _repeat_progression(progression: str, bars: int) -> str:
    """Repeat a '-'-joined progression (one chord per bar) until it covers `bars` bars,
    then trim to exactly `bars` chords."""
    chords = progression.split("-")
    bars = max(1, bars)
    reps = -(-bars // len(chords))  # ceil division
    return "-".join((chords * reps)[:bars])


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_chord_progression(
        track: str,
        chords: str,
        key: str | None = None,
        mode: str = "minor",
        bars_per_chord: int = 1,
        octave: int = 4,
    ) -> dict:
        """Write a chord progression as voiced MIDI. `chords` is Roman ('i-iv-V-i', needs
        `key`) or absolute symbols ('Cm7, Fm7, Bb7'). Auto-loads ReaSynth so it's audible."""
        voiced = voice_lead(resolve_progression(chords, key=key, mode=mode, octave=octave))
        beats_per_chord = 4.0 * bars_per_chord
        notes = _chord_notes(voiced, beats_per_chord)
        bridge = BridgeClient()
        written = _write_notes(bridge, track, notes)
        bridge.call("add_instrument", track=track, kind="named", name="ReaSynth")
        return {"track": track, "chords_written": len(voiced), "notes_written": written}

    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_bassline(
        track: str,
        chords: str,
        key: str | None = None,
        mode: str = "minor",
        style: str = "root",
        octave: int = 2,
        bars_per_chord: int = 1,
    ) -> dict:
        """Write a bass line following `chords` (same notation as create_chord_progression).
        `style`: 'root' | 'root_fifth' | 'octave'. Auto-loads ReaSynth (bass register)."""
        from orpheus_mcp.theory.patterns import bassline_notes

        chord_pitches = resolve_progression(chords, key=key, mode=mode, octave=octave)
        notes = bassline_notes(chord_pitches, style=style, bars_per_chord=bars_per_chord)
        bridge = BridgeClient()
        written = _write_notes(bridge, track, notes)
        bridge.call("add_instrument", track=track, kind="named", name="ReaSynth")
        return {"track": track, "notes_written": written}

    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_drum_pattern(track: str, pattern: str, steps_per_bar: int = 16) -> dict:
        """Write a drum pattern from a step grid (rows 'kick:'/'snare:'/'hat:', 'x'=hit).
        Loads a stock 3-voice kit so it's audible immediately."""
        from orpheus_mcp.drumkit import load_drumkit
        from orpheus_mcp.theory.patterns import parse_drum_grid

        notes = parse_drum_grid(pattern, steps_per_bar=steps_per_bar)
        bridge = BridgeClient()
        written = _write_notes(bridge, track, notes)
        load_drumkit(bridge, track)
        return {"track": track, "hits_written": written}

    @mcp.tool(annotations=_DESTRUCTIVE)
    def humanize_pass(
        track: str,
        timing_ms: float = 12.0,
        velocity_jitter: int = 6,
        swing: float = 0.0,
        seed: int = 0,
    ) -> dict:
        """Add seeded human feel: timing/velocity jitter + optional swing on offbeat 16ths.
        Reads the track's notes, transforms, and REPLACES them (deterministic given `seed`)."""
        import random

        bridge = BridgeClient()
        current = bridge.call("get_track_midi", track=track).get("notes", [])
        if not current:
            return {"track": track, "humanized": 0}

        rng = random.Random(seed)
        tempo = bridge.call("get_project_info").get("tempo", 120.0)
        beats_per_ms = tempo / 60.0 / 1000.0  # ms -> beats
        out: list[dict] = []
        for n in current:
            start = n["start_beat"]
            # swing: delay the 2nd/4th 16th of each beat toward a triplet feel.
            if swing and round((start * 4) % 4) in (1, 3):
                start += swing * (1.0 / 6.0)
            start += rng.uniform(-timing_ms, timing_ms) * beats_per_ms
            start = max(0.0, start)
            vel = max(1, min(127, int(n["velocity"])
                                   + rng.randint(-velocity_jitter, velocity_jitter)))
            out.append({"pitch": n["pitch"], "start_beat": round(start, 6),
                        "duration_beats": n["duration_beats"], "velocity": vel})

        bridge.call("clear_track_midi", track=track)
        _write_notes(bridge, track, out)
        return {"track": track, "humanized": len(out)}

    @mcp.tool(annotations=_DESTRUCTIVE)
    def compose_section(genre: str, bars: int = 8, key: str | None = None) -> dict:
        """Build a full, audible section in one call: sets tempo, creates drums/chords/bass
        tracks, lays a genre-appropriate groove, and loads the best available instrument per
        role (preferring the user's own installed plugins). Returns the tempo, tracks, and
        the instrument chosen per role."""
        from orpheus_mcp.drumkit import load_drumkit
        from orpheus_mcp.instruments import select_instrument
        from orpheus_mcp.theory.genre_profiles import get_profile
        from orpheus_mcp.theory.patterns import bassline_notes, parse_drum_grid

        profile = get_profile(genre)  # raises ValueError on unknown genre
        tonic = key or {"lofi": "A", "hiphop": "A", "classical": "C"}.get(genre, "A")
        mode = profile["typical_modes"][0]
        bpm = sum(profile["bpm_range"]) // 2
        full_prog = _repeat_progression(profile["progressions"][0], bars)

        bridge = BridgeClient()
        bridge.call("set_tempo", bpm=float(bpm))
        inventory = bridge.call("list_installed_fx").get("fx", [])

        for track_name in ("drums", "chords", "bass"):
            bridge.call("create_track", name=track_name)

        # drums: one-bar backbeat, tiled across the full section length.
        one_bar = ("kick:  x...x...x...x...\n"
                   "snare: ....x.......x...\n"
                   "hat:   x.x.x.x.x.x.x.x.")
        backbone = parse_drum_grid(one_bar)
        drum_notes = [
            {**hit, "start_beat": hit["start_beat"] + b * 4.0}
            for b in range(bars)
            for hit in backbone
        ]
        _write_notes(bridge, "drums", drum_notes)
        drums_instrument = select_instrument("drums", inventory)
        if drums_instrument["kind"] == "drumkit":
            load_drumkit(bridge, "drums")
        else:
            bridge.call("add_instrument", track="drums", kind="named",
                        name=drums_instrument["name"])

        # chords: voice-led block chords, one bar per chord.
        voiced = voice_lead(resolve_progression(full_prog, key=tonic, mode=mode, octave=4))
        _write_notes(bridge, "chords", _chord_notes(voiced, beats_per_chord=4.0))
        chords_instrument = select_instrument("keys", inventory)
        bridge.call("add_instrument", track="chords", kind="named",
                    name=chords_instrument["name"])

        # bass: root notes following the same progression, one octave register down.
        bass_chords = resolve_progression(full_prog, key=tonic, mode=mode, octave=2)
        _write_notes(bridge, "bass", bassline_notes(bass_chords, style="root"))
        bass_instrument = select_instrument("bass", inventory)
        bridge.call("add_instrument", track="bass", kind="named",
                    name=bass_instrument["name"])

        return {
            "genre": genre,
            "tempo": float(bpm),
            "key": tonic,
            "tracks": ["drums", "chords", "bass"],
            "instruments": {
                "drums": drums_instrument,
                "chords": chords_instrument,
                "bass": bass_instrument,
            },
        }
