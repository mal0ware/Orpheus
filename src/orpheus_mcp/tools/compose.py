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
        notes: list[dict] = []
        for i, chord in enumerate(voiced):
            start = i * beats_per_chord
            for pitch in chord:
                notes.append({"pitch": pitch, "start_beat": start,
                              "duration_beats": beats_per_chord, "velocity": 90})
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
        import tempfile
        from pathlib import Path

        from orpheus_mcp.drumkit import ensure_drum_samples
        from orpheus_mcp.theory.patterns import parse_drum_grid

        notes = parse_drum_grid(pattern, steps_per_bar=steps_per_bar)
        bridge = BridgeClient()
        written = _write_notes(bridge, track, notes)
        samples = ensure_drum_samples(Path(tempfile.gettempdir()) / "orpheus_drumkit")
        bridge.call("add_instrument", track=track, kind="drumkit", samples=samples)
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
            vel = max(1, min(127, int(n["velocity"]) + rng.randint(-velocity_jitter, velocity_jitter)))
            out.append({"pitch": n["pitch"], "start_beat": round(start, 6),
                        "duration_beats": n["duration_beats"], "velocity": vel})

        bridge.call("clear_track_midi", track=track)
        _write_notes(bridge, track, out)
        return {"track": track, "humanized": len(out)}
