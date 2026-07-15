"""In-key scaffolding so the LLM doesn't drift out of key. A read-only knowledge oracle
backed by reimplemented theory tables + ported genre profiles — pure functions, no bridge."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.theory.genre_profiles import get_profile
from orpheus_mcp.theory.music_theory_data import (
    progression_triads,
    scale_notes,
    snap_to_scale,
)

_RO = {"readOnlyHint": True}

# Sensible fallbacks when no genre is given: the most vanilla functional progressions.
_DEFAULT_PROGRESSIONS = {"major": "I-IV-V-I", "minor": "i-iv-V-i"}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:  # noqa: ARG001 - uniform signature
    @mcp.tool(annotations=_RO)
    def get_scale_notes(key: str, mode: str = "major", octave: int = 4) -> dict:
        """MIDI notes for a key/mode so generated notes stay diatonic."""
        midi = scale_notes(key, mode, octave)
        return {
            "key": key,
            "mode": mode,
            "octave": octave,
            "midi_notes": midi,
            "pitch_classes": [n % 12 for n in midi],
        }

    @mcp.tool(annotations=_RO)
    def suggest_chord_progression(key: str, mode: str = "major", genre: str | None = None) -> dict:
        """A genre-typical diatonic progression as Roman numerals + concrete MIDI.

        With a genre, picks the profile progression whose numeral case matches the mode
        (minor progressions start lowercase); without one, falls back to the vanilla
        functional default for the mode.
        """
        if genre is not None:
            candidates = get_profile(genre)["progressions"]
            wants_minor = mode == "minor"
            progression = next(
                (p for p in candidates if p.split("-")[0].strip()[0].islower() == wants_minor),
                candidates[0],
            )
        else:
            progression = _DEFAULT_PROGRESSIONS.get(mode, _DEFAULT_PROGRESSIONS["major"])
        chords = progression_triads(key, mode, progression)
        return {
            "key": key,
            "mode": mode,
            "genre": genre,
            "progression": progression,
            "roman_numerals": [numeral for numeral, _ in chords],
            "chords": [{"roman": numeral, "midi": pitches} for numeral, pitches in chords],
        }

    @mcp.tool(annotations=_RO)
    def constrain_to_key(notes: list[int], key: str, mode: str = "major") -> dict:
        """Snap a proposed note set to the nearest in-key pitches before writing.

        Ties resolve downward (see snap_to_scale) so constrained passages never creep up.
        """
        snapped = snap_to_scale(notes, key, mode)
        return {
            "key": key,
            "mode": mode,
            "notes": snapped,
            "changed": sum(1 for a, b in zip(notes, snapped, strict=True) if a != b),
        }

    @mcp.tool(annotations=_RO)
    def get_genre_profile(genre: str) -> dict:
        """A style's typical progressions / scales / BPM range / instruments / feel.
        The RECOMMEND-side lookup."""
        return dict(get_profile(genre))
