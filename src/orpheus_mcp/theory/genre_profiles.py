"""Genre/style profiles — the RECOMMEND-side knowledge base.

Ported (reimplemented) from the genre-intelligence pattern in asume21/music-theory-mcp:
typical BPM range, scales, chord progressions, instrumentation, and feel per style. This
backs both `get_genre_profile` and the primitive "why does this sound like X" classifier.

These are intentionally OPINIONATED defaults (taste, not ground truth). Real fingerprints
in data/fingerprints/ override them with measured per-era reference data.
"""

from __future__ import annotations

from typing import TypedDict


class GenreProfile(TypedDict):
    bpm_range: tuple[int, int]
    typical_modes: list[str]
    progressions: list[str]          # Roman-numeral strings
    instruments: list[str]
    feel: str


GENRE_PROFILES: dict[str, GenreProfile] = {
    "classical": {
        "bpm_range": (60, 90),
        "typical_modes": ["major", "minor", "harmonic_minor"],
        "progressions": ["I-IV-V-I", "ii-V-I", "i-iv-V-i"],
        "instruments": ["strings", "piano", "woodwinds", "brass"],
        "feel": "functional harmony, voice-led, dynamic phrasing that breathes",
    },
    "hiphop": {
        "bpm_range": (80, 100),
        "typical_modes": ["minor", "dorian", "phrygian"],
        "progressions": ["i-iv", "i-VI-VII", "i-v"],
        "instruments": ["sub_bass", "drums", "keys", "sampled_loop"],
        "feel": "drum-forward, heavy sub, sparse harmony, swung hats",
    },
    "lofi": {
        "bpm_range": (60, 90),
        "typical_modes": ["minor", "dorian"],
        "progressions": ["ii-V-I", "i-iv-VII-III"],
        "instruments": ["keys", "sub_bass", "drums", "guitar"],
        "feel": "jazzy 7th chords, swing, tape warmth, relaxed pocket",
    },
}


def get_profile(genre: str) -> GenreProfile:
    key = genre.strip().lower().replace("-", "").replace(" ", "")
    if key not in GENRE_PROFILES:
        raise ValueError(f"No profile for {genre!r}; known: {sorted(GENRE_PROFILES)}")
    return GENRE_PROFILES[key]
