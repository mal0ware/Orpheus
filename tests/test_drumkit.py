# tests/test_drumkit.py
"""ensure_drum_samples: bundle-or-synthesize the 3 one-shots, license-clean."""
from __future__ import annotations

import wave

from orpheus_mcp.drumkit import ensure_drum_samples


def test_creates_three_wavs(tmp_path):
    got = ensure_drum_samples(tmp_path)
    assert set(got) == {"kick", "snare", "hat"}
    for path in got.values():
        with wave.open(path, "rb") as w:
            assert w.getnframes() > 0


def test_idempotent(tmp_path):
    first = ensure_drum_samples(tmp_path)
    second = ensure_drum_samples(tmp_path)
    assert first == second
