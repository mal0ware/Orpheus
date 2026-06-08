"""THE load-bearing invariant: a note written at beat B reads back at beat B.

PPQ/tempo/take conversion is the single most correctness-sensitive code in Orpheus. This
test guards it (mirrors waveform-MCP's 250 XML round-trip tests). Marked xfail until the
M1 MIDI primitive lands; flip to a hard requirement the moment insert_midi_notes exists.
"""

from __future__ import annotations

import pytest

from orpheus_mcp.models import Note


@pytest.mark.xfail(reason="insert_midi_notes / get_track_midi land in M1", strict=False)
def test_note_roundtrips_in_beats():
    written = [
        Note(pitch=60, start_beat=0.0, duration_beats=1.0, velocity=100),
        Note(pitch=64, start_beat=1.5, duration_beats=0.5, velocity=80),
    ]
    # TODO(M1): write `written` into a take via the bridge, read it back, and assert
    # the beats survive the PPQ/tempo round-trip within tolerance.
    read_back: list[Note] = []
    assert read_back == written
