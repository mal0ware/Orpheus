"""LeadSheet -> notes per track (spec 2026-10-24 §6), pure: no REAPER."""

from __future__ import annotations

import pytest

from orpheus_mcp.models import LeadSheet, LeadSheetSection
from orpheus_mcp.notation.leadsheet import resolve_lead_sheet
from orpheus_mcp.notation.realize import chord_pitches, realize_section


@pytest.mark.parametrize(
    ("symbol", "pcs"),
    [("C", {0, 4, 7}), ("Am", {9, 0, 4}), ("Fmaj7", {5, 9, 0, 4}), ("E7", {4, 8, 11, 2}),
     ("Bm7b5", {11, 2, 5, 9}), ("Gsus4", {7, 0, 2}), ("Cadd9", {0, 4, 7, 2}), ("C5", {0, 7}),
     ("Bbmaj7", {10, 2, 5, 9}), ("Ebm", {3, 6, 10}), ("Am(maj7)", {9, 0, 4, 8}),
     ("Fmaj9", {5, 9, 0, 4}), ("C6/9", {0, 4, 7, 9})],
)
def test_chord_pitches_cover_the_owners_vocabulary(symbol, pcs):
    pitches = chord_pitches(symbol)
    assert {p % 12 for p in pitches} >= pcs
    assert all(40 <= p <= 84 for p in pitches)


def test_slash_chord_puts_its_bass_lowest():
    pitches = chord_pitches("C/E")
    assert min(pitches) % 12 == 4 and {p % 12 for p in pitches} == {0, 4, 7}


def test_unknown_symbols_raise():
    with pytest.raises(ValueError):
        chord_pitches("Hm7")


def _section(**kw):
    sheet = LeadSheet(key="A", mode="minor", tempo_bpm=90,
                      sections=[LeadSheetSection(name="Verse 1", bars=2, **kw)])
    resolved, _ = resolve_lead_sheet(sheet)
    return realize_section(resolved.sections[0], resolved.meter)


def test_chords_riff_melody_and_drums_land_on_their_beats():
    out = _section(chart="Am / F | C",
                   tab="Am | A0 -  A0 -  A0 -  A0 - |\nC | A3 -  -  -  -  -  -  - |",
                   melody="A4:h C5:h E5:w", syllables="la la la", drums="backbeat",
                   lyrics=["first line"])
    chords = out.tracks["chords"]
    assert sorted({n["start_beat"] for n in chords}) == [0.0, 2.0, 4.0]
    assert {n["duration_beats"] for n in chords if n["start_beat"] == 4.0} == {4.0}
    assert [n["pitch"] for n in out.tracks["bass"]][:4] == [33, 33, 33, 33]
    assert [(n["pitch"], n["start_beat"]) for n in out.tracks["lead"]] == [
        (69, 0.0), (72, 2.0), (76, 4.0)]
    assert max(n["start_beat"] for n in out.tracks["drums"]) >= 4.0  # tiled over both bars
    assert out.lyric_marker == "first line"


def test_empty_parts_are_absent_and_lyrics_fall_back_to_syllables():
    out = _section(chart="Am | F", melody="A4:h A4:h A4:w", syllables="be-fore long")
    assert set(out.tracks) == {"chords", "lead"}
    assert out.lyric_marker == "before long"


def test_drums_outside_four_four_are_skipped_with_a_warning():
    sheet = LeadSheet(key="C", meter=(3, 4), tempo_bpm=100, sections=[
        LeadSheetSection(name="Verse", bars=1, chart="C", drums="backbeat")])
    resolved, _ = resolve_lead_sheet(sheet)
    out = realize_section(resolved.sections[0], resolved.meter)
    assert "drums" not in out.tracks
    assert any("4/4" in w for w in out.warnings)
