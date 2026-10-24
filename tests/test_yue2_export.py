"""LeadSheet -> YuE2's native ABC dialect (spec 2026-10-24 §4). Expected scores are written
out by hand from the dialect rules in docs/pins/2026-10-yue2-backend.md and the YuE examples."""

from __future__ import annotations

import pytest

from orpheus_mcp.models import LeadSheet, LeadSheetSection
from orpheus_mcp.notation.leadsheet import resolve_lead_sheet
from orpheus_mcp.notation.yue2_abc import export_yue2, reduce_chord, yue2_key

HEADER_4_4 = (
    "X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=88\n"
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
    'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
)


def _export(*sections, key="C", mode="major", meter=(4, 4), tempo=88, style=None):
    sheet = LeadSheet(key=key, mode=mode, meter=meter, tempo_bpm=tempo, style=style,
                      sections=[LeadSheetSection(**s) for s in sections])
    resolved, _ = resolve_lead_sheet(sheet)
    return export_yue2(resolved)


def _body(abc: str) -> str:
    """Everything from the K: line on (the 7 lines before it are the fixed header)."""
    return abc.split("\n", 7)[7]


def test_golden_two_bar_verse():
    out = _export({"name": "Verse 1", "bars": 2, "chart": "C | G",
                   "melody": "E4:e G4:e A4:e G4:e E4:e D4:e C4:q D4:h r:h",
                   "syllables": "Neon fades a-long the lane foot-steps"})
    assert out.abc == HEADER_4_4 + (
        "K:C\n% verse\nV: Vocal\n"
        '"C"E2G2A2G2E2D2C4|"G"D8z8|\n'
        "V: Ins\nZ2|\n"
    )
    assert out.warnings == []


def test_chord_change_inside_a_held_note_uses_a_tie():
    out = _export({"name": "Verse", "bars": 1, "chart": "C / G", "melody": "C4:w"})
    assert '"C"C8-"G"C8|' in out.abc


def test_note_across_the_bar_line_is_tied_and_odd_lengths_split():
    out = _export({"name": "Verse", "bars": 2, "chart": "C | G",
                   "melody": "C4:h. E4:e. G4:q."})
    assert _body(out.abc).splitlines()[3] == '"C"C12E3G-|"G"G4-Gz8z3|'


def test_accidentals_are_written_so_both_readings_agree():
    # YuE2 carries ^F to every octave's F in the bar; standard ABC (music21) does not.
    out = _export({"name": "Verse", "bars": 1, "chart": "C",
                   "melody": "F#4:q F#5:q F4:q C4:q"})
    assert '"C"^F4^f4=F4C4|' in out.abc


def test_accidental_state_resets_at_the_bar_line():
    out = _export({"name": "Verse", "bars": 2, "chart": "C | C",
                   "melody": "F#4:w F#4:w"})
    assert '"C"^F16|"C"^F16|' in out.abc


def test_tied_continuation_repeats_its_accidental():
    out = _export({"name": "Verse", "bars": 2, "chart": "C | C", "melody": "C4:h. F#4:h C4:h."})
    assert '"C"C12^F4-|"C"^F4C12|' in out.abc


@pytest.mark.parametrize(
    ("key", "mode", "melody", "expect"),
    [("A", "minor", "G#4:q A4:q B4:q C5:q", '"Am"^G4A4B4c4|'),
     ("F", "major", "Bb4:q B4:q F4:q C5:q", '"F"B4=B4F4c4|'),
     ("D", "major", "F#4:q F4:q C#5:q D5:q", '"D"F4=F4c4d4|')],
)
def test_key_signature_spelling(key, mode, melody, expect):
    chord = "Am" if mode == "minor" else key
    out = _export({"name": "Verse", "bars": 1, "chart": chord, "melody": melody},
                  key=key, mode=mode)
    assert expect in out.abc


@pytest.mark.parametrize(
    ("tonic", "mode", "key"),
    [("C", "major", "C"), ("A", "minor", "Am"), ("A#", "major", "Bb"), ("A#", "minor", "A#m"),
     ("Gb", "major", "Gb"), ("F#", "major", "F#"), ("Db", "minor", "C#m"), ("Bb", "minor", "Bbm")],
)
def test_yue2_key_names(tonic, mode, key):
    assert yue2_key(tonic, mode) == key


@pytest.mark.parametrize(
    ("symbol", "reduced"),
    [("Fmaj9", "Fmaj7"), ("Cadd9", "C"), ("G13", "G7"), ("Em9", "Em7"), ("C5", "C"),
     ("Dsus", "Dsus4"), ("Cmaj7#11", "Cmaj7"), ("E7#9", "E7"), ("Am(maj7)", "Am(maj7)"),
     ("Bm7b5", "Bm7b5"), ("F#m7/C#", "F#m7/C#"), ("C/E", "C/E"), ("Gmaj9/B", "Gmaj7/B"),
     ("Am", "Am"), ("Cmin7", "Cm7"), ("CM7", "Cmaj7"), ("C6/9", "C6"), ("Bbo7", "Bbdim7")],
)
def test_chord_reduction(symbol, reduced):
    assert reduce_chord(symbol)[0] == reduced


def test_reductions_are_reported():
    out = _export({"name": "Verse", "bars": 1, "chart": "Fmaj9"})
    assert out.reductions == ["Fmaj9 -> Fmaj7"]
    assert '"Fmaj7"z16|' in out.abc


def test_sections_get_labels_groups_of_four_and_lyrics_tags():
    out = _export(
        {"name": "Intro", "bars": 2, "chart": "Am | Am"},
        {"name": "Verse 1", "bars": 6, "chart": "Am | F | C | G | Am | E",
         "lyrics": ["Line one here", "Line two here"]},
        {"name": "Chorus A", "bars": 1, "chart": "F",
         "melody": "A4:h A4:h", "syllables": "Hold on"},
        key="A", mode="minor",
    )
    body = _body(out.abc).splitlines()
    assert body[0] == "K:Am"
    assert [line for line in body if line.startswith("%")] == ["% intro", "% verse", "% chorus"]
    vocal_lines = [body[i + 1] for i, line in enumerate(body) if line == "V: Vocal"]
    assert [line.count("|") for line in vocal_lines] == [2, 4, 2, 1]  # verse: 4 + 2
    assert out.lyrics == ("[Intro]\n\n[Verse]\nLine one here\nLine two here\n\n"
                          "[Chorus]\nHold on")


def test_sidecar_maps_syllables_to_sung_notes_with_melisma():
    out = _export({"name": "Chorus", "bars": 1, "chart": "C",
                   "melody": "C4:q D4:q E4:q G4:q", "syllables": "Shine _ a-way"})
    assert [(e["syllable"], e["note_indices"]) for e in out.sidecar] == [
        ("Shine", [0, 1]), ("a-", [2]), ("way", [3])]
    assert out.sidecar[0]["section"] == "Chorus" and out.sidecar[0]["voice"] == "Vocal"
    assert out.lyrics == "[Chorus]\nShine away"


def test_riff_goes_to_the_instrument_line_only_when_asked():
    tab = "Am | A0 -  A0 -  A0 -  A0 - |"
    quiet = _export({"name": "Intro", "bars": 1, "chart": "Am", "tab": tab}, key="A", mode="minor")
    loud = _export({"name": "Intro", "bars": 1, "chart": "Am", "tab": tab, "riff_to_ins": True},
                   key="A", mode="minor")
    assert _body(quiet.abc).splitlines()[-1] == "Z|"
    assert _body(loud.abc).splitlines()[-1] == "A,,,4A,,,4A,,,4A,,,4|"  # A1 = MIDI 33


def test_six_eight_uses_a_thirty_second_grid():
    out = _export({"name": "Verse", "bars": 1, "chart": "C", "melody": "C4:q. E4:q."},
                  meter=(6, 8))
    assert "M:6/8\nL:1/32\n" in out.abc
    assert '"C"C12E12|' in out.abc


def test_off_grid_notes_are_snapped_and_warned():
    from orpheus_mcp.models import SheetNote

    sheet = LeadSheet(key="C", tempo_bpm=90, sections=[LeadSheetSection(
        name="Verse", label="verse", bars=1, chords=[{"symbol": "C", "bar": 1, "beats": 4}],
        melody_notes=[SheetNote(pitch=60, start_beat=0.1, duration_beats=1.0)])])
    out = export_yue2(sheet)
    assert any("grid" in w for w in out.warnings)
    assert '"C"C4z12|' in out.abc
