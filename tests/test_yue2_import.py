"""YuE2 score -> LeadSheet (spec 2026-10-24 §5, reader decision 2026-10-24): notes and chords
from Orpheus's YuE2-rules parser, lyrics from the sidecar; music21 independently re-reads our own
exports here, so two readers must agree."""

from __future__ import annotations

import pytest

from orpheus_mcp.models import LeadSheet, LeadSheetSection
from orpheus_mcp.notation.leadsheet import resolve_lead_sheet
from orpheus_mcp.notation.yue2_abc import export_yue2, reduce_chord
from orpheus_mcp.notation.yue2_read import import_yue2

SHEETS = [
    LeadSheet(key="A", mode="minor", tempo_bpm=92, sections=[
        LeadSheetSection(name="Intro", bars=2, chart="Am | Fmaj7",
                         tab="Am | A0 -  A0 -  A0 -  A0 - |\nF | E1 -  E1 -  E1 -  E1 - |",
                         riff_to_ins=True),
        LeadSheetSection(name="Verse 1", bars=6, chart="Am / Fmaj9 | C/E | G13 | E7#9 | Am | E---",
                         melody="A4:q G#4:e. A4:s B4:h C5:q. B4:e A4:h r:w E4:w F#4:h G#4:h",
                         syllables="Some-thing in the way you _ lea-ving now"),
        LeadSheetSection(name="Chorus A", bars=1, chart="F", melody="A4:h A4:h",
                         syllables="Hold on"),
    ]),
    LeadSheet(key="D", mode="major", meter=(6, 8), tempo_bpm=70, sections=[
        LeadSheetSection(name="Verse", bars=2, chart="D | A/C#",
                         melody="F#4:q. F4:q. C#5:q. D5:q."),
    ]),
    LeadSheet(key="Bb", mode="minor", tempo_bpm=120, sections=[
        LeadSheetSection(name="Bridge", bars=2, chart="Bbm | Gb",
                         melody="Db5:q C5:q Bb4:q A4:q Gb4:w", syllables="o pen the door to-day"),
    ]),
]


@pytest.mark.parametrize("sheet", SHEETS, ids=["A-minor", "D-6/8", "Bb-minor"])
def test_round_trip_keeps_the_music(sheet):
    original, _ = resolve_lead_sheet(sheet)
    out = export_yue2(original)
    back, warnings = import_yue2(out.abc, out.sidecar, lyrics=out.lyrics)

    assert (back.key, back.mode, back.meter, back.tempo_bpm) == (
        original.key, original.mode, original.meter, round(original.tempo_bpm))
    assert [(s.label, s.bars) for s in back.sections] == [
        (s.label, s.bars) for s in original.sections]
    for a, b in zip(original.sections, back.sections, strict=True):
        assert [(c.symbol, c.bar, c.beat) for c in b.chords] == [
            (reduce_chord(c.symbol)[0], c.bar, c.beat) for c in a.chords]
        assert [(n.pitch, n.start_beat, n.duration_beats, n.syllable)
                for n in b.melody_notes] == [
            (n.pitch, n.start_beat, n.duration_beats, n.syllable) for n in a.melody_notes]
        if a.riff_to_ins:
            assert b.riff_to_ins
            assert [(n.pitch, n.start_beat) for n in b.riff] == [
                (n.pitch, n.start_beat) for n in a.riff]
    assert warnings == []


def test_section_names_count_repeats():
    sheet = LeadSheet(key="C", tempo_bpm=100, sections=[
        LeadSheetSection(name="V", label="verse", bars=1, chart="C"),
        LeadSheetSection(name="C", label="chorus", bars=1, chart="F"),
        LeadSheetSection(name="V2", label="verse", bars=5, chart="C | C | C | C | G")])
    back, _ = import_yue2(export_yue2(resolve_lead_sheet(sheet)[0]).abc)
    assert [(s.name, s.bars) for s in back.sections] == [
        ("Verse 1", 1), ("Chorus 1", 1), ("Verse 2", 5)]
    assert back.sections[2].chords[-1].bar == 5


def test_chords_held_across_bars_keep_their_length():
    sheet = LeadSheet(key="C", tempo_bpm=100, sections=[
        LeadSheetSection(name="Solo", bars=3, chart="F | E---")])
    back, _ = import_yue2(export_yue2(resolve_lead_sheet(sheet)[0]).abc)
    assert [(c.symbol, c.bar, c.beats) for c in back.sections[0].chords] == [
        ("F", 1, 4.0), ("E", 2, 8.0)]


def test_lyrics_text_fills_section_lines():
    sheet = LeadSheet(key="C", tempo_bpm=100, sections=[
        LeadSheetSection(name="Verse", bars=1, chart="C", lyrics=["a line", "another"]),
        LeadSheetSection(name="Chorus", bars=1, chart="F", lyrics=["the hook"])])
    out = export_yue2(resolve_lead_sheet(sheet)[0])
    back, _ = import_yue2(out.abc, lyrics=out.lyrics)
    assert [s.lyrics for s in back.sections] == [["a line", "another"], ["the hook"]]


def test_a_yue2_planned_score_reads_with_yue2_accidentals():
    """YuE2's own scores rely on its rule: after ^F, every F in the bar is sharp."""
    abc = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=90\n"
           'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
           'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
           'K:C\n% verse\nV: Vocal\n"D7"^F4f4A8|\nV: Ins\nZ|\n')
    back, _ = import_yue2(abc)
    assert [n.pitch for n in back.sections[0].melody_notes] == [66, 78, 69]  # f is F#5


def test_invalid_scores_are_rejected_with_the_checker_message():
    with pytest.raises(ValueError, match="T:"):
        import_yue2("X:1\nT:Title\nM:4/4\nL:1/16\nQ:1/4=90\n")


# --------------------------------------------------------------------------- music21 agrees


def _music21_vocal(abc: str) -> list[tuple[int, float, float]]:
    """music21 re-reading our export's Vocal voice on its own (it merges voices otherwise)."""
    import music21 as m21

    lines = abc.splitlines()
    head = [ln for ln in lines[:8] if not ln.startswith("V:")]
    vocal, take = [], False
    for ln in lines[8:]:
        if ln.startswith("V: "):
            take = ln == "V: Vocal"
        elif take and not ln.startswith("%"):
            vocal.append(ln)
    score = m21.converter.parse("\n".join(head + vocal), format="abc")
    notes = []
    for n in score.flatten().notes:
        if isinstance(n, m21.harmony.ChordSymbol):
            continue
        if n.tie is not None and n.tie.type in ("continue", "stop"):
            prev = notes[-1]
            notes[-1] = (prev[0], prev[1], prev[2] + float(n.quarterLength))
            continue
        notes.append((int(n.pitch.midi), float(n.offset), float(n.quarterLength)))
    return notes


@pytest.mark.parametrize("sheet", SHEETS, ids=["A-minor", "D-6/8", "Bb-minor"])
def test_music21_reads_our_exports_the_same_way(sheet):
    original, _ = resolve_lead_sheet(sheet)
    out = export_yue2(original)
    back, _ = import_yue2(out.abc)
    ours, offset = [], 0.0
    bar_beats = original.meter[0] * 4.0 / original.meter[1]
    for sec in back.sections:
        ours += [(n.pitch, offset + n.start_beat, n.duration_beats) for n in sec.melody_notes]
        offset += sec.bars * bar_beats
    assert _music21_vocal(out.abc) == ours


def test_windows_line_endings_are_normalised_on_import():
    sheet = LeadSheet(key="C", tempo_bpm=100, sections=[
        LeadSheetSection(name="Verse", bars=1, chart="C", melody="C4:w")])
    abc = export_yue2(resolve_lead_sheet(sheet)[0]).abc.replace("\n", "\r\n")
    back, _ = import_yue2(abc)
    assert [n.pitch for n in back.sections[0].melody_notes] == [60]
