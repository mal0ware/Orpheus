"""resolve_lead_sheet: the sheet as written -> the expanded LeadSheet (spec 2026-10-24 §3)."""

from __future__ import annotations

import pytest

from orpheus_mcp.models import LeadSheet, LeadSheetSection
from orpheus_mcp.notation.leadsheet import attach_syllables, derive_label, resolve_lead_sheet


def _sheet(**section) -> LeadSheet:
    return LeadSheet(key="A", mode="minor", tempo_bpm=90,
                     sections=[LeadSheetSection(name="Verse 1", bars=2, **section)])


@pytest.mark.parametrize(
    ("name", "label"),
    [("Intro (after drums)", "intro"), ("Verse 2", "verse"), ("Pre-chorus", "pre-chorus"),
     ("Prechorus", "pre-chorus"), ("Chorus A", "chorus"), ("Final chorus", "chorus"),
     ("Post-chorus", "post-chorus"), ("Bridge", "bridge"), ("Solo", "solo"),
     ("Interlude", "interlude"), ("Outro", "outro"), ("Instrumental break", "instrumental"),
     ("Hook", "chorus")],
)
def test_section_labels_come_from_names(name, label):
    assert derive_label(name) == label


def test_unknown_names_fall_back_to_verse_with_a_warning():
    ls, warnings = resolve_lead_sheet(LeadSheet(
        key="C", tempo_bpm=100, sections=[LeadSheetSection(name="Part B", bars=1, chart="C")]))
    assert ls.sections[0].label == "verse"
    assert any("Part B" in w for w in warnings)


def test_chart_tab_and_melody_expand():
    ls, warnings = resolve_lead_sheet(_sheet(
        chart="Am / Fmaj7 | C / E",
        tab="Am | A0 -  A0 -  A0 -  A0 - |\nC | A3 -  -  -  E0 -  -  - |",
        melody="A4:q C5:q E5:h G#4:h A4:h",
        syllables="Hold the eve-ning still",
    ))
    sec = ls.sections[0]
    assert warnings == []
    assert [(c.symbol, c.bar, c.beat) for c in sec.chords] == [
        ("Am", 1, 1.0), ("Fmaj7", 1, 3.0), ("C", 2, 1.0), ("E", 2, 3.0)]
    assert len(sec.riff) == 6 and sec.riff[0].pitch == 33
    assert [n.pitch for n in sec.melody_notes] == [69, 72, 76, 68, 69]  # G#4 kept: no snapping
    assert [n.syllable for n in sec.melody_notes] == ["Hold", "the", "eve-", "ning", "still"]
    assert sec.label == "verse"


def test_chart_notes_are_kept_on_the_section():
    ls, _ = resolve_lead_sheet(_sheet(chart="Fmaj7 | E (stop)"))
    assert ls.sections[0].notes == ["bar 2: stop"]


def test_mismatched_lengths_are_warned():
    ls, warnings = resolve_lead_sheet(_sheet(chart="Am | F | C", melody="A4:w A4:w A4:w"))
    joined = " ".join(warnings)
    assert "chart has 3 bars" in joined
    assert "runs past bar 2" in joined


def test_expanded_fields_are_left_alone_when_no_raw_text():
    sec = LeadSheetSection(name="Verse", bars=1, chords=[{"symbol": "C", "bar": 1, "beats": 4}])
    ls, _ = resolve_lead_sheet(LeadSheet(key="C", tempo_bpm=100, sections=[sec]))
    assert [c.symbol for c in ls.sections[0].chords] == ["C"]


def test_syllables_melisma_and_count_warnings():
    from orpheus_mcp.models import SheetNote

    notes = [SheetNote(pitch=60 + i, start_beat=i, duration_beats=1) for i in range(4)]
    out, warnings = attach_syllables(notes, "Shine _ a-")
    assert [n.syllable for n in out] == ["Shine", "_", "a-", None]
    assert any("4 notes but 3 syllables" in w for w in warnings)
    out2, warnings2 = attach_syllables(notes[:1], "too many words")
    assert [n.syllable for n in out2] == ["too"]
    assert any("1 notes but 3 syllables" in w for w in warnings2)
