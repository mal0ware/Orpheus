"""YuE2's native ABC dialect: checker and reader (spec 2026-10-24 lead sheet §5).

Every rule of the dialect has a failing example below, and YuE's own example scores must pass
and read back to the right notes, chords and groups.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from orpheus_mcp.notation.yue2_check import (
    AbcChord,
    AbcNote,
    Yue2AbcError,
    check_yue2_abc,
    parse_yue2_abc,
)

# Example scores from https://github.com/multimodal-art-projection/YuE (Apache-2.0), used as test
# data. EXAMPLE_INS_ABC is the repo's instrumental/assets/example-ins.abc.
SCORE_ABC = """\
X:1
T:
M:4/4
L:1/16
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal
"C"E2G2A2G2E2D2C4|"G"D2E2G2E2D2C2D4|"Am"E2G2A2c2B2A2G4|"F"F2E2D2E2G2E2C4|
V: Ins
Z4|
% chorus
V: Vocal
"C"G2A2c2B2A2G2E4|"F"F2A2G2E2D2E2G4|"G"A2c2B2A2G2E2D4|"C"E2G2A2G2E2D2C4|
V: Ins
Z4|
"""

MELODY_ABC = """\
X:1
T:
M:4/4
L:1/16
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal
E2G2A2G2E2D2C4|D2E2G2E2D2C2D4|E2G2A2c2B2A2G4|F2E2D2E2G2E2C4|
V: Ins
Z4|
% chorus
V: Vocal
G2A2c2B2A2G2E4|F2A2G2E2D2E2G4|A2c2B2A2G2E2D4|E2G2A2G2E2D2C4|
V: Ins
Z4|
"""

EXAMPLE_INS_ABC = """\
X:1
T:
M:4/4
L:1/32
Q:1/4=96
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:G
% verse
V: Vocal
"G"z32|"D7"z32|"G"z32|"D7"z32|
V: Ins
d8e6d2B8z8|F8F8=F8z8|d8e6d2B8z8|F8F8=F8z8|
% outro
V: Vocal
M:5/8
"Em"z16z4|z4"G"z16|
V: Ins
M:5/8
E8G8B4-|B4G8D8|
"""

# Written for these tests in the shape of the model-planned score from the 2026-10-23 spike
# (docs/pins/2026-10-yue2-backend.md): L:1/32, ties across bars, Z/Z2 rests, four sections.
MODEL_STYLE_ABC = """\
X:1
T:
M:4/4
L:1/32
Q:1/4=92
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% intro
V: Vocal
"Dm7"z32|"G7"z32|
V: Ins
d8f8a8f8|g8f8e8d8|
% verse
V: Vocal
"Cmaj7"z8e8g8e8-|"Dm7"e16d8c8|"G7"B16-B8D8|"Cmaj7"C32|
V: Ins
Z4|
% chorus
V: Vocal
"Dm7"f8a8a8g8|"G7"g16f8e8|"Cmaj7"e32-|e32|
V: Ins
Z2|c16e16|g32|
% interlude
V: Vocal
"Dm7"z32|"G7"z16"Cmaj7"z16|
V: Ins
Z|c8d8e8g8|
"""

HEADER = (
    "X:1",
    "T:",
    "M:4/4",
    "L:1/16",
    "Q:1/4=88",
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
    'V: Ins clef=treble name="Ins Melody" snm="Inst."',
    "K:C",
)
GROUP = ("% verse", "V: Vocal", '"C"E4G4A4G4|', "V: Ins", "Z|")


def abc(*body: str, header: tuple[str, ...] = HEADER, key: str | None = None) -> str:
    lines = list(header)
    if key is not None:
        lines[7] = f"K:{key}"
    return "\n".join([*lines, *body]) + "\n"


def group(vocal: str, ins: str = "Z|") -> tuple[str, ...]:
    return ("V: Vocal", vocal, "V: Ins", ins)


def one(vocal: str, ins: str = "Z|", key: str | None = None) -> str:
    """A score with one group: the Vocal music line is line 10, the Ins one line 12."""
    return abc(*group(vocal, ins), key=key)


def header_with(index: int, line: str) -> str:
    lines = list(HEADER)
    lines[index] = line
    return abc(*GROUP, header=tuple(lines))


def notes(seq: tuple[AbcNote, ...]) -> list[tuple[F, int, F]]:
    return [(n.onset_q, n.pitch, n.duration_q) for n in seq]


def chords(seq: tuple[AbcChord, ...]) -> list[tuple[F, str]]:
    return [(c.onset_q, c.symbol) for c in seq]


def pitches(text: str) -> list[int]:
    return [n.pitch for n in parse_yue2_abc(text).vocal]


# --- YuE's own examples -----------------------------------------------------------------------


@pytest.mark.parametrize("text", [SCORE_ABC, MELODY_ABC, EXAMPLE_INS_ABC, MODEL_STYLE_ABC])
def test_example_scores_pass(text: str) -> None:
    assert check_yue2_abc(text) == []


def test_score_abc_reads_back() -> None:
    score = parse_yue2_abc(SCORE_ABC)
    assert (score.meter, score.unit, score.tempo_bpm, score.key) == ((4, 4), F(1, 16), 88, "C")
    assert [g.labels for g in score.groups] == [("verse",), ("chorus",)]
    assert [g.bar_count for g in score.groups] == [4, 4]
    assert [g.start_q for g in score.groups] == [0, 16]
    assert [g.meter for g in score.groups] == [(4, 4), (4, 4)]
    # Bar 1: "C"E2G2A2G2E2D2C4, each 2 = an eighth (1/2 quarter) at L:1/16.
    half = F(1, 2)
    assert notes(score.vocal)[:7] == [
        (0, 64, half), (half, 67, half), (1, 69, half), (F(3, 2), 67, half),
        (2, 64, half), (F(5, 2), 62, half), (3, 60, 1),
    ]
    assert notes(score.vocal)[-1] == (31, 60, 1)
    assert len(score.vocal) == 56
    assert sum(n.duration_q for n in score.vocal) == 32
    assert score.ins == ()
    assert chords(score.chords) == [
        (0, "C"), (4, "G"), (8, "Am"), (12, "F"), (16, "C"), (20, "F"), (24, "G"), (28, "C"),
    ]


def test_melody_abc_is_score_abc_without_chords() -> None:
    melody = parse_yue2_abc(MELODY_ABC)
    assert melody.vocal == parse_yue2_abc(SCORE_ABC).vocal
    assert melody.chords == ()


def test_example_ins_reads_back() -> None:
    score = parse_yue2_abc(EXAMPLE_INS_ABC)
    assert (score.meter, score.unit, score.tempo_bpm, score.key) == ((4, 4), F(1, 32), 96, "G")
    assert [g.labels for g in score.groups] == [("verse",), ("outro",)]
    assert [g.bar_count for g in score.groups] == [4, 2]
    assert [g.start_q for g in score.groups] == [0, 16]
    assert [g.meter for g in score.groups] == [(4, 4), (5, 8)]
    assert score.vocal == ()
    # Chords on rests; the outro's 5/8 bars last 5/2 quarters, so "G" sits at 16 + 5/2 + 1/2.
    assert chords(score.chords) == [
        (0, "G"), (4, "D7"), (8, "G"), (12, "D7"), (16, "Em"), (19, "G"),
    ]
    ins = notes(score.ins)
    assert len(ins) == 19
    assert ins[:4] == [(0, 74, 1), (1, 76, F(3, 4)), (F(7, 4), 74, F(1, 4)), (2, 71, 1)]
    # K:G makes F sharp; =F naturalises it for the rest of the bar; the barline resets it.
    assert ins[4:7] == [(4, 66, 1), (5, 66, 1), (6, 65, 1)]
    assert ins[11] == (12, 66, 1)
    # Outro: B4- | B4 merges into one sounding B across the barline.
    assert ins[14:] == [(16, 64, 1), (17, 67, 1), (18, 71, 1), (19, 67, 1), (20, 62, 1)]


def test_model_style_score_reads_back() -> None:
    score = parse_yue2_abc(MODEL_STYLE_ABC)
    assert [g.labels for g in score.groups] == [("intro",), ("verse",), ("chorus",), ("interlude",)]
    assert [g.bar_count for g in score.groups] == [2, 4, 4, 2]
    assert [g.start_q for g in score.groups] == [0, 8, 24, 40]
    vocal = notes(score.vocal)
    assert (11, 76, 3) in vocal  # e8- | e16 tied across the barline
    assert (16, 71, 3) in vocal  # B16-B8 tied inside a bar
    assert (32, 76, 8) in vocal  # e32- | e32
    assert notes(score.ins)[-4:] == [(44, 72, 1), (45, 74, 1), (46, 76, 1), (47, 79, 1)]
    assert {c.symbol for c in score.chords} == {"Dm7", "G7", "Cmaj7"}


# --- One failing example per rule -------------------------------------------------------------

BAD_CASES = [
    # Header
    pytest.param(header_with(0, "X:2"), "x:1", id="x-not-1"),
    pytest.param(header_with(1, "T:My Song"), "title", id="title-not-empty"),
    pytest.param(header_with(2, "M:C"), "meter", id="meter-common-time"),
    pytest.param(header_with(2, "M:4/3"), "meter", id="meter-denominator"),
    pytest.param(header_with(3, "L:1/12"), "unit", id="unit-not-power-of-two"),
    pytest.param(header_with(3, "L:1/2048"), "unit", id="unit-too-fine"),
    pytest.param(header_with(4, "Q:1/8=120"), "tempo", id="tempo-not-quarter"),
    pytest.param(header_with(4, "Q:1/4=0"), "tempo", id="tempo-zero"),
    pytest.param(
        header_with(5, 'V:Vocal clef=treble name="Vocal Melody" snm="Vocal"'), "voice",
        id="vocal-header-not-exact",
    ),
    pytest.param(
        header_with(6, 'V: Ins clef=treble name="Ins" snm="Inst."'), "voice",
        id="ins-header-not-exact",
    ),
    pytest.param(header_with(7, "K:Ddor"), "key", id="key-mode"),
    pytest.param(header_with(7, "K:Amin"), "key", id="key-amin"),
    pytest.param(header_with(7, "K:none"), "key", id="key-none"),
    pytest.param("\n".join([*HEADER, "V: Vocal", "C16|", "V: Ins"]) + "\n", "12 lines",
                 id="too-short"),
    pytest.param(SCORE_ABC.replace("\n", "\r\n"), "carriage return", id="crlf"),
    # Body structure
    pytest.param(abc("% verse", "", *group("C16|")), "blank", id="blank-line"),
    pytest.param(abc("%%MIDI program 1", *GROUP), "directive", id="midi-directive"),
    pytest.param(abc("V: Vocal", "C16|", "w: la", "V: Ins", "Z|"), "lyric", id="w-line"),
    pytest.param(abc(*GROUP, "W: words"), "lyric", id="W-line"),
    pytest.param(abc("V: Bass", "C16|", "V: Ins", "Z|"), "voice", id="other-voice"),
    pytest.param(abc(*GROUP, "% outro"), "comment", id="dangling-comment"),
    pytest.param(abc("%verse", *group("C16|")), "comment", id="comment-without-space"),
    pytest.param(abc("V: Vocal", "C16|", "V: Vocal", "C16|"), "v: ins", id="missing-ins"),
    pytest.param(abc("V: Vocal", "C16|", "% verse", "V: Ins", "Z|"), "v: ins",
                 id="comment-between-voices"),
    pytest.param(abc("V: Vocal", "Q:1/4=100", "C16|", "V: Ins", "Z|"), "field",
                 id="field-in-body"),
    pytest.param(abc("V: Vocal", "C16|", "M:3/4", "V: Ins", "Z|"), "right after",
                 id="misplaced-meter-line"),
    pytest.param(one("C16|C16|C16|C16|C16|", "Z4|Z|"), "measures", id="five-measures"),
    pytest.param(one("C16|D16|", "Z|"), "measures", id="measure-count-mismatch"),
    pytest.param(abc("V: Vocal", "M:3/4", "C12|", "V: Ins", "z12|"), "change",
                 id="meter-change-in-one-voice"),
    pytest.param(abc("V: Vocal", "K:G", "C16|", "V: Ins", "K:D", "Z|"), "change",
                 id="key-changes-differ"),
    pytest.param(abc("V: Vocal", "M:3/4", "M:3/4", "C12|", "V: Ins", "M:3/4", "z12|"), "once",
                 id="two-meter-lines"),
    pytest.param(abc("V: Vocal", "K:G", "[K:G]C16|", "V: Ins", "K:G", "Z|"), "once",
                 id="line-and-inline-key-change"),
    pytest.param(one("C8[K:G]C8|"), "start of a group", id="key-change-mid-group"),
    pytest.param(abc("V: Vocal", "K:Dmix", "C16|", "V: Ins", "K:Dmix", "Z|"), "key",
                 id="body-key-mode"),
    # Barlines and bar sums
    pytest.param(one("C16"), "end with", id="no-final-barline"),
    pytest.param(one("C16||D16|", "Z2|"), "barline", id="double-barline"),
    pytest.param(one("|:C16|"), "barline", id="repeat-start"),
    pytest.param(one("C16:|"), "barline", id="repeat-end"),
    pytest.param(one("C16|]"), "barline", id="final-barline"),
    pytest.param(one("[1C16|"), "barline", id="alternate-ending"),
    pytest.param(one("C12|"), "meter duration", id="underfull-bar"),
    pytest.param(one("C16C4|"), "meter duration", id="overfull-bar"),
    # Notes, rests, ties
    pytest.param(one("C10C6|"), "multiplier", id="duration-not-allowed"),
    pytest.param(one("C/2C12C3|"), "/", id="slash-duration"),
    pytest.param(one("C,'16|"), "octave", id="mixed-octave-marks"),
    pytest.param(one("C,,,,,,16|"), "range", id="pitch-out-of-range"),
    pytest.param(one("^z16|"), "rest", id="rest-with-accidental"),
    pytest.param(one("z,16|"), "rest", id="rest-with-octave-mark"),
    pytest.param(one("z8-z8|"), "rest", id="tied-rest"),
    pytest.param(one("C8-z8|"), "tie", id="tie-into-rest"),
    pytest.param(one("C16-|Z|", "Z2|"), "tie", id="tie-into-whole-rest"),
    pytest.param(one("C8-D8|"), "tie", id="tie-different-pitches"),
    pytest.param(one("C8-c8|"), "tie", id="tie-different-octave"),
    pytest.param(one("C16-|"), "tie", id="unresolved-tie-at-end"),
    pytest.param(one("^F16-|=F16|", "Z2|"), "accidental", id="tie-contradictory-accidental"),
    pytest.param(one('"C"Z|'), "whole-measure rest", id="whole-rest-with-chord"),
    pytest.param(abc("V: Vocal", "[K:G]Z|", "V: Ins", "[K:G]Z|"), "whole-measure rest",
                 id="whole-rest-with-key-change"),
    pytest.param(one("Z5|", "Z5|"), "whole-measure rest", id="whole-rest-z5"),
    pytest.param(one("x16|"), "unsupported", id="invisible-rest"),
    # Chord symbols
    pytest.param(one("C16|", '"C"z16|'), "belong in vocal", id="chord-in-ins"),
    pytest.param(one('""C16|'), "chord symbol", id="empty-chord"),
    pytest.param(one('"C C16|'), "chord symbol", id="unclosed-chord"),
    # Rejected syntax
    pytest.param(one("[CEG]16|"), "polyphonic", id="polyphonic-stack"),
    pytest.param(one("(3C4C4C4C4|"), "tuplet", id="tuplet"),
    pytest.param(one("{g}C16|"), "grace", id="grace-note"),
    pytest.param(one("(C8D8)|"), "slur", id="slur"),
    pytest.param(one("C8>D8|"), "broken rhythm", id="broken-rhythm"),
    pytest.param(one("!trill!C16|"), "decoration", id="decoration-bang"),
    pytest.param(one(".C16|"), "decoration", id="decoration-staccato"),
    pytest.param(one("~C16|"), "decoration", id="decoration-roll"),
]

UNSUPPORTED_CHORDS = [
    "Cmaj9", "Cadd9", "C9", "C11", "C13", "C5", "A7alt", "C:maj", "intro riff", "H7", "Cm/9", "c",
]


@pytest.mark.parametrize(("text", "keyword"), BAD_CASES)
def test_each_rule_has_a_failing_example(text: str, keyword: str) -> None:
    errors = check_yue2_abc(text)
    assert errors, "expected the checker to reject this score"
    assert keyword in errors[0].lower(), errors
    with pytest.raises(Yue2AbcError):
        parse_yue2_abc(text)


@pytest.mark.parametrize("symbol", UNSUPPORTED_CHORDS)
def test_unsupported_chord_symbols_fail(symbol: str) -> None:
    errors = check_yue2_abc(one(f'"{symbol}"C16|'))
    assert errors and "chord symbol" in errors[0].lower(), errors


@pytest.mark.parametrize(
    "symbol",
    ["C", "Cm", "Cdim", "Caug", "C7", "Cmaj7", "Cm7", "Cdim7", "Cm7b5", "Csus4", "Csus2", "C6",
     "Cm6", "C7sus4", "Cm(maj7)", "F#m7/C#", "Dbaug", "Gm6/Bb", "A7/E", "Bbbm", "C##7"],
)
def test_supported_chord_symbols_pass(symbol: str) -> None:
    assert check_yue2_abc(one(f'"{symbol}"C16|')) == []


# --- Errors -----------------------------------------------------------------------------------


def test_errors_name_the_line_bar_and_voice() -> None:
    [error] = check_yue2_abc(one("C12|"))
    assert error.startswith("line 10, bar 1 (Vocal):")
    assert "12 != meter duration 16" in error


def test_check_reports_every_broken_bar() -> None:
    errors = check_yue2_abc(one("C12|D12|", "Z2|"))
    assert len(errors) == 2
    assert "bar 1" in errors[0] and "bar 2" in errors[1]


def test_parse_raises_the_first_error_check_reports() -> None:
    text = one("C12|D12|", "Z2|")
    with pytest.raises(Yue2AbcError) as raised:
        parse_yue2_abc(text)
    assert str(raised.value) == check_yue2_abc(text)[0]
    assert isinstance(raised.value, ValueError)


# --- Pitch, accidentals, keys -----------------------------------------------------------------


def test_pitch_letters_and_octave_marks() -> None:
    assert pitches(one("C2c2C,2c'2c,2B,2b2C'2|")) == [60, 72, 48, 84, 60, 59, 83, 72]


def test_accidental_spellings() -> None:
    assert pitches(one("^C4__D4_E4^^F4|")) == [61, 60, 63, 67]


def test_accidental_carries_by_letter_across_octaves() -> None:
    # Not standard ABC: after ^F, the f an octave up is sharp too (78, not 77).
    assert pitches(one("^F4f4F4f4|")) == [66, 78, 66, 78]


def test_key_signature_applies() -> None:
    assert pitches(one("F4f4G4c4|", key="G")) == [66, 78, 67, 72]
    assert pitches(one("B4E4A4D4|", key="Bbm")) == [70, 63, 68, 61]
    assert pitches(one("B4E4A4D4|", key="Am")) == [71, 64, 69, 62]


def test_natural_cancels_the_key_until_the_barline() -> None:
    assert pitches(one("=F8F8|F16|", "Z2|", key="G")) == [65, 65, 66]


def test_accidentals_reset_at_the_barline() -> None:
    assert pitches(one("^F16|F16|", "Z2|")) == [66, 65]


# --- Ties -------------------------------------------------------------------------------------


def test_tie_merges_within_a_bar() -> None:
    assert notes(parse_yue2_abc(one("C8-C8|")).vocal) == [(0, 60, 4)]


def test_tie_merges_across_a_barline() -> None:
    score = parse_yue2_abc(one("E8G8-|G16|", "Z2|"))
    assert notes(score.vocal) == [(0, 64, 2), (2, 67, 6)]


def test_tie_merges_across_groups() -> None:
    score = parse_yue2_abc(abc(*group("C16-|"), *group("C16|")))
    assert notes(score.vocal) == [(0, 60, 8)]


def test_unmarked_tied_continuation_keeps_the_tied_pitch() -> None:
    # The barline resets the sharp, but the tied F stays F#.
    assert notes(parse_yue2_abc(one("^F16-|F16|", "Z2|")).vocal) == [(0, 66, 8)]
    assert notes(parse_yue2_abc(one("^F16-|^F16|", "Z2|")).vocal) == [(0, 66, 8)]


def test_chord_changes_inside_a_held_note() -> None:
    score = parse_yue2_abc(one('"C"E8-"Am7"E8|'))
    assert notes(score.vocal) == [(0, 64, 4)]
    assert chords(score.chords) == [(0, "C"), (2, "Am7")]


def test_chord_on_a_rest() -> None:
    score = parse_yue2_abc(one('"Am7"z16|'))
    assert score.vocal == ()
    assert chords(score.chords) == [(0, "Am7")]


# --- Whole-measure rests and groups -----------------------------------------------------------


def test_z2_counts_as_two_measures() -> None:
    score = parse_yue2_abc(one("C16|D16|", "Z2|"))
    assert score.groups[0].bar_count == 2
    assert score.ins == ()


def test_z_rests_in_both_voices() -> None:
    score = parse_yue2_abc(one("Z4|", "Z3|Z|"))
    assert score.groups[0].bar_count == 4
    assert (score.vocal, score.ins) == ((), ())


def test_whole_rest_then_notes() -> None:
    score = parse_yue2_abc(one("C16|D16|", "Z|c16|"))
    assert notes(score.ins) == [(4, 72, 4)]


def test_labels_collect_before_their_group() -> None:
    score = parse_yue2_abc(abc("% intro", "% verse", *group("C16|"), *group("D16|")))
    assert [g.labels for g in score.groups] == [("intro", "verse"), ()]


def test_inline_key_change_at_group_start() -> None:
    text = abc(
        *group("F16|"),
        "V: Vocal", "[K:G]F16|", "V: Ins", "K:G", "z16|",
        *group("F16|"),
    )
    assert pitches(text) == [65, 66, 66]


def test_key_change_lines() -> None:
    text = abc(*group("F16|"), "V: Vocal", "K:G", "F16|", "V: Ins", "K:G", "Z|")
    assert pitches(text) == [65, 66]


def test_inline_meter_change_at_group_start() -> None:
    text = abc(*group("C16|"), "V: Vocal", "[M:3/4]C12|", "V: Ins", "M:3/4", "z12|")
    score = parse_yue2_abc(text)
    assert [(g.start_q, g.meter, g.bar_count) for g in score.groups] == [
        (0, (4, 4), 1), (4, (3, 4), 1),
    ]


def test_onsets_accumulate_across_groups_and_meter_changes() -> None:
    text = abc(
        "% verse", *group("C16|D16|", "Z2|"),
        "% bridge", "V: Vocal", "M:3/4", "E12|", "V: Ins", "M:3/4", "Z|",
        "% outro", "V: Vocal", "M:4/4", "F16|", "V: Ins", "M:4/4", "c16|",
    )
    score = parse_yue2_abc(text)
    assert [g.start_q for g in score.groups] == [0, 8, 11]
    assert [g.meter for g in score.groups] == [(4, 4), (3, 4), (4, 4)]
    assert score.meter == (4, 4)
    assert notes(score.vocal) == [(0, 60, 4), (4, 62, 4), (8, 64, 3), (11, 65, 4)]
    assert notes(score.ins) == [(11, 72, 4)]
