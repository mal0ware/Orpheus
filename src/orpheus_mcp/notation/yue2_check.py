"""Checker and reader for YuE2's native ABC dialect.

YuE2 plans and reads songs as a small, fixed subset of ABC. A score is an exact 8-line header
(``X:1``, an empty ``T:``, ``M:``, ``L:``, ``Q:1/4=``, two fixed ``V:`` lines, ``K:``) followed by
groups of one to four measures: optional ``% label`` lines, a ``V: Vocal`` block (the sung melody
and the chord symbols) and a ``V: Ins`` block (one instrumental melody line, or ``Z`` rests).
Each bar fills its meter exactly from a fixed set of note lengths. Unlike standard ABC, an
accidental applies to its letter in every octave until the barline.

yue2.cpp tokenizes whatever ABC it is given without checking it, so Orpheus checks a score here
before rendering it. This module is an independent MIT implementation written from the
documented rules of the dialect. It contains no code from YuE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import NoReturn

__all__ = [
    "CHORD_QUALITIES",
    "DURATIONS",
    "INS_HEADER",
    "MAJOR_KEYS",
    "MINOR_KEYS",
    "VOCAL_HEADER",
    "AbcChord",
    "AbcGroup",
    "AbcNote",
    "Yue2AbcError",
    "Yue2Score",
    "check_yue2_abc",
    "parse_yue2_abc",
]

VOCAL_HEADER = 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"'
INS_HEADER = 'V: Ins clef=treble name="Ins Melody" snm="Inst."'

# Ordered by fifths from 7 flats to 7 sharps, so each minor sits under its relative major.
MAJOR_KEYS = ("Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#")
MINOR_KEYS = (
    "Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m",
    "A#m",
)
DURATIONS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
CHORD_QUALITIES = (
    "", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "m7b5", "sus4", "sus2", "6", "m6", "7sus4",
    "m(maj7)",
)

_ROOT = r"[A-G](?:bb|##|b|#)?"
_QUALITY = "|".join(re.escape(q) for q in sorted(CHORD_QUALITIES, key=len, reverse=True))
_CHORD_RE = re.compile(rf"{_ROOT}(?:{_QUALITY})(?:/{_ROOT})?")
_NOTE_RE = re.compile(r"(\^\^|\^|__|_|=)?([A-Ga-g])([,']*)(\d*)(-?)")
_REST_RE = re.compile(r"z(\d*)")
_WHOLE_REST_RE = re.compile(r"Z([2-4]?)")
_INLINE_RE = re.compile(r"\[([KM]):([^\]]*)\]")
_FIELD_RE = re.compile(r"[A-Za-z]:")
_BAD_BARLINE_RE = re.compile(r"\|\||\|:|:\||\|\]|\[\||::|\[\d|\|\d")
_METER_RE = re.compile(r"(\d+)/(\d+)")
_UNIT_RE = re.compile(r"L:1/(\d+)")
_TEMPO_RE = re.compile(r"Q:1/4=(\d+)")

_LETTER_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTALS = {"^^": 2, "^": 1, "=": 0, "_": -1, "__": -2}
_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"


@dataclass(frozen=True)
class AbcNote:
    """One sounding note, with tied notes merged."""

    onset_q: Fraction  # whole-score onset in quarter notes
    pitch: int  # MIDI; C = 60, c = 72, C, = 48
    duration_q: Fraction


@dataclass(frozen=True)
class AbcChord:
    """A chord symbol from the Vocal voice, as written."""

    onset_q: Fraction
    symbol: str


@dataclass(frozen=True)
class AbcGroup:
    """One Vocal/Ins group of 1-4 measures."""

    labels: tuple[str, ...]  # the "% ..." comment texts before this group
    bar_count: int  # measures, with Z/Z2/Z3/Z4 expanded
    start_q: Fraction
    meter: tuple[int, int]  # meter in force for the group


@dataclass(frozen=True)
class Yue2Score:
    """A score read from YuE2's ABC dialect."""

    meter: tuple[int, int]  # header M:
    unit: Fraction  # header L:, as a fraction of a whole note
    tempo_bpm: int  # header Q:1/4=N
    key: str  # header K:
    groups: tuple[AbcGroup, ...]
    vocal: tuple[AbcNote, ...]
    ins: tuple[AbcNote, ...]
    chords: tuple[AbcChord, ...]


class Yue2AbcError(ValueError):
    """A score breaks a rule of YuE2's ABC dialect."""


def parse_yue2_abc(text: str) -> Yue2Score:
    """Read a YuE2 score; raise Yue2AbcError naming the first rule it breaks."""
    scan = _Scan(text)
    score = scan.run()
    if scan.errors or score is None:
        raise Yue2AbcError(scan.errors[0])
    return score


def check_yue2_abc(text: str) -> list[str]:
    """Return the rules a YuE2 score breaks, first violation first; [] when it is valid."""
    scan = _Scan(text)
    scan.run()
    return list(scan.errors)


# --- internals --------------------------------------------------------------------------------


class _Stop(Exception):
    """Ends the scan after a structural error."""


class _BarError(Exception):
    """Abandons one bar."""


@dataclass
class _Tie:
    onset: Fraction
    pitch: int
    natural: int  # letter and octave without any accidental
    duration: Fraction
    lineno: int


@dataclass
class _Voice:
    name: str
    onset: Fraction = Fraction(0)
    tie: _Tie | None = None
    notes: list[AbcNote] = field(default_factory=list)


@dataclass
class _Block:
    voice: str
    vline: int  # line number of "V: ..."
    lineno: int  # line number of the music line
    music: str
    meter: tuple[int, int] | None
    key: str | None


def _is_power_of_two(n: int) -> bool:
    return 0 < n <= 1024 and n & (n - 1) == 0


def _meter_error(value: str) -> str | None:
    if value in ("C", "C|"):
        return f"meter M:{value} is not supported; write it as n/d, e.g. M:4/4"
    m = _METER_RE.fullmatch(value)
    if not m or int(m.group(1)) < 1 or not _is_power_of_two(int(m.group(2))):
        return (
            f"meter M:{value} is not supported; write n/d with d a power of two up to 1024, "
            "e.g. M:4/4 or M:5/8"
        )
    return None


def _parse_meter(value: str) -> tuple[int, int]:
    n, d = value.split("/")
    return int(n), int(d)


def _key_error(value: str) -> str | None:
    if value in MAJOR_KEYS or value in MINOR_KEYS:
        return None
    return (
        f"key {value!r} is not supported; use a major key ({' '.join(MAJOR_KEYS)}) or a minor "
        f"key ({' '.join(MINOR_KEYS)}), with no modes"
    )


def _key_signature(key: str) -> dict[str, int]:
    keys = MAJOR_KEYS if key in MAJOR_KEYS else MINOR_KEYS
    fifths = keys.index(key) - 7
    if fifths >= 0:
        return {letter: 1 for letter in _SHARP_ORDER[:fifths]}
    return {letter: -1 for letter in _FLAT_ORDER[:-fifths]}


def _multiplier(digits: str) -> int:
    value = int(digits) if digits else 1
    if value not in DURATIONS:
        raise _BarError(
            f"duration multiplier {value} is not allowed; use 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 "
            "or 48 and tie longer notes"
        )
    return value


def _unsupported(text: str, pos: int) -> str:
    """Explain the token at text[pos], which is not part of the dialect."""
    ch = text[pos]
    nxt = text[pos + 1 : pos + 2]
    if ch == "(":
        if nxt.isdigit():
            return "tuplets like (3 are not supported; write the rhythm on the straight grid"
        return "slurs ( ) are not supported"
    if ch == ")":
        return "slurs ( ) are not supported"
    if ch in "{}":
        return "grace notes { } are not supported"
    if ch in "<>":
        return "broken rhythm (> and <) is not supported; write the lengths out"
    if ch in "!+":
        return "decorations like !trill! are not supported"
    if ch in ".~HLMOPSTuv":
        return f"decorations like {ch!r} are not supported"
    if ch == "/":
        return "'/' durations are not supported; use whole multiples of the L: unit"
    if ch == "[":
        if nxt.isalpha() and text[pos + 2 : pos + 3] == ":":
            return f"inline field [{nxt}:...] is not supported; only [K:...] and [M:...]"
        return "polyphonic stacks like [CEG] are not supported; each voice is one melody line"
    if ch == "Z":
        return (
            "a Z whole-measure rest must be alone in its bar; it can't carry a chord, a key "
            "change or notes (use z rests instead)"
        )
    if ch in "^_=":
        end = pos
        while end < len(text) and text[end] in "^_=":
            end += 1
        if text[end : end + 1] == "z":
            return "a rest takes no accidental"
        return f"accidental {text[pos:end]!r} is malformed or has no note after it"
    if ch == "-":
        return "a tie '-' must follow a note directly"
    if ch in ",'":
        return "octave marks , and ' must follow a note letter"
    if ch.isdigit():
        return "a duration must follow a note or rest directly"
    token = text[pos:].split()[0][:12]
    return f"unsupported token {token!r}"


class _Scan:
    """One pass over a score, collecting errors."""

    def __init__(self, text: str) -> None:
        self.lines = text.split("\n")
        if self.lines and self.lines[-1] == "":
            self.lines.pop()  # a final newline is not a blank line
        self.errors: list[str] = []
        self.chords: list[AbcChord] = []
        self.unit = Fraction(1, 16)
        self.lineno = 0

    def run(self) -> Yue2Score | None:
        try:
            return self._score()
        except _Stop:
            return None

    # -- reporting

    def _fail(self, lineno: int, message: str) -> NoReturn:
        self.errors.append(f"line {lineno}: {message}")
        raise _Stop

    def _report(self, lineno: int, message: str) -> None:
        self.errors.append(f"line {lineno}: {message}")

    # -- structure

    def _score(self) -> Yue2Score:
        for lineno, line in enumerate(self.lines, 1):
            if "\r" in line:
                self._fail(lineno, "carriage return found; YuE2 scores use plain \\n line endings")
        meter, unit, tempo, key = self._header()
        if len(self.lines) < 12:
            self._fail(
                len(self.lines),
                "a score needs at least 12 lines: the 8 header lines and one Vocal/Ins group",
            )
        self.unit = unit
        vocal_voice, ins_voice = _Voice("Vocal"), _Voice("Ins")
        groups: list[AbcGroup] = []
        labels: list[str] = []
        group_meter, group_key = meter, key
        i = 8
        while i < len(self.lines):
            line = self._line(i, "a group")
            if line.startswith("%"):
                self._check_comment(i + 1, line)
                labels.append(line[2:].strip())
                i += 1
                continue
            vocal, i = self._block(i, "Vocal")
            ins, i = self._block(i, "Ins")
            if (vocal.meter, vocal.key) != (ins.meter, ins.key):
                self._report(
                    ins.vline,
                    "meter and key changes must be identical in both voices: Vocal has "
                    f"{_describe(vocal)}, Ins has {_describe(ins)}",
                )
            group_meter = vocal.meter or group_meter
            group_key = vocal.key or group_key
            start = vocal_voice.onset
            counts = [
                self._music(block, voice, group_meter, group_key)
                for block, voice in ((vocal, vocal_voice), (ins, ins_voice))
            ]
            if counts[0] != counts[1]:
                self._report(
                    ins.vline,
                    f"Vocal has {counts[0]} measures but Ins has {counts[1]}; a group needs the "
                    "same number in both voices",
                )
            for block, count in zip((vocal, ins), counts, strict=True):
                if not 1 <= count <= 4:
                    self._report(
                        block.lineno,
                        f"the {block.voice} line has {count} measures; a group holds 1 to 4",
                    )
            groups.append(AbcGroup(tuple(labels), counts[0], start, group_meter))
            labels = []
        if labels:
            self._fail(
                len(self.lines),
                f"comment {self.lines[-1]!r} at the end of the file is not followed by a group",
            )
        for voice in (vocal_voice, ins_voice):
            if voice.tie is not None:
                self._report(
                    voice.tie.lineno, f"the score ends on an unresolved tie in {voice.name}"
                )
        return Yue2Score(
            meter=meter,
            unit=unit,
            tempo_bpm=tempo,
            key=key,
            groups=tuple(groups),
            vocal=tuple(vocal_voice.notes),
            ins=tuple(ins_voice.notes),
            chords=tuple(self.chords),
        )

    def _header(self) -> tuple[tuple[int, int], Fraction, int, str]:
        lines = self.lines
        if len(lines) < 8:
            self._fail(
                len(lines) or 1, "the header must be 8 lines: X:, T:, M:, L:, Q:, two V:, K:"
            )
        errors: list[tuple[int, str]] = []
        if lines[0] != "X:1":
            errors.append((1, "the first line must be 'X:1'"))
        if lines[1] != "T:":
            errors.append((2, "the title must be empty: write exactly 'T:'"))
        meter_line = lines[2]
        meter_problem = (
            _meter_error(meter_line[2:])
            if meter_line.startswith("M:")
            else "expected the meter, M:<n>/<d>"
        )
        if meter_problem:
            errors.append((3, meter_problem))
        unit_match = _UNIT_RE.fullmatch(lines[3])
        if not unit_match or not _is_power_of_two(int(unit_match.group(1))):
            errors.append(
                (4, "expected the unit length L:1/<power of two up to 1024>, e.g. L:1/16")
            )
        tempo_match = _TEMPO_RE.fullmatch(lines[4])
        if not tempo_match or int(tempo_match.group(1)) < 1:
            errors.append((5, "expected the tempo Q:1/4=<positive beats per minute>"))
        if lines[5] != VOCAL_HEADER:
            errors.append((6, f"expected the Vocal voice header, exactly: {VOCAL_HEADER}"))
        if lines[6] != INS_HEADER:
            errors.append((7, f"expected the Ins voice header, exactly: {INS_HEADER}"))
        key_line = lines[7]
        key_problem = (
            _key_error(key_line[2:])
            if key_line.startswith("K:")
            else "expected the key, K:<key>"
        )
        if key_problem:
            errors.append((8, key_problem))
        if errors:
            for lineno, message in errors:
                self._report(lineno, message)
            raise _Stop
        assert unit_match and tempo_match
        return (
            _parse_meter(meter_line[2:]),
            Fraction(1, int(unit_match.group(1))),
            int(tempo_match.group(1)),
            key_line[2:],
        )

    def _line(self, i: int, expected: str) -> str:
        if i >= len(self.lines):
            self._fail(len(self.lines), f"the file ends where {expected} was expected")
        line = self.lines[i]
        if not line.strip():
            self._fail(i + 1, "blank lines are not allowed in a score")
        return line

    def _check_comment(self, lineno: int, line: str) -> None:
        if line.startswith("%%"):
            self._fail(lineno, f"directives such as %%MIDI are not allowed: {line!r}")
        if not line.startswith("% "):
            self._fail(lineno, f"a section comment must be '% <label>', with a space: {line!r}")

    def _misplaced(self, line: str, expected: str) -> str:
        if line.startswith("%%"):
            return f"directives such as %%MIDI are not allowed: {line!r}"
        if line.startswith("%"):
            return f"expected {expected}, got the comment {line!r}"
        if line.startswith(("w:", "W:")):
            return "lyric lines (w: and W:) are not supported; lyrics go in the request"
        if line.startswith("V:"):
            if line in ("V: Vocal", "V: Ins"):
                return f"expected {expected}, got {line!r}"
            return f"unknown voice line {line!r}; only 'V: Vocal' and 'V: Ins' are allowed"
        if line.startswith(("M:", "K:")):
            return f"{line[:2]} change lines must come right after 'V: Vocal' or 'V: Ins'"
        if _FIELD_RE.match(line):
            return f"field {line[:2]!r} is not allowed in the body"
        return f"expected {expected}, got {line!r}"

    def _block(self, i: int, voice: str) -> tuple[_Block, int]:
        """Read 'V: <voice>', its optional M:/K: changes and its music line."""
        expected = f"'V: {voice}'"
        line = self._line(i, expected)
        if line != f"V: {voice}":
            self._fail(i + 1, self._misplaced(line, expected))
        vline = i + 1
        changes: dict[str, str] = {}
        i += 1
        music_name = f"the {voice} music line"
        while True:
            line = self._line(i, music_name)
            if line.startswith("%") or not _FIELD_RE.match(line):
                break
            if line[0] not in "MK":
                self._fail(i + 1, self._misplaced(line, music_name))
            self._change(i + 1, voice, changes, line[0], line[2:])
            i += 1
        if line.startswith("%"):
            self._fail(i + 1, self._misplaced(line, music_name))
        pos = 0
        while True:
            while pos < len(line) and line[pos].isspace():
                pos += 1
            inline = _INLINE_RE.match(line, pos)
            if not inline:
                break
            self._change(i + 1, voice, changes, inline.group(1), inline.group(2))
            pos = inline.end()
        meter = _parse_meter(changes["M"]) if "M" in changes else None
        return _Block(voice, vline, i + 1, line, meter, changes.get("K")), i + 1

    def _change(
        self, lineno: int, voice: str, changes: dict[str, str], kind: str, value: str
    ) -> None:
        name = "meter" if kind == "M" else "key"
        if kind in changes:
            self._fail(lineno, f"more than one {name} change in the {voice} block; write it once")
        problem = _meter_error(value) if kind == "M" else _key_error(value)
        if problem:
            self._fail(lineno, problem)
        changes[kind] = value

    # -- music

    def _music(self, block: _Block, voice: _Voice, meter: tuple[int, int], key: str) -> int:
        """Read one music line; return its measure count."""
        line = block.music
        self.lineno = block.lineno
        bad = _BAD_BARLINE_RE.search(line)
        if bad:
            self._fail(
                block.lineno,
                f"unsupported barline {bad.group()!r}: only plain '|' is allowed (no repeats, "
                "alternate endings or double bars)",
            )
        stripped = line.rstrip()
        if not stripped.endswith("|"):
            self._fail(block.lineno, "a music line must end with '|'")
        measure_q = Fraction(4 * meter[0], meter[1])
        signature = _key_signature(key)
        count = 0
        for index, text in enumerate(stripped[:-1].split("|")):
            bar_start = voice.onset
            try:
                count += self._bar(text, index == 0, voice, signature, meter, measure_q)
            except _BarError as exc:
                self.errors.append(f"line {block.lineno}, bar {index + 1} ({voice.name}): {exc}")
                voice.tie = None
                voice.onset = bar_start + measure_q
                count += 1
        return count

    def _bar(
        self,
        text: str,
        first: bool,
        voice: _Voice,
        signature: dict[str, int],
        meter: tuple[int, int],
        measure_q: Fraction,
    ) -> int:
        """Read one bar; return how many measures it spans."""
        body = text.strip()
        if not body:
            raise _BarError("empty measure; write Z for a silent bar")
        whole = _WHOLE_REST_RE.fullmatch(body)
        if whole:
            if voice.tie is not None:
                raise _BarError("a tie can't go into a Z rest")
            measures = int(whole.group(1) or 1)
            voice.onset += measures * measure_q
            return measures
        if re.fullmatch(r"Z\d*", body):
            raise _BarError(f"whole-measure rest {body!r} is not allowed; use Z, Z2, Z3 or Z4")
        unit_q = self.unit * 4
        bar_start = voice.onset
        accidentals: dict[str, int] = {}
        units = 0
        started = False
        pos = 0
        while pos < len(text):
            ch = text[pos]
            if ch.isspace():
                pos += 1
                continue
            onset = bar_start + units * unit_q
            if ch == '"':
                end = text.find('"', pos + 1)
                if end < 0:
                    raise _BarError("unclosed chord symbol")
                symbol = text[pos + 1 : end]
                if voice.name != "Vocal":
                    raise _BarError(f"chord symbols belong in Vocal, not Ins (found {symbol!r})")
                if not _CHORD_RE.fullmatch(symbol):
                    raise _BarError(
                        f"chord symbol {symbol!r} is not supported; use a root A-G (with b, #, "
                        "bb or ##), one of the qualities "
                        f"{', '.join(repr(q) for q in CHORD_QUALITIES)} and an optional /bass"
                    )
                self.chords.append(AbcChord(onset, symbol))
                pos, started = end + 1, True
                continue
            inline = _INLINE_RE.match(text, pos)
            if inline:
                if not first or started:
                    raise _BarError("key and meter changes only go at the start of a group")
                pos = inline.end()  # already applied by _block
                continue
            note = _NOTE_RE.match(text, pos)
            if note:
                units += self._note(note, voice, onset, accidentals, signature)
                pos, started = note.end(), True
                continue
            rest = _REST_RE.match(text, pos)
            if rest:
                if voice.tie is not None:
                    raise _BarError("a tie can't go into a rest")
                units += _multiplier(rest.group(1))
                pos, started = rest.end(), True
                follow = text[pos : pos + 1]
                if follow and follow in ",'":
                    raise _BarError("a rest takes no octave mark")
                if follow == "-":
                    raise _BarError("a rest can't be tied")
                continue
            raise _BarError(_unsupported(text, pos))
        meter_units = Fraction(meter[0], meter[1]) / self.unit
        if units != meter_units:
            raise _BarError(
                f"bar duration {units} != meter duration {meter_units} "
                f"({meter[0]}/{meter[1]} at L:1/{self.unit.denominator})"
            )
        voice.onset = bar_start + measure_q
        return 1

    def _note(
        self,
        match: re.Match[str],
        voice: _Voice,
        onset: Fraction,
        accidentals: dict[str, int],
        signature: dict[str, int],
    ) -> int:
        """Read one note token; return its length in L units."""
        accidental, letter, marks, digits, tie = match.groups()
        written = match.group()
        if "," in marks and "'" in marks:
            raise _BarError(f"note {written!r} mixes , and ' octave marks")
        multiplier = _multiplier(digits)
        upper = letter.upper()
        octave = (1 if letter.islower() else 0) + marks.count("'") - marks.count(",")
        natural = 60 + _LETTER_STEPS[upper] + 12 * octave
        if accidental:
            accidentals[upper] = _ACCIDENTALS[accidental]  # every octave, until the barline
        pitch = natural + accidentals.get(upper, signature.get(upper, 0))
        if not 0 <= pitch <= 127:
            raise _BarError(f"note {written!r} is pitch {pitch}, out of the MIDI range 0-127")
        held = voice.tie
        if held is not None and pitch != held.pitch:
            if natural != held.natural:
                raise _BarError(
                    f"a tie must join equal pitches, but {written!r} differs from the note it is "
                    "tied from"
                )
            if accidental:
                raise _BarError(
                    f"the explicit accidental on {written!r} contradicts the note it is tied from"
                )
            pitch = held.pitch  # an unmarked continuation keeps the tied pitch
        duration = multiplier * self.unit * 4
        if held is not None:
            held.duration += duration
            if not tie:
                voice.notes.append(AbcNote(held.onset, held.pitch, held.duration))
                voice.tie = None
        elif tie:
            voice.tie = _Tie(onset, pitch, natural, duration, self.lineno)
        else:
            voice.notes.append(AbcNote(onset, pitch, duration))
        return multiplier


def _describe(block: _Block) -> str:
    parts = []
    if block.meter:
        parts.append(f"M:{block.meter[0]}/{block.meter[1]}")
    if block.key:
        parts.append(f"K:{block.key}")
    return " ".join(parts) or "no change"
