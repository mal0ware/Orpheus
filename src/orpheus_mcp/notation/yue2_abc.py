"""LeadSheet → YuE2's native ABC dialect (spec 2026-10-24 §4).

YuE2 (an open music model, run here through yue2.cpp) plans a song as an ABC score in a narrow
dialect: a fixed 8-line header, exactly two monophonic voices (``V: Vocal`` with the melody and
the chord symbols, ``V: Ins`` with an instrument line or rests), sections as ``% label`` comments,
groups of 1-4 bars, durations from a fixed set, ties instead of tuplets or dotted values, and
chords from 15 qualities. Rules: docs/pins/2026-10-yue2-backend.md. This writer is Orpheus's own
MIT code; no YuE code is used.

Accidentals are written so YuE2 (accidentals carry by LETTER across octaves within a bar) and
standard ABC readers such as music21 (per letter AND octave) read the same pitch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from orpheus_mcp.models import LeadSheet, SheetNote

MAJOR_KEYS = {"Cb": -7, "Gb": -6, "Db": -5, "Ab": -4, "Eb": -3, "Bb": -2, "F": -1, "C": 0,
              "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7}
MINOR_KEYS = {"Abm": -7, "Ebm": -6, "Bbm": -5, "Fm": -4, "Cm": -3, "Gm": -2, "Dm": -1, "Am": 0,
              "Em": 1, "Bm": 2, "F#m": 3, "C#m": 4, "G#m": 5, "D#m": 6, "A#m": 7}
ALLOWED_QUALITIES = ("", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "m7b5", "sus4", "sus2",
                     "6", "m6", "7sus4", "m(maj7)")
PIECES = (48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1)  # allowed duration multipliers, largest first
LYRIC_TAGS = {"intro": "[Intro]", "verse": "[Verse]", "pre-chorus": "[Pre-Chorus]",
              "chorus": "[Chorus]", "post-chorus": "[Post-Chorus]", "bridge": "[Bridge]",
              "interlude": "[Interlude]", "solo": "[Interlude]", "instrumental": "[Interlude]",
              "outro": "[Outro]"}
HEADER_VOICES = ('V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
                 'V: Ins clef=treble name="Ins Melody" snm="Inst."')

_LETTERS = "CDEFGAB"
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACC = {-2: "__", -1: "_", 0: "=", 1: "^", 2: "^^"}
# Chromatic spellings by interval from the tonic: (scale degree 0-6, alteration vs. the key).
_CHROMATIC = {
    "major": {1: (1, -1), 3: (2, -1), 6: (3, 1), 8: (5, -1), 10: (6, -1)},
    "minor": {1: (1, -1), 4: (2, 1), 6: (3, 1), 9: (5, 1), 11: (6, 1)},
}


def _pc(name: str) -> int:
    return (_LETTER_PC[name[0].upper()] + name.count("#") - name.count("b")) % 12


def yue2_key(tonic: str, mode: str) -> str:
    """The allowed YuE2 key name for a tonic and mode, preferring the written spelling and
    otherwise the enharmonic with fewer accidentals ("A#" major -> "Bb", "Db" minor -> "C#m")."""
    table = MINOR_KEYS if mode == "minor" else MAJOR_KEYS
    want = tonic + ("m" if mode == "minor" else "")
    if want in table:
        return want
    pc = _pc(tonic)
    same = [k for k in table if _pc(k.rstrip("m")) == pc]
    return min(same, key=lambda k: abs(table[k]))


def _key_signature(key: str) -> dict[str, int]:
    fifths = (MINOR_KEYS if key.endswith("m") else MAJOR_KEYS)[key]
    sig = dict.fromkeys(_LETTERS, 0)
    for letter in ("FCGDAEB"[:fifths] if fifths > 0 else "BEADGCF"[:-fifths]):
        sig[letter] = 1 if fifths > 0 else -1
    return sig


class _Speller:
    """Letter + alteration for every MIDI pitch in one key."""

    def __init__(self, key: str, mode: str) -> None:
        self.sig = _key_signature(key)
        tonic = key.rstrip("m")
        self.tonic_pc = _pc(tonic)
        start = _LETTERS.index(tonic[0])
        self.degrees = [_LETTERS[(start + d) % 7] for d in range(7)]
        self.mode = mode
        self.diatonic = {(_LETTER_PC[ltr] + self.sig[ltr]) % 12: (ltr, self.sig[ltr])
                         for ltr in _LETTERS}

    def spell(self, pitch: int) -> tuple[str, int, int]:
        """(letter, alteration, octave) with C4 = 60."""
        pc = pitch % 12
        if pc in self.diatonic:
            letter, alt = self.diatonic[pc]
        else:
            degree, offset = _CHROMATIC[self.mode][(pc - self.tonic_pc) % 12]
            letter = self.degrees[degree]
            alt = self.sig[letter] + offset
        natural = pitch - alt
        return letter, alt, natural // 12 - 1


def _note_text(letter: str, octave: int) -> str:
    if octave >= 5:
        return letter.lower() + "'" * (octave - 5)
    return letter + "," * (4 - octave)


# --------------------------------------------------------------------------- chords

_SYMBOL = re.compile(r"^([A-G](?:#|b)?)(.*?)(?:/([A-G](?:#|b)?))?$")
_QUALITY_TABLE = {
    "": "", "M": "", "maj": "", "ma": "", "min": "m", "mi": "m", "-": "m", "m": "m",
    "dim": "dim", "o": "dim", "aug": "aug", "+": "aug", "7": "7", "dom7": "7",
    "maj7": "maj7", "M7": "maj7", "ma7": "maj7", "Δ": "maj7", "Δ7": "maj7",
    "m7": "m7", "min7": "m7", "mi7": "m7", "-7": "m7", "dim7": "dim7", "o7": "dim7", "°7": "dim7",
    "°": "dim", "m7b5": "m7b5", "min7b5": "m7b5", "m7(b5)": "m7b5", "ø": "m7b5", "ø7": "m7b5",
    "sus4": "sus4", "sus": "sus4", "sus2": "sus2", "6": "6", "m6": "m6", "min6": "m6",
    "7sus4": "7sus4", "7sus": "7sus4", "m(maj7)": "m(maj7)", "mmaj7": "m(maj7)",
    "mM7": "m(maj7)", "minmaj7": "m(maj7)", "5": "", "add9": "", "add2": "", "add4": "",
    "2": "", "6/9": "6", "69": "6", "m6/9": "m6", "madd9": "m",
}


def _reduce_quality(q: str) -> str:
    if q in _QUALITY_TABLE:
        return _QUALITY_TABLE[q]
    if q.startswith(("maj", "M", "ma", "Δ")):
        return "maj7"  # maj9, maj11, maj13, maj7#11, M9
    if q.startswith(("min", "m", "-")):
        return "m7" if re.search(r"7|9|11|13", q) else ("m6" if "6" in q else "m")
    if q.startswith(("dim", "o", "°")):
        return "dim7" if "7" in q else "dim"
    if q.startswith(("aug", "+")):
        return "aug"
    if q.startswith("sus"):
        return "sus2" if "2" in q else "sus4"
    if q[:1].isdigit():
        return "7sus4" if "sus" in q else "7"  # 9, 11, 13, 7#9, 7b9, 7alt
    return ""


def reduce_chord(symbol: str) -> tuple[str, bool]:
    """A chord symbol mapped onto YuE2's 15 qualities; (symbol, was_reduced)."""
    m = _SYMBOL.match(symbol.strip())
    if not m:
        return symbol, False
    root, quality, bass = m.groups()
    q = quality if quality in ALLOWED_QUALITIES else _reduce_quality(quality)
    out = root + q + (f"/{bass}" if bass else "")
    return out, out != symbol


# --------------------------------------------------------------------------- export


@dataclass
class Yue2Export:
    abc: str
    lyrics: str
    style: str | None
    sidecar: list[dict]
    reductions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Event:
    onset: int
    dur: int
    pitch: int | None
    syllable: str | None = None
    index: int | None = None  # sounding-note index in its voice
    section: str = ""


def _grid(meter: tuple[int, int]) -> int:
    """L: denominator: 16, or finer when the meter's beat is shorter (x/8 -> 32)."""
    den = 16
    while den < meter[1] * 4:
        den *= 2
    return den


def _events(
    notes: list[tuple[str, int, SheetNote]], upq: int, warnings: list[str], index: bool
) -> list[_Event]:
    out: list[_Event] = []
    for section, offset, n in sorted(notes, key=lambda x: x[1] + x[2].start_beat * upq):
        start_f = n.start_beat * upq
        end_f = (n.start_beat + n.duration_beats) * upq
        start, end = round(start_f), round(end_f)
        if abs(start_f - start) > 1e-6 or abs(end_f - end) > 1e-6:
            warnings.append(f"{section}: note at beat {n.start_beat:g} moved to the 1/"
                            f"{upq * 4} grid")
        end = max(end, start + 1)
        ev = _Event(offset + start, end - start, n.pitch, n.syllable, section=section)
        if out and out[-1].onset + out[-1].dur > ev.onset:
            warnings.append(f"{section}: overlapping notes at beat {n.start_beat:g}; the "
                            "earlier one is shortened (YuE2 voices are monophonic)")
            out[-1].dur = max(1, ev.onset - out[-1].onset)
            if out[-1].onset >= ev.onset:
                out.pop()
        out.append(ev)
    if index:
        for i, ev in enumerate(out):
            ev.index = i
    return out


def _with_rests(events: list[_Event], total: int) -> list[_Event]:
    out, t = [], 0
    for ev in events:
        if ev.onset > t:
            out.append(_Event(t, ev.onset - t, None))
        out.append(ev)
        t = ev.onset + ev.dur
    if t < total:
        out.append(_Event(t, total - t, None))
    return out


def _render_voice(
    events: list[_Event], chords: dict[int, str], bar_units: int, n_bars: int,
    speller: _Speller,
) -> list[str]:
    """One string per bar (each ending in "|"); whole-bar rests with no chord come back as "Z"."""
    cuts = {b * bar_units for b in range(1, n_bars)} | set(chords)
    bars: list[list[str]] = [[] for _ in range(n_bars)]
    rest_only = [True] * n_bars
    state_yue: dict[str, int] = {}
    state_std: dict[tuple[str, int], int] = {}
    current_bar = -1
    for ev in events:
        bounds = sorted({ev.onset, ev.onset + ev.dur} | {c for c in cuts
                                                          if ev.onset < c < ev.onset + ev.dur})
        segments = list(zip(bounds, bounds[1:], strict=False))
        for s_i, (a, b) in enumerate(segments):
            bar = a // bar_units
            if bar != current_bar:
                current_bar, state_yue, state_std = bar, {}, {}
            text = f'"{chords[a]}"' if a in chords else ""
            if a in chords:
                rest_only[bar] = False
            length = b - a
            parts = []
            while length:
                piece = next(p for p in PIECES if p <= length)
                parts.append(piece)
                length -= piece
            if ev.pitch is None:
                text += "".join(f"z{p if p > 1 else ''}" for p in parts)
            else:
                rest_only[bar] = False
                letter, alt, octave = speller.spell(ev.pitch)
                keyed = speller.sig[letter]
                mark = ""
                if (state_yue.get(letter, keyed) != alt
                        or state_std.get((letter, octave), keyed) != alt):
                    mark = _ACC[alt]
                state_yue[letter] = alt
                state_std[(letter, octave)] = alt
                last_segment = s_i == len(segments) - 1
                for p_i, piece in enumerate(parts):
                    tie = not (last_segment and p_i == len(parts) - 1)
                    text += (mark if p_i == 0 else "") + _note_text(letter, octave)
                    text += (str(piece) if piece > 1 else "") + ("-" if tie else "")
            bars[bar].append(text)
    return ["Z" if rest_only[i] else "".join(parts) for i, parts in enumerate(bars)]


def _join_group(bars: list[str]) -> str:
    out, i = [], 0
    while i < len(bars):
        if bars[i] == "Z":
            j = i
            while j < len(bars) and bars[j] == "Z" and j - i < 4:
                j += 1
            n = j - i
            out.append(f"Z{n if n > 1 else ''}|")
            i = j
        else:
            out.append(bars[i] + "|")
            i += 1
    return "".join(out)


def _lyric_line(notes: list[SheetNote]) -> str:
    words = ""
    for n in notes:
        s = n.syllable
        if not s or s == "_":
            continue
        words += s[:-1] if s.endswith("-") else s + " "
    return words.strip()


def export_yue2(sheet: LeadSheet) -> Yue2Export:
    """A resolved LeadSheet as a YuE2 request: ABC score, lyrics text, syllable sidecar."""
    warnings: list[str] = []
    key = yue2_key(sheet.key, sheet.mode)
    speller = _Speller(key, sheet.mode)
    den = _grid(sheet.meter)
    upq = den // 4
    bar_units = sheet.meter[0] * den // sheet.meter[1]

    vocal_notes, ins_notes, chord_at, reductions = [], [], {}, []
    bar_offset = 0
    section_bars: list[tuple[str, str, int, int]] = []  # (name, label, first bar, bars)
    for sec in sheet.sections:
        offset = bar_offset * bar_units
        section_bars.append((sec.name, sec.label or "verse", bar_offset, sec.bars))
        vocal_notes += [(sec.name, offset, n) for n in sec.melody_notes]
        if sec.riff_to_ins:
            ins_notes += [(sec.name, offset, n) for n in sec.riff]
        for c in sec.chords:
            at = offset + round(((c.bar - 1) * sheet.meter[0] * 4 / sheet.meter[1]
                                 + (c.beat - 1)) * upq)
            symbol, reduced = reduce_chord(c.symbol)
            if reduced:
                reductions.append(f"{c.symbol} -> {symbol}")
            chord_at[at] = symbol
        bar_offset += sec.bars
    total_bars = bar_offset
    total = total_bars * bar_units

    vocal = _with_rests(_events(vocal_notes, upq, warnings, index=True), total)
    ins = _with_rests(_events(ins_notes, upq, warnings, index=False), total)
    vocal_bars = _render_voice(vocal, chord_at, bar_units, total_bars, speller)
    ins_bars = _render_voice(ins, {}, bar_units, total_bars, speller)

    lines = ["X:1", "T:", f"M:{sheet.meter[0]}/{sheet.meter[1]}", f"L:1/{den}",
             f"Q:1/4={round(sheet.tempo_bpm)}", *HEADER_VOICES, f"K:{key}"]
    for _name, label, first, n in section_bars:
        for g, start in enumerate(range(first, first + n, 4)):
            stop = min(start + 4, first + n)
            if g == 0:
                lines.append(f"% {label}")
            lines += ["V: Vocal", _join_group(vocal_bars[start:stop]),
                      "V: Ins", _join_group(ins_bars[start:stop])]

    sidecar: list[dict] = []
    for ev in vocal:
        if ev.pitch is None or ev.syllable is None:
            continue
        if ev.syllable == "_" and sidecar:
            sidecar[-1]["note_indices"].append(ev.index)
            sidecar[-1]["pitches"].append(ev.pitch)
            continue
        sidecar.append({"voice": "Vocal", "note_index_base": 0, "section": ev.section,
                        "syllable": ev.syllable, "note_indices": [ev.index],
                        "pitches": [ev.pitch], "onset_q": ev.onset / upq,
                        "duration_q": ev.dur / upq})

    lyric_blocks = []
    for sec in sheet.sections:
        body = list(sec.lyrics) or ([_lyric_line(sec.melody_notes)]
                                    if any(n.syllable for n in sec.melody_notes) else [])
        lyric_blocks.append("\n".join([LYRIC_TAGS[sec.label or "verse"], *body]))
    return Yue2Export(abc="\n".join(lines) + "\n", lyrics="\n\n".join(lyric_blocks),
                      style=sheet.style, sidecar=sidecar, reductions=reductions,
                      warnings=warnings)
