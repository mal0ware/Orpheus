"""The owner's chord-chart notation → timed chords (spec 2026-10-24 §2).

    Am | Am | Fmaj7 | C/E (x2)      one chord per bar, a slash chord, the line twice
    Am / Fmaj7 | C / E (x4)         " / " splits a bar evenly
    F | C | G | Am | F | C | E---   "E---" holds to the end of the section
    Fmaj7 | C | E | E (stop)        "(stop)" is a note on bar 4

Anything that is not a chord, a bar line, a split, a hold or a repeat becomes a note: free text
from the first unreadable word to the end of the line is kept whole, never guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from orpheus_mcp.models import ChordEvent

# Root, then a quality spelled from known parts (so words like "EEAm" or "but" are not chords),
# then an optional slash bass.
_QUALITY = r"(?:maj|min|dim|aug|sus|add|alt|m|M|[0-9]|#|b|\+|°|ø|\(|\))*"
_CHORD = re.compile(rf"^[A-G](?:#|b)?{_QUALITY}(?:/[A-G](?:#|b)?)?$")
_TOKEN = re.compile(r"(?:(?<=\s)|^)\([^)]*\)|\S+")
_REPEAT = re.compile(r"^[x×]\s*(\d+)$", re.IGNORECASE)
_HOLD = re.compile(r"^(.+?)(-{2,})$")


@dataclass
class ChartResult:
    chords: list[ChordEvent]
    bars: int
    notes: list[str] = field(default_factory=list)


def _bar_ranges(bars: list[int]) -> str:
    return f"bar {bars[0]}" if len(bars) == 1 else f"bars {bars[0]}-{bars[-1]}"


def parse_chord_chart(
    chart: str, *, bars: int | None = None, meter: tuple[int, int] = (4, 4)
) -> ChartResult:
    """Chords with 1-based bar/beat positions in quarter-note beats. ``bars`` is the section's
    length from the sheet: a held last chord fills up to it, missing bars are reported as having
    no chord, and a chart longer than ``bars`` is reported (and kept)."""
    bar_beats = meter[0] * 4.0 / meter[1]
    tokens = _TOKEN.findall(chart.replace("|", " | "))
    per_bar: list[list[tuple[str, bool]]] = [[]]
    bar_notes: list[tuple[int, str]] = []
    notes: list[str] = []
    repeat = 1

    def current_bar() -> int:
        return len(per_bar) if per_bar[-1] or len(per_bar) == 1 else len(per_bar) - 1

    for i, tok in enumerate(tokens):
        if tok.startswith("(") and tok.endswith(")") and not _CHORD.match(tok):
            inner = tok[1:-1].strip()
            rep = _REPEAT.match(inner)
            if rep:
                repeat = int(rep.group(1))
            elif inner:
                bar_notes.append((current_bar(), inner))
            continue
        if tok == "|":
            if per_bar[-1]:
                per_bar.append([])
            continue
        if tok == "/":
            continue
        if tok in ("...", "…"):
            notes.append(f"'...' after bar {current_bar()} (an unspecified repeat) not expanded")
            continue
        if re.fullmatch(r"-+", tok):
            if per_bar[-1]:
                sym, _ = per_bar[-1][-1]
                per_bar[-1][-1] = (sym, True)
            continue
        hold = False
        held = _HOLD.match(tok)
        if held and _CHORD.match(held.group(1)):
            tok, hold = held.group(1), True
        if tok.startswith("/") and _CHORD.match(tok[1:]):
            tok = tok[1:]  # "F /C": the split, then C
        elif tok.endswith("/") and _CHORD.match(tok[:-1]):
            tok = tok[:-1]  # "F/ C"
        if not _CHORD.match(tok):
            notes.append(" ".join(tokens[i:]).strip())  # free text: keep the rest whole
            break
        per_bar[-1].append((tok, hold))

    if not per_bar[-1]:
        per_bar.pop()
    if not per_bar:
        raise ValueError(f"no chords in chart: {chart!r}")
    one_pass = len(per_bar)
    all_bars = per_bar * repeat

    chords: list[ChordEvent] = []
    for bar_index, bar in enumerate(all_bars, start=1):
        if not bar:
            continue
        each = bar_beats / len(bar)
        for k, (sym, _hold) in enumerate(bar):
            chords.append(ChordEvent(symbol=sym, bar=bar_index, beat=1.0 + k * each, beats=each))

    total = len(all_bars)
    last_held = bool(all_bars[-1]) and all_bars[-1][-1][1]
    out_notes = [f"bar {b}: {text}" for b, text in bar_notes if b <= one_pass]
    if bars is not None and total < bars:
        if last_held and chords:
            last = chords[-1]
            chords[-1] = last.model_copy(update={"beats": last.beats + (bars - total) * bar_beats})
        else:
            out_notes.append(f"{_bar_ranges(list(range(total + 1, bars + 1)))}: no chord")
        total = bars
    elif bars is not None and total > bars:
        out_notes.append(f"chart has {total} bars, section says {bars}")
    return ChartResult(chords=chords, bars=total, notes=out_notes + notes)
