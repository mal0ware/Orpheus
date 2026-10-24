"""The owner's riff tab → notes (spec 2026-10-24 §2).

    Am   | A0 -  A0 -  A0 -  A0 -  |        open A on the beat, held to the next eighth
    C/E  | A3 -  -  -  E0 -  -  -  |  (x4)  C for half a bar, then E; the line four times
    stop | -  -  -  -  -  -  -  -  |        a silent bar

Each line is ``label | slots | slots |``: text before the first ``|`` is a label, each bar's
slots divide it evenly, ``<string><fret>`` plays a note, ``-`` holds the previous note within the
bar, and a bar that starts with ``-`` is silent until its first note. Standard 4-string bass
tuning unless ``tuning="guitar"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from orpheus_mcp.models import SheetNote

TUNINGS: dict[str, dict[str, int]] = {
    "bass": {"E": 28, "A": 33, "D": 38, "G": 43},  # E1 A1 D2 G2
    "guitar": {"E": 40, "A": 45, "D": 50, "G": 55, "B": 59, "e": 64},  # E2 A2 D3 G3 B3 E4
}
_SLOT = re.compile(r"^([A-Ga-g])(\d{1,2})$")
_REPEAT = re.compile(r"\(\s*[x×]\s*(\d+)\s*\)\s*$", re.IGNORECASE)


@dataclass
class TabResult:
    notes: list[SheetNote]
    bars: int
    labels: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_riff_tab(
    tab: str, *, meter: tuple[int, int] = (4, 4), tuning: str = "bass"
) -> TabResult:
    """Notes in quarter-note beats from the start of the tab; one line is one or more bars."""
    strings = TUNINGS[tuning]
    bar_beats = meter[0] * 4.0 / meter[1]
    notes: list[SheetNote] = []
    labels: list[str] = []
    warnings: list[str] = []
    bar_no = 0

    for raw in tab.splitlines():
        line = raw.strip()
        if "|" not in line:
            continue
        repeat = 1
        rep = _REPEAT.search(line)
        if rep:
            repeat = int(rep.group(1))
            line = line[: rep.start()].strip()
        label, _, rest = line.partition("|")
        bars = [b.split() for b in rest.split("|") if b.strip()]
        if not bars:
            continue
        labels.append(" ".join(label.split()))
        for _ in range(repeat):
            for slots in bars:
                start = bar_no * bar_beats
                each = bar_beats / len(slots)
                current: SheetNote | None = None
                for k, slot in enumerate(slots):
                    if slot == "-":
                        if current is not None:
                            current.duration_beats += each
                        continue
                    m = _SLOT.match(slot)
                    if not m or m.group(1) not in strings:
                        warnings.append(f"bar {bar_no + 1}: unknown slot {slot!r} (left silent)")
                        current = None
                        continue
                    current = SheetNote(
                        pitch=strings[m.group(1)] + int(m.group(2)),
                        start_beat=start + k * each,
                        duration_beats=each,
                    )
                    notes.append(current)
                bar_no += 1

    if bar_no == 0:
        raise ValueError(f"no bars in tab: {tab!r}")
    return TabResult(notes=notes, bars=bar_no, labels=labels, warnings=warnings)
