"""YuE2 score → LeadSheet (spec 2026-10-24 §5; reader decision 2026-10-24).

Notes and chords come from Orpheus's YuE2-rules parser (``yue2_check.parse_yue2_abc``), which
reads accidentals the way YuE2 does: scores YuE2 writes itself rely on that rule, and standard
ABC readers (music21) would misread them. The tests have music21 re-read Orpheus's own exports
as an independent check. Lyrics are not in the score: syllables come from the sidecar and lines
from the request's lyrics text.
"""

from __future__ import annotations

import re
from fractions import Fraction

from orpheus_mcp.models import ChordEvent, LeadSheet, LeadSheetSection, SectionLabel, SheetNote
from orpheus_mcp.notation.leadsheet import derive_label
from orpheus_mcp.notation.yue2_check import Yue2AbcError, parse_yue2_abc

_KNOWN_LABELS: set[str] = {"intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge",
                           "interlude", "solo", "instrumental", "outro"}
_TAG = re.compile(r"^\[([^\]]+)\]\s*$")


def _label(text: str, warnings: list[str]) -> SectionLabel:
    low = text.strip().lower()
    if low in _KNOWN_LABELS:
        return low  # type: ignore[return-value]
    derived = derive_label(low)
    if derived is None:
        warnings.append(f"unknown section label {text!r}: read as a verse")
        return "verse"
    return derived


def _lyric_blocks(lyrics: str | None) -> list[tuple[str, list[str]]]:
    """'[Verse]\\nline\\n\\n[Chorus]\\nline' -> [("verse", ["line"]), ("chorus", ["line"])]."""
    blocks: list[tuple[str, list[str]]] = []
    for raw in (lyrics or "").splitlines():
        line = raw.strip()
        tag = _TAG.match(line)
        if tag:
            blocks.append((tag.group(1).strip().lower(), []))
        elif line and blocks:
            blocks[-1][1].append(line)
    return blocks


def import_yue2(
    abc: str, sidecar: list[dict] | None = None, *, lyrics: str | None = None,
    title: str | None = None,
) -> tuple[LeadSheet, list[str]]:
    """A LeadSheet from a YuE2 score. Raises ``ValueError`` (with the checker's message) for a
    score outside the dialect."""
    abc = abc.replace('\r\n', '\n').replace('\r', '\n')  # files saved on Windows
    try:
        score = parse_yue2_abc(abc)
    except Yue2AbcError as exc:
        raise ValueError(f"not a valid YuE2 score: {exc}") from exc
    warnings: list[str] = []
    num, den = score.meter
    bar_q = Fraction(num * 4, den)
    if any(g.meter != score.meter for g in score.groups):
        warnings.append("the score changes meter; the lead sheet keeps the header meter only")

    # Sections: a labelled group starts one; unlabelled groups continue the previous.
    spans: list[list] = []  # [label, start_q, bars]
    for g in score.groups:
        if g.labels or not spans:
            spans.append([_label(g.labels[0] if g.labels else "verse", warnings), g.start_q, 0])
        spans[-1][2] += g.bar_count
    ends = [s[1] + s[2] * bar_q for s in spans]

    syllable_of: dict[int, str] = {}
    for entry in sidecar or []:
        indices = entry.get("note_indices") or []
        for k, idx in enumerate(indices):
            syllable_of[int(idx)] = str(entry.get("syllable") or "") if k == 0 else "_"

    def section_of(onset: Fraction) -> int:
        return next(i for i, end in enumerate(ends) if onset < end)

    melody: list[list[SheetNote]] = [[] for _ in spans]
    for i, n in enumerate(score.vocal):
        s = section_of(n.onset_q)
        melody[s].append(SheetNote(pitch=n.pitch, start_beat=float(n.onset_q - spans[s][1]),
                                   duration_beats=float(n.duration_q),
                                   syllable=syllable_of.get(i)))
    riff: list[list[SheetNote]] = [[] for _ in spans]
    for n in score.ins:
        s = section_of(n.onset_q)
        riff[s].append(SheetNote(pitch=n.pitch, start_beat=float(n.onset_q - spans[s][1]),
                                 duration_beats=float(n.duration_q)))
    chords: list[list[ChordEvent]] = [[] for _ in spans]
    for k, c in enumerate(score.chords):
        s = section_of(c.onset_q)
        nxt = score.chords[k + 1].onset_q if k + 1 < len(score.chords) else ends[s]
        rel = c.onset_q - spans[s][1]
        chords[s].append(ChordEvent(symbol=c.symbol, bar=int(rel // bar_q) + 1,
                                    beat=float(rel % bar_q) + 1.0,
                                    beats=float(min(nxt, ends[s]) - c.onset_q)))

    blocks = _lyric_blocks(lyrics)
    counts: dict[str, int] = {}
    sections = []
    for i, (label, _start, bars) in enumerate(spans):
        counts[label] = counts.get(label, 0) + 1
        lines: list[str] = []
        if blocks:
            tag, lines = blocks.pop(0)
            if tag.replace(" ", "-") not in (label, "interlude"):
                warnings.append(f"lyrics block [{tag}] paired with section {label!r}")
        sections.append(LeadSheetSection(
            name=f"{label.capitalize()} {counts[label]}", label=label, bars=bars,
            chords=chords[i], melody_notes=melody[i], riff=riff[i],
            riff_to_ins=bool(riff[i]), lyrics=lines,
        ))
    key = score.key
    minor = key.endswith("m")
    sheet = LeadSheet(title=title, key=key[:-1] if minor else key,
                      mode="minor" if minor else "major",
                      meter=score.meter, tempo_bpm=score.tempo_bpm, sections=sections)
    return sheet, warnings
