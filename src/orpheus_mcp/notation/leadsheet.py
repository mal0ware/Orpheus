"""Expand a LeadSheet written like the owner's sheet into timed chords, riff and melody
(spec 2026-10-24 §3). Pure; every problem comes back as a warning, never a silent guess."""

from __future__ import annotations

import re

from orpheus_mcp.models import LeadSheet, LeadSheetSection, SectionLabel, SheetNote
from orpheus_mcp.notation.chart import parse_chord_chart
from orpheus_mcp.notation.tab import parse_riff_tab
from orpheus_mcp.theory.melody import parse_melody

# Checked in order: "pre-chorus" before "chorus", "post-chorus" before "chorus".
_LABEL_WORDS: list[tuple[str, SectionLabel]] = [
    (r"pre-?chorus", "pre-chorus"),
    (r"post-?chorus", "post-chorus"),
    (r"chorus|hook|refrain", "chorus"),
    (r"intro", "intro"),
    (r"outro|coda|ending", "outro"),
    (r"verse", "verse"),
    (r"bridge|middle 8", "bridge"),
    (r"solo", "solo"),
    (r"interlude|break(?!down)", "interlude"),
    (r"instrumental", "instrumental"),
]


def derive_label(name: str) -> SectionLabel | None:
    """YuE2's section label from a section name ("Chorus A" -> "chorus"), or None."""
    low = name.lower()
    if "instrumental" in low:
        return "instrumental"
    for pattern, label in _LABEL_WORDS:
        if re.search(rf"\b(?:{pattern})\b", low):
            return label
    return None


def _syllable_tokens(text: str) -> list[str]:
    out: list[str] = []
    for word in text.split():
        if word == "_":
            out.append("_")
            continue
        parts = [p for p in word.split("-") if p]
        if not parts:
            continue
        last = parts[-1] + ("-" if word.endswith("-") else "")  # "a- bout": a word continues
        out.extend([p + "-" for p in parts[:-1]] + [last])
    return out


def attach_syllables(notes: list[SheetNote], text: str) -> tuple[list[SheetNote], list[str]]:
    """One syllable per note in order: words split at hyphens ("ra-ther" -> "ra-", "ther"),
    "_" continues the previous syllable. Count mismatches are warned, never padded or merged."""
    tokens = _syllable_tokens(text)
    warnings = []
    if len(tokens) != len(notes):
        warnings.append(f"{len(notes)} notes but {len(tokens)} syllables")
    out = [n.model_copy(update={"syllable": tokens[i] if i < len(tokens) else None})
           for i, n in enumerate(notes)]
    return out, warnings


def _resolve_section(
    sec: LeadSheetSection, meter: tuple[int, int]
) -> tuple[LeadSheetSection, list[str]]:
    warnings: list[str] = []
    update: dict = {}
    bar_beats = meter[0] * 4.0 / meter[1]
    span = sec.bars * bar_beats

    label = sec.label or derive_label(sec.name)
    if label is None:
        label = "verse"
        warnings.append(f"no section type in the name {sec.name!r}: treated as a verse")
    update["label"] = label

    if sec.chart:
        chart = parse_chord_chart(sec.chart, bars=sec.bars, meter=meter)
        update["chords"] = chart.chords
        update["notes"] = list(sec.notes) + chart.notes
        warnings += [n for n in chart.notes if n.startswith("chart has")]
    if sec.tab:
        tab = parse_riff_tab(sec.tab, meter=meter)
        update["riff"] = tab.notes
        warnings += tab.warnings
        if tab.bars != sec.bars:
            warnings.append(f"tab has {tab.bars} bars, section says {sec.bars}")
    if sec.melody:
        notes = [SheetNote(pitch=n["pitch"], start_beat=n["start_beat"],
                           duration_beats=n["duration_beats"])
                 for n in parse_melody(sec.melody)]  # no key snapping: chromatic notes are meant
        if sec.syllables:
            notes, syl_warn = attach_syllables(notes, sec.syllables)
            warnings += syl_warn
        update["melody_notes"] = notes
        end = max((n.start_beat + n.duration_beats for n in notes), default=0.0)
        if end > span + 1e-9:
            warnings.append(f"melody runs past bar {sec.bars} ({end:g} of {span:g} beats)")
    return sec.model_copy(update=update), [f"{sec.name}: {w}" for w in warnings]


def resolve_lead_sheet(sheet: LeadSheet) -> tuple[LeadSheet, list[str]]:
    """Every section expanded from its as-written fields; returns (sheet, warnings)."""
    sections, warnings = [], []
    for sec in sheet.sections:
        resolved, w = _resolve_section(sec, sheet.meter)
        sections.append(resolved)
        warnings += w
    return sheet.model_copy(update={"sections": sections}), warnings
