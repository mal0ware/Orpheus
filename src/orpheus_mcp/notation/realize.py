"""A resolved LeadSheet section -> MIDI notes per track (spec 2026-10-24 §6). Pure: the
``arrange_lead_sheet`` tool does the bridge calls. Beats are quarter notes from the section's
first bar, the same anchor ``insert_midi_notes(at_bar=...)`` uses."""

from __future__ import annotations

from dataclasses import dataclass, field

from orpheus_mcp.models import LeadSheetSection
from orpheus_mcp.theory.chords import parse_chord_symbol, voice_lead
from orpheus_mcp.theory.patterns import DRUM_PATTERNS, parse_drum_grid

CHORD_VELOCITY, BASS_VELOCITY, LEAD_VELOCITY = 80, 95, 100


def _music21_pitches(symbol: str) -> list[int]:
    from music21 import harmony

    def fix(part: str) -> str:  # music21 spells flats '-': "Bb" -> "B-"
        return part[0] + "-" + part[2:] if len(part) > 1 and part[1] == "b" else part

    def m21(name: str) -> str:
        root, slash, bass = name.partition("/")
        return fix(root) + (slash + fix(bass) if slash else "")

    cs = harmony.ChordSymbol(m21(symbol))
    return [int(p.midi) for p in cs.pitches]


def chord_pitches(symbol: str) -> list[int]:
    """MIDI pitches for a chord symbol, voiced around middle C with any slash bass lowest.
    music21 reads most symbols; ones it doesn't (maj9, 6/9, m(maj7)) fall back to YuE2's
    simplified form, then to Orpheus's triad parser. Unreadable symbols raise ValueError."""
    from orpheus_mcp.notation.yue2_abc import reduce_chord

    pitches: list[int] = []
    for candidate in (symbol, reduce_chord(symbol)[0], symbol.replace("(maj7)", "M7")):
        try:
            pitches = _music21_pitches(candidate)
            break
        except Exception:  # noqa: BLE001 - music21 raises several types; try the next form
            continue
    if not pitches:
        pitches = parse_chord_symbol(reduce_chord(symbol)[0].split("/")[0])  # raises if hopeless
    centre = sum(pitches) / len(pitches)
    shift = 12 * round((62 - centre) / 12)  # centre the block near D4
    voiced = sorted(p + shift for p in pitches)
    while voiced[0] < 40:
        voiced = [p + 12 for p in voiced]
    while voiced[-1] > 84:
        voiced = [p - 12 for p in voiced]
    return voiced


@dataclass
class RealizedSection:
    tracks: dict[str, list[dict]]  # role ("chords", "bass", "lead", "drums") -> notes
    lyric_marker: str | None = None
    warnings: list[str] = field(default_factory=list)


def realize_section(sec: LeadSheetSection, meter: tuple[int, int]) -> RealizedSection:
    bar_beats = meter[0] * 4.0 / meter[1]
    tracks: dict[str, list[dict]] = {}
    warnings: list[str] = []

    if sec.chords:
        blocks, starts = [], []
        for c in sec.chords:
            try:
                blocks.append(chord_pitches(c.symbol))
                starts.append(c)
            except ValueError:
                warnings.append(f"{sec.name}: can't voice chord {c.symbol!r}; skipped")
        tracks["chords"] = [
            {"pitch": p, "start_beat": (c.bar - 1) * bar_beats + (c.beat - 1),
             "duration_beats": c.beats, "velocity": CHORD_VELOCITY}
            for c, block in zip(starts, voice_lead(blocks, low=40, high=84), strict=True)
            for p in block
        ]
    if sec.riff:
        tracks["bass"] = [{"pitch": n.pitch, "start_beat": n.start_beat,
                           "duration_beats": n.duration_beats, "velocity": BASS_VELOCITY}
                          for n in sec.riff]
    if sec.melody_notes:
        tracks["lead"] = [{"pitch": n.pitch, "start_beat": n.start_beat,
                           "duration_beats": n.duration_beats, "velocity": LEAD_VELOCITY}
                          for n in sec.melody_notes]
    if sec.drums:
        if meter != (4, 4):
            warnings.append(f"{sec.name}: drum patterns are 4/4 only; no drums written")
        elif sec.drums not in DRUM_PATTERNS:
            warnings.append(f"{sec.name}: unknown drum pattern {sec.drums!r} "
                            f"(have {', '.join(DRUM_PATTERNS)})")
        else:
            one_bar = parse_drum_grid(DRUM_PATTERNS[sec.drums])
            tracks["drums"] = [{**hit, "start_beat": hit["start_beat"] + b * bar_beats}
                               for b in range(sec.bars) for hit in one_bar]

    lyric = sec.lyrics[0] if sec.lyrics else None
    if lyric is None and any(n.syllable for n in sec.melody_notes):
        from orpheus_mcp.notation.yue2_abc import _lyric_line

        lyric = _lyric_line(sec.melody_notes) or None
    return RealizedSection(tracks=tracks, lyric_marker=lyric, warnings=warnings)
