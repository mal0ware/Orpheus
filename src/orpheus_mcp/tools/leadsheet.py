"""Lead-sheet tools (spec 2026-10-24): read the owner's chart and tab notation, preview a
LeadSheet, export it as a YuE2 score, and lay it into REAPER."""

from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient
from orpheus_mcp.models import LeadSheet
from orpheus_mcp.notation.chart import parse_chord_chart as _parse_chart
from orpheus_mcp.notation.leadsheet import resolve_lead_sheet
from orpheus_mcp.notation.realize import realize_section
from orpheus_mcp.notation.tab import parse_riff_tab as _parse_tab

_RO = {"readOnlyHint": True}
_DESTRUCTIVE = {"destructiveHint": True}

TRACKS = {"chords": "Chords", "bass": "Bass", "lead": "Lead", "drums": "Drums"}
ROLE_INSTRUMENT = {"chords": "keys", "bass": "bass", "lead": "keys", "drums": "drums"}


def arrange_lead_sheet_impl(bridge, sheet: LeadSheet) -> dict:
    """Tempo, meter, a marker per section (plus its first lyric line), and chords / bass riff /
    melody / drums on shared tracks. Instruments load only on tracks this call creates."""
    from orpheus_mcp.instruments import choose_instrument, load_instrument
    from orpheus_mcp.tools.compose import _write_notes

    sheet, warnings = resolve_lead_sheet(sheet)
    realized = [realize_section(sec, sheet.meter) for sec in sheet.sections]
    for r in realized:
        warnings += r.warnings
    roles = [role for role in TRACKS if any(role in r.tracks for r in realized)]

    bridge.call("set_tempo", bpm=float(sheet.tempo_bpm))
    bridge.call("set_time_signature", numerator=sheet.meter[0], denominator=sheet.meter[1])
    existing = {t["name"] for t in (bridge.call("list_tracks") or [])}
    created = []
    for role in roles:
        if TRACKS[role] not in existing:
            bridge.call("create_track", name=TRACKS[role])
            created.append(role)

    bar, placed = 1, []
    for sec, r in zip(sheet.sections, realized, strict=True):
        bridge.call("add_marker", name=sec.name, bar=bar)
        if r.lyric_marker:
            bridge.call("add_marker", name=r.lyric_marker, bar=bar)
        counts = {}
        for role, notes in r.tracks.items():
            counts[role] = _write_notes(bridge, TRACKS[role], notes, at_bar=bar)
        placed.append({"section": sec.name, "label": sec.label, "bar": bar, "bars": sec.bars,
                       "notes": counts})
        bar += sec.bars

    instruments = {}
    if created:
        inventory = bridge.call("list_installed_fx").get("fx", [])
        for role in created:
            choice = choose_instrument(ROLE_INSTRUMENT[role], inventory)
            load_instrument(bridge, TRACKS[role], choice)
            instruments[TRACKS[role]] = choice
    return {
        "tempo_bpm": sheet.tempo_bpm,
        "meter": list(sheet.meter),
        "tracks": [TRACKS[r] for r in roles],
        "sections": placed,
        "instruments": instruments,
        "warnings": warnings,
    }


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:  # noqa: ARG001
    @mcp.tool(annotations=_RO)
    def parse_chord_chart(
        chart: str, bars: int | None = None, meter: tuple[int, int] = (4, 4)
    ) -> dict:
        """Read one chord chart in the owner's notation: '|' bar line, a lone chord fills its
        bar, ' / ' splits a bar evenly, 'C/E' is a slash chord, '(x4)' repeats the line,
        'E---' holds to the section's end, '(stop)' and free text come back as notes. Pass
        the section's bar count from the sheet as ``bars``."""
        r = _parse_chart(chart, bars=bars, meter=meter)
        return {"chords": [c.model_dump() for c in r.chords], "bars": r.bars, "notes": r.notes}

    @mcp.tool(annotations=_RO)
    def parse_riff_tab(
        tab: str, meter: tuple[int, int] = (4, 4), tuning: Literal["bass", "guitar"] = "bass"
    ) -> dict:
        """Read riff tab in the owner's notation: one line per bar group, 'label | slots |',
        string letter + fret per slot ('A0'), '-' holds the previous note within the bar,
        a bar starting with '-' is silent, '(x4)' repeats. Standard 4-string bass unless
        ``tuning='guitar'``. Beats are quarter notes from the start of the tab."""
        r = _parse_tab(tab, meter=meter, tuning=tuning)
        return {"notes": [n.model_dump() for n in r.notes], "bars": r.bars, "labels": r.labels,
                "warnings": r.warnings}

    @mcp.tool(annotations=_RO)
    def preview_lead_sheet(lead_sheet: LeadSheet) -> dict:
        """Expand a LeadSheet written like the owner's sheet (chart, tab, melody, syllables)
        and return it with every warning, without touching REAPER. Melody notation:
        'A4:q C5:e. r:h' (w h q e s, a dot adds half); syllables: one per pitched note, split
        at hyphens ('ra-ther'), '_' continues the previous syllable."""
        resolved, warnings = resolve_lead_sheet(lead_sheet)
        return {"lead_sheet": resolved.model_dump(), "warnings": warnings}

    @mcp.tool(annotations=_RO)
    def export_yue2_score(lead_sheet: LeadSheet) -> dict:
        """The LeadSheet as a YuE2 request: the ABC score in YuE2's dialect, the lyrics text,
        the style text, a sidecar mapping each syllable to its notes, chord simplifications,
        warnings, and Orpheus's own dialect check (``valid`` / ``errors``). Read-only."""
        from orpheus_mcp.notation.yue2_abc import export_yue2

        resolved, warnings = resolve_lead_sheet(lead_sheet)
        out = export_yue2(resolved)
        errors = _check(out.abc)
        return {"abc": out.abc, "lyrics": out.lyrics, "style": out.style,
                "sidecar": out.sidecar, "reductions": out.reductions,
                "warnings": warnings + out.warnings, "valid": not errors, "errors": errors}

    @mcp.tool(annotations=_DESTRUCTIVE)
    def arrange_lead_sheet(lead_sheet: LeadSheet) -> dict:
        """Lay a LeadSheet into REAPER: tempo and meter, a marker per section plus its first
        lyric line, voice-led chords on 'Chords', the riff on 'Bass', the melody on 'Lead', and
        drum patterns on 'Drums' for sections that name one. Bars follow REAPER's tempo and
        meter map. Instruments (installed plugin, then sound pack, then stock) load only on
        tracks this call creates. Ask the user first: it changes the project."""
        return arrange_lead_sheet_impl(BridgeClient(), lead_sheet)


def _check(abc: str) -> list[str]:
    try:
        from orpheus_mcp.notation.yue2_check import check_yue2_abc
    except ImportError:  # pragma: no cover - the checker ships with this module
        return ["Orpheus's YuE2 checker is not available"]
    return check_yue2_abc(abc)
