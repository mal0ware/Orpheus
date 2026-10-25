"""Sung sketches (spec 2026-10-24 sketch): a lead sheet's sections go to YuE2 and the result
lands on a new REAPER track at the first section's bar. Registered only when
``ORPHEUS_NC_BACKENDS`` names ``yue2``, because YuE2's weights are CC BY-NC 4.0."""

from __future__ import annotations

import re
import secrets
import time
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from orpheus_mcp.backends import yue2
from orpheus_mcp.bridge import BridgeClient
from orpheus_mcp.models import LeadSheet, LeadSheetSection
from orpheus_mcp.notation.leadsheet import resolve_lead_sheet
from orpheus_mcp.notation.yue2_abc import export_yue2
from orpheus_mcp.notation.yue2_check import check_yue2_abc

_DESTRUCTIVE = {"destructiveHint": True}

SKETCH_PREFIX = "SKETCH (YuE2) "
TAIL_S = 4.0  # room after the last bar for the final note to ring
# YuE2 often keeps writing past the owner's last bar; when the tail cuts it off, the REAPER item
# fades out over this long (owner decision 2026-10-25). The WAV itself is left as rendered.
FADE_OUT_S = 2.0
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def render_sketch_impl(bridge, sheet: LeadSheet, *, sections: list[str] | None = None,
                       style: str | None = None, seed: int | None = None,
                       install: yue2.Yue2Install | None = None) -> dict:
    """Score check, then REAPER's bar time (so a closed REAPER fails before the GPU runs),
    then one yue-synth render, then the import as a new time-locked track."""
    resolved, warnings = resolve_lead_sheet(sheet)
    first, stop = _select(resolved.sections, sections)
    chosen = resolved.sections[first:stop]
    start_bar = 1 + sum(s.bars for s in resolved.sections[:first])
    sub = resolved.model_copy(update={"sections": chosen,
                                      "style": style if style is not None else resolved.style})
    score = export_yue2(sub)
    errors = check_yue2_abc(score.abc)
    if errors:
        raise ValueError("the score fails YuE2's rules, nothing was rendered: "
                         + "; ".join(errors))
    warnings += score.warnings
    if not sub.style:
        warnings.append("the sheet has no style text, so YuE2 picks its own sound and voice")
    for sec in chosen:
        if not sec.melody_notes:
            warnings.append(f"no melody in {sec.name}: YuE2 has no tune to sing there")
    cot = "full" if any(sec.chords for sec in chosen) else "melody"

    score_bpm = round(sheet.tempo_bpm)  # the score's Q: line
    if score_bpm != sheet.tempo_bpm:
        warnings.append(f"YuE2's score rounds {sheet.tempo_bpm:g} BPM to {score_bpm} BPM, so "
                        "the sketch drifts slowly against REAPER's grid")
    quarters = sum(s.bars for s in chosen) * sub.meter[0] * 4 / sub.meter[1]
    duration = quarters * 60.0 / score_bpm + TAIL_S
    seed = seed if seed is not None else secrets.randbelow(2**31)
    request = yue2.build_request(style=sub.style or "", lyrics=score.lyrics, abc=score.abc,
                                 cot=cot, duration=duration, lm_seed=seed, seed=seed)
    install = install or yue2.find_install()

    at = bridge.call("get_bar_time", bar=start_bar)
    warnings += _grid_warnings(at, sheet, start_bar)

    label = chosen[0].name if len(chosen) == 1 else f"{chosen[0].name} to {chosen[-1].name}"
    stem = f"{_safe(label)} #{seed}"
    result = yue2.render(install, request, _new_dir(seed), stem)
    if result.truncated:
        ending = (f"YuE2 kept playing past the last bar, so the sketch stops {TAIL_S:g} s after "
                  f"it with a {FADE_OUT_S:g} s fade-out on the REAPER item")
    else:
        ending = "YuE2 ended the sketch on its own"

    params: dict = {"files": [str(result.wav)], "at_s": float(at["time_s"]),
                    "name_prefix": SKETCH_PREFIX}
    if result.truncated:
        params["fade_out_s"] = FADE_OUT_S
    imported = bridge.call("import_audio_files", **params)
    done = list(imported.get("imported") or [])
    failed = list(imported.get("failed") or [])
    import_error = None
    if not done:
        import_error = "; ".join(str(f.get("error")) for f in failed) or "nothing was imported"
        warnings.append(f"the sketch rendered but REAPER did not import it ({import_error}); "
                        f"drag {result.wav} onto bar {start_bar} by hand")
    return {
        "track": SKETCH_PREFIX + stem if done else None,
        "track_guid": done[0].get("track_guid") if done else None,
        "sections": [s.name for s in chosen],
        "start_bar": start_bar,
        "start_s": float(at["time_s"]),
        "wav": str(result.wav),
        "sketch_dir": str(result.wav.parent),
        "render_s": round(result.render_s, 2),
        "audio_s": result.audio_s,
        "duration_s": duration,
        "truncated": result.truncated,
        "ending": ending,
        "cot": cot,
        "lm_seed": result.lm_seed,
        "seed": result.seed,
        "reductions": score.reductions,
        "warnings": warnings,
        "import_error": import_error,
    }


def _select(sections: list[LeadSheetSection], wanted: list[str] | None) -> tuple[int, int]:
    """All sections, or the first contiguous run whose names match ``wanted`` (any case)."""
    if wanted is None:
        return 0, len(sections)
    if not wanted:
        raise ValueError("name at least one section, or leave sections out to sing them all")
    names = [s.name for s in sections]
    have = [n.casefold() for n in names]
    want = [n.casefold() for n in wanted]
    unknown = [n for n in wanted if n.casefold() not in have]
    if unknown:
        raise ValueError(f"no section named {', '.join(unknown)}; the sheet has: "
                         + ", ".join(names))
    for i in range(len(have) - len(want) + 1):
        if have[i:i + len(want)] == want:
            return i, i + len(want)
    raise ValueError("sections must be a contiguous run in the sheet's order "
                     f"(the sheet has: {', '.join(names)})")


def _grid_warnings(at: dict, sheet: LeadSheet, bar: int) -> list[str]:
    out = []
    bpm = at.get("bpm")
    if bpm is not None and abs(float(bpm) - sheet.tempo_bpm) > 0.01:
        out.append(f"REAPER's tempo at bar {bar} is {float(bpm):g} BPM but the sheet's is "
                   f"{sheet.tempo_bpm:g} BPM: the sketch will not line up with the bars")
    meter = at.get("meter")
    if meter and tuple(int(m) for m in meter) != tuple(sheet.meter):
        out.append(f"REAPER's meter at bar {bar} is {meter[0]}/{meter[1]} but the sheet is in "
                   f"{sheet.meter[0]}/{sheet.meter[1]}: the sketch will not line up with the "
                   "bars")
    return out


def _safe(name: str) -> str:
    """A section name as a file name (Windows-safe); it also becomes the track name."""
    cleaned = re.sub(r"\s+", " ", _UNSAFE.sub("-", name)).strip(" .")
    return cleaned[:60] or "sketch"


def _new_dir(seed: int) -> Path:
    base = yue2.sketch_root() / f"{time.strftime('%Y%m%d-%H%M%S')}-{seed}"
    path, n = base, 2
    while path.exists():
        path, n = base.with_name(f"{base.name}-{n}"), n + 1
    return path


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:  # noqa: ARG001
    if not yue2.nc_enabled("yue2"):
        return  # non-commercial weights: the owner opts in with ORPHEUS_NC_BACKENDS=yue2

    @mcp.tool(annotations=_DESTRUCTIVE)
    def render_sketch(
        lead_sheet: LeadSheet,
        sections: list[str] | None = None,
        style: str | None = None,
        seed: Annotated[int, Field(ge=0, lt=2**31)] | None = None,
    ) -> dict:
        """Sing a lead sheet with YuE2 (local, on the GPU) and put the result on a NEW REAPER
        track named 'SKETCH (YuE2) <section> #<seed>' at the first chosen section's bar (bars
        count from 1 in sheet order, as arrange_lead_sheet lays them out). ``sections``: names
        of a contiguous run (any case); leave it out for the whole sheet. ``style`` overrides
        the sheet's style text. The same ``seed`` gives the same take.

        Takes roughly half the sketch's length on the RTX 3080 and adds a track, so ask the
        user first. YuE2's weights are CC BY-NC 4.0: sketches are for the owner's own
        writing, not for release. YuE2 often keeps playing past the last bar; the sketch then
        stops 4 s after it with a 2 s fade-out on the REAPER item (``ending`` says which).
        Returns the track, the WAV path (under ~/.orpheus/sketches), render time, seeds, chord
        simplifications and warnings (tempo or meter that differ between REAPER and the sheet,
        sections with no melody)."""
        return render_sketch_impl(BridgeClient(), lead_sheet, sections=sections, style=style,
                                  seed=seed)
