"""A minimal SFZ reader for the FreePats sound pack.

SFZ is the plain-text instrument format FreePats publishes: ``<global>`` / ``<group>`` /
``<region>`` headers, each followed by ``opcode=value`` pairs. This reader understands what
REAPER's sampler (RS5k) can use: which sample plays, over which notes, from which root note, at
which velocities. It is not a full SFZ engine: envelopes, filters, loops and round robins are
kept in ``SfzRegion.opcodes`` but not interpreted.

Handled: headers anywhere on a line; several opcodes per line; ``//`` and ``/* */`` comments;
values with spaces or ``#`` (``sample=Button Accordion HN B3.flac``, ``samples/pick/F#.flac``);
inheritance global → master → group → region, where a repeated ``<global>`` resets the scope;
``key=`` shorthand; note names (``C4`` = 60, ``Db2``, ``F#3``); ``<control> default_path``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HEADER = re.compile(r"<\s*([A-Za-z]+)\s*>")
_OPCODE = re.compile(r"(?:(?<=\s)|^)([A-Za-z_][A-Za-z0-9_]*)=")
_NOTE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_PITCH_CLASS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def note_number(value: str) -> int:
    """An SFZ key value as a MIDI note: a number (``60``) or a name with C4 = 60 (``Db2``)."""
    v = value.strip()
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    m = _NOTE.match(v)
    if not m:
        raise ValueError(f"not an SFZ note: {value!r}")
    letter, accidental, octave = m.groups()
    semis = _PITCH_CLASS[letter.lower()] + {"": 0, "#": 1, "b": -1}[accidental]
    return (int(octave) + 1) * 12 + semis


@dataclass(frozen=True)
class SfzRegion:
    """One playable region, with everything it inherited already merged in."""

    sample: str
    lokey: int
    hikey: int
    pitch_keycenter: int
    lovel: int = 1
    hivel: int = 127
    loop_mode: str | None = None
    trigger: str = "attack"
    opcodes: dict[str, str] = field(default_factory=dict)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def _opcodes(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in body.splitlines():
        found = list(_OPCODE.finditer(line))
        for i, m in enumerate(found):
            end = found[i + 1].start() if i + 1 < len(found) else len(line)
            out[m.group(1)] = line[m.end():end].strip()
    return out


def _region(op: dict[str, str], default_path: str) -> SfzRegion | None:
    sample = op.get("sample")
    if not sample:
        return None
    lokey, hikey, center = 0, 127, 60
    if "key" in op:
        lokey = hikey = center = note_number(op["key"])
    if "lokey" in op:
        lokey = note_number(op["lokey"])
    if "hikey" in op:
        hikey = note_number(op["hikey"])
    if "pitch_keycenter" in op:
        try:
            center = note_number(op["pitch_keycenter"])
        except ValueError:  # e.g. "sample" (read the root from the file): not supported
            center = lokey
    return SfzRegion(
        sample=(default_path + sample).replace("\\", "/"),
        lokey=lokey,
        hikey=hikey,
        pitch_keycenter=center,
        lovel=int(op.get("lovel", 1)),
        hivel=int(op.get("hivel", 127)),
        loop_mode=op.get("loop_mode"),
        trigger=op.get("trigger", "attack"),
        opcodes=dict(op),
    )


def parse_sfz(text: str) -> list[SfzRegion]:
    """Every ``<region>`` in ``text``, in file order, with inherited opcodes merged."""
    text = _strip_comments(text)
    scopes: dict[str, dict[str, str]] = {"global": {}, "master": {}, "group": {}}
    default_path = ""
    regions: list[SfzRegion] = []

    pieces = _HEADER.split(text)  # [preamble, header1, body1, header2, body2, ...]
    for name, body in zip(pieces[1::2], pieces[2::2], strict=True):
        header = name.lower()
        op = _opcodes(body)
        if header == "control":
            default_path = op.get("default_path", default_path)
        elif header == "global":
            scopes = {"global": op, "master": {}, "group": {}}
        elif header == "master":
            scopes["master"], scopes["group"] = op, {}
        elif header == "group":
            scopes["group"] = op
        elif header == "region":
            merged = {**scopes["global"], **scopes["master"], **scopes["group"], **op}
            region = _region(merged, default_path)
            if region is not None:
                regions.append(region)
        # <curve>, <effect>, <midi>, ...: nothing RS5k can use
    return regions


def select_layer(regions: list[SfzRegion], velocity: int = 100) -> list[SfzRegion]:
    """The attack-triggered regions of the velocity layer containing ``velocity``, by note."""
    layer = [r for r in regions if r.trigger == "attack" and r.lovel <= velocity <= r.hivel]
    return sorted(layer, key=lambda r: (r.lokey, r.hikey))


def rs5k_zones(
    regions: list[SfzRegion],
    *,
    base_dir: Path,
    velocity: int = 100,
    widen: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Zones for the bridge's ``add_sampler_zones``: the one velocity layer that contains
    ``velocity``, attack-triggered regions only, sorted by note. ``widen=(lo, hi)`` stretches the
    lowest and highest zones out to those notes (their root notes stay put, so RS5k pitch-shifts).
    Every zone obeys note-offs: these are long sampled notes, not one-shots.
    """
    layer = select_layer(regions, velocity)
    zones: list[dict[str, Any]] = [
        {
            "file": str(Path(base_dir) / r.sample),
            "note_lo": r.lokey,
            "note_hi": r.hikey,
            "root": r.pitch_keycenter,
            "obey": True,
        }
        for r in layer
    ]
    if zones and widen is not None:
        zones[0]["note_lo"] = min(zones[0]["note_lo"], widen[0])
        zones[-1]["note_hi"] = max(zones[-1]["note_hi"], widen[1])
    return zones
