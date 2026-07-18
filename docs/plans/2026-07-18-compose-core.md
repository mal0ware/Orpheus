# Orpheus Compose Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Claude the ability to build an audible, editable musical section in the user's live REAPER session from a natural-language ask, in one command, with zero manual setup.

**Architecture:** Pure, REAPER-free theory helpers produce `Note`-shaped data; thin FastMCP compose tools glue that data to the already-proven `insert_midi_notes` bridge primitive; a small new bridge surface (`list_installed_fx`, `add_instrument`, `clear_track_midi`) makes generated tracks audible by preferring the user's own installed instruments and falling back to stock plugins. All logic is TDD'd against pure unit tests + the `FakeReaperBridge` wire-protocol fake + the `lupa` Lua handler suite; a single live-REAPER smoke proves the end-to-end.

**Tech Stack:** Python 3.11+ (FastMCP, Pydantic), Lua 5.4 ReaScript (the in-REAPER bridge), pytest, lupa (in-process Lua), ruff, mypy.

## Global Constraints

- **Python floor:** 3.11 (`target-version = "py311"`, `from __future__ import annotations` at the top of every module).
- **Depends only on M1** (done). No analyze/recommend/transform, no audio/MP3, no FX *editing* verbs (Slice 2). Only read-only `list_installed_fx` + instrument *loading* are in scope.
- **Audibility ladder** (per role keys/bass/drums): user override → best installed match → optional curated pack → stock (`ReaSynth` for pitched, 3× `ReaSamplOmatic5000` for drums). The chosen instrument is always reported back.
- **Stock is the always-works floor:** never make sound depend on a third-party plugin being present.
- **No auto-install of arbitrary/licensed VSTs.** The only download path is `install_sound_pack()`: one pinned, checksum-verified, BSD-licensed sfizz `.vst3` + a CC0 patch, into a user-writable folder, consent-gated, never automatic.
- **Registry honesty rule:** a tool is exposed in `explain`/`default` profiles only once implemented. Stubs live only in `full` with a `[NOT IMPLEMENTED]` docstring. Remove a stub in the same commit its real implementation lands.
- **Determinism:** every generator (voicing, humanize) is deterministic given its inputs (humanize takes an explicit `seed`) so unit tests assert exact output.
- **Musical time is beats.** Tools speak `Note(pitch, start_beat, duration_beats, velocity)`; all PPQ/tempo math stays in the bridge.
- **Commits:** no Claude attribution trailers (repo sets `includeCoAuthoredBy: false`). Conventional-commit messages, one per task minimum.
- **Test commands** run from the Orpheus repo root with the dev extras installed (`uv run pytest ...`, `uv run python scripts/run_lua_tests.py`).

---

## Module Map (created/modified — lock these names)

**Created (pure, REAPER-free):**
- `src/orpheus_mcp/theory/chords.py` — `parse_chord_symbol`, `resolve_progression`, `voice_lead`
- `src/orpheus_mcp/theory/patterns.py` — `GM_DRUMS`, `parse_drum_grid`, `bassline_notes`
- `src/orpheus_mcp/instruments.py` — `ROLE_ALLOWLIST`, `select_instrument`
- `src/orpheus_mcp/drumkit.py` — `ensure_drum_samples` (bundle + synthesized WAV fallback)
- `src/orpheus_mcp/soundpack.py` — `install_sound_pack` core (pinned download + checksum + placement)

**Created (bridge-facing tools):**
- `src/orpheus_mcp/tools/instruments.py` — `list_installed_fx`, `add_instrument`, `install_sound_pack` tools

**Modified:**
- `src/orpheus_mcp/bridge/lua/orpheus_bridge.lua` — add `list_installed_fx`, `add_instrument`, `clear_track_midi` handlers
- `src/orpheus_mcp/tools/compose.py` — replace stubs with the 5 real compose tools
- `src/orpheus_mcp/tools/project.py` (or `midi.py`) — no change expected; reuse `get_track_midi`
- `src/orpheus_mcp/registry.py` — add `instruments` category; graduate `compose` into `default`/`full`
- `tests/fake_reaper.py` — model installed FX + per-track FX; add fake handlers for the 3 new verbs
- `tests/lua/test_m1_handlers.lua` (or a new `tests/lua/test_compose_handlers.lua`) + `scripts/run_lua_tests.py` `SUITES`
- `data/drumkit/` — bundled CC0 one-shots (or generated on first use)
- `docs/dev-log.md`, `README.md`, `docs/roadmap.md` — status advance

**Signatures other tasks rely on (verbatim):**
```python
# theory/chords.py
def parse_chord_symbol(symbol: str, octave: int = 4) -> list[int]: ...
def resolve_progression(spec: str, key: str | None = None,
                        mode: str = "minor", octave: int = 4) -> list[list[int]]: ...
def voice_lead(chords: list[list[int]], low: int = 48, high: int = 84) -> list[list[int]]: ...

# theory/patterns.py
GM_DRUMS: dict[str, int]  # {"kick":36,"snare":38,"hat":42}
def parse_drum_grid(pattern: str, steps_per_bar: int = 16) -> list[dict]: ...
def bassline_notes(chords: list[list[int]], style: str = "root",
                   bars_per_chord: int = 1) -> list[dict]: ...
# each note dict: {"pitch":int,"start_beat":float,"duration_beats":float,"velocity":int}

# instruments.py
ROLE_ALLOWLIST: dict[str, list[str]]
def select_instrument(role: str, inventory: list[str], override: str | None = None,
                      pack_installed: bool = False) -> dict: ...
# returns {"kind": "named"|"drumkit", "name": str|None, "source": str}

# drumkit.py
def ensure_drum_samples(dest_dir: "pathlib.Path") -> dict[str, str]: ...  # {voice: abs_path}

# bridge verbs (BridgeClient().call names)
# "list_installed_fx" -> {"fx": [str, ...]}
# "add_instrument" (track, kind, name?, samples?) -> {"track","loaded","already_present"}
# "clear_track_midi" (track) -> {"track","cleared"}
```

---

## Phase 1 — Pure theory (no REAPER, no bridge)

### Task 1: Chord-symbol parser + progression resolver

**Files:**
- Create: `src/orpheus_mcp/theory/chords.py`
- Test: `tests/test_chords.py`

**Interfaces:**
- Consumes: `NOTE_TO_PC`, `TRIAD_INTERVALS`, `progression_triads` from `orpheus_mcp.theory.music_theory_data`.
- Produces: `parse_chord_symbol`, `resolve_progression` (see Module Map).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chords.py
"""Pure chord parsing/resolution — no REAPER, no optional deps."""
from __future__ import annotations

import pytest

from orpheus_mcp.theory.chords import parse_chord_symbol, resolve_progression


def test_major_triad():
    assert parse_chord_symbol("C", octave=4) == [60, 64, 67]


def test_minor_triad():
    assert parse_chord_symbol("Am", octave=4) == [69, 72, 76]


def test_dominant_seventh_adds_flat7():
    assert parse_chord_symbol("C7", octave=4) == [60, 64, 67, 70]


def test_minor_seventh():
    assert parse_chord_symbol("Cm7", octave=4) == [60, 63, 67, 70]


def test_major_seventh():
    assert parse_chord_symbol("Cmaj7", octave=4) == [60, 64, 67, 71]


def test_flat_root():
    assert parse_chord_symbol("Bb", octave=4) == [70, 74, 77]


def test_rejects_garbage():
    for bad in ("H", "", "C#b", "Xmaj"):
        with pytest.raises(ValueError):
            parse_chord_symbol(bad)


def test_resolve_symbol_progression_by_comma():
    prog = resolve_progression("Cm7, Fm7, Bb7")
    assert prog[0] == [60, 63, 67, 70]
    assert len(prog) == 3


def test_resolve_roman_progression_needs_key():
    prog = resolve_progression("i-iv-V-i", key="A", mode="minor")
    assert len(prog) == 4
    # i in A minor = A minor triad at octave 4 -> A4=69
    assert prog[0] == [69, 72, 76]


def test_resolve_roman_without_key_raises():
    with pytest.raises(ValueError):
        resolve_progression("i-iv-V-i")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chords.py -v`
Expected: FAIL — `ModuleNotFoundError: orpheus_mcp.theory.chords`.

- [ ] **Step 3: Write the implementation**

```python
# src/orpheus_mcp/theory/chords.py
"""Chord-symbol parsing, dual-notation progression resolution, and voice-leading.

Pure and REAPER-free. Builds on music_theory_data's pitch-class + triad tables and the
existing Roman-numeral resolver (progression_triads) so the two notations agree.
"""
from __future__ import annotations

import re

from orpheus_mcp.theory.music_theory_data import (
    NOTE_TO_PC,
    TRIAD_INTERVALS,
    progression_triads,
    roman_to_degree,
)

# quality token -> triad key in TRIAD_INTERVALS
_QUALITIES: dict[str, str] = {
    "": "maj", "maj": "maj", "M": "maj",
    "m": "min", "min": "min", "-": "min",
    "dim": "dim", "aug": "aug", "+": "aug",
}
# seventh token -> semitones above root
_SEVENTHS: dict[str, int] = {"7": 10, "maj7": 11}

_ROOT_RE = re.compile(r"^([A-Ga-g][#b]?)(.*)$")


def parse_chord_symbol(symbol: str, octave: int = 4) -> list[int]:
    """'Cm7' -> MIDI pitches for the voiced chord at `octave` (C4 = 60).

    Grammar: root(A-G)(#|b)? quality(m|min|-|dim|aug|+|maj|M|"") seventh(7|maj7)?.
    Dim/aug take no seventh in v1; a trailing '7' is a (dominant) flat-7 except after
    'maj'. Unknown input raises ValueError.
    """
    token = symbol.strip()
    m = _ROOT_RE.match(token)
    if not m:
        raise ValueError(f"not a chord symbol: {symbol!r}")
    root_name, rest = m.group(1), m.group(2)
    try:
        root_pc = NOTE_TO_PC[root_name[:1].upper() + root_name[1:].lower()]
    except KeyError as exc:
        raise ValueError(f"bad chord root: {symbol!r}") from exc

    seventh: int | None = None
    if rest.endswith("maj7"):
        seventh, rest = _SEVENTHS["maj7"], rest[:-4]
    elif rest.endswith("7"):
        seventh, rest = _SEVENTHS["7"], rest[:-1]

    if rest not in _QUALITIES:
        raise ValueError(f"bad chord quality in {symbol!r}: {rest!r}")
    quality = _QUALITIES[rest]

    base = root_pc + 12 * (octave + 1)  # MIDI: C-1 = 0, so C4 = 60
    pitches = [base + i for i in TRIAD_INTERVALS[quality]]
    if seventh is not None:
        pitches.append(base + seventh)
    return pitches


def _looks_roman(spec: str) -> bool:
    tokens = [t.strip() for t in spec.split("-") if t.strip()]
    if not tokens:
        return False
    try:
        for t in tokens:
            roman_to_degree(t)
        return True
    except ValueError:
        return False


def resolve_progression(
    spec: str, key: str | None = None, mode: str = "minor", octave: int = 4
) -> list[list[int]]:
    """Resolve either notation to a list of chord pitch-lists.

    - Comma-separated OR non-Roman '-'-separated  -> absolute chord symbols.
    - Roman numerals ('i-iv-V-i')                 -> requires `key`; uses progression_triads.
    """
    if "," in spec:
        return [parse_chord_symbol(tok, octave) for tok in spec.split(",") if tok.strip()]
    if _looks_roman(spec):
        if key is None:
            raise ValueError("Roman-numeral progressions require a `key`.")
        return [pitches for _numeral, pitches in progression_triads(key, mode, spec, octave)]
    # dash-separated symbols, e.g. "Cm7-Fm7-Bb7"
    return [parse_chord_symbol(tok, octave) for tok in spec.split("-") if tok.strip()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chords.py -v`
Expected: PASS (all 10).

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/theory/chords.py tests/test_chords.py
git commit -m "feat(theory): chord-symbol parser + dual-notation progression resolver"
```

---

### Task 2: Voice-leading pass

**Files:**
- Modify: `src/orpheus_mcp/theory/chords.py` (append `voice_lead`)
- Test: `tests/test_chords.py` (append)

**Interfaces:**
- Produces: `voice_lead(chords, low=48, high=84) -> list[list[int]]`.

- [ ] **Step 1: Write the failing test (append to tests/test_chords.py)**

```python
from orpheus_mcp.theory.chords import voice_lead


def test_voice_lead_keeps_first_chord():
    chords = [[60, 64, 67], [65, 69, 72]]
    assert voice_lead(chords)[0] == [60, 64, 67]


def test_voice_lead_minimizes_movement():
    # A big upward jump should be pulled down an octave to sit near the previous chord.
    chords = [[60, 64, 67], [77, 81, 84]]  # C major, then F major an octave high
    out = voice_lead(chords)
    prev_centroid = sum(out[0]) / 3
    next_centroid = sum(out[1]) / 3
    assert abs(next_centroid - prev_centroid) <= 6  # stayed close


def test_voice_lead_respects_register_band():
    chords = [[60, 64, 67], [65, 69, 72]]
    for chord in voice_lead(chords, low=48, high=84):
        assert all(48 <= p <= 84 for p in chord)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_chords.py -k voice_lead -v`
Expected: FAIL — `cannot import name 'voice_lead'`.

- [ ] **Step 3: Implement (append to chords.py)**

```python
def voice_lead(chords: list[list[int]], low: int = 48, high: int = 84) -> list[list[int]]:
    """Octave-shift each chord (as a block) to minimize centroid movement vs the previous
    voicing, staying inside [low, high]. Deterministic; the first chord is unchanged."""
    if not chords:
        return []
    out: list[list[int]] = [list(chords[0])]
    for chord in chords[1:]:
        prev_centroid = sum(out[-1]) / len(out[-1])
        best: list[int] | None = None
        best_cost: float | None = None
        for shift in (-24, -12, 0, 12, 24):
            cand = [p + shift for p in chord]
            if any(p < low or p > high for p in cand):
                continue
            cost = abs(sum(cand) / len(cand) - prev_centroid)
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, cand
        out.append(best if best is not None else list(chord))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_chords.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/theory/chords.py tests/test_chords.py
git commit -m "feat(theory): deterministic voice-leading pass"
```

---

### Task 3: Drum-grid parser + bassline generator

**Files:**
- Create: `src/orpheus_mcp/theory/patterns.py`
- Test: `tests/test_patterns.py`

**Interfaces:**
- Produces: `GM_DRUMS`, `parse_drum_grid`, `bassline_notes` (see Module Map).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_patterns.py
"""Pure drum-grid + bassline generation — no REAPER."""
from __future__ import annotations

import pytest

from orpheus_mcp.theory.patterns import GM_DRUMS, bassline_notes, parse_drum_grid


def test_kick_on_every_quarter():
    pattern = "kick: x...x...x...x..."
    notes = parse_drum_grid(pattern, steps_per_bar=16)
    assert len(notes) == 4
    assert all(n["pitch"] == GM_DRUMS["kick"] for n in notes)
    assert [n["start_beat"] for n in notes] == [0.0, 1.0, 2.0, 3.0]


def test_multi_voice_grid():
    pattern = "kick:  x...x...x...x...\nsnare: ....x.......x...\nhat:   x.x.x.x.x.x.x.x."
    notes = parse_drum_grid(pattern)
    pitches = {n["pitch"] for n in notes}
    assert pitches == {GM_DRUMS["kick"], GM_DRUMS["snare"], GM_DRUMS["hat"]}
    snares = [n for n in notes if n["pitch"] == GM_DRUMS["snare"]]
    assert [n["start_beat"] for n in snares] == [1.0, 3.0]


def test_unknown_voice_raises():
    with pytest.raises(ValueError):
        parse_drum_grid("cowbell: x...")


def test_bassline_root_style_one_note_per_chord():
    chords = [[60, 64, 67], [65, 69, 72]]  # C, F
    notes = bassline_notes(chords, style="root", bars_per_chord=1)
    assert [n["pitch"] for n in notes] == [60, 65]
    assert [n["start_beat"] for n in notes] == [0.0, 4.0]
    assert notes[0]["duration_beats"] == 4.0


def test_bassline_root_fifth():
    chords = [[60, 64, 67]]
    notes = bassline_notes(chords, style="root_fifth", bars_per_chord=1)
    assert [n["pitch"] for n in notes] == [60, 67]
    assert [n["start_beat"] for n in notes] == [0.0, 2.0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_patterns.py -v`
Expected: FAIL — `ModuleNotFoundError: orpheus_mcp.theory.patterns`.

- [ ] **Step 3: Implement**

```python
# src/orpheus_mcp/theory/patterns.py
"""Drum-grid parsing + bassline generation. Pure; emits Note-shaped dicts in BEATS."""
from __future__ import annotations

# General-MIDI percussion notes the stock/kit instruments respond to.
GM_DRUMS: dict[str, int] = {"kick": 36, "snare": 38, "hat": 42}

_BEATS_PER_BAR = 4.0  # v1 is 4/4 only


def parse_drum_grid(pattern: str, steps_per_bar: int = 16) -> list[dict]:
    """Multi-line step grid -> note dicts. 'x' = hit, '.'/' ' = rest. One row per voice,
    'voice: xxxx'. Row label must be in GM_DRUMS. Step length = 4 beats / steps_per_bar."""
    step_beats = _BEATS_PER_BAR / steps_per_bar
    notes: list[dict] = []
    for raw in pattern.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"drum row needs 'voice: steps': {raw!r}")
        voice, cells = (s.strip() for s in line.split(":", 1))
        if voice not in GM_DRUMS:
            raise ValueError(f"unknown drum voice {voice!r}; known: {sorted(GM_DRUMS)}")
        vel = 100 if voice != "hat" else 80
        for i, ch in enumerate(cells.replace(" ", "")):
            if ch == "x":
                notes.append(
                    {
                        "pitch": GM_DRUMS[voice],
                        "start_beat": round(i * step_beats, 6),
                        "duration_beats": round(step_beats, 6),
                        "velocity": vel,
                    }
                )
            elif ch != ".":
                raise ValueError(f"drum cell must be 'x' or '.', got {ch!r}")
    return notes


def bassline_notes(
    chords: list[list[int]], style: str = "root", bars_per_chord: int = 1
) -> list[dict]:
    """Turn resolved chords into a bass line. Root = lowest chord tone dropped to bass
    register is the caller's job (pass already-registered chords); here root = chords[i][0]."""
    span = _BEATS_PER_BAR * bars_per_chord
    notes: list[dict] = []
    for i, chord in enumerate(chords):
        root = chord[0]
        start = i * span
        if style == "root":
            notes.append({"pitch": root, "start_beat": start,
                          "duration_beats": span, "velocity": 100})
        elif style == "root_fifth":
            notes.append({"pitch": root, "start_beat": start,
                          "duration_beats": span / 2, "velocity": 100})
            notes.append({"pitch": root + 7, "start_beat": start + span / 2,
                          "duration_beats": span / 2, "velocity": 96})
        elif style == "octave":
            notes.append({"pitch": root, "start_beat": start,
                          "duration_beats": span / 2, "velocity": 100})
            notes.append({"pitch": root + 12, "start_beat": start + span / 2,
                          "duration_beats": span / 2, "velocity": 96})
        else:
            raise ValueError(f"unknown bass style {style!r}")
    return notes
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_patterns.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/theory/patterns.py tests/test_patterns.py
git commit -m "feat(theory): drum-grid parser + bassline generator"
```

---

### Task 4: Instrument-selection ladder

**Files:**
- Create: `src/orpheus_mcp/instruments.py`
- Test: `tests/test_instruments.py`

**Interfaces:**
- Produces: `ROLE_ALLOWLIST`, `select_instrument` (see Module Map). Consumed by the compose tools (Task 12).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_instruments.py
"""Pure instrument-selection ladder — no REAPER."""
from __future__ import annotations

from orpheus_mcp.instruments import select_instrument


def test_override_wins():
    got = select_instrument("keys", inventory=["Vital"], override="My Piano")
    assert got == {"kind": "named", "name": "My Piano", "source": "override"}


def test_prefers_installed_allowlisted():
    got = select_instrument("keys", inventory=["ReaSynth", "Surge XT"])
    assert got["source"] == "installed"
    assert got["name"] == "Surge XT"
    assert got["kind"] == "named"


def test_pack_when_no_install_match():
    got = select_instrument("keys", inventory=["ReaSynth"], pack_installed=True)
    assert got == {"kind": "named", "name": "sfizz", "source": "pack"}


def test_stock_pitched_fallback():
    got = select_instrument("bass", inventory=["ReaSynth"])
    assert got == {"kind": "named", "name": "ReaSynth", "source": "stock"}


def test_stock_drum_fallback_is_drumkit():
    got = select_instrument("drums", inventory=["ReaSynth"])
    assert got == {"kind": "drumkit", "name": None, "source": "stock"}


def test_installed_drum_match():
    got = select_instrument("drums", inventory=["MT-PowerDrumKit"])
    assert got["source"] == "installed"
    assert got["kind"] == "named"
    assert got["name"] == "MT-PowerDrumKit"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_instruments.py -v`
Expected: FAIL — `ModuleNotFoundError: orpheus_mcp.instruments`.

- [ ] **Step 3: Implement**

```python
# src/orpheus_mcp/instruments.py
"""Deterministic instrument-selection ladder: prefer the user's own good instruments,
guarantee sound with stock. Discovery only — this module installs nothing."""
from __future__ import annotations

# Curated per-role allowlist of quality instruments to prefer if already installed.
# Substring-matched against the installed-FX inventory (names vary by version/format).
ROLE_ALLOWLIST: dict[str, list[str]] = {
    "keys": ["Vital", "Surge XT", "Pianoteq", "Kontakt", "Decent Sampler"],
    "bass": ["Vital", "Surge XT", "Kontakt"],
    "drums": ["MT-PowerDrumKit", "Battery", "EZdrummer", "Superior Drummer", "Kontakt"],
}


def _match(inventory: list[str], candidates: list[str]) -> str | None:
    for cand in candidates:
        for fx in inventory:
            if cand.lower() in fx.lower():
                return cand
    return None


def select_instrument(
    role: str,
    inventory: list[str],
    override: str | None = None,
    pack_installed: bool = False,
) -> dict:
    """Pick an instrument for `role` ('keys'|'bass'|'drums') against the installed
    inventory. Ladder: override -> installed allowlist match -> curated pack -> stock."""
    if override:
        return {"kind": "named", "name": override, "source": "override"}

    match = _match(inventory, ROLE_ALLOWLIST.get(role, []))
    if match:
        return {"kind": "named", "name": match, "source": "installed"}

    if pack_installed:
        return {"kind": "named", "name": "sfizz", "source": "pack"}

    if role == "drums":
        return {"kind": "drumkit", "name": None, "source": "stock"}
    return {"kind": "named", "name": "ReaSynth", "source": "stock"}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_instruments.py -v`
Expected: PASS (all 6).

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/instruments.py tests/test_instruments.py
git commit -m "feat(instruments): deterministic instrument-selection ladder"
```

---

## Phase 2 — Bridge verbs (Lua + fake + Python tool + contract test)

> Each task here touches THREE agreeing surfaces: the Lua handler (`orpheus_bridge.lua`), the Python behavioural fake (`tests/fake_reaper.py`), and the Python tool wrapper. The fake + a Lua-suite assertion are the executable spec; the live smoke (Task 15) is the final proof.

### Task 5: `list_installed_fx` bridge verb + tool

**Files:**
- Modify: `tests/fake_reaper.py` (model `installed_fx`; add fake handler)
- Modify: `src/orpheus_mcp/bridge/lua/orpheus_bridge.lua` (add handler)
- Create: `src/orpheus_mcp/tools/instruments.py` (tool `list_installed_fx`)
- Test: `tests/test_instruments_tools.py`
- Modify: `tests/lua/test_m1_handlers.lua` (assert the handler shape) + confirm `scripts/run_lua_tests.py`

**Interfaces:**
- Produces bridge fn `list_installed_fx() -> {"fx": [str, ...]}`; MCP tool `list_installed_fx() -> {"fx":[...]}`.
- Consumed by Task 12 (`compose_section`) + Task 6 (idempotency check).

- [ ] **Step 1: Write the failing contract test**

```python
# tests/test_instruments_tools.py
"""Bridge-contract + tool tests for the instrument verbs, over the real wire protocol."""
from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.bridge.client import BridgeClient
from orpheus_mcp.registry import register_tools


@pytest.fixture
def project():
    p = FakeReaperProject(tempo=120.0)
    p.installed_fx = ["VSTi: ReaSynth (Cockos)", "VSTi: Surge XT (Surge Synth Team)"]
    return p


@pytest.fixture
def client(project, tmp_path):
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield BridgeClient(bridge_dir=tmp_path)


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    mcp = FastMCP(name="OrpheusTest")
    register_tools(mcp, profile="full")
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield Client(mcp)


def test_list_installed_fx_contract(client):
    res = client.call("list_installed_fx")
    assert "Surge XT" in " ".join(res["fx"])


async def test_list_installed_fx_tool(mcp_client):
    async with mcp_client as c:
        res = await c.call_tool("list_installed_fx", {})
    assert any("ReaSynth" in name for name in res.data["fx"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_instruments_tools.py -v`
Expected: FAIL — fake bridge returns `unknown fn: list_installed_fx` (and the tool doesn't exist).

- [ ] **Step 3a: Model installed FX in the fake (tests/fake_reaper.py)**

In `FakeReaperProject`, add a field near `play_state`:
```python
    installed_fx: list[str] = field(default_factory=list)
```
In `make_handlers`, add and register the handler:
```python
    def list_installed_fx(_):
        return {"fx": list(project.installed_fx)}
```
```python
        "list_installed_fx": list_installed_fx,
```

- [ ] **Step 3b: Add the Lua handler (orpheus_bridge.lua, near the other HANDLERS)**

```lua
-- Enumerate installed FX/instruments. EnumInstalledFX(index) -> ret, name, ident.
-- Read-only; installs nothing. Cached by REAPER internally.
HANDLERS.list_installed_fx = function(_)
  local out = {}
  local i = 0
  while true do
    local ok, name = reaper.EnumInstalledFX(i)
    if not ok then break end
    out[#out + 1] = name
    i = i + 1
    if i > 20000 then break end  -- hard safety cap
  end
  return { fx = out }
end
```
> NOTE (verify live): `reaper.EnumInstalledFX` returns `(retval, name, ident)` in REAPER 7.x. If unavailable on the target build, fall back to reading `reaper-vstplugins*.ini`. The fake + Lua-stub tests do not depend on the real API; the live smoke (Task 15) confirms it.

- [ ] **Step 3c: Add the Python tool (src/orpheus_mcp/tools/instruments.py)**

```python
# src/orpheus_mcp/tools/instruments.py
"""Instrument discovery + loading tools (Slice 1). `list_installed_fx` is read-only;
`add_instrument` loads a synth or a stock drum kit. Both are thin bridge wrappers."""
from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient

_RO = {"readOnlyHint": True}
_DESTRUCTIVE = {"destructiveHint": True}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(annotations=_RO)
    def list_installed_fx() -> dict:
        """List the plugins/instruments installed in REAPER (names). Read-only."""
        result = BridgeClient().call("list_installed_fx")
        return {"fx": result.get("fx", [])}
```

- [ ] **Step 3d: Register the new category (src/orpheus_mcp/registry.py)**

Add to `_CATEGORY_IMPORTS`:
```python
    "instruments": "orpheus_mcp.tools.instruments",
```
Add `"instruments"` to the `default` and `full` profile tuples (leave `explain` as-is).

- [ ] **Step 3e: Add a Lua-suite assertion (tests/lua/test_m1_handlers.lua)**

Add a stub for `EnumInstalledFX` to the `reaper` table and an assertion:
```lua
-- in the reaper stub table:
  EnumInstalledFX = function(i)
    local fx = { "VSTi: ReaSynth (Cockos)", "VSTi: Surge XT" }
    if fx[i + 1] then return true, fx[i + 1], "ident" end
    return false
  end,
```
```lua
-- with the other dispatch assertions:
do
  local r = M.dispatch("list_installed_fx", {})
  eq(#r.fx, 2, "list_installed_fx returns both installed")
end
```

- [ ] **Step 4: Run all three surfaces**

Run: `uv run pytest tests/test_instruments_tools.py -v`
Expected: PASS.
Run: `uv run python scripts/run_lua_tests.py`
Expected: PASS (existing + new assertion).

- [ ] **Step 5: Commit**

```bash
git add tests/fake_reaper.py src/orpheus_mcp/bridge/lua/orpheus_bridge.lua \
        src/orpheus_mcp/tools/instruments.py src/orpheus_mcp/registry.py \
        tests/test_instruments_tools.py tests/lua/test_m1_handlers.lua
git commit -m "feat(bridge): list_installed_fx verb + tool (pulled forward from Slice 2)"
```

---

### Task 6: `add_instrument` bridge verb + tool (synth + stock drum kit)

**Files:**
- Create: `src/orpheus_mcp/drumkit.py` (`ensure_drum_samples`)
- Modify: `tests/fake_reaper.py` (model per-track FX; fake handler)
- Modify: `src/orpheus_mcp/bridge/lua/orpheus_bridge.lua` (`add_instrument` handler)
- Modify: `src/orpheus_mcp/tools/instruments.py` (`add_instrument` tool)
- Test: `tests/test_instruments_tools.py` (append), `tests/test_drumkit.py`
- Modify: `tests/lua/test_m1_handlers.lua`

**Interfaces:**
- Consumes: `ensure_drum_samples` (this task), `list_installed_fx` (Task 5).
- Produces bridge fn `add_instrument(track, kind, name?, samples?) -> {"track","loaded","already_present"}`; tool `add_instrument(track, kind, name=None) -> dict`.

- [ ] **Step 1a: Write the failing drumkit test (tests/test_drumkit.py)**

```python
# tests/test_drumkit.py
"""ensure_drum_samples: bundle-or-synthesize the 3 one-shots, license-clean."""
from __future__ import annotations

import wave

from orpheus_mcp.drumkit import ensure_drum_samples


def test_creates_three_wavs(tmp_path):
    got = ensure_drum_samples(tmp_path)
    assert set(got) == {"kick", "snare", "hat"}
    for path in got.values():
        with wave.open(path, "rb") as w:
            assert w.getnframes() > 0


def test_idempotent(tmp_path):
    first = ensure_drum_samples(tmp_path)
    second = ensure_drum_samples(tmp_path)
    assert first == second
```

- [ ] **Step 1b: Write the failing add_instrument contract test (append tests/test_instruments_tools.py)**

```python
def test_add_synth_contract(client, project):
    project.tracks.append(_make_track(project, "keys"))
    res = client.call("add_instrument", track="keys", kind="named", name="ReaSynth")
    assert res["loaded"] == "ReaSynth"
    assert res["already_present"] is False


def test_add_synth_is_idempotent(client, project):
    project.tracks.append(_make_track(project, "keys"))
    client.call("add_instrument", track="keys", kind="named", name="ReaSynth")
    res = client.call("add_instrument", track="keys", kind="named", name="ReaSynth")
    assert res["already_present"] is True


def test_add_drumkit_loads_three_samplers(client, project):
    project.tracks.append(_make_track(project, "drums"))
    res = client.call(
        "add_instrument", track="drums", kind="drumkit",
        samples={"kick": "/x/kick.wav", "snare": "/x/snare.wav", "hat": "/x/hat.wav"},
    )
    assert res["loaded"] == "drumkit"
    tr = project.resolve_track("drums")
    assert len(tr.fx) == 3
```

Add this helper at the top of the test module (after imports):
```python
from fake_reaper import FakeTrack


def _make_track(project, name):
    return FakeTrack(guid=project._next_guid(), name=name)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_drumkit.py tests/test_instruments_tools.py -v`
Expected: FAIL — no `orpheus_mcp.drumkit`; `unknown fn: add_instrument`.

- [ ] **Step 3a: Implement ensure_drum_samples (src/orpheus_mcp/drumkit.py)**

```python
# src/orpheus_mcp/drumkit.py
"""Provide the 3 stock-kit one-shots. If bundled CC0 samples exist in data/drumkit/ they
are used; otherwise tiny synthesized WAVs are generated (license-clean, no download)."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

_SR = 44100
_VOICES = ("kick", "snare", "hat")


def _synth(voice: str) -> list[int]:
    """Return int16 PCM samples for a short one-shot."""
    import random  # local: only for noise voices

    rng = random.Random({"kick": 1, "snare": 2, "hat": 3}[voice])
    dur = {"kick": 0.18, "snare": 0.16, "hat": 0.05}[voice]
    n = int(_SR * dur)
    out: list[int] = []
    for i in range(n):
        t = i / _SR
        env = math.exp(-t * {"kick": 22, "snare": 30, "hat": 90}[voice])
        if voice == "kick":
            freq = 120 * math.exp(-t * 8) + 45  # pitch drop
            s = math.sin(2 * math.pi * freq * t)
        elif voice == "snare":
            s = 0.5 * math.sin(2 * math.pi * 180 * t) + 0.5 * (rng.uniform(-1, 1))
        else:  # hat: filtered noise
            s = rng.uniform(-1, 1)
        out.append(int(max(-1.0, min(1.0, s * env)) * 30000))
    return out


def ensure_drum_samples(dest_dir: Path) -> dict[str, str]:
    """Ensure kick/snare/hat WAVs exist in dest_dir; return {voice: absolute_path}."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    bundled = Path(__file__).resolve().parent.parent.parent / "data" / "drumkit"
    paths: dict[str, str] = {}
    for voice in _VOICES:
        src = bundled / f"{voice}.wav"
        target = dest_dir / f"{voice}.wav"
        if src.exists():
            if not target.exists():
                target.write_bytes(src.read_bytes())
        elif not target.exists():
            samples = _synth(voice)
            with wave.open(str(target), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(_SR)
                w.writeframes(b"".join(struct.pack("<h", s) for s in samples))
        paths[voice] = str(target)
    return paths
```

- [ ] **Step 3b: Model per-track FX in the fake (tests/fake_reaper.py)**

In `FakeTrack`, add:
```python
    fx: list[str] = field(default_factory=list)
```
In `make_handlers`, add + register:
```python
    def add_instrument(p):
        tr = project.resolve_track(p["track"])
        kind = p["kind"]
        if kind == "drumkit":
            for _voice, _path in (p.get("samples") or {}).items():
                tr.fx.append("ReaSamplOmatic5000")
            return {"track": tr.guid, "loaded": "drumkit", "already_present": False}
        name = p["name"]
        if name in tr.fx:
            return {"track": tr.guid, "loaded": name, "already_present": True}
        tr.fx.append(name)
        return {"track": tr.guid, "loaded": name, "already_present": False}
```
```python
        "add_instrument": add_instrument,
```

- [ ] **Step 3c: Add the Lua handler (orpheus_bridge.lua)**

```lua
-- Load an instrument on a track. kind="named" adds one FX by name (idempotent);
-- kind="drumkit" adds one ReaSamplOmatic5000 per sample, note-filtered to its GM note.
HANDLERS.add_instrument = function(p)
  local tr, err = resolve_track(p.track)
  if not tr then error(err) end

  if p.kind == "drumkit" then
    local GM = { kick = 36, snare = 38, hat = 42 }
    for voice, file in pairs(p.samples or {}) do
      local fx = reaper.TrackFX_AddByName(tr, "ReaSamplOmatic5000", false, -1)
      if fx >= 0 then
        -- Load the sample and pin it to one MIDI note. NOTE (verify live): the config
        -- parm for the sample file is "FILE0"; note-range params are indices 3 (min) and
        -- 4 (max) as 0..1 normalized pitch (n/127). Confirm on REAPER 7.x in Task 15.
        reaper.TrackFX_SetNamedConfigParm(tr, fx, "FILE0", file)
        local n = GM[voice] or 36
        reaper.TrackFX_SetParamNormalized(tr, fx, 3, n / 127.0)
        reaper.TrackFX_SetParamNormalized(tr, fx, 4, n / 127.0)
      end
    end
    return { track = reaper.GetTrackGUID(tr), loaded = "drumkit", already_present = false }
  end

  -- named synth: idempotent (don't stack duplicates).
  local existing = reaper.TrackFX_AddByName(tr, p.name, false, 0)  -- 0 = find only
  if existing >= 0 then
    return { track = reaper.GetTrackGUID(tr), loaded = p.name, already_present = true }
  end
  local added = reaper.TrackFX_AddByName(tr, p.name, false, -1)  -- -1 = add
  if added < 0 then error("could not add instrument: " .. tostring(p.name)) end
  return { track = reaper.GetTrackGUID(tr), loaded = p.name, already_present = false }
end
```

- [ ] **Step 3d: Add the Python tool (append to tools/instruments.py register())**

```python
    @mcp.tool(annotations=_DESTRUCTIVE)
    def add_instrument(track: str, kind: str = "named", name: str | None = None) -> dict:
        """Load an instrument so a track is audible. kind='named' adds `name` (idempotent);
        kind='drumkit' loads a stock 3-voice kit (kick/snare/hat) from bundled samples."""
        if kind == "drumkit":
            import tempfile
            from pathlib import Path

            from orpheus_mcp.drumkit import ensure_drum_samples

            samples = ensure_drum_samples(Path(tempfile.gettempdir()) / "orpheus_drumkit")
            result = BridgeClient().call(
                "add_instrument", track=track, kind="drumkit", samples=samples
            )
        else:
            if not name:
                raise ValueError("kind='named' requires an instrument name")
            result = BridgeClient().call(
                "add_instrument", track=track, kind="named", name=name
            )
        return {
            "track": result.get("track"),
            "loaded": result.get("loaded"),
            "already_present": result.get("already_present", False),
        }
```

- [ ] **Step 3e: Lua-suite assertion (tests/lua/test_m1_handlers.lua)**

Extend the `reaper` stub with FX table + funcs and assert drumkit adds three:
```lua
  -- add to a track model: tr.fx = {}
  TrackFX_AddByName = function(tr, name, rec, mode)
    tr.fx = tr.fx or {}
    if mode == 0 then  -- find-only
      for i, n in ipairs(tr.fx) do if n == name then return i - 1 end end
      return -1
    end
    tr.fx[#tr.fx + 1] = name
    return #tr.fx - 1
  end,
  TrackFX_SetNamedConfigParm = function() return true end,
  TrackFX_SetParamNormalized = function() return true end,
```
```lua
do
  local tr = make_track("drums")   -- use the suite's existing track factory
  local r = M.dispatch("add_instrument",
    { track = tr_ref, kind = "drumkit", samples = { kick="a", snare="b", hat="c" } })
  eq(r.loaded, "drumkit", "drumkit loaded")
  eq(#tr.fx, 3, "three samplers added")
end
```
> If the suite lacks a track factory, mirror the pattern already used by the `create_track`/`insert_midi_notes` assertions in that file.

- [ ] **Step 4: Run all surfaces**

Run: `uv run pytest tests/test_drumkit.py tests/test_instruments_tools.py -v`
Expected: PASS.
Run: `uv run python scripts/run_lua_tests.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/drumkit.py tests/fake_reaper.py \
        src/orpheus_mcp/bridge/lua/orpheus_bridge.lua \
        src/orpheus_mcp/tools/instruments.py \
        tests/test_drumkit.py tests/test_instruments_tools.py tests/lua/test_m1_handlers.lua
git commit -m "feat(bridge): add_instrument verb (ReaSynth + stock RS5k drum kit) + synth one-shots"
```

---

### Task 7: `clear_track_midi` bridge verb (enables replace-semantics for humanize)

**Files:**
- Modify: `tests/fake_reaper.py` (fake handler)
- Modify: `src/orpheus_mcp/bridge/lua/orpheus_bridge.lua` (handler)
- Test: `tests/test_instruments_tools.py` (append a small contract test) or `tests/test_m1_tools.py`
- Modify: `tests/lua/test_m1_handlers.lua`

**Interfaces:**
- Produces bridge fn `clear_track_midi(track) -> {"track","cleared"}`. Consumed by Task 11 (`humanize_pass`).

- [ ] **Step 1: Write the failing contract test (append tests/test_instruments_tools.py)**

```python
def test_clear_track_midi_empties_take(client, project):
    from fake_reaper import FakeNote, FakeTake, FakeTrack

    tr = FakeTrack(guid=project._next_guid(), name="keys")
    tr.takes.append(FakeTake(notes=[FakeNote(pitch=60, start_ppq=0, end_ppq=480)]))
    project.tracks.append(tr)
    res = client.call("clear_track_midi", track="keys")
    assert res["cleared"] == 1
    assert project.resolve_track("keys").takes[0].notes == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_instruments_tools.py::test_clear_track_midi_empties_take -v`
Expected: FAIL — `unknown fn: clear_track_midi`.

- [ ] **Step 3a: Fake handler (tests/fake_reaper.py)**

```python
    def clear_track_midi(p):
        tr = project.resolve_track(p["track"])
        cleared = 0
        if tr.takes:
            cleared = len(tr.takes[0].notes)
            tr.takes[0].notes.clear()
        return {"track": tr.guid, "cleared": cleared}
```
```python
        "clear_track_midi": clear_track_midi,
```

- [ ] **Step 3b: Lua handler (orpheus_bridge.lua)**

```lua
-- Delete every note in a track's first take (so humanize can rewrite in place).
HANDLERS.clear_track_midi = function(p)
  local tr, err = resolve_track(p.track)
  if not tr then error(err) end
  local take = first_take(tr)
  if not take then return { track = reaper.GetTrackGUID(tr), cleared = 0 } end
  local _, note_count = reaper.MIDI_CountEvts(take)
  for i = note_count - 1, 0, -1 do
    reaper.MIDI_DeleteNote(take, i)
  end
  reaper.MIDI_Sort(take)
  return { track = reaper.GetTrackGUID(tr), cleared = note_count }
end
```

- [ ] **Step 3c: Lua-suite assertion (tests/lua/test_m1_handlers.lua)**

Ensure the stub has `MIDI_DeleteNote` (remove by index) and assert clearing empties a take. Mirror the existing MIDI assertions.

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_instruments_tools.py -v && uv run python scripts/run_lua_tests.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/fake_reaper.py src/orpheus_mcp/bridge/lua/orpheus_bridge.lua \
        tests/test_instruments_tools.py tests/lua/test_m1_handlers.lua
git commit -m "feat(bridge): clear_track_midi verb (replace-semantics for humanize)"
```

---

## Phase 3 — Compose tools (Python over primitives; MCP contract tests)

> These replace the stubs in `tools/compose.py`. They run through the in-memory MCP client against the fake bridge, so they are fully tested without REAPER. Note dicts from Phase 1 map 1:1 to the `Note` fields `insert_midi_notes` expects.

### Task 8: `create_chord_progression`

**Files:**
- Modify: `src/orpheus_mcp/tools/compose.py` (remove chord stub; implement)
- Test: `tests/test_compose.py`

**Interfaces:**
- Consumes: `resolve_progression`, `voice_lead` (Task 1–2); bridge `create_track`/`insert_midi_notes` (M1); `add_instrument` (Task 6).
- Produces: tool `create_chord_progression(track, chords, key=None, mode="minor", bars_per_chord=1, octave=4) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose.py
"""Compose tools end-to-end through the in-memory MCP client + fake bridge."""
from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.registry import register_tools


@pytest.fixture
def project():
    p = FakeReaperProject(tempo=120.0)
    p.installed_fx = ["VSTi: ReaSynth (Cockos)"]
    return p


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    mcp = FastMCP(name="OrpheusTest")
    register_tools(mcp, profile="full")
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield Client(mcp)


async def test_create_chord_progression_writes_notes(mcp_client, project):
    async with mcp_client as c:
        await c.call_tool(
            "create_track", {"name": "keys"}
        )
        res = await c.call_tool(
            "create_chord_progression",
            {"track": "keys", "chords": "i-iv-V-i", "key": "A", "mode": "minor"},
        )
    assert res.data["chords_written"] == 4
    tr = project.resolve_track("keys")
    assert len(tr.takes[0].notes) == 12  # 4 triads * 3 notes
    assert "ReaSynth" in tr.fx  # instrument auto-loaded
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_compose.py::test_create_chord_progression_writes_notes -v`
Expected: FAIL — stub raises `NotImplementedError` (or tool absent once stub removed).

- [ ] **Step 3: Implement (rewrite src/orpheus_mcp/tools/compose.py)**

```python
# src/orpheus_mcp/tools/compose.py
"""Generate-from-scratch composers — thin orchestrators over the theory layer + the M1
MIDI writer. Composers never get private superpowers the model couldn't call directly."""
from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient
from orpheus_mcp.theory.chords import resolve_progression, voice_lead

_DESTRUCTIVE = {"destructiveHint": True}


def _write_notes(bridge: BridgeClient, track: str, notes: list[dict], at_bar: int = 1) -> int:
    """Batch notes through insert_midi_notes (<=512/call)."""
    written = 0
    for i in range(0, len(notes), 512):
        chunk = notes[i : i + 512]
        bridge.call("insert_midi_notes", track=track, notes=chunk, at_bar=at_bar)
        written += len(chunk)
    return written


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_chord_progression(
        track: str,
        chords: str,
        key: str | None = None,
        mode: str = "minor",
        bars_per_chord: int = 1,
        octave: int = 4,
    ) -> dict:
        """Write a chord progression as voiced MIDI. `chords` is Roman ('i-iv-V-i', needs
        `key`) or absolute symbols ('Cm7, Fm7, Bb7'). Auto-loads ReaSynth so it's audible."""
        voiced = voice_lead(resolve_progression(chords, key=key, mode=mode, octave=octave))
        beats_per_chord = 4.0 * bars_per_chord
        notes: list[dict] = []
        for i, chord in enumerate(voiced):
            start = i * beats_per_chord
            for pitch in chord:
                notes.append({"pitch": pitch, "start_beat": start,
                              "duration_beats": beats_per_chord, "velocity": 90})
        bridge = BridgeClient()
        written = _write_notes(bridge, track, notes)
        bridge.call("add_instrument", track=track, kind="named", name="ReaSynth")
        return {"track": track, "chords_written": len(voiced), "notes_written": written}
```

> The chord/drum/humanize stubs are removed as each real tool lands (this task removes the chord stub). `create_bassline`/`create_drum_pattern`/`humanize_pass`/`compose_section` are added in Tasks 9–12 to this same `register()`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_compose.py::test_create_chord_progression_writes_notes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/tools/compose.py tests/test_compose.py
git commit -m "feat(compose): create_chord_progression (voiced, audible)"
```

---

### Task 9: `create_bassline`

**Files:**
- Modify: `src/orpheus_mcp/tools/compose.py`
- Test: `tests/test_compose.py` (append)

**Interfaces:**
- Consumes: `resolve_progression` (Task 1), `bassline_notes` (Task 3), M1 writer, `add_instrument`.
- Produces: tool `create_bassline(track, chords, key=None, mode="minor", style="root", octave=2, bars_per_chord=1) -> dict`.

- [ ] **Step 1: Write the failing test (append tests/test_compose.py)**

```python
async def test_create_bassline_root_style(mcp_client, project):
    async with mcp_client as c:
        await c.call_tool("create_track", {"name": "bass"})
        res = await c.call_tool(
            "create_bassline",
            {"track": "bass", "chords": "Am, Dm, E, Am", "style": "root", "octave": 2},
        )
    tr = project.resolve_track("bass")
    assert res.data["notes_written"] == 4
    # Am root at octave 2 -> A2 = 45
    assert tr.takes[0].notes[0].pitch == 45
    assert "ReaSynth" in tr.fx
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_compose.py::test_create_bassline_root_style -v`
Expected: FAIL — unknown tool `create_bassline`.

- [ ] **Step 3: Implement (append inside register() in compose.py)**

```python
    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_bassline(
        track: str,
        chords: str,
        key: str | None = None,
        mode: str = "minor",
        style: str = "root",
        octave: int = 2,
        bars_per_chord: int = 1,
    ) -> dict:
        """Write a bass line following `chords` (same notation as create_chord_progression).
        `style`: 'root' | 'root_fifth' | 'octave'. Auto-loads ReaSynth (bass register)."""
        from orpheus_mcp.theory.patterns import bassline_notes

        chord_pitches = resolve_progression(chords, key=key, mode=mode, octave=octave)
        notes = bassline_notes(chord_pitches, style=style, bars_per_chord=bars_per_chord)
        bridge = BridgeClient()
        written = _write_notes(bridge, track, notes)
        bridge.call("add_instrument", track=track, kind="named", name="ReaSynth")
        return {"track": track, "notes_written": written}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_compose.py::test_create_bassline_root_style -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/tools/compose.py tests/test_compose.py
git commit -m "feat(compose): create_bassline"
```

---

### Task 10: `create_drum_pattern`

**Files:**
- Modify: `src/orpheus_mcp/tools/compose.py`
- Test: `tests/test_compose.py` (append)

**Interfaces:**
- Consumes: `parse_drum_grid` (Task 3), M1 writer, `add_instrument(kind="drumkit")` (Task 6).
- Produces: tool `create_drum_pattern(track, pattern, steps_per_bar=16) -> dict`.

- [ ] **Step 1: Write the failing test (append tests/test_compose.py)**

```python
async def test_create_drum_pattern_writes_and_loads_kit(mcp_client, project):
    pattern = "kick:  x...x...x...x...\nsnare: ....x.......x...\nhat:   x.x.x.x.x.x.x.x."
    async with mcp_client as c:
        await c.call_tool("create_track", {"name": "drums"})
        res = await c.call_tool(
            "create_drum_pattern", {"track": "drums", "pattern": pattern}
        )
    tr = project.resolve_track("drums")
    assert res.data["hits_written"] == 4 + 2 + 8
    assert len(tr.fx) == 3  # three RS5k voices
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_compose.py::test_create_drum_pattern_writes_and_loads_kit -v`
Expected: FAIL — unknown tool `create_drum_pattern`.

- [ ] **Step 3: Implement (append inside register())**

```python
    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_drum_pattern(track: str, pattern: str, steps_per_bar: int = 16) -> dict:
        """Write a drum pattern from a step grid (rows 'kick:'/'snare:'/'hat:', 'x'=hit).
        Loads a stock 3-voice kit so it's audible immediately."""
        from orpheus_mcp.theory.patterns import parse_drum_grid

        notes = parse_drum_grid(pattern, steps_per_bar=steps_per_bar)
        bridge = BridgeClient()
        written = _write_notes(bridge, track, notes)
        bridge.call("add_instrument", track=track, kind="drumkit")
        return {"track": track, "hits_written": written}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_compose.py::test_create_drum_pattern_writes_and_loads_kit -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/tools/compose.py tests/test_compose.py
git commit -m "feat(compose): create_drum_pattern (stock kit auto-loaded)"
```

---

### Task 11: `humanize_pass`

**Files:**
- Modify: `src/orpheus_mcp/tools/compose.py`
- Test: `tests/test_compose.py` (append)

**Interfaces:**
- Consumes: bridge `get_track_midi` (M1), `clear_track_midi` (Task 7), `insert_midi_notes` (M1).
- Produces: tool `humanize_pass(track, timing_ms=12, velocity_jitter=6, swing=0.0, seed=0) -> dict`.

- [ ] **Step 1: Write the failing test (append tests/test_compose.py)**

```python
async def test_humanize_is_deterministic_and_preserves_count(mcp_client, project):
    async with mcp_client as c:
        await c.call_tool("create_track", {"name": "keys"})
        await c.call_tool(
            "create_chord_progression",
            {"track": "keys", "chords": "i-iv-V-i", "key": "A"},
        )
        before = len(project.resolve_track("keys").takes[0].notes)
        r1 = await c.call_tool("humanize_pass", {"track": "keys", "seed": 42})
    after = len(project.resolve_track("keys").takes[0].notes)
    assert after == before  # replaced, not appended
    assert r1.data["humanized"] == before
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_compose.py::test_humanize_is_deterministic_and_preserves_count -v`
Expected: FAIL — unknown tool `humanize_pass` (and, if implemented naively with append, the count would double — the test guards that).

- [ ] **Step 3: Implement (append inside register())**

```python
    @mcp.tool(annotations=_DESTRUCTIVE)
    def humanize_pass(
        track: str,
        timing_ms: float = 12.0,
        velocity_jitter: int = 6,
        swing: float = 0.0,
        seed: int = 0,
    ) -> dict:
        """Add seeded human feel: timing/velocity jitter + optional swing on offbeat 16ths.
        Reads the track's notes, transforms, and REPLACES them (deterministic given `seed`)."""
        import random

        bridge = BridgeClient()
        current = bridge.call("get_track_midi", track=track).get("notes", [])
        if not current:
            return {"track": track, "humanized": 0}

        rng = random.Random(seed)
        tempo = bridge.call("get_project_info").get("tempo", 120.0)
        beats_per_ms = tempo / 60.0 / 1000.0  # ms -> beats
        out: list[dict] = []
        for n in current:
            start = n["start_beat"]
            # swing: delay the 2nd/4th 16th of each beat toward a triplet feel.
            if swing and round((start * 4) % 4) in (1, 3):
                start += swing * (1.0 / 6.0)
            start += rng.uniform(-timing_ms, timing_ms) * beats_per_ms
            start = max(0.0, start)
            vel = max(1, min(127, int(n["velocity"]) + rng.randint(-velocity_jitter, velocity_jitter)))
            out.append({"pitch": n["pitch"], "start_beat": round(start, 6),
                        "duration_beats": n["duration_beats"], "velocity": vel})

        bridge.call("clear_track_midi", track=track)
        _write_notes(bridge, track, out)
        return {"track": track, "humanized": len(out)}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_compose.py -v`
Expected: PASS (all compose tests).

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/tools/compose.py tests/test_compose.py
git commit -m "feat(compose): humanize_pass (seeded, replace-in-place)"
```

---

### Task 12: `compose_section` orchestrator

**Files:**
- Modify: `src/orpheus_mcp/tools/compose.py`
- Test: `tests/test_compose.py` (append)

**Interfaces:**
- Consumes: `get_profile` (`genre_profiles`), `select_instrument` (Task 4), `list_installed_fx` (Task 5), the four compose tools above, bridge `set_tempo`/`create_track`.
- Produces: tool `compose_section(genre, bars=8, key=None) -> dict` (reports tempo, tracks, and the instrument chosen per role).

- [ ] **Step 1: Write the failing test (append tests/test_compose.py)**

```python
async def test_compose_section_builds_audible_lofi(mcp_client, project):
    async with mcp_client as c:
        res = await c.call_tool("compose_section", {"genre": "lofi", "bars": 4})
    names = [t.name for t in project.tracks]
    assert {"drums", "chords", "bass"}.issubset(set(names))
    # tempo set into the lofi range (60-90)
    assert 60 <= project.tempo <= 90
    # instrument choice reported per role
    assert set(res.data["instruments"]) == {"drums", "chords", "bass"}
    # every pitched track has an instrument; drums has the 3-voice kit
    assert "ReaSynth" in project.resolve_track("chords").fx
    assert len(project.resolve_track("drums").fx) == 3


async def test_compose_section_unknown_genre_raises(mcp_client):
    async with mcp_client as c:
        with pytest.raises(Exception):
            await c.call_tool("compose_section", {"genre": "polka"})
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_compose.py::test_compose_section_builds_audible_lofi -v`
Expected: FAIL — unknown tool `compose_section`.

- [ ] **Step 3: Implement (append inside register())**

```python
    @mcp.tool(annotations=_DESTRUCTIVE)
    def compose_section(genre: str, bars: int = 8, key: str | None = None) -> dict:
        """Build a full, audible section in one call: sets tempo, creates drums/chords/bass,
        lays a genre-appropriate groove, humanizes, and loads the best available instruments
        (preferring the user's own). Returns the tempo, tracks, and instrument chosen per role."""
        from orpheus_mcp.instruments import select_instrument
        from orpheus_mcp.theory.genre_profiles import get_profile

        profile = get_profile(genre)  # raises ValueError on unknown genre
        tonic = key or {"lofi": "A", "hiphop": "A", "classical": "C"}.get(genre, "A")
        mode = profile["typical_modes"][0]
        progression = profile["progressions"][0]
        bpm = (profile["bpm_range"][0] + profile["bpm_range"][1]) // 2

        bridge = BridgeClient()
        bridge.call("set_tempo", bpm=float(bpm))
        inventory = bridge.call("list_installed_fx").get("fx", [])

        # Standard drum pattern per section length (repeat a 1-bar backbeat).
        one_bar = "kick:  x...x...x...x...\nsnare: ....x.......x...\nhat:   x.x.x.x.x.x.x.x."
        drum_pattern = "\n".join(
            "\n".join(f"{row}" for row in one_bar.splitlines()) for _ in range(max(1, bars))
        )  # bars are concatenated by the 16-step grid repeating; see note below

        chosen: dict[str, dict] = {}
        for role, track_name in (("drums", "drums"), ("keys", "chords"), ("bass", "bass")):
            bridge.call("create_track", name=track_name)
            chosen[track_name if role != "keys" else "chords"] = select_instrument(role, inventory)

        # Lay content via the atomic tools (their own add_instrument calls make them audible).
        # chords + bass repeat the profile progression to fill `bars`.
        reps = max(1, bars // (progression.count("-") + 1))
        full_prog = "-".join([progression] * reps)

        # NOTE: drum grid is 1 bar; to fill `bars`, call create_drum_pattern per bar at_bar.
        # For v1 simplicity we lay one bar and rely on the user to loop the item in REAPER;
        # multi-bar drum tiling is a Slice-2 arrangement concern.
        # chords
        # (call the already-registered tools directly via the MCP layer is not available here;
        #  replicate their bodies through the bridge + theory to avoid re-entrancy.)
        from orpheus_mcp.theory.chords import resolve_progression as _rp, voice_lead as _vl
        from orpheus_mcp.theory.patterns import bassline_notes as _bl, parse_drum_grid as _dg

        # drums
        _write_notes(bridge, "drums", _dg(one_bar))
        d = select_instrument("drums", inventory)
        if d["kind"] == "drumkit":
            bridge.call("add_instrument", track="drums", kind="drumkit")
        else:
            bridge.call("add_instrument", track="drums", kind="named", name=d["name"])

        # chords
        voiced = _vl(_rp(full_prog, key=tonic, mode=mode, octave=4))
        cnotes: list[dict] = []
        for i, chord in enumerate(voiced):
            for p in chord:
                cnotes.append({"pitch": p, "start_beat": i * 4.0,
                               "duration_beats": 4.0, "velocity": 88})
        _write_notes(bridge, "chords", cnotes)
        ci = select_instrument("keys", inventory)
        bridge.call("add_instrument", track="chords", kind="named",
                    name=ci["name"] if ci["kind"] == "named" else "ReaSynth")

        # bass
        bnotes = _bl(_rp(full_prog, key=tonic, mode=mode, octave=2), style="root")
        _write_notes(bridge, "bass", bnotes)
        bi = select_instrument("bass", inventory)
        bridge.call("add_instrument", track="bass", kind="named",
                    name=bi["name"] if bi["kind"] == "named" else "ReaSynth")

        return {
            "genre": genre,
            "tempo": float(bpm),
            "key": tonic,
            "tracks": ["drums", "chords", "bass"],
            "instruments": {"drums": d, "chords": ci, "bass": bi},
        }
```

> **Design note for the implementer:** the orchestrator re-uses the *theory helpers* directly (not the sibling MCP tools) to avoid MCP re-entrancy inside a tool call. Keep the note-building logic identical to Tasks 8–10 — if you change a generator there, change it here. (If this duplication grows, extract a private `_lay_chords/_lay_bass/_lay_drums(bridge, ...)` helper shared by both; do that only if a third caller appears — YAGNI for now.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_compose.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/tools/compose.py tests/test_compose.py
git commit -m "feat(compose): compose_section orchestrator (one call -> audible section)"
```

---

## Phase 4 — Optional curated sound pack

### Task 13: `install_sound_pack` (consent-gated, pinned + checksum)

**Files:**
- Create: `src/orpheus_mcp/soundpack.py`
- Modify: `src/orpheus_mcp/tools/instruments.py` (add `install_sound_pack` tool)
- Test: `tests/test_soundpack.py`

**Interfaces:**
- Produces: `soundpack.install_sound_pack(dest_dir, fetch=None) -> dict` and MCP tool `install_sound_pack() -> dict`.
- The `fetch` seam lets tests inject bytes instead of hitting the network.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_soundpack.py
"""install_sound_pack: pinned + checksum-verified placement into a user-writable dir."""
from __future__ import annotations

import hashlib

import pytest

from orpheus_mcp.soundpack import PACK, install_sound_pack


def _fake_fetch(payloads):
    def fetch(url):
        return payloads[url]
    return fetch


def test_installs_when_checksums_match(tmp_path):
    payloads = {item["url"]: item["_bytes"] for item in _fixture_items()}
    got = install_sound_pack(tmp_path, fetch=_fake_fetch(payloads))
    assert got["installed"] is True
    for item in PACK:
        assert (tmp_path / item["filename"]).exists()


def test_rejects_on_checksum_mismatch(tmp_path):
    bad = {item["url"]: b"tampered" for item in PACK}
    with pytest.raises(ValueError, match="checksum"):
        install_sound_pack(tmp_path, fetch=_fake_fetch(bad))


def _fixture_items():
    # Build payloads whose sha256 matches PACK, so the happy path is self-consistent.
    items = []
    for item in PACK:
        data = b"CONTENT:" + item["filename"].encode()
        item = dict(item)
        item["_bytes"] = data
        item["sha256"] = hashlib.sha256(data).hexdigest()
        items.append(item)
    return items
```
> Implementer: after writing `soundpack.py`, set each `PACK` entry's real `sha256` from the pinned release artifact. For the unit test, monkeypatch `PACK`'s checksums to the fixture values (or expose a `_set_checksums_for_test`); the point under test is the verify-then-place logic, not the real bytes.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_soundpack.py -v`
Expected: FAIL — `ModuleNotFoundError: orpheus_mcp.soundpack`.

- [ ] **Step 3a: Implement soundpack.py**

```python
# src/orpheus_mcp/soundpack.py
"""Optional curated sound pack: ONE BSD-licensed sfizz .vst3 + a CC0 SFZ patch, pinned by
URL + sha256, placed into a user-writable folder. Never runs automatically. No installer
.exe is executed; no licensed software is redistributed."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

# Pinned artifacts. Fill sha256 from the exact pinned release before shipping (Task 15).
PACK: list[dict] = [
    {
        "name": "sfizz",
        "filename": "sfizz.vst3",
        "url": "https://github.com/sfztools/sfizz/releases/download/PINNED/sfizz.vst3",
        "sha256": "<PIN_BEFORE_SHIP>",
        "license": "BSD-2-Clause",
    },
    {
        "name": "cc0-gm-patch",
        "filename": "orpheus_gm.sfz",
        "url": "https://example.org/cc0/orpheus_gm.sfz",  # CC0 patch, pinned in Task 15
        "sha256": "<PIN_BEFORE_SHIP>",
        "license": "CC0-1.0",
    },
]


def install_sound_pack(
    dest_dir: Path, fetch: Callable[[str], bytes] | None = None
) -> dict:
    """Download each pinned artifact, verify its sha256, and place it in dest_dir (which the
    caller has pointed REAPER at). Raises ValueError on any checksum mismatch — nothing is
    written for a mismatched item. `fetch` is injectable for tests."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if fetch is None:
        fetch = _http_get

    placed: list[str] = []
    for item in PACK:
        data = fetch(item["url"])
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(
                f"checksum mismatch for {item['name']}: got {digest}, want {item['sha256']}"
            )
        (dest_dir / item["filename"]).write_bytes(data)
        placed.append(item["filename"])
    return {"installed": True, "placed": placed, "dir": str(dest_dir)}


def _http_get(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (pinned https)
        return resp.read()
```

- [ ] **Step 3b: Add the consent-gated tool (tools/instruments.py register())**

```python
    @mcp.tool(annotations=_DESTRUCTIVE)
    def install_sound_pack(confirm: bool = False) -> dict:
        """OPTIONAL, opt-in: download ONE BSD-licensed sfizz .vst3 + a CC0 patch into a
        user-writable VST folder for a nicer default sound. Requires confirm=True. No admin,
        no installer, no licensed software. Falls back to stock if declined."""
        if not confirm:
            return {
                "installed": False,
                "note": "Set confirm=True to install the open-source sfizz + CC0 pack. "
                        "Stock instruments are used until then.",
            }
        import os
        from pathlib import Path

        from orpheus_mcp.soundpack import install_sound_pack as _install

        vst_dir = Path(os.path.expanduser("~")) / ".orpheus" / "vst"
        result = _install(vst_dir)
        # Point REAPER at the folder + rescan (best-effort; stock still works if this fails).
        try:
            BridgeClient().call("add_vst_path_and_rescan", path=str(vst_dir))
        except Exception:  # noqa: BLE001
            result["rescan"] = "manual — add the folder to REAPER's VST paths + rescan"
        return result
```
> The `add_vst_path_and_rescan` bridge verb is a thin best-effort helper (set `vst_path` in `reaper.ini` via `reaper.SNM_SetStringConfigVar` / `BR_Win32_WritePrivateProfileString`, then `Main_OnCommand(40312, 0)` "rescan"). If you choose not to implement it in Slice 1, keep the `except` branch's manual instruction — the pack still installs; only the auto-rescan is skipped. Implement the verb only if the live smoke shows it's needed; otherwise document the one-time manual path.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_soundpack.py -v`
Expected: PASS (with the checksum-fixture shim described in Step 1's note).

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/soundpack.py src/orpheus_mcp/tools/instruments.py tests/test_soundpack.py
git commit -m "feat(soundpack): consent-gated, pinned+checksummed open-source sound pack"
```

---

## Phase 5 — Wiring, full-suite gate, docs, live smoke

### Task 14: Registry graduation + full suite green + lint/type gates

**Files:**
- Modify: `src/orpheus_mcp/registry.py` (ensure `compose` + `instruments` in `default`/`full`)
- Modify: `src/orpheus_mcp/tools/dsl.py` (leave stubs; out of scope) — no change
- Test: `tests/test_registry.py` (extend)

**Interfaces:**
- Confirms the compose + instrument tools are advertised in `default`/`full` and NOT in `explain`.

- [ ] **Step 1: Write/extend the failing test (tests/test_registry.py)**

```python
def test_compose_and_instruments_in_default_profile():
    from fastmcp import FastMCP
    from orpheus_mcp.registry import register_tools

    mcp = FastMCP(name="t")
    cats = register_tools(mcp, profile="default")
    assert "compose" in cats
    assert "instruments" in cats


def test_compose_not_in_explain_profile():
    from fastmcp import FastMCP
    from orpheus_mcp.registry import register_tools

    mcp = FastMCP(name="t")
    cats = register_tools(mcp, profile="explain")
    assert "compose" not in cats
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL until `compose`/`instruments` are added to the `default` profile tuple.

- [ ] **Step 3: Implement — update PROFILES in registry.py**

```python
    "default": (
        "bridge", "project", "transport", "tracks", "midi",
        "theory", "analyze", "style", "apply", "render",
        "instruments", "compose",
    ),
```
(Keep `full` = all categories; it already includes them. `compose.register` no longer early-returns on `include_stubs=False` because the tools are real now — confirm the stub guard is gone.)

- [ ] **Step 4: Run the FULL suite + lint + types**

Run: `uv run pytest -q`
Expected: PASS (all, incl. `tests/test_midi_roundtrip.py` the non-negotiable gate).
Run: `uv run python scripts/run_lua_tests.py`
Expected: PASS.
Run: `uv run ruff check . && uv run mypy src`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/orpheus_mcp/registry.py tests/test_registry.py
git commit -m "feat(registry): graduate compose + instruments into the default profile"
```

---

### Task 15: Live-REAPER smoke + docs

**Files:**
- Modify: `docs/dev-log.md` (append a dated entry)
- Modify: `README.md` (advance the status line), `docs/roadmap.md` (mark M4 compose slice + pulled-forward `list_installed_fx`)
- Create: `examples/make_a_lofi_beat.md` (the demo transcript)

**This task is a MANUAL live verification — it is the real proof for the bridge verbs whose ReaScript constants were stubbed in tests.**

- [ ] **Step 1: Verify the empirically-uncertain ReaScript calls against live REAPER 7.x**
  Open REAPER, run the Orpheus bridge action, then from an MCP client:
  - `list_installed_fx` returns a non-empty list containing "ReaSynth".
  - `add_instrument(track, kind="named", name="ReaSynth")` makes the track show ReaSynth; a second call reports `already_present=True`.
  - `add_instrument(track, kind="drumkit")` adds three ReaSamplOmatic5000 instances, each playing its sample on the right note.
  Fix `FILE0` / note-range param indices in the Lua handler if the observed behavior differs from the Task 6 note; re-run `scripts/run_lua_tests.py`.

- [ ] **Step 2: Run the end-to-end demo**
  `compose_section("lofi", bars=4)` → confirm: tempo set (60–90), three tracks created, notes present, and **pressing play produces audible drums + chords + bass**. Confirm one Ctrl+Z is sane (wrap the orchestrator's writes in an undo block if the live test shows multiple undo steps — add `Undo_BeginBlock`/`Undo_EndBlock` around the orchestrator's bridge calls via a `batch`/undo verb if needed).

- [ ] **Step 3: Pin the sound-pack checksums** (if shipping the pack) — download the exact sfizz release + CC0 patch, record their sha256 into `soundpack.PACK`, and re-run `tests/test_soundpack.py` against the real bytes once.

- [ ] **Step 4: Write the docs**
  - `docs/dev-log.md`: dated entry — what was verified live, any ReaScript constant corrections.
  - `README.md`: advance the status line to note the compose slice ships (NL → audible editable section) + `list_installed_fx`.
  - `docs/roadmap.md`: check off the M4 compose bullet + note `list_installed_fx` pulled forward.
  - `examples/make_a_lofi_beat.md`: the literal transcript of the demo.

- [ ] **Step 5: Commit**

```bash
git add docs/dev-log.md README.md docs/roadmap.md examples/make_a_lofi_beat.md \
        src/orpheus_mcp/bridge/lua/orpheus_bridge.lua src/orpheus_mcp/soundpack.py
git commit -m "docs: compose-core live smoke PASS + M4 compose slice shipped"
```

---

## Self-Review (completed against the spec)

- **Spec coverage:** §4 tools → Tasks 8–12; §5 audibility ladder + `add_instrument`/`list_installed_fx` → Tasks 4–6; §5.4 `install_sound_pack` → Task 13; §6 theory helpers → Tasks 1–3; §7 assets → Task 6 (`ensure_drum_samples`); §8 testing → every task's test steps + Task 14 full-suite gate + Task 15 live smoke; §10 registry honesty → Task 14; §11 DoD → Tasks 14–15. **Gap found & fixed during review:** humanize's replace-semantics needed a `clear_track_midi` verb → added as Task 7 (not in the original spec's tool list but required by §4.4).
- **Placeholder scan:** the only `<PIN_BEFORE_SHIP>` / `add_vst_path_and_rescan` items are genuine ship-time/live-only values (checksums, an optional best-effort verb), each with an explicit fallback — not logic placeholders. The two "verify live" ReaScript notes (EnumInstalledFX, RS5k FILE0/note-range params) are known-unknown constants flagged in the spec's risk section, resolved in Task 15, and do not block the tested surfaces.
- **Type consistency:** `select_instrument` returns `{"kind","name","source"}` used consistently in Tasks 4/6/12; note dicts use `pitch/start_beat/duration_beats/velocity` everywhere; bridge fns `add_instrument`/`clear_track_midi`/`list_installed_fx` have one signature used by fake, Lua, and tool.
