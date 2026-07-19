# Orpheus Songwriting System (Slice 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** From a natural-language description (the model fills in the musical settings), build a complete, editable song in live REAPER — sectioned timeline with markers, chords, bass, drums, an in-key lead melody, and original lyrics placed on the timeline.

**Architecture:** Pure, REAPER-free helpers (`parse_melody`, named `DRUM_PATTERNS`) produce note data; a new `tools/arrange.py` module adds `add_marker` (bridge verb), a private `_build_section` placement engine, and the `build_section` / `arrange_song` / `place_lyric_markers` tools — all thin orchestrators over the Slice-1 helpers (`_write_notes`, `load_drumkit`, `resolve_progression`, `voice_lead`, `bassline_notes`, `parse_drum_grid`, `select_instrument`) and the M1 MIDI writer. Tested via pure unit tests + the `FakeReaperBridge` wire-protocol fake + the lupa Lua suite.

**Tech Stack:** Python 3.11 (FastMCP, Pydantic), Lua 5.4 ReaScript, pytest, lupa, ruff, mypy.

## Global Constraints

- **Python floor 3.11**; every module starts `from __future__ import annotations`.
- **Depends only on Slice 1 (merged) + M1.** No audio ingestion, no mixing, no vocal synth, no mid-song tempo/key change, 4/4 only.
- **No genre requirement.** New tools take explicit musical settings; the model supplies them from the user's words. `compose_section(genre)` stays untouched as legacy.
- **Reuse, don't duplicate:** use the existing helpers — `_write_notes` (module-level in `tools/compose.py`), `load_drumkit` (in `drumkit.py`), `resolve_progression`/`voice_lead` (`theory/chords.py`), `bassline_notes`/`parse_drum_grid`/`GM_DRUMS` (`theory/patterns.py`), `snap_to_scale`/`note_to_pc`/`NOTE_TO_PC` (`theory/music_theory_data.py`), `select_instrument` (`orpheus_mcp/instruments.py`).
- **No re-entrant tool calls:** tools never call sibling MCP tools; shared logic goes in module-level private functions (`_build_section`) that both a tool and the orchestrator call, using `BridgeClient` + theory helpers directly.
- **Musical time is beats;** notes are dicts `{"pitch","start_beat","duration_beats","velocity"}`; PPQ/tempo math stays in the bridge.
- **Registry honesty:** `arrange` category is added to `default`+`full` when it first has a real tool (Task 3). Tools in `default` need no `IMPLEMENTED`-set entry. Leave `explain` unchanged.
- **Lyrics are always original** text the model authors — never a referenced song's copyrighted lyrics.
- **Commits:** no Claude attribution trailers. Conventional messages, ≥1 per task.
- **Test commands** from repo root: `uv run pytest ...`, `uv run python scripts/run_lua_tests.py`, `uv run ruff check .`, `uv run mypy src`.
- **KNOWN FLAKE:** transient `os.replace`/`PermissionError` in `FakeReaperBridge` on Windows — rerun on hit, never "fix" it.

---

## Module Map (lock these names)

**Created:**
- `src/orpheus_mcp/theory/melody.py` — `DURATIONS`, `parse_melody`
- `src/orpheus_mcp/tools/arrange.py` — `_build_section` (private), `add_marker`, `build_section`, `arrange_song`, `place_lyric_markers` tools
- `tests/test_melody.py`, `tests/test_arrange.py`

**Modified:**
- `src/orpheus_mcp/theory/patterns.py` — add `DRUM_PATTERNS`
- `src/orpheus_mcp/tools/compose.py` — add `create_melody` tool
- `src/orpheus_mcp/bridge/lua/orpheus_bridge.lua` — add `add_marker` handler
- `src/orpheus_mcp/registry.py` — register `arrange` category (default + full)
- `tests/fake_reaper.py` — model `markers`; add `add_marker` handler
- `tests/lua/test_m1_handlers.lua`, `tests/test_registry.py`, docs

**Signatures other tasks rely on (verbatim):**
```python
# theory/melody.py
DURATIONS: dict[str, float]  # {"w":4.0,"h":2.0,"q":1.0,"e":0.5,"s":0.25}
def parse_melody(notation: str, key: str | None = None, mode: str | None = None,
                 velocity: int = 90) -> list[dict]: ...

# theory/patterns.py
DRUM_PATTERNS: dict[str, str]  # name -> one-bar step grid string

# bridge verb
# "add_marker" (name, bar) -> {"name","bar","index"}

# tools/arrange.py
def _build_section(bridge, key: str, mode: str, progression: str, bars: int,
                   at_bar: int, drums: str = "backbeat", melody: str | None = None,
                   inventory: list[str] | None = None,
                   drum_track: str = "drums", chord_track: str = "chords",
                   bass_track: str = "bass", lead_track: str = "lead") -> dict: ...
```

---

## Phase 1 — Pure helpers

### Task 1: `parse_melody`

**Files:** Create `src/orpheus_mcp/theory/melody.py`, `tests/test_melody.py`

**Interfaces:** Consumes `note_to_pc`, `snap_to_scale` from `music_theory_data`. Produces `DURATIONS`, `parse_melody`.

- [ ] **Step 1: Failing test**

```python
# tests/test_melody.py
"""Pure melody-notation parsing — no REAPER."""
from __future__ import annotations

import pytest

from orpheus_mcp.theory.melody import parse_melody


def test_simple_line_pitches_and_timing():
    notes = parse_melody("A4:q C5:q E5:h")
    assert [n["pitch"] for n in notes] == [69, 72, 76]
    assert [n["start_beat"] for n in notes] == [0.0, 1.0, 2.0]
    assert [n["duration_beats"] for n in notes] == [1.0, 1.0, 2.0]


def test_rest_advances_time_without_a_note():
    notes = parse_melody("A4:q r:q A4:q")
    assert [n["start_beat"] for n in notes] == [0.0, 2.0]
    assert len(notes) == 2


def test_default_octave_is_4():
    assert parse_melody("C:q")[0]["pitch"] == 60


def test_accidentals():
    assert parse_melody("F#4:q")[0]["pitch"] == 66
    assert parse_melody("Bb3:q")[0]["pitch"] == 58


def test_in_key_snapping():
    # F#4 (66) is out of C major; snapped to nearest in-scale pitch.
    notes = parse_melody("F#4:q", key="C", mode="major")
    assert notes[0]["pitch"] in (65, 67)  # F or G


def test_rejects_bad_token():
    for bad in ("H4:q", "A4:z", "A4", "A4:"):
        with pytest.raises(ValueError):
            parse_melody(bad)
```

- [ ] **Step 2: Run, verify fail** — `uv run pytest tests/test_melody.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/orpheus_mcp/theory/melody.py
"""Parse a model-friendly melody notation into Note-shaped dicts (beats). Pure."""
from __future__ import annotations

import re

from orpheus_mcp.theory.music_theory_data import NOTE_TO_PC, snap_to_scale

DURATIONS: dict[str, float] = {"w": 4.0, "h": 2.0, "q": 1.0, "e": 0.5, "s": 0.25}

_TOKEN = re.compile(r"^([A-Ga-g][#b]?)(-?\d+)?:([whqes])$")
_REST = re.compile(r"^[rR]:([whqes])$")


def parse_melody(
    notation: str, key: str | None = None, mode: str | None = None, velocity: int = 90
) -> list[dict]:
    """'A4:q C5:q E5:h' -> sequential note dicts. `r:dur` is a rest. Tokens are
    whitespace-separated `Name[octave]:dur` (octave defaults to 4). If key+mode given,
    non-rest pitches are snapped into the scale so the line stays in key."""
    notes: list[dict] = []
    beat = 0.0
    tokens = notation.split()
    if not tokens:
        raise ValueError("empty melody")
    for tok in tokens:
        rest = _REST.match(tok)
        if rest:
            beat += DURATIONS[rest.group(1)]
            continue
        m = _TOKEN.match(tok)
        if not m:
            raise ValueError(f"bad melody token: {tok!r}")
        name, octave, dur = m.group(1), m.group(2), m.group(3)
        try:
            pc = NOTE_TO_PC[name[:1].upper() + name[1:].lower()]
        except KeyError as exc:
            raise ValueError(f"bad note name in {tok!r}") from exc
        octn = int(octave) if octave is not None else 4
        pitch = pc + 12 * (octn + 1)  # MIDI: C4 = 60
        if key is not None and mode is not None:
            pitch = snap_to_scale([pitch], key, mode)[0]
        notes.append({"pitch": pitch, "start_beat": round(beat, 6),
                      "duration_beats": DURATIONS[dur], "velocity": velocity})
        beat += DURATIONS[dur]
    return notes
```

- [ ] **Step 4: Run, verify pass** — `uv run pytest tests/test_melody.py -v`.

- [ ] **Step 5: Commit**
```bash
git add src/orpheus_mcp/theory/melody.py tests/test_melody.py
git commit -m "feat(theory): melody-notation parser (in-key, rests)"
```

---

### Task 2: `DRUM_PATTERNS`

**Files:** Modify `src/orpheus_mcp/theory/patterns.py`; Test `tests/test_patterns.py` (append)

**Interfaces:** Produces `DRUM_PATTERNS: dict[str,str]` (each value a one-bar grid parseable by the existing `parse_drum_grid`).

- [ ] **Step 1: Failing test (append tests/test_patterns.py)**

```python
from orpheus_mcp.theory.patterns import DRUM_PATTERNS, parse_drum_grid


def test_named_patterns_parse_and_are_nonempty():
    assert {"backbeat", "halftime", "fourfloor"}.issubset(DRUM_PATTERNS)
    for name, grid in DRUM_PATTERNS.items():
        hits = parse_drum_grid(grid)
        assert hits, f"{name} produced no hits"


def test_backbeat_has_snare_on_2_and_4():
    snares = [h for h in parse_drum_grid(DRUM_PATTERNS["backbeat"])
              if h["pitch"] == 38]
    assert [s["start_beat"] for s in snares] == [1.0, 3.0]
```

- [ ] **Step 2: Run, verify fail** — `uv run pytest tests/test_patterns.py -k DRUM -v`.

- [ ] **Step 3: Implement (append to patterns.py)**

```python
# Named one-bar drum grids (16 steps = 4/4 bar). Values parse via parse_drum_grid.
DRUM_PATTERNS: dict[str, str] = {
    "backbeat": (
        "kick:  x...x...x...x...\n"
        "snare: ....x.......x...\n"
        "hat:   x.x.x.x.x.x.x.x."
    ),
    "halftime": (
        "kick:  x.......x.......\n"
        "snare: ........x.......\n"
        "hat:   x.x.x.x.x.x.x.x."
    ),
    "fourfloor": (
        "kick:  x...x...x...x...\n"
        "snare: ....x...x...x...\n"
        "hat:   ..x...x...x...x."
    ),
}
```

- [ ] **Step 4: Run, verify pass** — `uv run pytest tests/test_patterns.py -v`.

- [ ] **Step 5: Commit**
```bash
git add src/orpheus_mcp/theory/patterns.py tests/test_patterns.py
git commit -m "feat(theory): named drum patterns (backbeat/halftime/fourfloor)"
```

---

## Phase 2 — `add_marker` bridge verb + `arrange` module

### Task 3: `add_marker` (Lua + fake + tool + registry)

**Files:** Modify `tests/fake_reaper.py`, `src/orpheus_mcp/bridge/lua/orpheus_bridge.lua`, `src/orpheus_mcp/registry.py`, `tests/lua/test_m1_handlers.lua`; Create `src/orpheus_mcp/tools/arrange.py`, `tests/test_arrange.py`

**Interfaces:** Produces bridge fn `add_marker(name, bar) -> {"name","bar","index"}`; tool `add_marker`; new `arrange` category.

- [ ] **Step 1: Failing contract test**

```python
# tests/test_arrange.py
"""Contract + tool tests for the arrange module, over the real wire protocol."""
from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.bridge.client import BridgeClient
from orpheus_mcp.registry import register_tools


@pytest.fixture
def project():
    p = FakeReaperProject(tempo=120.0)
    p.installed_fx = ["VSTi: ReaSynth (Cockos)"]
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


def test_add_marker_contract(client, project):
    res = client.call("add_marker", name="Verse 1", bar=5)
    assert res["name"] == "Verse 1"
    assert res["bar"] == 5
    assert project.markers == [{"name": "Verse 1", "bar": 5}]


async def test_add_marker_tool(mcp_client, project):
    async with mcp_client as c:
        res = await c.call_tool("add_marker", {"name": "Chorus", "bar": 9})
    assert res.data["bar"] == 9
    assert project.markers[-1]["name"] == "Chorus"
```

- [ ] **Step 2: Run, verify fail** — `uv run pytest tests/test_arrange.py -v` → `unknown fn: add_marker`.

- [ ] **Step 3a: Fake (tests/fake_reaper.py)** — add field to `FakeReaperProject`:
```python
    markers: list[dict] = field(default_factory=list)
```
add + register handler in `make_handlers`:
```python
    def add_marker(p):
        entry = {"name": p["name"], "bar": int(p["bar"])}
        project.markers.append(entry)
        return {"name": entry["name"], "bar": entry["bar"], "index": len(project.markers)}
```
```python
        "add_marker": add_marker,
```

- [ ] **Step 3b: Lua handler (orpheus_bridge.lua)**
```lua
-- Place a named project marker at the start of a bar.
HANDLERS.add_marker = function(p)
  -- NOTE (verify live): AddProjectMarker2(proj, isrgn, pos, rgnend, name, wantidx, color)
  -- returns the created marker index on REAPER 7.x. The fake + lua-stub tests do not
  -- depend on the real API; confirmed by the live smoke.
  local qn = bar_start_qn(p.bar or 1)
  local pos = reaper.TimeMap2_QNToTime(0, qn)
  local idx = reaper.AddProjectMarker2(0, false, pos, 0, p.name or "", -1, 0)
  return { name = p.name or "", bar = p.bar or 1, index = idx }
end
```

- [ ] **Step 3c: Tool + registration (src/orpheus_mcp/tools/arrange.py)**
```python
# src/orpheus_mcp/tools/arrange.py
"""Song arrangement: section markers + section/song builders (Slice 2)."""
from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient

_DESTRUCTIVE = {"destructiveHint": True}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(annotations=_DESTRUCTIVE)
    def add_marker(name: str, bar: int = 1) -> dict:
        """Place a named marker at the start of `bar` (1-based) on the REAPER timeline."""
        result = BridgeClient().call("add_marker", name=name, bar=bar)
        return {"name": result.get("name"), "bar": result.get("bar"),
                "index": result.get("index")}
```

- [ ] **Step 3d: Registry (registry.py)** — add to `_CATEGORY_IMPORTS`:
```python
    "arrange": "orpheus_mcp.tools.arrange",
```
add `"arrange"` to the `default` profile tuple (after `"compose"`). `full` picks it up automatically. Leave `explain` unchanged.

- [ ] **Step 3e: Lua-suite assertion (tests/lua/test_m1_handlers.lua)** — add stubs + assertion:
```lua
  -- in the reaper stub table:
  AddProjectMarker2 = function(_, _, _, _, _, wantidx) return wantidx == -1 and 1 or wantidx end,
  TimeMap2_QNToTime = reaper.TimeMap2_QNToTime or function(_, qn) return qn * 0.5 end,
```
```lua
do
  local r = M.dispatch("add_marker", { name = "Verse", bar = 3 })
  eq(r.ok, true, "add_marker ok")
  eq(r.result.name, "Verse", "marker name")
  eq(r.result.bar, 3, "marker bar")
end
```
> If `TimeMap2_QNToTime` already exists in the stub, don't redefine it — reuse it.

- [ ] **Step 4: Run** — `uv run pytest tests/test_arrange.py -v` and `uv run python scripts/run_lua_tests.py` → green.

- [ ] **Step 5: Commit**
```bash
git add tests/fake_reaper.py src/orpheus_mcp/bridge/lua/orpheus_bridge.lua \
        src/orpheus_mcp/tools/arrange.py src/orpheus_mcp/registry.py \
        tests/test_arrange.py tests/lua/test_m1_handlers.lua
git commit -m "feat(arrange): add_marker bridge verb + tool + arrange category"
```

---

## Phase 3 — melody + section + song

### Task 4: `create_melody` tool

**Files:** Modify `src/orpheus_mcp/tools/compose.py`; Test `tests/test_compose.py` (append)

**Interfaces:** Consumes `parse_melody` (Task 1), `_write_notes` (compose.py), `select_instrument`. Produces tool `create_melody(track, notation, key=None, mode=None, at_bar=1) -> dict`.

- [ ] **Step 1: Failing test (append tests/test_compose.py)**

```python
async def test_create_melody_writes_in_key_line(mcp_client, project):
    async with mcp_client as c:
        await c.call_tool("create_track", {"name": "lead"})
        res = await c.call_tool(
            "create_melody",
            {"track": "lead", "notation": "A4:q C5:q E5:h", "key": "A", "mode": "minor"},
        )
    tr = project.resolve_track("lead")
    assert res.data["notes_written"] == 3
    assert [n.pitch for n in tr.takes[0].notes] == [69, 72, 76]
    assert "ReaSynth" in tr.fx
```

- [ ] **Step 2: Run, verify fail** — unknown tool `create_melody`.

- [ ] **Step 3: Implement (append inside compose.py `register()`)**
```python
    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_melody(
        track: str, notation: str, key: str | None = None,
        mode: str | None = None, at_bar: int = 1,
    ) -> dict:
        """Write a lead melody from note+rhythm notation (e.g. 'A4:q C5:q E5:h'), kept in
        key if key+mode are given. Auto-loads a lead synth so it's audible."""
        from orpheus_mcp.theory.melody import parse_melody

        notes = parse_melody(notation, key=key, mode=mode)
        bridge = BridgeClient()
        written = _write_notes(bridge, track, notes, at_bar=at_bar)
        bridge.call("add_instrument", track=track, kind="named", name="ReaSynth")
        return {"track": track, "notes_written": written}
```

- [ ] **Step 4: Run, verify pass** — `uv run pytest tests/test_compose.py -k melody -v`.

- [ ] **Step 5: Commit**
```bash
git add src/orpheus_mcp/tools/compose.py tests/test_compose.py
git commit -m "feat(compose): create_melody (in-key lead line)"
```

---

### Task 5: `_build_section` + `build_section` tool

**Files:** Modify `src/orpheus_mcp/tools/arrange.py`; Test `tests/test_arrange.py` (append)

**Interfaces:** Consumes `resolve_progression`/`voice_lead`, `bassline_notes`/`parse_drum_grid`/`DRUM_PATTERNS`, `parse_melody`, `_write_notes` (import from compose), `load_drumkit` (import from drumkit), `select_instrument`, bridge `create_track`/`add_instrument`. Produces module-level `_build_section(...)` + tool `build_section(...)`.

- [ ] **Step 1: Failing test (append tests/test_arrange.py)**

```python
async def test_build_section_lays_all_parts_at_offset(mcp_client, project):
    async with mcp_client as c:
        res = await c.call_tool(
            "build_section",
            {"key": "A", "mode": "minor", "progression": "i-iv-V-i",
             "bars": 4, "at_bar": 1, "drums": "backbeat", "melody": "A4:q C5:q E5:h"},
        )
    names = {t.name for t in project.tracks}
    assert {"drums", "chords", "bass", "lead"}.issubset(names)
    assert len(project.resolve_track("drums").fx) == 3  # stock kit
    assert project.resolve_track("chords").takes[0].notes  # chords written
    assert project.resolve_track("lead").takes[0].notes    # melody written
    assert res.data["at_bar"] == 1 and res.data["bars"] == 4


async def test_build_section_second_section_offsets(mcp_client, project):
    async with mcp_client as c:
        await c.call_tool("build_section",
            {"key": "A", "mode": "minor", "progression": "i-iv", "bars": 2, "at_bar": 1})
        await c.call_tool("build_section",
            {"key": "A", "mode": "minor", "progression": "VI-VII", "bars": 2, "at_bar": 3})
    # chords track holds notes from both sections (bar 1 and bar 3 => beat >= 8)
    starts = [n.start_ppq for n in project.resolve_track("chords").takes[0].notes]
    assert max(starts) >= 8 * 960  # a note at/after bar 3 (8 beats in)
```

- [ ] **Step 2: Run, verify fail** — unknown tool `build_section`.

- [ ] **Step 3: Implement (arrange.py — add module-level helper ABOVE register(), and the tool INSIDE register())**

Module level (top of file, after imports add these imports):
```python
from orpheus_mcp.drumkit import load_drumkit
from orpheus_mcp.instruments import select_instrument
from orpheus_mcp.theory.chords import resolve_progression, voice_lead
from orpheus_mcp.theory.melody import parse_melody
from orpheus_mcp.theory.patterns import DRUM_PATTERNS, bassline_notes, parse_drum_grid
from orpheus_mcp.tools.compose import _write_notes
```
```python
def _build_section(
    bridge, key: str, mode: str, progression: str, bars: int, at_bar: int,
    drums: str = "backbeat", melody: str | None = None,
    inventory: list[str] | None = None,
    drum_track: str = "drums", chord_track: str = "chords",
    bass_track: str = "bass", lead_track: str = "lead",
) -> dict:
    """Lay one section (chords+bass+drums, optional melody) at `at_bar` on shared tracks.
    Creates missing tracks (idempotent by name) and loads instruments. Returns placement."""
    inv = inventory if inventory is not None else bridge.call("list_installed_fx").get("fx", [])

    # tracks (create is idempotent-enough for the fake/live; name lookup drives writes)
    existing = {t["name"] for t in bridge.call("list_tracks")}
    needed = [drum_track, chord_track, bass_track] + ([lead_track] if melody else [])
    for name in needed:
        if name not in existing:
            bridge.call("create_track", name=name)

    # chords: 1 bar/chord, progression repeated to fill `bars`
    voiced = voice_lead(resolve_progression(progression, key=key, mode=mode, octave=4))
    filled = [voiced[i % len(voiced)] for i in range(bars)]
    chord_notes = [
        {"pitch": p, "start_beat": i * 4.0, "duration_beats": 4.0, "velocity": 88}
        for i, chord in enumerate(filled) for p in chord
    ]
    _write_notes(bridge, chord_track, chord_notes, at_bar=at_bar)
    ci = select_instrument("keys", inv)
    bridge.call("add_instrument", track=chord_track, kind="named",
                name=ci["name"] if ci["kind"] == "named" else "ReaSynth")

    # bass: roots from a low-octave resolution
    bass_chords = resolve_progression(progression, key=key, mode=mode, octave=2)
    bass_filled = [bass_chords[i % len(bass_chords)] for i in range(bars)]
    _write_notes(bridge, bass_track, bassline_notes(bass_filled, style="root",
                                                    bars_per_chord=1), at_bar=at_bar)
    bi = select_instrument("bass", inv)
    bridge.call("add_instrument", track=bass_track, kind="named",
                name=bi["name"] if bi["kind"] == "named" else "ReaSynth")

    # drums: named pattern (or raw grid), tiled across `bars`
    grid = DRUM_PATTERNS.get(drums, drums)
    one_bar = parse_drum_grid(grid)
    drum_notes = [
        {**hit, "start_beat": hit["start_beat"] + b * 4.0}
        for b in range(bars) for hit in one_bar
    ]
    _write_notes(bridge, drum_track, drum_notes, at_bar=at_bar)
    di = select_instrument("drums", inv)
    if di["kind"] == "drumkit":
        load_drumkit(bridge, drum_track)
    else:
        bridge.call("add_instrument", track=drum_track, kind="named", name=di["name"])

    # optional melody
    if melody:
        mel = parse_melody(melody, key=key, mode=mode)
        _write_notes(bridge, lead_track, mel, at_bar=at_bar)
        li = select_instrument("keys", inv)
        bridge.call("add_instrument", track=lead_track, kind="named",
                    name=li["name"] if li["kind"] == "named" else "ReaSynth")

    return {"at_bar": at_bar, "bars": bars, "tracks": needed}
```
Inside `register()`:
```python
    @mcp.tool(annotations=_DESTRUCTIVE)
    def build_section(
        key: str, mode: str, progression: str, bars: int = 4, at_bar: int = 1,
        drums: str = "backbeat", melody: str | None = None,
    ) -> dict:
        """Lay one section — chords, bass, drums, and an optional in-key melody — at `at_bar`
        on the shared drums/chords/bass/lead tracks. `drums` is a named pattern
        (backbeat/halftime/fourfloor) or a raw step grid."""
        return _build_section(BridgeClient(), key=key, mode=mode, progression=progression,
                              bars=bars, at_bar=at_bar, drums=drums, melody=melody)
```

- [ ] **Step 4: Run, verify pass** — `uv run pytest tests/test_arrange.py -v`; then `uv run pytest -q`.

- [ ] **Step 5: Commit**
```bash
git add src/orpheus_mcp/tools/arrange.py tests/test_arrange.py
git commit -m "feat(arrange): _build_section + build_section (chords/bass/drums/melody at offset)"
```

---

### Task 6: `arrange_song` orchestrator

**Files:** Modify `src/orpheus_mcp/tools/arrange.py`; Test `tests/test_arrange.py` (append)

**Interfaces:** Consumes `_build_section` (Task 5), bridge `set_tempo`/`create_track`/`add_marker`/`list_installed_fx`. Produces tool `arrange_song(tempo, key, mode, sections, humanize=False) -> dict`.

- [ ] **Step 1: Failing test (append)**

```python
async def test_arrange_song_places_sections_and_markers(mcp_client, project):
    sections = [
        {"name": "Verse", "bars": 2, "progression": "i-iv"},
        {"name": "Chorus", "bars": 2, "progression": "VI-VII", "melody": "A4:q C5:q E5:h"},
    ]
    async with mcp_client as c:
        res = await c.call_tool(
            "arrange_song",
            {"tempo": 72, "key": "A", "mode": "minor", "sections": sections},
        )
    assert 71 <= project.tempo <= 73
    assert [m["name"] for m in project.markers] == ["Verse", "Chorus"]
    assert [m["bar"] for m in project.markers] == [1, 3]  # Verse@1, Chorus after 2 bars
    assert res.data["sections"] == [
        {"name": "Verse", "at_bar": 1, "bars": 2},
        {"name": "Chorus", "at_bar": 3, "bars": 2},
    ]
    assert len(project.resolve_track("drums").fx) == 3
```

- [ ] **Step 2: Run, verify fail** — unknown tool `arrange_song`.

- [ ] **Step 3: Implement (inside register())**
```python
    @mcp.tool(annotations=_DESTRUCTIVE)
    def arrange_song(
        tempo: float, key: str, mode: str, sections: list[dict], humanize: bool = False,
    ) -> dict:
        """Build a full song: set tempo, then place each section end-to-end on shared tracks
        with a marker at its start. `sections` is a list of
        {name, bars, progression, drums?, melody?}. The model composes this list + the
        melodies/lyrics from the user's description."""
        bridge = BridgeClient()
        bridge.call("set_tempo", bpm=float(tempo))
        inv = bridge.call("list_installed_fx").get("fx", [])
        bar = 1
        placed: list[dict] = []
        for s in sections:
            bridge.call("add_marker", name=s["name"], bar=bar)
            _build_section(bridge, key=key, mode=mode, progression=s["progression"],
                           bars=int(s["bars"]), at_bar=bar,
                           drums=s.get("drums", "backbeat"), melody=s.get("melody"),
                           inventory=inv)
            placed.append({"name": s["name"], "at_bar": bar, "bars": int(s["bars"])})
            bar += int(s["bars"])
        if humanize:
            for tr in ("drums", "chords", "bass", "lead"):
                try:
                    bridge.call("get_track_midi", track=tr)  # skip empty/missing gracefully
                except Exception:  # noqa: BLE001
                    pass
        return {"tempo": float(tempo), "key": key, "sections": placed,
                "markers": [{"name": p["name"], "bar": p["at_bar"]} for p in placed]}
```
> `humanize` is threaded but intentionally minimal in v1 (per-track humanize can be added once the live smoke confirms track state); keep the flag so the API is stable. If the reviewer prefers, drop the flag rather than ship a no-op — flag this in your report for the controller to decide.

- [ ] **Step 4: Run, verify pass** — `uv run pytest tests/test_arrange.py -v`; then `uv run pytest -q`.

- [ ] **Step 5: Commit**
```bash
git add src/orpheus_mcp/tools/arrange.py tests/test_arrange.py
git commit -m "feat(arrange): arrange_song orchestrator (sections + markers -> full song)"
```

---

### Task 7: `place_lyric_markers`

**Files:** Modify `src/orpheus_mcp/tools/arrange.py`; Test `tests/test_arrange.py` (append)

**Interfaces:** Consumes bridge `add_marker`. Produces tool `place_lyric_markers(lines, at_bars) -> dict`.

- [ ] **Step 1: Failing test (append)**

```python
async def test_place_lyric_markers(mcp_client, project):
    async with mcp_client as c:
        res = await c.call_tool(
            "place_lyric_markers",
            {"lines": ["Verse: first line here", "Chorus: hook line here"],
             "at_bars": [1, 9]},
        )
    assert res.data["placed"] == 2
    assert [m["bar"] for m in project.markers] == [1, 9]
    assert "first line" in project.markers[0]["name"]
```

- [ ] **Step 2: Run, verify fail** — unknown tool `place_lyric_markers`.

- [ ] **Step 3: Implement (inside register())**
```python
    @mcp.tool(annotations=_DESTRUCTIVE)
    def place_lyric_markers(lines: list[str], at_bars: list[int]) -> dict:
        """Place model-authored lyric lines as timeline markers. `lines[i]` goes at
        `at_bars[i]`. Lyrics must be original text, never a copyrighted song's lyrics."""
        if len(lines) != len(at_bars):
            raise ValueError("lines and at_bars must be the same length")
        bridge = BridgeClient()
        for line, bar in zip(lines, at_bars):
            bridge.call("add_marker", name=line, bar=int(bar))
        return {"placed": len(lines)}
```

- [ ] **Step 4: Run, verify pass** — `uv run pytest tests/test_arrange.py -v`.

- [ ] **Step 5: Commit**
```bash
git add src/orpheus_mcp/tools/arrange.py tests/test_arrange.py
git commit -m "feat(arrange): place_lyric_markers (original lyrics on timeline)"
```

---

## Phase 4 — gate + docs

### Task 8: Full gate + registry confirm + docs

**Files:** Modify `tests/test_registry.py`, `docs/dev-log.md`, `docs/roadmap.md`, `README.md`

**Interfaces:** Confirms `arrange` in `default`/`full`, not in `explain`; full suite/lint/type gate green; docs updated.

- [ ] **Step 1: Failing/added test (tests/test_registry.py)**
```python
def test_arrange_in_default_not_explain():
    from fastmcp import FastMCP
    from orpheus_mcp.registry import register_tools

    m1 = FastMCP(name="t")
    assert "arrange" in register_tools(m1, profile="default")
    m2 = FastMCP(name="t")
    assert "arrange" not in register_tools(m2, profile="explain")
```

- [ ] **Step 2: Run, verify** — fails only if `arrange` wasn't added to `default` in Task 3; if already green, note that and proceed.

- [ ] **Step 3: Run the FULL gate**
```
uv run pytest -q
uv run python scripts/run_lua_tests.py
uv run ruff check .
uv run mypy src
```
Fix any ruff/mypy issue in files this slice created/modified (e.g. line length, blind excepts). Report pre-existing unrelated issues rather than fixing broadly.

- [ ] **Step 4: Docs**
- `docs/dev-log.md`: dated entry — Slice 2 songwriting system shipped (melody parser, named drum patterns, `add_marker`, `build_section`, `arrange_song`, `place_lyric_markers`); tested via unit + fake + lua; **live smoke pending** (`AddProjectMarker2` live behavior + audible multi-section song). State find/download of un-owned reference audio was **rejected** (legal), and audio stem ingestion remains deferred Slice 4 (owned files only).
- `docs/roadmap.md`: note Slice 2 shipped; Slice 4 (`reference-ingest`) still deferred and scoped to owned files.
- `README.md`: advance status to mention the full-song composer (description → sectioned, audible song), live-verify pending.

- [ ] **Step 5: Commit**
```bash
git add tests/test_registry.py docs/dev-log.md docs/roadmap.md README.md
git commit -m "docs/registry: Slice 2 songwriting system shipped (live smoke pending)"
```

---

## Self-Review (against the spec)

- **Coverage:** §5 add_marker → Task 3; §6 create_melody/parse_melody → Tasks 1,4; §7 build_section → Task 5; §8 arrange_song → Task 6; §9 lyrics → Task 7 (+ model authorship, not a tool); §10 tests → each task + Task 8 gate; §11 limits documented; DRUM_PATTERNS (§7) → Task 2.
- **Placeholders:** the `AddProjectMarker2` "verify live" note is a known live-only unknown (resolved in the user's live smoke), not a logic gap. The `arrange_song` `humanize` flag is a deliberate stable-API no-op with an explicit reviewer call-out — the one thing to adjudicate.
- **Type/name consistency:** note dicts use `pitch/start_beat/duration_beats/velocity` throughout; `_build_section` signature is shared by the `build_section` tool and `arrange_song`; `add_marker` bridge fn shape `{name,bar,index}` matches fake + lua + tool.
- **Reuse:** every builder reuses Slice-1 helpers (`_write_notes`, `load_drumkit`, `resolve_progression`, `voice_lead`, `bassline_notes`, `parse_drum_grid`, `select_instrument`) — no duplicated logic beyond the small, section-local note assembly.
