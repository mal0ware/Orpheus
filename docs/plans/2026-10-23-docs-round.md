# Docs Round (Step 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every doc and Claude's operating instructions describe Orpheus as agreed in
`docs/specs/2026-10-23-orpheus-definition-design.md`, drop the unused GPL Matchering dependency,
bring the dev-log up to date, and leave `feat/reference-analysis` ready to merge into `main`.

**Architecture:** One code change: `server.py` loads its MCP instructions from
`data/instructions/00_core.md` (installed-package path first, repo path as fallback, the same
pattern as `analysis/config.py`), guarded by a test that the instructions only name tools the
default profile really registers. Everything else is docs and `pyproject.toml`.

**Tech Stack:** Python 3.12, FastMCP 3.x, uv, pytest, hatchling (wheel force-includes `data/`).

## Global Constraints

- Definition text: §1 of the spec, verbatim in meaning; the README may shorten it.
- No lyrics, song titles or sheet IDs from the owner's Google Sheets in the repo.
- Commits: no Claude attribution trailers.
- This desktop's environment is `uv sync --locked --extra separation --extra dev`. Never run
  `uv run --extra analysis` here: it swaps torch to the CPU build and breaks torchaudio and
  faster-whisper (happened 2026-07-30; repaired 2026-10-23). Use `uv run --no-sync` for checks.
- Keep `[tool.uv] constraint-dependencies` (numba/llvmlite) and the analysis/separation conflict
  partition intact when re-locking.
- Numbers quoted in docs come from commands run that day, with the date next to them.

---

### Task 1: Server loads one instructions file

**Files:**
- Modify: `src/orpheus_mcp/server.py:16-31` (replace the hard-coded `INSTRUCTIONS`)
- Rewrite: `data/instructions/00_core.md`
- Test: `tests/test_server_instructions.py` (new)

**Interfaces:**
- Produces: `orpheus_mcp.server.load_instructions() -> str`; `mcp.instructions` equals it.

- [ ] **Step 1: Write the failing test**

```python
"""The server's MCP instructions come from data/instructions/00_core.md and only name real tools."""
from __future__ import annotations

import re

from fastmcp import FastMCP

from orpheus_mcp.registry import register_tools


async def _tool_names(profile: str) -> set[str]:
    m = FastMCP(name="probe")
    register_tools(m, profile=profile)
    return {t.name for t in await m.list_tools()}


def test_instructions_load_from_core_file():
    from orpheus_mcp import server

    text = server.load_instructions()
    assert text.startswith("# Orpheus")
    assert server.mcp.instructions == text


async def test_instructions_name_only_default_profile_tools():
    from orpheus_mcp import server

    default = await _tool_names("default")
    every_tool = await _tool_names("full")
    named = set(re.findall(r"`([a-z][a-z0-9_]*)`", server.load_instructions()))
    # Only identifiers that ARE tool names somewhere (incl. stubs) are checked.
    mentioned_tools = named & every_tool
    assert mentioned_tools, "instructions should name at least one tool"
    assert mentioned_tools <= default, f"not in default profile: {mentioned_tools - default}"
```

- [ ] **Step 2: Run it, expect FAIL** — `uv run --no-sync pytest tests/test_server_instructions.py -v`
  → `AttributeError: module 'orpheus_mcp.server' has no attribute 'load_instructions'`.

- [ ] **Step 3: Implement**

In `server.py`, replace the `INSTRUCTIONS = """..."""` block with:

```python
from pathlib import Path

_CORE = Path("instructions") / "00_core.md"


def load_instructions() -> str:
    """Claude's operating instructions, from data/instructions/00_core.md.

    Installed wheel: orpheus_mcp/data/... (hatch force-include). Source checkout: <repo>/data/...
    """
    package_path = Path(__file__).resolve().parent / "data" / _CORE
    if package_path.exists():
        return package_path.read_text(encoding="utf-8")
    return (Path(__file__).resolve().parents[2] / "data" / _CORE).read_text(encoding="utf-8")


INSTRUCTIONS = load_instructions()
```

Rewrite `data/instructions/00_core.md` to the definition: what Orpheus is (one paragraph);
**Always** (ground first with `get_connection_status` then reads; beats not ticks; stay in key with
`get_scale_notes`/`constrain_to_key`; ask before any destructive tool and say what it will change;
batch notes); **Listening** (`analyze_reference` routes questions about owned audio; report
confidence; owned audio only, never fetch); **Songwriting** (lyrics come from the user's Google
Sheet via the Drive connector, passed as text; `arrange_song` + `create_melody` +
`place_lyric_markers`); **Be honest** (detection is probabilistic; taste is not measurement;
MIDI rebuilt from a produced master is an editable sketch); **Never** (claim success without a
tool result; suggest plugins as owned unless `list_installed_fx` shows them — `suggest_free_vst`
recommends free ones explicitly; touch audio the user doesn't own). No stub tool names.

- [ ] **Step 4: Run it, expect PASS**, plus `uv run --no-sync pytest tests/test_registry.py -q`.

- [ ] **Step 5: Commit** — `git commit -m "feat(server): load MCP instructions from data/instructions/00_core.md"`

### Task 2: Remove Matchering from core dependencies

**Files:**
- Modify: `pyproject.toml:27`, `uv.lock` (regenerated), `src/orpheus_mcp/analysis/match.py:1-15`
  (docstring only), `src/orpheus_mcp/tools/apply.py:31` (docstring only)
- Test: `tests/test_packaging.py` (add one test)

- [ ] **Step 1: Write the failing test** (append to `tests/test_packaging.py`)

```python
def test_core_dependencies_have_no_gpl_matchering():
    # MIT package: GPL-3.0 Matchering may only ever be an optional extra (spec 2026-10-23 §3).
    deps = _pyproject()["project"]["dependencies"]
    assert not any(d.lower().startswith("matchering") for d in deps)
```

- [ ] **Step 2: Run it, expect FAIL** — `uv run --no-sync pytest tests/test_packaging.py -v`.
- [ ] **Step 3: Implement** — delete `"matchering>=2.0",` from `[project] dependencies`; run
  `uv lock`; confirm `git diff uv.lock` only drops matchering and its now-orphaned deps; run
  `uv sync --locked --extra separation --extra dev`. Docstrings: match.py says Matchering is not a
  dependency and comes back as an optional `[master]` extra when M3 builds it; apply.py's
  `apply_master_match` docstring keeps "[NOT IMPLEMENTED]".
- [ ] **Step 4: Run** `uv run --no-sync pytest tests/test_packaging.py -q` and
  `grep -rn matchering src pyproject.toml` (only the match.py docstring). Expect PASS.
- [ ] **Step 5: Commit** — `git commit -m "build: drop unused GPL-3.0 matchering from core deps"`

### Task 3: README, architecture and roadmap match the definition

**Files:** `README.md`, `docs/architecture.md`, `docs/roadmap.md`, `docs/frontier-analysis.md`
(Matchering wording only)

- [ ] **Step 1: Gather numbers** — `uv run --no-sync pytest --collect-only -q | tail -1`,
  `uv run --no-sync python scripts/run_lua_tests.py` (sum the per-suite passes), the default-profile
  tool count, and the bridge handler count (`grep -c` of `HANDLERS.` definitions).
- [ ] **Step 2: README** — sections: one-line pitch + the §1 definition; "What works today"
  (listen / explain / work, honest about what is fake-tested vs live-verified, with the numbers and
  date); "How a song gets made" (Sheets → Claude → Orpheus → REAPER → record → refine);
  "What it won't do"; "Not built yet" (recommend/apply loop, render tools, lead sheet + ABC, YuE2
  sketches, beat-map export for Premiere); install (unchanged, still "planned"); optional
  non-commercial backends (off by default, `ORPHEUS_NC_BACKENDS`, weights downloaded separately,
  CC BY-NC); credits with Matchering moved to "optional outside tool, not a dependency" and the
  "vendors Matchering" sentence removed; licence.
- [ ] **Step 3: architecture.md** — replace "The spine: analyze → recommend → apply" with
  listen → explain → work (approval gate + undo), keep the bridge sections, add hot reload and the
  isolated-env backends (audio-separator, basic-pitch; YuE2 later over HTTP/CLI), drop Matchering
  from the module list.
- [ ] **Step 4: roadmap.md** — mark reference-analysis + live July verbs as done with dates; add an
  "Upgrades 2026-10" block listing the spec's §4 steps 2–5 plus the beat-map/OTIO export and the
  lyric-engine fold-in; keep M3 recommend/apply as later.
- [ ] **Step 5: Humanizer pass** on all three, then re-read for factual drift.
- [ ] **Step 6: Commit** — `git commit -m "docs: README, architecture and roadmap match the agreed definition"`

### Task 4: Dev-log catch-up

**Files:** `docs/dev-log.md` (two new entries at the top)

- [ ] **Step 1: 2026-07-25 → 08-01 entry** from `git log` and `docs/pins/2026-07-model-pins.md`:
  live REAPER 7.78 smoke and FX-map corrections; first real-music run (mp4 ingest, timings, the
  three new issues); `import_audio_files` + the timebase fix; the 15 bridge verbs added 07-26..31
  and hot reload; section-diff; the vocal session (tuning, doubles, telephone FX); the MIDI
  rebuild and sampler instruments; the 08-01 lesson ("MIDI reconstruction of a produced master is
  an editable sketch"); and an honest note that bass/guitar tabs were fetched from Ultimate Guitar
  and UkuTabs on 07-30/31 to cross-check transcriptions (text only, no audio).
- [ ] **Step 2: 2026-10-23 entry** — the definition and decisions (link the spec), the venv drift
  found and repaired (torch 2.13 CPU vs torchaudio 2.6 cu124; setuptools 83; 12 order-dependent
  failures + exit 139), Matchering removal, instructions file, numbers, what's next.
- [ ] **Step 3: Commit** — `git commit -m "docs(dev-log): catch up 2026-07-25..08-01 and 2026-10-23"`

### Task 5: Merge readiness

- [ ] **Step 1:** `uv run --no-sync pytest -q -m "not desktop and not gpu"` → 0 failed;
  `uv run --no-sync python scripts/run_lua_tests.py` → 0 failed;
  `uv run --no-sync ruff check .` and `uv run --no-sync mypy` → clean.
- [ ] **Step 2:** Show the owner the summary and ask for the merge OK (merge commit vs squash).
- [ ] **Step 3:** On yes: `git checkout main && git merge --no-ff feat/reference-analysis`, re-run
  Step 1 on main, and ask before pushing.
