# Orpheus — Development Log

A running record of what's built, what's verified, and where to pick up next. Newest entry first. This is the session-to-session handoff doc — read the top entry to resume.

---

## ▶ NEXT SESSION — resume here

**Picking up on:** the **Windows Desktop PC** (the primary REAPER machine). Everything below was built on a Mac; the bridge is cross-platform but needs a fresh setup on Windows.

### Windows setup (first time on this machine)

```powershell
# 1. Get the code
git clone https://github.com/mal0ware/Orpheus
cd Orpheus

# 2. Install uv if you don't have it
#    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Install deps (fastmcp, pydantic, music21, …) + dev tools
uv sync --extra dev          # add --extra analysis later for librosa

# 4. Install the in-REAPER bridge script (finds %APPDATA%\REAPER automatically)
uv run orpheus-mcp install-bridge
#    → copies orpheus_bridge.lua into %APPDATA%\REAPER\Scripts\orpheus\
```

### Smoke test M0 against your real REAPER (do this first to confirm the bridge works on Windows)

1. Open REAPER. **Actions → Show action list → "ReaScript: Run…" → `orpheus_bridge.lua`.**
   Console should print: `Orpheus bridge listening in: C:\Users\<you>\.orpheus_bridge`.
2. In a terminal:
   ```powershell
   uv run python -c "from orpheus_mcp.bridge.client import BridgeClient; print(BridgeClient().call('get_connection_status'))"
   ```
   Expect `{'ok': True, 'reaper_version': '7.x/win64', 'bridge_dir': '…\\.orpheus_bridge'}`.
   A clean `BridgeTimeout` (not a hang) means the .lua isn't running in REAPER.

### Run the test suite
```powershell
uv run pytest -q          # 27 passed, 1 xfailed expected
uv run ruff check .       # should be clean
# Lua-side tests need a lua interpreter (optional): the integration test auto-skips without one.
```

### Windows correctness notes (already handled — just FYI)
- Default bridge dir resolves to `%USERPROFILE%\.orpheus_bridge` on both sides (stable across processes).
- Atomic writes use `os.replace` (Python) and `os.remove`+`os.rename` (Lua) — both Windows-safe.
- Path separator via `package.config` in Lua — `\` on Windows.

### ⏭ The actual task: **M1 — Construction core**
Goal: the agent can build/modify a project correctly. Build **test-first** (TDD), same as M0.

Start with the load-bearing primitive and let the failing test drive it:
1. **`tests/test_midi_roundtrip.py`** is currently `xfail` — make it pass. A note written at beat B must read back at beat B through the PPQ/tempo/take conversion. This is the single most correctness-sensitive piece in Orpheus.
2. Implement the Lua handlers + `tools/midi.py`: `insert_midi_notes` (beats→PPQ, batched), `create_midi_item`, `transpose_notes`. Mind: PPQ is per-take and relative to the item's start; need a valid take before `MIDI_InsertNote`; beats↔PPQ depends on tempo.
3. Add read handlers: `get_project_info`, `list_tracks` (`tools/project.py`) — reads-before-writes.
4. Extend `tests/lua/test_bridge.lua` with stubbed-`reaper` handler tests, and do an in-REAPER verification (you're on the REAPER machine now — actually insert notes and eyeball the piano roll).

Reference: [`architecture.md`](architecture.md) (the bridge + the spine), [`roadmap.md`](roadmap.md) (M1 checklist), [`frontier-analysis.md`](frontier-analysis.md) (bonfire-audio's chord/drum recipes are the closest reference for the MIDI math).

---

## 2026-06-07 — M0: the REAPER bridge ✅ (commit `8bd4495`)

**Built + verified the file-IPC bridge** — the foundation everything stands on.

- `src/orpheus_mcp/bridge/client.py` — `BridgeClient`: atomic writes (temp→`os.replace`), heartbeat liveness, timeout (clean `BridgeTimeout`, no hangs), sequential request ids, `batch()` composites.
- `src/orpheus_mcp/bridge/lua/orpheus_bridge.lua` — persistent `reaper.defer()` poll loop, **static** dispatch (no `loadstring`), embedded pure-Lua JSON, atomic writes, `EnumerateFiles` cache invalidation (`fileindex=-1`), integer-id filenames (a `1.0` would've broken the rendezvous), `__batch__`.
- `src/orpheus_mcp/install.py` + `orpheus-mcp install-bridge` — per-OS REAPER resource-dir discovery.
- `tools/bridge_status.py` — `get_connection_status` round-trips a live ping (version + latency).
- Default bridge dir moved to `~/.orpheus_bridge` (was a temp dir → could differ between REAPER and the server and silently break).

**Verified:** 12 client + 8 installer Python tests, 19 Lua-side assertions, and a cross-language integration test (real `BridgeClient` ↔ a real `lua` subprocess running the actual bridge). ruff clean. Built test-first.

**Not done (later milestones):** all music logic. The Lua handler table has exactly one entry (`get_connection_status`); analysis/compose/recommend/apply are M1–M3 stubs.

## 2026-06-07 — Initial scaffold (commit `872fe96`)

Repo created from the frontier research: full architecture, `docs/frontier-analysis.md`, professional package skeleton (~30 tools as documented stubs), implemented + tested music-theory core, MIT license, CI. See [`roadmap.md`](roadmap.md) for the M0–M6 plan.
