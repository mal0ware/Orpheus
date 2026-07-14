# Orpheus — Development Log

A running record of what's built, what's verified, and where to pick up next. Newest entry first. This is the session-to-session handoff doc — read the top entry to resume.

---

## ▶ NEXT SESSION — resume here

**Where things stand:** the Windows Desktop PC is fully set up (deps synced, tests green, bridge script installed into `%APPDATA%\REAPER\Scripts\orpheus\`, REAPER 7.73 detected at `C:\Program Files\REAPER (x64)`). M0 **and the M1 core** (tracks, transport, PPQ-correct MIDI write/read/transpose) are merged and verified against the behavioural REAPER fake. The one thing no machine has done yet: run the handlers against **live REAPER**.

### ⏱ Your 5-minute task: the in-REAPER smoke test (M0 + M1)

1. Open REAPER. **Actions → Show action list → "ReaScript: Run…" → `orpheus_bridge.lua`**
   (it's in `%APPDATA%\REAPER\Scripts\orpheus\`; the installed copy already includes the M1 handlers).
   Console should print: `Orpheus bridge listening in: C:\Users\mal0s\.orpheus_bridge`.
2. M0 check — in a terminal at the repo root:
   ```powershell
   uv run python -c "from orpheus_mcp.bridge.client import BridgeClient; print(BridgeClient().call('get_connection_status'))"
   ```
   Expect `reaper_version: '7.73/win64'`. A clean `BridgeTimeout` (not a hang) means the .lua isn't running.
3. M1 check — insert a C-E-G arpeggio and read it back:
   ```powershell
   uv run python -c "from orpheus_mcp.bridge.client import BridgeClient; c = BridgeClient(); g = c.call('create_track', name='Orpheus M1 smoke')['guid']; c.call('insert_midi_notes', track=g, notes=[{'pitch': 60, 'start_beat': 0.0, 'duration_beats': 1.0, 'velocity': 100}, {'pitch': 64, 'start_beat': 1.0, 'duration_beats': 1.0, 'velocity': 100}, {'pitch': 67, 'start_beat': 2.0, 'duration_beats': 2.0, 'velocity': 100}], at_bar=1); print(c.call('get_track_midi', track=g))"
   ```
   Expect: a new "Orpheus M1 smoke" track appears with a 1-bar MIDI item; the printed notes read back at beats 0.0 / 1.0 / 2.0; open the item and eyeball C4–E4–G4 in the piano roll.
4. Jot the result (pass/fail + REAPER version) in a new dev-log entry. If anything is off, the beats↔PPQ math in `orpheus_bridge.lua` vs `tests/fake_reaper.py` is the first place to diff.

### ⏭ After the smoke: what's next
- **M1 leftovers** (deliberately deferred): FX/mix verbs (`set_track_volume_pan`, `add_fx_by_name`, `set_fx_param`, `get_fx_params` — need the installed-plugin inventory + name→index resolution), `quantize_notes`, `render` tools.
- **M2** — the ANALYZE brain (see [`roadmap.md`](roadmap.md)).
- Optional: install a `lua` interpreter to un-skip the 2 lua-gated tests (`winget install DEVCOM.Lua` or similar); they run the real `orpheus_bridge.lua` handlers against a stubbed `reaper`.

---

## 2026-07-14 — Windows machine bring-up + M1 merged ✅ (branch `maint/2026-07-14`)

First session on the Windows Desktop PC (the primary REAPER machine). Executed the resume checklist minus the GUI steps, merged the M1 branch, and staged everything for the in-REAPER smoke.

- **`uv sync --extra dev` was broken on this machine**: a fresh resolution picked `numpy 2.5`, which forced the resolver to backtrack `numba` to 0.53.1 / `llvmlite` 0.36 (source-only, requires Python <3.10) — the build failed on Python 3.12. Fixed with `[tool.uv] constraint-dependencies = ["numba>=0.60", "llvmlite>=0.43"]` in `pyproject.toml`; committed the repo's first `uv.lock` (now pins numba 0.66 / llvmlite 0.48 / numpy 2.4.6, all wheels).
- **Merged `origin/claude/m1`** (M1 construction core, commit `829bfe0`) into `maint/2026-07-14` after a full-diff review. Clean merge; nothing trimmed. The 2026-06-12 entry below describes what it contains.
- **Fixed a Windows CLI crash**: `orpheus-mcp install-bridge` copied the script fine, then died with `UnicodeEncodeError` printing "✓/→" to a cp1252 console. Output now degrades via `sys.stdout.reconfigure(errors="replace")`; regression test in `tests/test_cli.py` (simulates a strict cp1252 stdout).
- **REAPER 7.73 confirmed installed** (`C:\Program Files\REAPER (x64)\reaper.exe`, registry + resource dir checked) — *not launched*. Bridge script installed via `uv run orpheus-mcp install-bridge` and verified byte-identical to the repo copy (M1 handlers included).
- **Fixed a live-REAPER-only bug in the merged M1 Lua** (found by API review, locked in test-first): `first_take` called `reaper.GetTrackMediaItem(0, tr, 0)`, but the ReaScript signature is `GetTrackMediaItem(tr, itemidx)` — no project arg (unlike `GetTrack`). Every test passed because the stub in `test_m1_handlers.lua` mirrored the wrong signature; against real REAPER, `insert_midi_notes`/`get_track_midi`/`transpose_notes` on a track that already has an item would have thrown. Corrected the stub first (suite went red: `insert_midi_notes ok (got false)`), then the bridge (green). Re-ran `install-bridge` afterwards, so the `%APPDATA%` copy includes the fix.

**Verified on this machine:** `uv run pytest -q` → **47 passed, 2 skipped** (the 2 lua-gated tests auto-skip — no standalone `lua` interpreter installed here); `uv run ruff check .` → clean. The MIDI round-trip gate (`tests/test_midi_roundtrip.py`) passes as a hard requirement (no longer xfail). The Lua-side M1 handler suite **was** executed here despite the missing interpreter — in-process under Lua 5.4 via `uv run --with lupa python <runner>` → **33 passed, 0 failed**. **Not verified:** anything against live REAPER — that's the 5-minute task at the top.

## 2026-06-12 — M1: construction core ✅

**Built + verified the construction core** — the APPLY verbs and the load-bearing MIDI primitive.

- `bridge/lua/orpheus_bridge.lua` — added static-dispatch handlers: `get_project_info`, `list_tracks`, `set_tempo`, `set_time_signature`, `play_stop_record`, `create_track`, `create_midi_item`, `insert_midi_notes`, `get_track_midi`, `transpose_notes`. The beats↔PPQ math lives here (project QN ↔ take PPQ via `MIDI_GetPPQPosFromProjQN`/`MIDI_GetProjQNFromPPQPos`); `at_bar` resolves the bar→QN offset from the project time signature. Tracks addressed by GUID/index/name (never live pointers); notes inserted with `noSortIn=true` then a single `MIDI_Sort`; per-call note cap enforced.
- `tools/project.py`, `tools/transport.py`, `tools/tracks.py`, `tools/midi.py` — thin FastMCP wrappers over the bridge, beats-in/typed-out. MIDI writes + transport + track creation carry `destructiveHint`; the project reads carry `readOnlyHint`.
- `tests/fake_reaper.py` — extracted `FakeReaperBridge` (wire-protocol harness) + added `FakeReaperProject` and `make_handlers`: a behavioural REAPER fake (tracks/items/takes/tempo/meter + the same 960-PPQ/QN math) that is the executable spec for the Lua handlers.
- `tests/test_midi_roundtrip.py` — flipped from xfail to a real gate: note-in-beats round-trip incl. fractional offsets, bar-relative `at_bar`, and transpose.
- `tests/test_m1_tools.py` — bridge-contract tests + end-to-end tests through an in-memory MCP `Client`.
- `tests/lua/test_m1_handlers.lua` — self-contained (no shell) Lua-side handler suite; `tests/test_bridge_integration.py` runs it through a real lua interpreter (auto-skips without one).
- `tests/lua/run_bridge.lua` — made cross-platform (separator-agnostic path, `dir /b` vs `ls -1`, portable sleep) so the Windows resume machine runs the lua tests too.

**Verified:** `pytest` 48 passed with a lua interpreter (46 + 2 lua-gated tests; the 2 auto-skip without lua). 33 Lua-side assertions green. `ruff check .` clean. No real REAPER launched. **Still TODO in M1's broader scope:** the FX/mix verbs (`set_track_volume_pan`, `add_fx_by_name`, `set_fx_param`, `get_fx_params`) and `quantize_notes` + `render` tools — they remain stubs.

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
