# Orpheus Roadmap

Built in milestones that each leave the repo in a working, demoable state. The ordering is deliberate: **get the riskiest infrastructure bulletproof first, then build understanding, then transformation.**

> **The honest framing.** The full *"make this sound like X → here's why → apply"* loop is a multi-month build (≈36 core tools; the hard ones — the PPQ-correct MIDI writer, the audio analyzer, `recommend_changes`, gated `apply_changes` — are all on the critical path). The *weeks*-scale first public release is the **understand-and-explain** half. So Orpheus ships in two acts:
> - **`v0.1` — it explains your track** (M0–M2)
> - **`v0.3` — it transforms your track** (M3), the headline differentiator

---

### M0 — Bridge + skeleton *(foundation)* ✅ **DONE**
A hardened, tested file-JSON IPC bridge and a professional FastMCP scaffold the rest stands on. Riskiest infrastructure — bulletproofed before anything musical.

- [x] src-layout `orpheus_mcp` package, `pyproject.toml` console-script, `fastmcp.json`, MIT license, CI.
- [x] The file-JSON IPC bridge (`orpheus_bridge.lua`), hardened with: persistent `reaper.defer()` loop, static dispatch (no `loadstring`), heartbeat lock-file, **atomic temp-then-rename writes**, `EnumerateFiles` cache invalidation, integer-id filenames, `REAPER_MCP_BRIDGE_DIR` override, stable `$HOME`-based default dir.
- [x] `BridgeClient` (sequential `request_id`, poll, timeout, `batch` composites) + `get_connection_status` + `install-bridge` command.
- [x] Wire `server.py` + `registry.py` + toolset gating.

**Verified:** 12 Python bridge tests, 8 installer tests, 19 Lua-side assertions, and a cross-language integration test (real `BridgeClient` ↔ a real `lua` subprocess running the actual bridge) all green. Remaining for the user: the in-REAPER smoke test (`get_connection_status` against a live REAPER) + the Claude Desktop hammer-icon check.

### M1 — Construction core *(the APPLY verbs + the load-bearing primitive)* ✅ **CORE DONE**
The agent can build and modify a project correctly. The load-bearing primitive (PPQ-correct
MIDI write) and the track/transport verbs ship here; the FX/mix verbs are deferred to a later
pass (they need the installed-plugin inventory + name→index resolution).

- [x] `project` reads-before-writes: `get_project_info`, `list_tracks`, `get_track_midi`. (`get_fx_params` deferred with the FX work.)
- [x] `transport`: `set_tempo`/`set_time_signature`/`play_stop_record`; `tracks`: `create_track`. (`set_track_volume_pan`, `add_fx_by_name` (inventory-validated), `set_fx_param` (name→index) deferred — FX/mix pass.)
- [x] `midi`: **`insert_midi_notes`** (beats in, PPQ out, batched) — the load-bearing primitive — plus `create_midi_item`, `transpose_notes`. (`quantize_notes` deferred.)
- [x] `tests/test_midi_roundtrip.py` is now the **non-negotiable** correctness gate (was xfail; flipped to a hard requirement — a note written at beat B reads back at beat B, through the real wire protocol + a behavioural REAPER fake, plus a self-contained Lua-side handler suite). `render` tools deferred to M2.

**Verified:** beats↔PPQ round-trip (incl. fractional/sixteenth-note offsets, bar-relative `at_bar` anchoring, transpose) green through the Python `FakeReaperBridge`/`FakeReaperProject` executable spec; 33 Lua-side assertions green against a stubbed `reaper`; full suite 48 passed with a lua interpreter (46 + 2 lua-gated tests auto-skip without one). The model speaks beats; every tick conversion lives in the bridge.

### M2 — Theory + the ANALYZE brain → **`v0.1`** *(north-star half 1)*
Orpheus can deeply *understand* a project and keep the LLM in-key.

- [ ] `theory`: embedded music21 + reimplemented `music_theory_data` + ported `genre_profiles` (`get_scale_notes`, `suggest_chord_progression`, `constrain_to_key`, `get_genre_profile`).
- [ ] `analysis/symbolic.py`: music21 key (+confidence/alternatives) / chordify / Roman-numeral harmony with **drum-track filtering**, plus the custom MIDI-PPQ **groove analyzer** → `analyze_harmony` / `analyze_groove` / `analyze_tempo_meter`.
- [ ] `analysis/audio.py`: librosa/numpy/pyloudnorm features + four objective DSP analyses → `analyze_audio_character`; `render_and_audit` loop.
- [ ] `build_project_spec` fuses both into `CompositionSpec(current)` + LLM-Readable Report. Unit tests on fixtures (no REAPER).
- [ ] `explain_style` → ships **`v0.1`: Orpheus explains your track.**

### M3 — RECOMMEND + APPLY + fingerprints → **`v0.3`** *(the differentiator)*
Close the loop with the approval gate — the thing nobody else ships.

- [ ] `analysis/fingerprint.py` + `scripts/build_fingerprint.py`; cache `classical.json`, `hiphop.json`, a per-era `dominic-fike-sunburn.json`; `list_style_fingerprints`.
- [ ] `recommend_changes`: diff vs fingerprint across v1 dimensions → `EditPlan` (`readOnlyHint`), `ToolResult` = prose + structured + meta.
- [ ] `apply_changes` (undo-block-wrapped, `destructiveHint`) + `apply_master_match` (Matchering → ReaEQ on master). `tests/test_recommend.py`.

### M4 — NL ergonomics + GENERATE stretch *(polish)*
- [ ] `dsl` resolvers (fuzzy track + role + `SessionContext` pronoun memory) + tool profiles defaulting to a small set.
- [ ] `compose` generators as thin orchestrators (`create_chord_progression`, `create_drum_pattern`, `humanize_pass`) + section-aware arrangement.
- [ ] `mix`: `apply_mix_balance` (genre×role dB table + chain-offset) + `list_installed_fx`.
- [ ] `data/instructions/00_core.md` system-prompt teaching notation + in-key rules.

### M5 — Ship
- [ ] `docs/` complete; `examples/make_it_classical.md`.
- [ ] PyPI publish for `uvx orpheus-mcp`; exact Claude Desktop JSON in README; record the `fastmcp dev` Inspector session as the demo GIF.
- [ ] OIDC `publish-mcp.yml` → PyPI + MCP Registry on `v*` tags.
- [ ] Post ONE jaw-dropping north-star demo video as the launch.

### M6 — Reach *(post-launch)*
- [ ] MIDI-first virtual-keyboard recording (`arm_virtual_keyboard_record`, `capture_midi`).
- [ ] `reference-ingest`: `ingest_reference_track` with opt-in BasicPitch transcription (graceful degradation) + `build_fingerprint` from raw audio.
- [ ] Groove/swing transfer + arrangement reshaping promoted from stretch to applied.
- [ ] "Here's what I'll do / here's what I did" post-apply action readback (the transparency steal from shiehn).
