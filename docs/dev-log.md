# Orpheus — Development Log

A running record of what's built, what's verified, and where to pick up next. Newest entry first. This is the session-to-session handoff doc — read the top entry to resume.

---

## 2026-07-24 — Reference Analysis Engine BUILT (Phases 0–12, branch `feat/reference-analysis`)

The whole engine designed in the 2026-07-22 spec is now implemented, on `feat/reference-analysis`,
against `docs/plans/2026-07-23-reference-analysis.md` (SDD-tracked task-by-task in the gitignored
`.superpowers/sdd/progress-refanalysis.md`). Twelve phases: **Phase 0** — packaging (conflict-
partitioned `[analysis]`/`[separation]` extras), `CacheFS` (decoded-PCM-hash keying, per-artifact
`meta.json` sidecars, LRU + pinning), the model registry with all 13 registry slots hash-pinned,
vendored MSST SCNet (GPU-verified), 24 committed audio fixtures (75.7 MB, `ci_sparse`/`ci_vocal`),
and both isolated envs (audio-separator's own uv env, D9; basic-pitch's ≤3.11 env). **Phase 1** —
`recon.py` (T1: LUFS, tempo/beat-grid/downbeat/meter + grid confidence, tuning offset, key +
alternates, sections, 8-band sketch) + `recon_reference`. **Phase 2** — `timeline.py` (PANNs SED),
`instrument_timeline` + the §3.1 exposure/absent pre-gate rules. **Phase 3** — the always-available
CPU chord baseline lane (`harmony.py`: template+Viterbi recognizer, extensions, bass-informed
slash/inversion, music21 theory) + `get_chords`. **Phase 4** — both separation engines
(audio-separator subprocess adapter + vendored MSST), §5 verification gates 1–3 (mix consistency,
bleed, silence honesty), `separate_region`. **Phase 5** — gate 4 (cross-model agreement) + the §6
R1–R6 escalation ladder, `verify_stem`, and full empirical calibration
(`data/calibrated_thresholds.json`, agreement fitted 0.997/J=0.76). **Phase 6** — `descriptors.py`
(§7 `SoundCharacter`: spectral/dynamics/harmonics/stereo/time-fx/pitch) + `characterize`.
**Phase 7** — the `render_track_region` bridge verb (§15.2), Lua data-chunk-aware silence scan.
**Phase 8** — the lyrics pipeline (VAD-gated faster-whisper ASR + whisperX alignment + line/section
assembly) + `transcribe_lyrics`/`locate_lyric` (§14.2's phonetic/masked-profanity fuzzy quote
resolver). **Phase 9** — the §13 chord MODEL lane (vendored BTC-ISMIR19; music-x-lab as a
no-engine skeleton, D12) engaging at T3. **Phase 10** — the 14-rule §15.3 descriptor→intent engine
(`data/intent_rules.json` + `analysis/intents.py`) + `compare_character`. **Phase 11** — the FX
apply verbs (`add_fx_by_name`/`set_fx_param`/`get_fx_params`) + ReaPlugs/JSFX param maps, and a
closed-loop convergence test that drives characterize→compare→apply→render→re-compare against the
fake bridge with a BLIND corrective Q. **Phase 12** — the §9 `analyze_reference` orchestrator (12.1:
answer envelope, question-class routing, the D15 T3→T4 tier-ceiling bump for `sound_at`/
`compare_track`) and, this task (12.2), the last build task: `data/vst_catalog.json` (§8.1's
instruments + effects tables transcribed, 26 entries — Valhalla's Supermassive/FreqEcho/
SpaceModulator and Voxengo SPAN/Youlean Loudness Meter split into per-plugin rows so intent
matching doesn't blur distinct effects together), `suggest_free_vst` (pure keyword lookup, zero
network calls, recommend-only), and `clear_analysis_cache` (wraps `CacheFS.clear`,
`destructiveHint`).

**Pins recorded:** engines (pin a, Phase 0 — all 13 registry slots hash-pinned; owned-mp3 timing
re-run still human-deferred, see below); the SED head (pin d, Phase 2 — resolved as an honest tie:
upstream `panns-inference` only ships `Cnn14_DecisionLevelMax` regardless of the head string
requested, so `DecisionLevelAtt` is recorded as that same engine, a documented scope limit, not a
fabricated distinction); calibration (Phase 5.3 — agreement gate fitted 0.997/J=0.76, five §16
constants live at their fallback values with provenance recorded, and the absent-fire guard TRIPPED
exactly as designed, refusing a fitted value that would have suppressed 24% of frames). Chord
ear-check (pin b) and lyrics ear-check (pin c) are **not** pinned — they need a human ear on owned
audio and are carried into the deferred checklist below.

**The 15 Deviations** (full table: `docs/plans/2026-07-23-reference-analysis.md`'s Deviations Log,
D1–D15; all deliberate, all reasoned there) — condensed: D1 reordered the spec's four pin
prerequisites to match when their slice deliverables actually exist. D2 split "gated stems" into
gates 1–3 (Phase 4, no second-model dependency) vs. the agreement gate + ladder (Phase 5). D3 gave
the tagger and the intent rule engine explicit module homes (`timeline.py`, `intents.py`) the spec's
module list never named. D4 renders test fixtures headlessly (MIDI → `sfizz_render` CLI + CC0
samples) instead of through a live REAPER, for CI-safety. D5 places FX-loop iteration/plateau
enforcement on Claude, not Orpheus (no session state to count against) — the two guard constants are
exported in tool docstrings, never silently enforced server-side. D6 lets the Phase-0 pin run swap a
404'd/broken registry slot for its documented alternate as a data change. D7 moves calibrated
thresholds into `data/calibrated_thresholds.json`, loaded as config overrides with a hash that busts
stale gate caches on re-calibration. D8 makes `ci_vocal`'s vocal lane deterministic Windows-SAPI TTS
reading known lines, so the lyrics accuracy bar is machine-checkable on any box. D9 (exec-discovered)
moved audio-separator into its own isolated env — every release transitively caps `fastmcp<3.4`,
incompatible with the MCP server's own floor. D10 (exec-discovered) split `BeatGrid.confidence`
(grid alignment) from an additive `meter_confidence` (3-vs-4 margin ratio) — one field was
overloaded for two signals. D11 (exec-discovered) scoped the ≥90% majmin baseline bar to
synthesized progressions and the fixture's chords STEM, not the dense full mix (~0.20, template
matching's honest ceiling there — the model lane's job). D12 (exec-discovered) recorded that
audio-separator cannot actually load BS-RoFormer ep17 (a false spec claim); it's marked
`engine_status:"no-engine"` and the 4-stem rank order became SCNet-XL (MSST) > htdemucs-ft
(audio-separator). D13 (exec-discovered) gave multi-target registry entries per-stem
`tagger_classes_by_target`, closing a bleed-gate blind spot where a union class list let vocal
energy read as clean on a drums stem. D14 (exec-discovered) shipped `verify_stem`'s contract as the
flat `GateSet` model instead of the spec's nested shape, for one shared shape across every §9 gate
surface. D15 (exec-discovered) has `_class_ceiling` permit a T3→T4 climb for `sound_at`/
`compare_track` — a literal `min(QUESTION_TIERS, max_tier)` would make the R3-complement escalation
rung the plan itself carries forward permanently unreachable.

**Carried, closed here:** Task 11.2's carry — §15.3 rule 1 (presence boost, ReaEQ bell) has no
encoded Q/bandwidth policy in `data/intent_rules.json`'s `apply` block. No data change: the Q Claude
sets on the actual `ReaEQ` band when applying the rule is Claude-side judgment, same as every
un-mapped third-party-plugin instruction — the rule table fixes the trigger, the shape, and the
center frequency; bandwidth is not part of the closed-form contract.

**Deferred to a human** (full detail in `.superpowers/sdd/progress-refanalysis.md`'s Deferred
section — gitignored, summarized here since nothing else records it): re-running the Phase-0 engine
pins against an OWNED mp3 with an actual ear-check on the resulting stems (timings/downloads are
already automated; only the listening is pending), plus an owned-audio spot-listen on the `ci_sparse`
fixture; the full desktop calibration run + a 6-song owned-audio ear panel to re-fit the §16 fallback
constants outside the fixture domain; live-REAPER render + FX smoke — specifically the ReaComp/
ReaXcomp param-ORDER warning carried since Task 11.1 (mapped as threshold/ratio/attack/release,
unverified against the real plugin), the `render_track_region` WAV-format blob, `I_FXEN` master-bypass
behavior, and real float-WAV header layout through the silence scan; the chord (pin b) and lyrics
(pin c) hand-check certifications against owned songs; and hardening `fetch-models`'s audio-separator
config-sidecar warm-up (Task 4.1's Important#1 carry — it currently fetches sidecars at first
separation, a cold-box §1.1 violation until pre-staged at install).

**Gate this session (Task 12.2 — the catalog/cache-admin/docs task closing Phase 12):**
`pytest -q -m "not desktop and not gpu"` → **684 passed, 2 skipped, 14 deselected** (was 655/2/14 at
Task 12.1; +29 for `data/vst_catalog.json` + `suggest_free_vst` + `clear_analysis_cache`);
`scripts/run_lua_tests.py` → **123 passed, 0 failed** (unchanged — this task touched no Lua); `ruff
check` clean; `mypy src` clean. Every build task in the plan is now complete; what remains is the
deferred-human list above plus the live end-to-end demo (12.3).

## 2026-07-22 — Reference Analysis Engine spec (docs-only session, off-site)

Hotel session, no code by design (home PC has local-only work in flight). Produced the full
design spec for the song-analyzer direction: **`docs/specs/2026-07-22-reference-analysis-design.md`**.
Read that top to bottom to resume — §12 is the exact next-session sequence (desktop
experiments → `writing-plans`). Highlights: local-only/free-only constraint set; staged
recon→route→separate→verify→escalate pipeline with reference-free quality gates; dual
inference engines (audio-separator + vendored MSST for SCNet — audio-separator lacks SCNet,
verified); RoFormer/SCNet-era model registry with checkpoint-availability verified per slot;
§13 harmony engine (complex chords first-class: BTC large-voca + music-x-lab ISMIR2019 +
bass-informed slash detection + in-house baseline; madmom excluded — no py3.12 install);
§14 lyric-anchored navigation ("after the line …" → faster-whisper + whisperX on the isolated
vocal stem, fuzzy `locate_lyric`); §15 closes every open question (calibration protocol,
render mechanics, the 14 launch FX-intent rules, VRAM guard). Spec survived four adversarial
verification passes (separation claims; chord/lyrics tooling; a design-logic/edge-case audit
that found 7 wrong-as-written behaviors — incl. the FX loop converging toward mp3 codec
artifacts, Whisper hallucinating lyrics on instrumentals, walking-bass false slash chords,
render-solo contamination — and ~19 unspecified policies; and a final implementer-coherence
audit that closed 6 patch contradictions, added normative tool contracts to §9, per-slice
acceptance bars to §12, and the §16 constants appendix). Status: authoritative; only the §12
empirical pins remain before writing-plans. **Not done here:** any code, any implementation
plan — that starts at the desktop per §12.

## 2026-07-18 — LIVE SMOKE PASS (REAPER 7.77, Slices 1+2)

First end-to-end run against a live REAPER. Bridge installed via `install-bridge` + run as a ReaScript action; `~/.orpheus_bridge` heartbeat confirmed (Lua falls back `HOME`→`USERPROFILE`, so it matches the Python default on Windows). Built a 5-section, 18-bar A-minor ballad (Intro/Verse/Chorus/Verse/Chorus) via `set_tempo` + `add_marker` + `_build_section` per section.

**Confirmed live:** connection (7.77/x64); `list_installed_fx` returned 252 plugins (`EnumInstalledFX` works live); tempo/4-4; 4 tracks (drums/chords/bass/lead); 5 section markers visible; **plays straight through all 18 bars with no drop-out** — each track has exactly ONE MIDI item spanning the whole song, so the multi-section item-length fix (`insert_midi_notes` D_LENGTH growth) is verified on real hardware. Drums audible. User verdict: "full pass."

**Open (quality, not correctness):** stock ReaSynth timbre + zero mixing/FX + algorithmic parts = "scaffold, not a produced record" (sounds basic/robotic). Next lever = instrument selection actually using the user's installed VSTs, then an effects/mixing layer (the deferred editing-parity slice). NOT a bug — the documented limit.

## 2026-07-18 — Slice 2: the songwriting system — full-song composer (tested against fake/lua; live smoke pending)

**Built the full-song composer** — the second "ask in words, get an editable, audible result" slice, this
time a whole sectioned song rather than one loop:

- `parse_melody` (`theory/melody.py`) — a compact melody notation (`"A4:q C5:q E5:h"`, `r:dur` for
  rests) into sequential note dicts; snaps into `key`/`mode` if given so a model-authored line stays
  in key without hand-computing scale degrees.
- `DRUM_PATTERNS` (`theory/patterns.py`) — named step-grid patterns (`backbeat`, `halftime`,
  `fourfloor`) resolved through the existing `parse_drum_grid`, so callers say "halftime" instead of
  spelling out a grid.
- `add_marker` — a new bridge verb + tool wrapping `AddProjectMarker2`, placing a named marker at a
  1-based bar.
- `create_melody` — thin tool wrapper writing a `parse_melody` line to a track.
- `_build_section` / `build_section` — lays one section (chords + bass + drums + optional melody) at
  a bar offset onto shared named tracks, reusing every Slice-1 helper (`resolve_progression`,
  `voice_lead`, `bassline_notes`, `parse_drum_grid`, `select_instrument`, `_write_notes`,
  `load_drumkit`). Takes an optional caller-owned `instruments_loaded` set so a multi-section caller
  loads each track's instrument at most once instead of once per section.
- `arrange_song` — the orchestrator: sets tempo, then walks a `sections` list (`{name, bars,
  progression, drums?, melody?}`), placing a marker + a full section per entry end-to-end, sharing
  one instrument inventory and one `instruments_loaded` set across all sections. One call, one
  audible multi-section song.
- `place_lyric_markers` — places model-authored lyric lines as markers (`lines[i]` at `at_bars[i]`);
  the tool docstring is explicit that lyrics must be original text, never a copyrighted song's.

**Mid-slice fix**: `add_instrument(kind="drumkit")` was not idempotent — every call stacked three more
`ReaSamplOmatic5000` instances onto the track instead of detecting an existing kit, unlike the
`kind="named"` path. `arrange_song` calling `_build_section` once per section would have piled up
duplicate drum voices on every section after the first. Fixed by probing with
`TrackFX_AddByName(..., false, 0)` (find-only) before adding, mirroring the named-instrument path;
regression-tested at the Lua-handler, fake-bridge, and tool level (`e3a7422`).

**Tested:** unit tests (`parse_melody` tokens/rests/key-snapping, `DRUM_PATTERNS` resolution) +
`FakeReaperBridge` contract tests for `add_marker`/`create_melody`/`build_section`/`arrange_song`/
`place_lyric_markers` + the lupa-driven Lua handler suite exercising `add_marker` and the idempotent
drumkit probe against a stubbed `reaper`. Also added `tests/test_registry.py::test_arrange_in_default_not_explain`
and `test_arrange_in_full_profile`, confirming the `arrange` category (wired into `default` in Task 3)
is present in `default`/`full` and absent from `explain`. Full gate run this session: **`pytest -q` →
186 passed, 2 skipped**; **`scripts/run_lua_tests.py` → 50 passed, 0 failed**; **`ruff check .` → clean**
(one `B905 zip() without strict=` finding in `tools/arrange.py`, fixed by adding `strict=True`, itself
already guarded by an explicit length check); **`mypy src` → clean, 37 source files**.

### Pending live verification (explicitly not done — do not read the above as a live pass)

The live-REAPER smoke test for this slice is the user's next step, not run here. Live-only unknowns
introduced by this slice:

- `AddProjectMarker2(proj, isrgn, pos, rgnend, name, wantidx, color)` — the Lua assumes this exact
  signature and return shape (created marker index) per the REAPER 7.x ReaScript docs; the fake and
  the lua-stub tests do not depend on the real API, so this is genuinely unconfirmed until a live
  `add_marker` call is made and the marker is eyeballed in the timeline.
- Whether an `arrange_song` call actually produces an audible, correctly-sectioned song end-to-end in
  a real project — placement, tempo, and note content are all asserted programmatically; nobody has
  pressed play and listened yet.

Carried forward from the compose-core slice (still open, unrelated to this slice's changes):
`EnumInstalledFX` return shape, `ReaSamplOmatic5000`'s `FILE0`/note-range param indices, the
`MIDI_DeleteNote` descending-delete idiom in `clear_track_midi`, and the sound pack's placeholder
`sha256: "<PIN_BEFORE_SHIP>"` checksums.

**Rejected during this slice:** finding or downloading a reference song the user does not own, to use
as a melodic/lyrical reference or fixture — this was considered and explicitly rejected on copyright
grounds; Orpheus does not fetch or embed audio it has no rights to. **Audio stem ingestion remains
deferred to Slice 4** (`reference-ingest`, tracked in `docs/roadmap.md`), and is scoped, when built, to
audio files the user legally owns — no scraping, no un-owned reference tracks.

---

## 2026-07-18 — Compose-core slice: NL → audible section (tested against fake/lua; live smoke pending)

**Built the M4 compose slice** — the first "ask in words, get an editable, audible section" path. Four
atomic compose tools plus one orchestrator:

- `create_chord_progression` — writes a voice-led chord progression as MIDI (Roman numerals with a
  `key`, or absolute chord symbols), auto-loads ReaSynth so it's audible on drop.
- `create_bassline` — follows the same chord notation, `style` root/root_fifth/octave, bass-register
  ReaSynth.
- `create_drum_pattern` — parses a `kick:`/`snare:`/`hat:` step grid into hits and loads the stock
  3-voice kit.
- `humanize_pass` — seeded timing/velocity jitter + optional swing; reads a track's notes, transforms
  them, and replaces them in place (deterministic given the same `seed` — asserted by a dedicated
  same-seed-same-output test).
- `compose_section(genre, bars, key)` — the orchestrator: sets tempo from the genre profile, creates
  `drums`/`chords`/`bass` tracks, writes a one-bar backbeat + a voice-led progression (repeated to
  cover `bars`) + a root-note bassline, and loads the best available instrument per role. One call,
  one audible section.

**Three new bridge verbs** (`src/orpheus_mcp/bridge/lua/orpheus_bridge.lua`) back the above:
`list_installed_fx` (wraps `EnumInstalledFX`, read-only, capped at 20000), `add_instrument`
(`kind="named"` idempotently adds an FX by name; `kind="drumkit"` adds three `ReaSamplOmatic5000`
instances, one per sample, each pinned to its GM note via `FILE0` + normalized param indices 3/4),
and `clear_track_midi` (descending `MIDI_DeleteNote` loop + `MIDI_Sort`, added specifically so
`humanize_pass` can replace notes rather than duplicate them).

**Instrument selection** (`src/orpheus_mcp/instruments.py`) is a deterministic ladder:
explicit override → the user's own installed instruments (curated per-role allowlist —
Vital/Surge XT/Pianoteq/Kontakt for keys, similar for bass, MT-PowerDrumKit/Battery/EZdrummer/
Superior Drummer/Kontakt for drums — substring-matched against `list_installed_fx`) → the optional
consent-gated sound pack → stock fallback (ReaSynth for melodic roles, 3× `ReaSamplOmatic5000` for
drums). `compose_section` calls `list_installed_fx` once up front and threads the inventory through
all three role picks, so composing prefers what the user already has installed before ever touching
stock or the pack.

**Drum one-shots** (`src/orpheus_mcp/drumkit.py`) are stdlib-synthesized (kick/snare/hat, plain
`math`/`wave`/`struct`, no external deps) unless bundled CC0 WAVs are present in `data/drumkit/` —
license-clean either way, no download required to compose.

**`install_sound_pack`** (`src/orpheus_mcp/soundpack.py`) is consent-gated and never runs
automatically: it downloads pinned artifacts (one BSD-2-Clause sfizz build, one CC0 SFZ patch),
verifies each against a pinned sha256, and raises on any mismatch before writing anything.

**Tested:** pure unit tests (theory/chords, patterns, instruments ladder, drumkit synthesis,
soundpack checksum logic) + `FakeReaperBridge` contract tests for all three new verbs and all five
compose tools + the lupa-driven Lua handler suite exercising the same verbs against a stubbed
`reaper`. Full suite: **167 passed, 2 skipped**; Lua suite: **44 passed**; `ruff check .` and `mypy`
both clean. No real REAPER was launched for this slice.

### Pending live verification (explicitly not done — do not read the above as a live pass)

The following are proven only against the fake bridge / stubbed Lua and are flagged in the bridge
source as "verify live" — none of this has been run against real REAPER yet:

- `EnumInstalledFX` return shape — the Lua assumes `(retval, name, ident)` per the REAPER 7.x
  ReaScript docs; unconfirmed against an actual installed-plugin list. Fallback path (reading
  `reaper-vstplugins*.ini`) is unimplemented if this doesn't hold.
- `ReaSamplOmatic5000`'s `FILE0` config-parm name and the note-range param indices (3 = min, 4 = max,
  normalized `n/127`) — copied from the Task 6 research note, not yet confirmed against a live RS5k
  instance.
- The `MIDI_DeleteNote` descending-delete idiom in `clear_track_midi` — correct against the fake and
  the Lua stub, unverified against real `MIDI_CountEvts`/`MIDI_DeleteNote` semantics.
- Actual audible playback of `compose_section("lofi")` — tempo range, track creation, and note
  content are all asserted programmatically; nobody has pressed play and listened yet.

Also still a placeholder: the sound pack's `PACK` entries in `soundpack.py` carry
`sha256: "<PIN_BEFORE_SHIP>"` — the real pinned checksums are not filled in, so `install_sound_pack`
cannot succeed against the real URLs yet. Pinning the checksums, running the live-REAPER smoke test,
and correcting any of the constants above if live behavior differs are the user's next steps (Task
15, steps 1–3 in `.superpowers/sdd/task-15-brief.md`).

---

## 2026-07-15 — LIVE SMOKE PASS (M0 + M1) + M2 exercised against real REAPER

First-ever contact with live REAPER, and everything held:

- Launched REAPER 7.77/x64 with the bridge script as a command-line
  argument (`reaper.exe -nosplash <path>\orpheus_bridge.lua`) — no manual
  action-list step needed; the defer loop and heartbeat came up on their own.
- `get_connection_status` -> `{ok: true, reaper_version: '7.77/x64'}`.
- M1 roundtrip: `create_track` returned a stable GUID; `create_midi_item` +
  `insert_midi_notes` wrote a C4-E4-G4 arpeggio in beats; `get_track_midi`
  read it back at exactly beats 0.0 / 1.0 / 2.0 with durations and
  velocities preserved. The beats<->PPQ math agrees with
  `tests/fake_reaper.py` on real REAPER.
- M2 live: `analyze_harmony` on the read-back notes detected C major
  (confidence 0.81) and correctly declined Roman numerals on monophonic
  material; `analyze_groove` read "hard-quantized, machine-flat dynamics" —
  the honest description of a programmatic insert.

The queued 5-minute task below is done; kept for the manual-run steps.

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
