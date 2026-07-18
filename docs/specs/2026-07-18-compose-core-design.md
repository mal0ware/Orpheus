# Orpheus Slice 1 — Compose Core

- **Date:** 2026-07-18
- **Status:** Approved design → implementation plan next
- **Milestone mapping:** pulls the *compose* half of roadmap **M4** forward; adds one slice of the deferred **M4-mix** instrument-loading surface. Depends only on **M1 (done)** — not on M2/M3.

---

## 1. Goal

A natural-language ask to Claude produces an **audible, editable** musical section in the user's **live REAPER** session, in one command, with **zero manual setup**. The user then polishes it by hand in REAPER.

Orpheus is the MCP server that is *how Claude touches REAPER*. This slice gives Claude the generative half of that reach: it can lay down drums, chords, and bass as real, editable MIDI on real tracks, and — critically — those tracks make sound the instant the user presses play.

## 2. Where this sits in the larger program

This is **Slice 1 of 6** (see the sequenced program in the brainstorm thread / `roadmap.md`):
1. **Compose core** ← this spec (incl. read-only `list_installed_fx`, pulled forward so composing prefers the user's own instruments)
2. Full editing parity (FX/mix/arrange/automation)
3. Analyze/understand brain (M2)
4. Song ingestion — MP3 → stems/BPM/key/structure (M6)
5. Lyrics — Whisper on the vocal stem
6. Transform — recommend + gated apply (M3)

Each later slice is its own spec. This slice is self-contained and shippable on its own.

## 3. Scope

### In
- Four atomic compose tools + one orchestrator (§4).
- An instrument-selection ladder that **prefers the user's own installed instruments**, with stock plugins as the always-works floor (§5), via new bridge verbs `add_instrument` + `list_installed_fx` (the latter pulled forward from Slice 2).
- An **optional, consent-gated** `install_sound_pack()` for a nicer open-source default sound — never automatic (§5.4).
- Theory-layer additions: chord-symbol parser, voice-leading, bassline patterns, step-grid parser (§6).
- Bundled CC0 drum one-shots with a synthesized fallback (§7).
- Tests mirroring the existing discipline (§8).

### Out (explicitly deferred to later slices — YAGNI)
- Any analyze / recommend / transform tooling (Slices 3, 6).
- MP3 / audio ingestion and lyrics (Slices 4, 5).
- FX **editing/mixing** verbs — `set_fx_param`, volume/pan, delete/move, automation (Slice 2). Note: only the read-only `list_installed_fx` and instrument *loading* are in scope here.
- Auto-downloading arbitrary/licensed VSTs (rejected outright — legal/security/permission walls; see §5.4 for the one safe, curated, consent-gated exception).
- Extended harmony (9th/11th/slash chords), mid-section tempo/meter changes, non-4/4 grids.

## 4. Tools

All compose tools are **thin orchestrators** over the existing theory layer + the proven `insert_midi_notes` writer. They never get private superpowers the model couldn't invoke directly.

### 4.1 `create_chord_progression(track, chords, key=None, mode="minor", bars_per_chord=1, octave=4)`
- **Dual notation.** Accepts either:
  - **Roman numerals** (`"i-iv-V-i"`) — resolved through the existing `progression_triads(key, mode, …)`. Requires `key`.
  - **Absolute chord symbols** (`"Cm7, Fm7, Bb7"`) — resolved through a new lightweight parser (§6.1): root pitch-class + quality (`maj`/`min`/`dim`/`aug`) + optional `7`/`maj7`. `key` optional.
- Emits voiced `Note` lists, one chord per `bars_per_chord` bars, then a **voice-leading pass** (§6.2) minimizes semitone motion between adjacent chords so voicings don't leap octaves.
- Auto-loads **ReaSynth** on the track (idempotent).
- Writes via `insert_midi_notes` (already under the round-trip gate).

### 4.2 `create_drum_pattern(track, pattern)`
- **Step-grid mini-language**, one row per voice, `x` = hit, `.` = rest:
  ```
  kick:  x...x...x...x...
  snare: ....x.......x...
  hat:   x.x.x.x.x.x.x.x.
  ```
  Default 16 steps = one bar of 16th notes. Rows map to GM notes **kick 36 / snare 38 / closed-hat 42**. Unknown voice labels raise with the known set.
- Drums land on their **own track** whose instrument (a stock drum kit, §5) responds to those GM notes; MIDI channel is therefore irrelevant (the `Note` model has no channel field — this design sidesteps that).
- Auto-loads the stock drum kit (idempotent).

### 4.3 `create_bassline(track, from_progression, key=None, mode="minor", style="root", octave=2)`
- Follows a progression (same dual notation as §4.1) and emits a bass `Note` line.
- `style`: `"root"` (whole-note roots), `"root_fifth"` (root+fifth), `"octave"` (root/octave pulse). Root-per-chord by default.
- Auto-loads **ReaSynth** (a bass-register patch) on the track.

### 4.4 `humanize_pass(track, timing_ms=12, velocity_jitter=6, swing=0.0, seed=0)`
- Reads the track's notes via the existing `get_track_midi`, applies **seeded** timing + velocity jitter, and optional `swing` (delays offbeat 16ths toward a triplet feel), then rewrites the notes.
- `seed` makes it **deterministic** — required so unit tests assert exact output.

### 4.5 `compose_section(genre, bars=8, key=None)` — the orchestrator (headline)
- Reads the existing `genre_profiles` entry (`bpm_range`, `progressions`, `instruments`, `feel`) for `genre` ∈ {`lofi`, `hiphop`, `classical`}.
- Steps: set tempo (mid of `bpm_range`) → create drums/chords/bass tracks → lay a genre-appropriate drum pattern + the profile's first progression + a bassline → `humanize_pass` each → load instruments.
- **Result: one call → press play → hear a full section.** This is the usability payoff and the demo.
- `key` optional; defaults to a sensible tonic per genre.

## 5. Audibility — prefer the user's own instruments, guarantee sound with stock

The goal is "hear something good with zero setup." The best sound is usually an instrument the user **already owns**, so instrument choice is a **preference ladder**, not a hardcode. Stock plugins are the floor that guarantees it always works.

### 5.1 Instrument-selection ladder (per role: `keys`/chords, `bass`, `drums`)
Resolved deterministically against the current plugin inventory:
1. **User-configured override** — an explicit per-role instrument the user set (e.g. "always use Vital for keys"). Honored first.
2. **Best installed match** — scan `list_installed_fx` (§5.3) for a name on a small curated **allowlist** of quality instruments per role (e.g. keys: Vital, Surge XT, Kontakt, Decent Sampler; drums: a known drum VSTi). Discovery only — installs nothing.
3. **Curated open-source pack** — if the optional pack (§5.4) was installed, use sfizz + its CC0 patch.
4. **Stock fallback (always available):**
   - **Pitched (keys/bass):** `ReaSynth` via `TrackFX_AddByName` (bass vs keys patch tweak).
   - **Drums:** **3× `ReaSamplOmatic5000`** on the drum track, each loaded with one bundled one-shot (kick/snare/hat) and note-range-filtered to its GM note (36/38/42) — no third-party player dependency.

The chosen instrument (and *why* — "used your Vital"; "fell back to stock kit") is **reported back** so the user is never guessing what made the sound.

### 5.2 New bridge verb: `add_instrument(track, kind, opts)`
- `kind="named"` → add a specific plugin by name (`TrackFX_AddByName`) — used when the ladder picks an installed/curated instrument.
- `kind="drumkit"` → add the 3 `ReaSamplOmatic5000` instances; set each sample via `TrackFX_SetNamedConfigParm` (`FILE0`) and its note range via params.
- **Idempotent:** detects an already-present instrument and does not stack duplicates. Returns what was loaded (or already present).

### 5.3 New bridge verb: `list_installed_fx()` — pulled forward from Slice 2
- Returns REAPER's installed-plugin inventory (name + type), cached. Read-only; installs nothing.
- Pulled forward because it is what turns "sounds synthy" into "sounds like the good instruments you already have." Written to be reused by Slice 2's full FX surface.

### 5.4 Optional, consent-gated: `install_sound_pack()` — nicer default sound, safely
- Downloads **one** curated, permissively-licensed, **loose-file** instrument — **sfizz** (BSD-2 SFZ player, ships as a plain `.vst3`, *no installer*) — plus a **CC0** SFZ/soundfont, into a **user-writable** VST folder that REAPER is pointed at, then triggers a rescan.
- **Never runs automatically.** Requires an explicit user call/approval each time. No admin rights (user-writable path, not `Program Files`), no third-party `.exe` executed, no redistribution of licensed software — this is why it clears the legal/security/permission walls that "auto-install arbitrary VSTs" does not.
- If the user declines or it fails, the ladder simply falls back to stock — sound is never blocked on it.

## 6. Theory-layer additions (pure, REAPER-free, unit-tested)

### 6.1 Chord-symbol parser
`"Cm7"` → root pitch-class + quality + optional 7th. Grammar: `root(A–G)(#|b)? (m|min|maj|dim|aug|"")? (7|maj7)?`. Reuses `NOTE_TO_PC` and `TRIAD_INTERVALS`; 7th adds min7 (10) or maj7 (11) per quality. Rejects junk with a clear error.

### 6.2 Voice-leading pass
Given consecutive triads/tetrads as pitch sets, choose inversions/octaves minimizing total semitone movement from the previous chord's voicing. Deterministic. Keeps voicings in a comfortable register band.

### 6.3 Bassline generator
Maps a resolved progression to a bass `Note` line per `style` (§4.3). Deterministic.

### 6.4 Step-grid parser
Parses the drum mini-language (§4.2) into `Note` lists at the correct beat offsets. Validates row labels and step counts.

## 7. Assets
- 3 tiny **CC0 / royalty-free** one-shots (`kick`, `snare`, `hat`) in `data/drumkit/`.
- **Synthesized fallback:** if a sample file is missing, generate a simple one-shot at load time (sine-thump kick, filtered-noise snare/hat). Guarantees sound and keeps the repo license-clean regardless.

## 8. Testing (mirrors existing discipline)
- **Pure unit tests (no REAPER):** chord-symbol parser, voice-leading, bassline, step-grid parser, `humanize_pass` determinism (fixed `seed` → exact output), and the **instrument-selection ladder** (given a fake inventory → deterministic role→instrument choice, incl. fallbacks).
- **Lua handler suite** for `add_instrument` and `list_installed_fx` via `lupa` against a stubbed `reaper` (as with existing handlers).
- The note-write path is **already** covered by `tests/test_midi_roundtrip.py` (the non-negotiable gate).
- **One live-REAPER smoke:** `compose_section("lofi")` → tracks created, best-available instruments loaded (with the choice reported), notes present and audible.
- `install_sound_pack()` download/placement is validated in isolation (mocked fetch → correct user-writable path + rescan trigger); not part of the default smoke.

## 9. Honest limits (surface these to the user, don't hide them)
- **Timbre is basic.** ReaSynth is a plain synth; the user hears a correct musical *idea* immediately, not a produced sound. Swapping in their own VSTi is one click. We optimize for "hear it now."
- **Genres = the existing 3** for v1; the structure is trivially extensible.
- Fixed 4/4, straight-16th grid; no mid-section tempo/meter change.

## 10. Registry / honesty rule
Replace the current `compose` stubs. These tools graduate into the `default`/`full` profiles once implemented; until then the registry's "advertise only implemented tools" rule holds (no half-wired tool is exposed).

## 11. Definition of done
- All five compose tools implemented and registered; stubs removed.
- `add_instrument` + `list_installed_fx` bridge verbs implemented + Lua suite green.
- Instrument-selection ladder implemented and unit-tested; `compose_section` uses the best available instrument and reports its choice.
- `install_sound_pack()` implemented, consent-gated, and validated in isolation.
- All new theory helpers unit-tested; full suite green (incl. the round-trip gate).
- `compose_section("lofi")` produces an audible section in a live REAPER smoke, wrapped so a single Ctrl+Z is sane.
- `dev-log.md` updated; README/roadmap M4 status advanced to reflect the shipped compose slice + the pulled-forward `list_installed_fx`.

## 12. Open risks
- **RS5k config-parm names** (`FILE0`, note-range params) must be verified against live REAPER 7.x during implementation; the Lua suite stubs them, so the live smoke is the real proof.
- **ReaSynth patch control** for a convincing bass vs keys distinction may be limited; acceptable for v1 (timbre caveat already stated).
- **Plugin-name matching** for the installed-instrument allowlist is fuzzy (names vary by version/vendor/format); start with a small exact/substring allowlist and expand — a miss only means falling back to stock, never a failure.
- **`install_sound_pack()` download source** must be a pinned, checksum-verified URL for sfizz + the CC0 patch; no arbitrary fetch. Treat a checksum mismatch as a hard failure.
