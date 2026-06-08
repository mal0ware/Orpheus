# Walkthrough: "Make this beat sound Classical"

The north-star demo — the analyze → recommend-with-reasons → apply loop end to end. (This
is the `v0.3` target; today it's the design, not a runnable feature.)

### Setup
1. Open an 8-bar beat in REAPER (with at least one **chordal MIDI** track — see the note below).
2. Run `orpheus_bridge.lua`. In Claude: *"Check the Orpheus connection."*

### The conversation

> **You:** Make this beat sound Classical.

Behind the scenes the agent:

1. **Grounds** — `get_connection_status`, `list_tracks`, `get_track_midi`.
2. **Analyzes** — `analyze_harmony` (key + confidence), `analyze_groove`, `analyze_tempo_meter`, `analyze_audio_character`; fuses via `build_project_spec`.
3. **Recommends** — `recommend_changes("classical")` diffs your project against `classical.json` and returns an `EditPlan`:

```
Plan to move toward "classical" (your A-minor beat → classical idiom):
  1. set_tempo 120 → 72              — classical phrasing breathes slower
  2. transpose  0                    — keep A minor; it's already a fitting tonal center
  3. rewrite    i–iv  → ii°–V–i      — functional harmony w/ a secondary dominant is THE signature
  4. add_track  "Strings" (from your installed FX)  — idiomatic voice
  5. master_match classical          — your low band is +8 dB hot; cut ~3 dB @ 120 Hz

Caveats: key detected at 0.71 confidence (correct me if it's wrong); "classical = slower +
functional" is an opinionated stylistic call, not a measurement.
```

4. **You approve.**
5. **Applies** — `apply_changes(plan)` (one undo block) + `apply_master_match("classical")`.
6. **Verifies** — `render_and_audit`: *"now within 1.2 dB of the reference across all three bands."*

One `Ctrl+Z` reverts the whole thing.

### Why the demo beat must have chordal MIDI
Functional-harmony detection needs actual chords to read. A pure trap beat (drums + a
monophonic 808 + an audio sample) has no MIDI chords to analyze, and inferring harmony from
audio is a hard, lossy problem reserved for `v2`. Pick a beat with a real chord track, and
let Orpheus surface its detection confidence so you can correct it.
