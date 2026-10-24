# YuE2 sung sketches (design)

- **Date:** 2026-10-24
- **Status:** Agreed with the owner.
- **Parents:** [`2026-10-23-orpheus-definition-design.md`](2026-10-23-orpheus-definition-design.md)
  §4 step 7; [`2026-10-24-lead-sheet-design.md`](2026-10-24-lead-sheet-design.md);
  [`../pins/2026-10-yue2-backend.md`](../pins/2026-10-yue2-backend.md) (GO, 2026-10-23).

## 1. What changes for the owner

`render_sketch` sends a lead sheet (or some of its sections) to YuE2 and puts the sung result on
a new REAPER track, lined up with the section it came from, so the owner hears the idea over the
arrangement before recording it.

## 2. Decisions (2026-10-24)

| Question | Decision |
|---|---|
| Runtime | **CLI per render**: `yue-synth.exe` runs once per sketch; nothing to start beforehand, GPU memory used only while rendering. |
| Placement | **A new track per render**, named `SKETCH (YuE2) <section> #<seed>`, starting at the first rendered section's bar. |
| Opt-in | The tool exists only when `ORPHEUS_NC_BACKENDS` names `yue2` (weights are CC BY-NC 4.0). |
| Provenance | Only Orpheus lead sheets are rendered (no raw ABC input), so every sketch comes from the owner's own composition and the owned-audio rule holds. |

## 3. How it works

1. **Install discovery.** `ORPHEUS_YUE2_DIR` (default `~/.orpheus/yue2`) must hold
   `yue2.cpp/build/Release/yue-synth.exe` and `models/YuE2-3B-*.gguf` (Q8_0 preferred) plus
   `models/YuE2-Vae-F32.gguf`. Missing pieces give setup steps, not a crash.
2. **Score.** The chosen sections (all, or a contiguous run) go through `export_yue2_score`; the
   checker must pass. `cot` is `"full"` when the score has chords, else `"melody"` (with chord
   symbols stripped). `duration` is the score's length in seconds plus a 4 s tail.
3. **Render.** Request JSON (`style`, `lyrics`, `abc`, `cot`, `duration`, `lm_seed`, `seed`,
   `output_format: "wav24"`) is written as `request.json`, never with the output's stem (yue-synth
   overwrites `<out-stem>.json`). Orpheus holds its GPU semaphore while the CLI runs, so its own GPU
   stages wait. Truncation and failures are read from the CLI's output and reported.
4. **Files.** Each render gets `~/.orpheus/sketches/<timestamp>-<seed>/` with the request, the WAV
   and the replay JSON. Outside the analysis cache, so clearing the cache never takes REAPER's sketch
   items offline.
5. **Import.** A new read-only bridge verb `bar_time(bar)` gives the bar's time on REAPER's map;
   `import_audio_files` places the WAV there as a new, time-locked track.

Result: track name, start bar and time, WAV path, render seconds, truncation flag, seeds, the
score's warnings and chord reductions.

## 4. Tests

Fake CLI (writes a short WAV and YuE-style log lines) for request contents, the replay-JSON name
trap, truncation parsing, failure reporting, GPU semaphore use; tool absent without the opt-in;
end to end through the fake bridge (track name, placement after a meter change). Live: one real
render on the 3080 (no REAPER needed), then an import check with the owner in REAPER.
