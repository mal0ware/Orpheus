# YuE2 sung sketches (design)

- **Date:** 2026-10-24
- **Status:** Agreed with the owner; built 2026-10-25 (§5 lists what differs from the draft).
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
4. **Files.** Each render gets `~/.orpheus/sketches/<timestamp>-<seed>/` with the request, the WAV,
   the replay JSON and the CLI's log (`yue-synth.log`). Outside the analysis cache, so clearing the
   cache never takes REAPER's sketch items offline. `ORPHEUS_SKETCH_DIR` moves it.
5. **Import.** A new read-only bridge verb `get_bar_time(bar)` gives the bar's time on REAPER's map,
   with the tempo and meter there; it is asked before the render, so a closed REAPER fails before
   the GPU runs. `import_audio_files` places the WAV there as a new, time-locked track.

Result: track name, start bar and time, WAV path, render seconds, truncation flag, seeds, the
score's warnings and chord reductions.

## 5. As built (2026-10-25)

- The verb is `get_bar_time`, not `bar_time`: read-only verbs start with `get_`, and the fake bridge
  counts every other verb as a project edit.
- A run of sections is named by its ends: `SKETCH (YuE2) Verse 1 to Chorus #<seed>`. Sections are
  chosen by name (any case) as a contiguous run; the first matching run wins. Characters Windows
  forbids in file names become `-`, because the WAV's name becomes the track's name.
- `duration` uses the score's own tempo (`Q:` rounds the sheet's BPM), plus the 4 s tail.
- **Endings (owner decision 2026-10-25).** The first real render (8 bars ending on the dominant)
  showed YuE2 does not stop at the score's end: it kept writing, and was still going with a 12 s
  tail (its louder new part started about 8 s after the last bar). YuE's own example, which ends
  on the tonic, stopped by itself 0.4 s after its last bar. So a cut at the 4 s tail is normal:
  when the log says truncated, the import gives the REAPER item a 2 s fade-out
  (`import_audio_files` gained an optional `fade_out_s`), the WAV stays as rendered, and the
  result's `ending` says so instead of a warning. A take that ends by itself is not faded.
- Warnings: REAPER's tempo or meter at the start bar differs from the sheet's; the score rounds a
  fractional BPM; a chosen section has no melody; the sheet has no style text.
- A render that succeeds but fails to import still returns the WAV path and the import error.
- A non-zero exit fails the render even when a WAV was written (yue-synth exits 1 when it cannot
  write the replay JSON after the audio). Timeout: 300 s plus 3 s per second of sketch.

## 4. Tests

Fake CLI (writes a short WAV and YuE-style log lines) for request contents, the replay-JSON name
trap, truncation parsing, failure reporting, GPU semaphore use; tool absent without the opt-in;
end to end through the fake bridge (track name, placement after a meter change). Live: one real
render on the 3080 (no REAPER needed), then an import check with the owner in REAPER.
