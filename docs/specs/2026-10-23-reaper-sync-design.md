# REAPER sync: reading the live session (design)

- **Date:** 2026-10-23
- **Status:** Agreed with the owner, decision by decision.
- **Parent:** [`2026-10-23-orpheus-definition-design.md`](2026-10-23-orpheus-definition-design.md)
  §1 ("REAPER synced", meanings 2 and 3) and §4 step 4. Sheet ↔ REAPER sync (meaning 4) is a
  later step.

## 1. What changes for the owner

Claude can ask "where are we?" and get the open project's name, its tempo and meter changes,
every marker and region with bar and beat, and whether REAPER is playing. Claude can ask "what did
I change?" and get a short list of the edits since its last look. Anything Orpheus writes at "bar
N" lands on bar N of REAPER's own grid, even after a meter change.

## 2. Pieces

### 2.1 `get_project_timeline` (bridge verb + read-only tool)

One call returns:

- `project`: name, file path (empty for an unsaved project), length in seconds, and
  `state_change_count` (REAPER's `GetProjectStateChangeCount`, which rises on every change).
- `transport`: playing, recording, paused, edit-cursor and play positions in seconds.
- `tempo_map`: every tempo/time-signature marker with time, bar, beat, quarter-note position,
  BPM, meter and whether the ramp is linear. A project without markers reports one entry built
  from the project tempo and meter at time 0, so the list is never empty.
- `markers`: displayed number, name, time, bar, beat, colour.
- `regions`: displayed number, name, start and end (time, bar, beat), colour.

Bars and beats are 1-based and come from REAPER's own `TimeMap2_timeToBeats`, so they match the
ruler. Capped at 1024 markers and regions per call, with a `truncated` flag.

### 2.2 Bars follow REAPER's map

The bridge's `bar_start_qn(bar)` today multiplies by one bar length, so it assumes a single meter.
It becomes `TimeMap2_beatsToTime(0, 0, bar - 1)` followed by `TimeMap2_timeToQN`, which follows
every meter change. `create_midi_item`'s end becomes the start of bar `start_bar + length_bars`,
not "start + length × one bar". Handler-level code, so hot reload delivers it. If the API call is
missing (old REAPER), the old formula is kept.

The Python fake gains a meter map so the same rule is tested on both sides.

### 2.3 `project_changes` (bridge verb `get_project_snapshot` + read-only tool)

The bridge returns a snapshot:

- tracks: GUID, index, name, mute, solo, volume (dB), pan, FX count;
- items per track: GUID, position, length, kind (`midi` / `audio` / `empty`), and for MIDI the
  take's `MIDI_GetHash` (so note edits show without sending notes), for audio the source file;
- markers, regions, tempo map (as in 2.1); project length; `state_change_count`.

Caps: 512 items per track, 4096 in total, with a `truncated` flag.

Python compares it with the previous snapshot of the same project and returns a list of changes
in plain words with structured fields, for example:

- `track_added` / `track_removed` / `track_renamed` / `track_moved`
- `track_mute` / `track_solo` / `track_volume` / `track_pan` / `track_fx_count`
- `item_added` / `item_removed` / `item_moved` / `item_resized` / `item_notes_changed` /
  `item_source_changed`
- `marker_added` / `marker_removed` / `marker_moved` / `marker_renamed` (and the same for regions)
- `tempo_map_changed`, `project_length_changed`

Each change carries bars and beats where it has a position. Small float noise (under 1 ms, under
0.05 dB) is ignored.

**Baseline:** on disk, one file per project, under `~/.orpheus_cache/snapshots/<sha1 of the
project path, or "unsaved">.json`. Every call compares with the stored snapshot and then stores the
new one, so changes are "since the last check". The first call on a project records a baseline and
says so. Changes Orpheus made itself are reported too; Claude knows which were its own.

If `state_change_count` equals the stored one, the tool answers "no changes" without diffing.

### 2.4 Heartbeat v2

The Lua loop still touches `heartbeat.lock` every tick (liveness by modification time, unchanged),
and about once a second writes JSON into it:
`{"v": 2, "t": <epoch>, "reaper_version": ..., "project_name": ..., "project_path": ...,
"playing": ..., "state_change_count": ...}`. Python's reader accepts the old plain-integer content
too. `get_connection_status` adds the project fields when present.

This is the loop's core, which hot reload does not swap: **the owner re-runs the bridge script in
REAPER once** after `orpheus-mcp install-bridge`.

### 2.5 Server instructions

`00_core.md` tells Claude: call `get_project_timeline` before placing anything by bar, and call
`project_changes` at the start of a session and whenever `state_change_count` has moved.

## 3. Testing

- Lua: the stub `reaper` gains tempo/time-signature markers, project markers/regions,
  `TimeMap2_timeToBeats` / `TimeMap2_beatsToTime` with a meter change, `MIDI_GetHash`,
  `GetProjectStateChangeCount`. Assertions for both new verbs and for meter-aware bars.
- Python: the fake mirrors all of it; contract tests through the real `BridgeClient`; the diff is
  a pure function with one test per change kind; persistence round-trips; heartbeat parser for
  both formats.
- **Live (owner present):** a project with a 4/4 → 3/4 change, two markers and a region; check
  bars against the ruler, edit by hand, and check `project_changes`.

## 4. Not in this step

Sheet ↔ REAPER sync; the Claude Code status panel (reads heartbeat v2 later); watching for
changes in the background (Claude asks; Orpheus doesn't push).
