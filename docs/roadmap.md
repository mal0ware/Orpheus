# Orpheus Roadmap

Orpheus keeps Claude in sync with the user's REAPER session first; on top of that it listens,
explains, and works in the session, with the user approving every change
([definition](specs/2026-10-23-orpheus-definition-design.md)). Work ships in steps that each leave
the repo working and demoable. The order puts the riskiest infrastructure first, then
understanding, then the songwriting path, then automated suggestions.

"Built" below means implemented and tested against the behavioural REAPER fake and the Lua handler
suite. "Live" means also checked on real REAPER or real owned audio.

---

## Done

### M0: bridge + skeleton (2026-06-07, live)
File-JSON bridge with atomic writes, heartbeat, static dispatch, `EnumerateFiles` cache
invalidation, integer-id filenames; `BridgeClient`; `get_connection_status`; `install-bridge`.

### M1: construction core (2026-06-12, merged 2026-07-14, live)
Project reads, tracks, transport, and the PPQ-correct MIDI writer (`insert_midi_notes`,
`get_track_midi`, `transpose_notes`). The MIDI round-trip test is a hard gate: a note written at
beat B reads back at beat B.

### M2: understand a project (2026-07, built)
Theory tools (`get_scale_notes`, `constrain_to_key`, `suggest_chord_progression`,
`get_genre_profile`); symbolic analysis of the project's MIDI (`analyze_harmony`,
`analyze_groove`); audio character of a rendered WAV (`analyze_audio_character`);
`build_project_spec`; `explain_style` against a fingerprint (one ships: `classical`).

### Reference-analysis engine (2026-07-22..24, live 2026-07-25)
Spec: [`2026-07-22-reference-analysis-design.md`](specs/2026-07-22-reference-analysis-design.md).
Recon (tempo, beat grid, key, loudness, sections), instrument timeline, chord lanes, gated stem
separation with an escalation ladder, `SoundCharacter` descriptors, lyrics with word timings and
`locate_lyric`, the `analyze_reference` router, `compare_character` (14 descriptor→FX rules), FX
apply verbs, `suggest_free_vst`, `clear_analysis_cache`. Run on an owned 200 s song on 2026-07-25.
Still owed by ear: stem, chord and lyric quality checks and the calibration panel.

### Composers (M4 slices 1 and 2, 2026-07-18, built)
`create_chord_progression`, `create_bassline`, `create_drum_pattern`, `humanize_pass`,
`compose_section`, `create_melody`, `build_section`, `arrange_song`, `add_marker`,
`place_lyric_markers`, and an instrument-selection ladder that prefers installed plugins.
A live multi-section `arrange_song` run is still to do.

### Live studio work (2026-07-25..08-01, live)
`import_audio_files` (time-locked), `render_track_region`, `clone_track_items`,
`set_track_params`, mp4 input, section-diff analysis, hot reload, and fifteen bridge-only verbs
(rename/colour/move/delete track, regions, envelopes, trim/window/nudge items, presets,
refresh media/peaks, sampler instruments). Calibrated FX maps for REAPER's stock plugins and
Surge XT. See the [dev-log](dev-log.md).

---

## Upgrades 2026-10 (now)

Agreed 2026-10-23, in this order:

1. **Docs + merge.** Definition everywhere, dev-log caught up, Matchering out of core deps,
   instructions loaded from one file, `feat/reference-analysis` merged into `main`.
2. **Bug fixes**, each with a regression test:
   - [x] bridge request ids unique across the process (2026-10-23)
   - [x] lyric ASR: child-process CUDA probe + `ORPHEUS_ASR_DEVICE` (2026-10-23)
   - [x] cp1252 `UnicodeDecodeError` in subprocess readers, plus basic-pitch's stdout leak (2026-10-23)
   - [x] section naming on sparsely punctuated sung lyrics (2026-10-23)
   - [x] FX envelopes in the parameter's native range (2026-10-23; checked live 2026-10-25)
   - [x] dead `add_vst_path_and_rescan` call removed; sound pack rebuilt on CC0 FreePats + RS5k (2026-10-23)
3. **YuE2 test: GO** (2026-10-23): 60 s in 28.8 s, 7.2 GB peak; see
   [`pins/2026-10-yue2-backend.md`](pins/2026-10-yue2-backend.md).
4. **REAPER sync:**
   - [x] read verb + tool for the live timeline (2026-10-24; checked live 2026-10-25)
   - [x] bar/beat math that follows REAPER's tempo and meter map (2026-10-24)
   - [x] heartbeat v2 with project identity (2026-10-24; live 2026-10-25)
   - [x] the bridge starts by itself: with REAPER (`__startup.lua`), `orpheus-mcp start-bridge`,
     and on the first call when REAPER is open; the newest copy wins (2026-10-25, live)
   - [x] notice hand edits: change counter + `project_changes` diff, baseline per project
     (2026-10-24)
5. **Lead sheet + notation** (2026-10-24, [spec](specs/2026-10-24-lead-sheet-design.md)):
   - [x] `LeadSheet` model, referenced from `CompositionSpec`
   - [x] readers for the owner's chord charts and bass riff tab
   - [x] YuE2 ABC writer, checker (agrees with the official `abc_tools.py` on 76 scores), reader
     (YuE2-rules parser; music21 cross-checks our exports), sidecar for syllables
   - [x] `arrange_lead_sheet` into REAPER (checked live 2026-10-25)
6. **Sheet ↔ REAPER in step:** sheet sections and chords become markers and regions; the live
   timeline can be written back to the sheet (Claude does the sheet side through the Drive
   connector, with a yes).
7. **Sung sketches** (2026-10-25, [spec](specs/2026-10-24-yue2-sketch-design.md)):
   - [x] `render_sketch`: chosen sections of a lead sheet through yue2.cpp's CLI, one process per
     sketch under the GPU semaphore, onto a new `SKETCH (YuE2) <section> #<seed>` track at the
     section's bar. Gated by `ORPHEUS_NC_BACKENDS=yue2`.
   - [x] live check with the owner (2026-10-25): a Chorus sketch landed on bar 5, lined up by
     ear, and faded out

## Next

- **"Sing it to me":** the owner sings a scratch take and Orpheus notates the melody (pitch
  tracking, as in the July sessions) into a lead sheet.

- **Beat-map export for music videos.** Built on step 4's timeline read: `export_beat_map` (JSON + CSV of beats, bars, sections, lyric lines); an
  OpenTimelineIO file Premiere opens with section and lyric markers. This replaces the
  standalone lyric-engine repo, which gets archived.
- **MCP tools for the fifteen bridge-only verbs**, plus a `set_track_mute` verb.
- **`apply_changes` v1:** run a Claude-written `EditPlan` in one undo block (an `__undo_batch__`
  handler so hot reload delivers it, and a delete-notes-in-range verb so one section can be
  rewritten), then a "make it jazzier" demo.
- **Taste file:** `~/.orpheus/taste.json` with banned plugins and intents and preferred
  instruments, enforced in `compare_character`, `suggest_free_vst` and instrument selection.
- **Bridge status in Claude Code:** a small status panel that reads heartbeat v2 (step 4).

## Later

- **M3, style suggestions:** fingerprints built from reference tracks the user owns,
  `recommend_changes` diffing the project against them, `apply_master_match` (Matchering as an
  optional `[master]` extra).
- **M4 leftovers:** fuzzy track/role resolvers, `apply_mix_balance`, `quantize_notes`.
- **M5, ship:** PyPI (`uvx orpheus-mcp`), MCP Registry, a recorded demo.
- **M6, reach:** MIDI recording from a virtual keyboard, groove transfer, post-apply readback.
