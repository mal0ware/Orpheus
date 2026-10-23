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
   - [ ] bridge request ids unique across the process (parallel read-only calls collide today)
   - [ ] lyric ASR: cuDNN probe + `ORPHEUS_ASR_DEVICE=cpu` (auto-CUDA aborts on the desktop)
   - [ ] cp1252 `UnicodeDecodeError` in a subprocess reader thread
   - [ ] section naming collapses a real song into one "verse 1"
   - [ ] envelopes on JSFX parameters land at the wrong scale
   - [ ] `add_vst_path_and_rescan` has no Lua handler
3. **YuE2 test** on the RTX 3080 via yue2.cpp: one 60 s render, VRAM and time recorded in
   `docs/pins/`, go/no-go.
4. **REAPER sync:**
   - [ ] read verb + tool for the live timeline: project name and path, tempo and time-signature
     markers, markers, regions, play state (fake bridge and Lua suite included)
   - [ ] bar/beat math that follows REAPER's tempo and meter map (today `bar_start_qn` assumes one
     meter for the whole song)
   - [ ] heartbeat v2 with project identity (needs one manual bridge re-run)
   - [ ] notice hand edits: a cheap project-state fingerprint Claude compares between calls
5. **Lead sheet + notation:**
   - [ ] `LeadSheet` model (key, meter, tempo, sections with timed chords, melody, lyrics,
     reasons), referenced from `CompositionSpec`
   - [ ] readers for the owner's chord-chart notation (`Am / Fmaj7 | C / E (x4)`) and bass riff
     tab (`A0 - E1 -`, eight slots per bar)
   - [ ] MIT ABC writer for YuE2's dialect, a reader (music21 for notes and chords, a sidecar for
     lyrics), a dialect checker, round-trip tests
6. **Sheet ↔ REAPER in step:** sheet sections and chords become markers and regions; the live
   timeline can be written back to the sheet (Claude does the sheet side through the Drive
   connector, with a yes).
7. **Sung sketches** (if step 3 says go): send an Orpheus-written score to yue2.cpp, import the
   result onto a "SKETCH (YuE2)" track. Gated by `ORPHEUS_NC_BACKENDS=yue2`.

## Next

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
