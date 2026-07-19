# Orpheus Slice 2 — Songwriting System (full-song composer)

- **Date:** 2026-07-18
- **Status:** Approved design → implementation plan next
- **Builds on:** Slice 1 (compose core — shipped: `create_chord_progression`, `create_bassline`, `create_drum_pattern`, `humanize_pass`, `compose_section`, the `list_installed_fx`/`add_instrument`/`clear_track_midi` bridge verbs, the instrument-selection ladder, `_chord_notes`/`_load_drumkit` helpers). Depends only on Slice 1 + M1.

---

## 1. Goal

A natural-language description (or a reference the model infers from) becomes a **complete, editable song in the user's live REAPER session**: a section-structured timeline (Intro/Verse/Chorus/…) with chords, bass, drums, a lead **melody**, section **markers**, and a sheet of **original suggested lyrics** aligned to the sections. The user then plays along and records a vocal.

The user is non-technical about music theory. **They describe; the model fills in the concrete musical settings (key, tempo, progression, drum feel, melody, lyrics) and drives the tools.** No genre picker, no note typing, no theory required of the user.

## 2. Design principle: no genre lock — the model is the translator

Slice 1's `compose_section(genre)` required the user to pick "lofi"/"hiphop"/"classical". That is removed as a *requirement* here. The new tools take **explicit musical settings**; the model infers those settings from the user's words or a named reference ("dreamy and slow", "upbeat pop", "the vibe of <song>"). Genre becomes one optional flavor of description, never a required argument. `compose_section` stays as a legacy convenience; the song composer does not depend on it.

## 3. Scope

### In
- `add_marker` bridge verb (Lua + fake + tool) — named markers at a bar position (§5).
- `create_melody` — model-authored, in-key lead line from a friendly note+rhythm notation (§6).
- `build_section` — lay one section (chords + bass + drums + optional melody) at a timeline offset, in explicit key/tempo/progression/feel (§7).
- `arrange_song` — the headline orchestrator: tempo + key + a section list → full timeline with per-section content and markers (§8).
- **Lyrics**: the model writes **original** lyrics matched to mood + structure, returns the sheet, and places each section's opening line onto its marker (§9).
- Tests mirroring Slice 1's discipline; live-smoke deferred to the user (§10).

### Out (explicitly — YAGNI / legal / later)
- **Finding or downloading a reference song the user does not own — REJECTED, not deferred.** Auto-fetching commercial audio off streaming/YouTube is copyright infringement; Orpheus will not do it. The model may *infer* a reference song's public musical traits (key/tempo/typical progression) to guide original composition, but never fetches its audio or reproduces its copyrighted lyrics.
- **Audio stem ingestion (MP3 → stems → tracks)** — the legal "decompose a song" path, valid only on files the user owns. Remains the deferred **Slice 4** (`reference-ingest`), out of scope here.
- No mixing/mastering, no vocal synthesis (the user sings), no mid-song key/tempo/meter changes, no non-4/4.

## 4. User experience (the story the tools serve)
> User: "a slow emotional piano ballad — verse, chorus, verse, chorus, bridge, chorus."
> Model: infers key (e.g. A minor), tempo (~68), a fitting progression per section, a gentle drum feel; calls `arrange_song`; generates a lead melody per section via `create_melody`; writes original lyrics; places section markers with the opening lyric lines.
> Result in REAPER: labeled timeline, drums/chords/bass/lead tracks, audible on play; a lyric sheet returned in chat. User sings over it.

## 5. `add_marker` bridge verb (new bridge surface)
- Lua `HANDLERS.add_marker(p)` → `reaper.AddProjectMarker2(0, false, timepos, 0, name, -1, 0)`, where `timepos = TimeMap2_QNToTime(0, bar_start_qn(bar))`. Returns `{ name, bar, index }`.
  - NOTE (verify live): `AddProjectMarker2` signature/return on REAPER 7.x; the fake + lua-stub tests do not depend on the real API — confirmed by the live smoke.
- Fake handler in `tests/fake_reaper.py`: model a `markers: list[dict]` on `FakeReaperProject`; append `{name, bar}`; return `{name, bar, index}`.
- Python tool `add_marker(name, bar)` in `tools/arrange.py` (new module) → `{name, bar, index}`.
- Lua handler suite assertion + a `MIDI`-free contract test.

## 6. `create_melody` — in-key lead line
- **Notation** (model-friendly, parsed pure/tested): a string of `NoteName[octave]:dur` tokens, `dur` ∈ {`w`,`h`,`q`,`e`,`s`} (whole/half/quarter/eighth/sixteenth), rests as `r:dur`. Example: `"A4:q C5:q E5:h r:q A4:e G4:e"`.
- New pure module `theory/melody.py`: `parse_melody(notation, key=None, mode=None) -> list[dict]` → note dicts (`pitch/start_beat/duration_beats/velocity`), sequential in time. If `key`/`mode` given, snap out-of-scale pitches via existing `snap_to_scale` so the line stays in key.
- Tool `create_melody(track, notation, key=None, mode=None, at_bar=1) -> dict` in `tools/compose.py`: parse → `_write_notes` at `at_bar` → auto-load a lead instrument via the selection ladder (stock ReaSynth fallback).
- Determinism: pure parse; no RNG.

## 7. `build_section` — one section, explicit settings
- `build_section(key, mode, progression, bars, at_bar=1, drums="backbeat", melody=None, drum_track="drums", chord_track="chords", bass_track="bass", lead_track="lead", inventory=None) -> dict`.
- Lays, at `at_bar` on the shared tracks: chords (`_chord_notes` + `voice_lead`), bass (`bassline_notes`), drums (a named built-in pattern e.g. `"backbeat"`/`"halftime"`/`"fourfloor"` → `parse_drum_grid`, tiled across `bars`), and — if `melody` (a notation string) is given — the lead via `create_melody`'s parser.
- Creates any missing track (idempotent by name) and loads the best instrument per role (reusing the Slice-1 ladder + `_load_drumkit`). Returns the section's `{at_bar, bars, tracks, instruments}`.
- Named drum patterns live in `theory/patterns.py` as a small `DRUM_PATTERNS: dict[str,str]` of step grids (keeps the model from having to hand-draw grids).

## 8. `arrange_song` — the full song (headline)
- `arrange_song(tempo, key, mode, sections, humanize=True) -> dict` where `sections: list[dict]`, each `{name, bars, progression, drums?, melody?}`.
- Steps: `set_tempo(tempo)`; create drums/chords/bass/lead tracks once; running `bar = 1`; for each section: `add_marker(name, bar)` then `build_section(key, mode, section.progression, section.bars, at_bar=bar, drums=section.get("drums","backbeat"), melody=section.get("melody"))`; advance `bar += section.bars`; optionally `humanize_pass` each track at the end.
- One `list_installed_fx` call threaded through (no per-section re-query). Returns `{tempo, key, sections:[{name,at_bar,bars}], markers, instruments}`.
- The model composes the `sections` list (and the melodies/lyrics) from the user's description — `arrange_song` is the deterministic placement engine, not the creative source.

## 9. Lyrics
- The **model writes original lyrics** (not a tool) matched to the requested mood and the section structure, and returns them as a plain sheet in chat / an optional file.
- Placement: for each section, `add_marker(f"{name}: {first_line}", bar)` (or a thin `place_lyric_markers(lines_by_section)` helper over `add_marker`) so the timeline shows the section + opening line.
- **Copyright:** lyrics are always original text the model authors. The system never reproduces a referenced song's copyrighted lyrics; for an actual cover the user supplies those themselves.

## 10. Testing (mirrors Slice 1)
- **Pure unit:** `parse_melody` (durations→beats, in-key snapping, rests), `DRUM_PATTERNS` grids parse, `arrange_song` section→bar math (each section's `at_bar` is the running sum; markers at the right bars) via a fake bridge.
- **Fake-bridge contract:** `add_marker` places a marker at the right bar; `build_section` writes chords/bass/drums(+melody) at the correct `at_bar` on shared tracks and loads instruments; `arrange_song` produces N section markers and content spanning the full length. Real wire protocol, not mocks.
- **Lua handler suite:** `add_marker` via lupa against the stubbed `reaper` (with an `AddProjectMarker2` stub).
- Full suite + ruff + mypy green.
- **Live smoke (user's step, pending):** `arrange_song(...)` for a short song → markers visible, all tracks audible on play, one Ctrl+Z sane. Verify `AddProjectMarker2` live behavior.

## 11. Honest limits (surface, don't hide)
- **Scaffold, not a hit.** Melody/chords are coherent and in-key; lyrics fit theme and rhyme — but this is an editable starting point, not a produced record.
- **Basic timbre** (stock synths) until the user swaps instruments — same as Slice 1.
- **The tune is a guide** the user will rewrite; that's the point (all editable MIDI).
- Genres/patterns are a small built-in set, extensible; 4/4 only in v1.

## 12. File map
- New: `src/orpheus_mcp/theory/melody.py` (`parse_melody`), `src/orpheus_mcp/tools/arrange.py` (`add_marker`, `build_section`, `arrange_song`, optional `place_lyric_markers`).
- Modified: `theory/patterns.py` (`DRUM_PATTERNS`), `tools/compose.py` (`create_melody`), `bridge/lua/orpheus_bridge.lua` (`add_marker` handler), `registry.py` (register `arrange` category in default+full), `tests/fake_reaper.py` (markers model + handler), Lua + Python tests.

## 13. Definition of done
- All tools implemented + registered; `add_marker` on three agreeing surfaces.
- `parse_melody`, `DRUM_PATTERNS`, section math unit-tested; contract tests green; Lua suite green; full suite + ruff + mypy clean.
- `arrange_song` demonstrably lays a multi-section song with markers + melody in the fake harness; live smoke handed to the user.
- Docs: dev-log entry; roadmap updated (Slice 2 shipped; Slice 4 still deferred; find/download explicitly rejected).

## 14. Open risks
- `AddProjectMarker2` return/signature is a live-only unknown (Task-15-style; flagged inline, resolved in the live smoke).
- Melody musicality from a pure parser is limited; acceptable for a guide line (limits stated). The model authoring good notation is where quality comes from, not the parser.
