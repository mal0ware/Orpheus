# Orpheus: definition, scope and the October 2026 upgrade

- **Date:** 2026-10-23
- **Status:** Agreed with the owner, decision by decision, on 2026-10-23. Definition revised the
  same day to put REAPER sync first, at the owner's direction.
- **Supersedes:** the README pitch that led with "make this sound Classical" (the style-fingerprint
  recommend/apply loop). That loop stays on the roadmap; it is no longer the headline.
- **Later specs:** the lead-sheet + ABC design and the YuE2 sketch-backend design get their own
  docs when that work starts. This one fixes what Orpheus is and the order of work.

---

## 1. Definition

Orpheus keeps Claude in sync with your REAPER session.

**REAPER is home base.** Orpheus reads the live project (tracks, tempo map, markers, regions,
what's playing), notices what you change by hand, and everything it does lands back in the
project as editable tracks, MIDI, markers and effects, on REAPER's own bars and beats. Your Google
Sheet stays in step: sections and chords from the sheet become markers and regions, and changes in
REAPER can flow back to the sheet.

On top of that, it **listens** to your project and to audio you own (tempo, key, chords, sections,
stems, lyrics and sound character, each with an honest confidence), **explains** what it hears in
plain language, and **works** in your session: building tracks, writing MIDI, arranging songs from
a lead sheet, setting effects, and (optionally) rendering a rough sung sketch with YuE2. It asks
before it changes anything, and every change can be undone.

**Who it's for:** one songwriter-producer working in REAPER, lyrics in Google Sheets. It is
published as open source (MIT) for anyone with the same setup.

**What it won't do:** fetch or rip music you don't own, need accounts or paid services, master for
release, or replace your ears.

### What "REAPER synced" means (agreed 2026-10-23)

All four of these, in this priority:

1. **Home base.** Everything is read from and lands in the live REAPER project. Nothing important
   lives anywhere else.
2. **Always knows the session.** Orpheus can report the open project's name and path, tempo map,
   time-signature changes, markers, regions, tracks and play state, and picks up edits made by
   hand. Today `get_project_info` reports the tempo, the starting meter, project length, track
   count and play state, but no markers, regions, tempo or meter changes, or project name; that gap
   is the REAPER-sync step (§4).
3. **Synced to the beat.** Bars and beats follow REAPER's own tempo and meter map, so nothing Orpheus
   writes drifts across a tempo or meter change. Today the bridge's `bar_start_qn` multiplies by one
   bar length, so `at_bar` assumes a single meter for the whole song.
4. **Sheet and REAPER in step.** Sections and chords from the Google Sheet become markers and regions;
   changes in REAPER can be written back to the sheet. Orpheus never talks to Google: Claude does
   the sheet side through the Drive connector, with the owner's yes before any write.

## 2. How it fits the songwriting workflow

The owner's songs start in a Google Sheet. A worked song tab holds:

- a **section table**: section name, bar count, and a chord chart per section;
- the **lyrics**, laid out section by section with bar counts and performance notes
  (`[stop, silence]`, `[held over the E]`);
- a **riff in bass tab**: one line per bar, string + fret tokens in eight slots
  (`Am | A0 -  A0 -  A0 -  A0 - |`);
- a column for the plugins (VSTs) the song should use.

Brainstorm tabs hold loose lines, hook candidates and notes on how sections could split.

So the sheet is most of a lead sheet already. The flow:

1. **Claude reads the sheet** through the Google Drive connector in the Claude app. Orpheus never
   talks to Google and needs no account. It receives plain text.
2. **Orpheus reads the owner's notation directly**, so bar timing never depends on Claude
   re-typing it:
   - chord charts: `|` is a bar line, `/` splits a bar between chords, `(x4)` repeats the line;
   - riff tab: standard 4-string bass tuning (E A D G), string letter + fret number per slot,
     `-` holds or rests, eight slots per bar.
3. Orpheus builds a **LeadSheet** (sections, timed chords, melody, lyrics, reasons), lays it into
   REAPER, and can export it as ABC for a YuE2 sketch.
4. The owner records; Orpheus analyses the take and Claude suggests changes.

The exact grammar for both notations is pinned in the lead-sheet spec, with tests built from
patterns like the ones above. No lyrics or song titles from the owner's sheets go into this repo.

## 3. Decisions (2026-10-23)

| Question | Decision | Why |
|---|---|---|
| What is Orpheus at its core? | **REAPER-synced first**, then a studio partner (§1): all four meanings of sync, plus listen / explain / work. | Owner: "first and foremost REAPER synced, but it should be all of those." |
| How do lyrics get in? | **Claude reads the sheet** via the Drive connector; Orpheus takes text. | Keeps Orpheus local-only with no accounts. |
| How much notation does Orpheus parse? | **Chord charts and riff tab.** | It is how the owner already writes; parsing it makes bar timing exact. |
| What is lyric-engine? | **Folded into Orpheus.** Orpheus will export a beat + section + lyric timeline that Premiere can open (OpenTimelineIO). lyric-engine gets archived. | Orpheus already finds beats, sections and word timings. Archiving the repo is a separate GitHub action and needs its own OK. |
| Unpushed work on the Mac? | **None.** This PC's branch is the full history. | Owner confirmed. |
| Matchering (GPL-3.0, unused core dependency)? | **Removed for now.** It comes back as an optional extra only when master-matching is built. | An MIT package should not hard-depend on GPL code it doesn't use. The README's "vendored" claim was wrong. |
| How are non-commercial models switched on? | **Per backend, by name:** `ORPHEUS_NC_BACKENDS=yue2` (comma-separated) in the MCP config. Off by default. | Turning on YuE2 must not also turn on LarsNet and change stem results. Feeds the existing `model_registry.resolve(allow_nc=...)` filter. |
| Where do Claude's operating instructions live? | **One file, loaded:** `data/instructions/00_core.md`, rewritten to §1 and read by the server at startup. | One place to edit; what you read is what Claude gets. Today's copies point at tools that don't exist. |
| Where does REAPER-sync work go? | **Right after the bug fixes**, before the lead sheet; sheet↔REAPER sync after the lead sheet. | The request-id fix must land before more read verbs; sheet sync needs the chord-chart reader. |
| Order of work | §4. | Each step can be checked on its own. |

## 4. Order of work

1. **Docs, then merge.** On `feat/reference-analysis`: this definition in the README,
   `docs/architecture.md`, `docs/roadmap.md` and the instructions file; a dev-log entry for
   2026-07-25..08-01; corrected numbers; Matchering removed. Full suite green, then the owner's OK,
   then merge into `main`.
2. **Bug fixes**, each with a regression test, on a branch from `main`:
   - bridge request ids collide across parallel tool calls (every `BridgeClient` starts at 1);
   - lyric ASR picks CUDA and aborts natively on this PC (needs a cuDNN probe and
     `ORPHEUS_ASR_DEVICE=cpu`);
   - cp1252 `UnicodeDecodeError` in a subprocess reader thread;
   - section naming collapses a real song into one "verse 1";
   - envelopes on JSFX parameters land at the wrong scale (1.0 arrives as 0.01);
   - `install_sound_pack` calls `add_vst_path_and_rescan`, which has no Lua handler.
3. **YuE2 test**, outside the repo, alongside step 2: build yue2.cpp, download the two weight files
   (about 4.3 GB) after the owner says yes, render one 60 s song on the RTX 3080, record VRAM and
   time in `docs/pins/`, and make the go/no-go call together.
4. **REAPER sync:** a read verb for the live timeline (project name and path, tempo and
   time-signature markers, markers, regions, play state) with a tool and fake-bridge support;
   bar/beat math that follows REAPER's tempo and meter map instead of assuming one meter;
   heartbeat v2 carrying project identity; a way to notice hand edits (a cheap project-state
   fingerprint Claude can compare between calls).
5. **Lead sheet and notation:** the LeadSheet model (referenced from `CompositionSpec`), chord-chart
   and riff-tab readers, an MIT ABC writer for YuE2's dialect, a reader (music21 for notes and
   chords, a sidecar for lyrics), a dialect checker, and round-trip tests.
6. **Sheet ↔ REAPER sync:** sections and chords from the sheet become markers and regions; the live
   timeline can be written back to the sheet by Claude (with a yes).
7. **Sketch renderer**, only if step 3 says go: Orpheus sends its own score to yue2.cpp and puts the
   result on a "SKETCH (YuE2)" track.

## 5. Corrections to the 2026-10-22 breakdown

Found while researching yue2.cpp (master `8af0174`, 2026-10-16) and YuE (`main`, 2026-10-06):

- **yue2.cpp has no ABC parser or validator.** A non-empty `abc` is tokenized straight into the
  prompt, and a malformed score just burns GPU time. The only checker is YuE's
  `skills/yue2-music/scripts/abc_tools.py` (Apache-2.0, stdlib-only). "Validate against yue2.cpp's
  own parser" is therefore not possible; the lead-sheet spec decides how Orpheus checks its output.
- **`abc` is silently ignored when `cot` is `"off"`.** Orpheus must send `cot: "full"` (score with
  chords) or `"melody"`.
- **`buildcuda.cmd` assumes Visual Studio 2026.** This PC has the 2022 Build Tools, CUDA 13.2,
  CMake and Ninja, which is enough; the build uses the 2022 commands from yue2.cpp's
  `docs/ARCHITECTURE.md`. There are no official release binaries.
- **The weights are not gated** (no sign-in). Use `Serveurperso/YuE2-GGUF`: its backbone and VAE are
  byte-identical to `ldov/YuE2-GGUF`, and ldov's SheetSage2 files are an older format current
  yue2.cpp won't load. Minimum set: `YuE2-3B-Q8_0.gguf` (3.81 GB) and `YuE2-Vae-F32.gguf` (530 MB).
- **No cuDNN is used** by yue2.cpp; it needs the CUDA 13 runtime DLLs, which are installed.
- **YuE2's lyric "sidecar" has no fixed schema** and the model never reads it. It is Orpheus's own
  record of which syllable sits on which note.
- **YuE2's accidentals carry by letter across octaves within a bar** (after `^F`, `f` is sharp too).
  Stock ABC readers, music21 included, read that differently, so the writer must spell accidentals
  so both readings agree.
- ComfyUI is not needed: yue2.cpp runs on its own.
