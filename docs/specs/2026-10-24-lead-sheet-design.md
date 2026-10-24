# Lead sheet, the owner's notation, and YuE2's ABC (design)

- **Date:** 2026-10-24
- **Status:** Agreed with the owner, decision by decision.
- **Parent:** [`2026-10-23-orpheus-definition-design.md`](2026-10-23-orpheus-definition-design.md)
  §2 and §4 step 5. YuE2 facts: [`../pins/2026-10-yue2-backend.md`](../pins/2026-10-yue2-backend.md).

## 1. What changes for the owner

A song's section table, chord charts, bass riffs, lyrics and a melody become one **LeadSheet**.
From it Orpheus can lay the song into REAPER (markers, chords, bass riff, melody, drums), export
it as a YuE2 score for a sung sketch, and read a YuE2 score back. Claude passes the sheet's own
text (chart, tab) and Orpheus does the bar arithmetic, so timing never depends on Claude
re-typing numbers.

## 2. Decisions (2026-10-24)

| Question | Decision |
|---|---|
| Scope | All five: LeadSheet; chart and tab readers; YuE2 ABC export; ABC import; laying a LeadSheet into REAPER. |
| Chord charts | `\|` bar line; a lone chord fills its bar; ` / ` (space on either side) splits a bar evenly; `C/E` (no spaces) is a slash chord; `(xN)` repeats the line; `E---` holds and fills the section's remaining bars; `(stop)` and other free text become notes, never guesses. |
| Riff tab | 4-string bass, standard tuning (E1 A1 D2 G2 = MIDI 28 33 38 43); 8 slots per bar; `A0` = string + fret; `-` holds the previous note; a bar starting with `-` is silent until its first note; text before the first `\|` is a label; `(xN)` repeats. |
| Melody | Claude writes it with the owner, in Orpheus's melody notation, one syllable per note. Roadmap: the owner sings a scratch take and Orpheus notates it. |
| YuE2 instrument line | Rests unless a section opts in (`riff_to_ins`), then that section's riff goes there. |
| Validation | Orpheus's own MIT checker of YuE2's dialect runs before every export/render; an opt-in test runs the official `abc_tools.py`, fetched at a pinned commit with a sha256 check, and both must agree. No YuE code in the repo. |

Fixed by YuE2's format (not choices): a 1/16 grid (1/32 for x/8 meters); no tuplets, grace notes,
repeats or slurs, so swing and triplets become a straight grid and the feel goes into the style
text; chords reduced to YuE2's 15 qualities, each reduction reported.

## 3. The model

`LeadSheet`: `title`, `key` (tonic, e.g. `A`), `mode` (`major` | `minor`), `meter`, `tempo_bpm`,
`style` (YuE2 style text), `sections`. `CompositionSpec` gains `lead_sheet: LeadSheet | None`.

`LeadSheetSection` takes the sheet as written **or** the expanded form:

| Field | As written (input) | Expanded |
|---|---|---|
| bars | `bars` (the sheet's Bars column: "4 + 1" means 5) | same |
| chords | `chart: "Am / Fmaj7 \| C / E (x4)"` | `chords: [ChordEvent(symbol, bar, beat, beats)]` |
| riff | `tab: "Am \| A0 -  A0 -  A0 -  A0 - \|"` | `riff: [SheetNote]` |
| melody | `melody: "A4:q C5:e. ..."` + `syllables: "You'd ra-ther talk"` | `melody_notes: [SheetNote(pitch, start_beat, duration_beats, syllable)]` |
| lyrics | `lyrics: ["line 1", "line 2"]` | same |
| other | `name` ("Chorus A"), `label` (YuE2 section: intro, verse, pre-chorus, chorus, bridge, interlude, outro, instrumental), `drums` (pattern name or none), `riff_to_ins`, `notes` (free text kept from the chart), `rationale` | same |

`resolve_lead_sheet()` expands every as-written field and returns warnings (chart text it could
not read, syllable/note count mismatches, bars that don't add up). Beats are quarter notes from
the section start; bars and beats are 1-based.

Melody notation gains dotted durations (`q.` = 1.5 beats, `e.` = 0.75, `h.` = 3); existing
notation is unchanged. Syllables: split on spaces and hyphens, one per pitched note in order;
`_` means "this note continues the previous syllable" (a melisma).

## 4. YuE2 export (`export_yue2_score`, read-only)

Returns the ABC text, the `lyrics` string (`[Verse]`-style tags from each section's label, then
its lines), a sidecar (one entry per sung note: section, syllable, note index after tie merging,
pitch, onset and duration in quarters), the chord reductions, warnings, and the checker's
verdict. Writing rules follow the pinned dialect: the exact 8-line header with an empty `T:`;
groups of up to 4 bars, each `% label` before its section's first group; `V: Vocal` carries the
melody and the chord symbols (on rests when nothing is sung); `V: Ins` carries `Z` rests or the
opted-in riff; every bar sums to the meter; durations split into 1, 2, 3, 4, 6, 8, 12, 16, 24,
32, 48 units with ties; keys from YuE2's major/minor list (modes become major/minor plus
accidentals).

**Accidentals** are written so YuE2 (whose accidentals carry by letter across octaves within a
bar) and standard ABC readers such as music21 (per octave) read the same pitch: a note gets an
explicit `^`, `_` or `=` whenever either reading would otherwise differ.

## 5. Checker and reader

- `check_yue2_abc(text)` returns a list of errors: header shape, group structure, equal Vocal/Ins
  bar counts, allowed durations, bar sums, rejected syntax (tuplets, grace notes, slurs,
  decorations, repeats, `w:`, `/` durations, blank lines), chords only in Vocal and only from the
  allowed set, allowed keys, ties joining equal pitches, no dangling tie at the end.
- `import_yue2_score(abc, sidecar=None)` (read-only) returns a LeadSheet: sections and bar counts
  from Orpheus's own scan of the `%` labels and groups, notes and chord symbols through music21,
  syllables from the sidecar (music21 does not attach ABC lyrics to notes, and YuE2 scores have
  none).

## 6. Into REAPER (`arrange_lead_sheet`, destructive)

Sets tempo and meter, places one marker per section (name), and writes onto shared tracks:
**Chords** (voice-led block chords at their exact beats), **Bass** (the riff), **Lead** (the
melody), **Drums** (only sections with a pattern), plus each section's first lyric line as a marker
in the same way `place_lyric_markers` does. Bars go through the bridge, so they follow REAPER's
meter map. Instruments come from the existing ladder (installed plugin, then sound pack, then
stock). Returns what it wrote per section and the resolver's warnings.

## 7. Tests

- Chart and tab readers on the owner's patterns (generic progressions, no lyrics), including
  `(x4)`, split bars, slash chords, `---` fill, `(stop)`, silent bars.
- Writer: golden ABC for small sheets; every exported score passes the checker; round trip
  LeadSheet → ABC → (music21 + sidecar) → LeadSheet keeps pitches, onsets, durations, meter, key,
  chord symbols (post-reduction) and syllables; accidental cases that differ between the two
  readings.
- Checker: each rule has a failing example; YuE's own example scores pass.
- Official cross-check (opt-in, network): our exports and the bad examples get the same verdict
  from `abc_tools.py inspect`.
- `arrange_lead_sheet` through the fake bridge, including a meter change.

## 8. Not in this step

The YuE2 render itself (step 7); "sing it, Orpheus notates it" (roadmap); writing back to the
Google Sheet (sheet ↔ REAPER sync step).
