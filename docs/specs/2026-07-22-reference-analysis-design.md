# Reference Analysis Engine — Design Spec (DRAFT FOR DISCUSSION)

**Date:** 2026-07-22 · **Status:** draft — written off-site (docs only); discuss + refine at the
desktop before writing the implementation plan. · **Target milestone:** replaces/absorbs the
roadmap's M2 `analysis/audio.py` line and Slice 4 (`reference-ingest`), and pulls the M1-deferred
FX verbs onto the critical path.

**The ask, verbatim-ish:** "I have a song (an mp3 I own). Tell me why it's better than what's in my
REAPER project. Find every part of it — instruments, chords, structure. And when there's a sound I
can't describe in words, measure it and tell me exactly how to get my track closer to it, using
plugins I actually have."

---

## 1. Hard constraints (non-negotiable, from the 2026-07-22 session)

1. **Local only.** All audio processing happens on the user's machine. The mp3, stems, renders,
   and every derived artifact never leave the box. The only thing that leaves Orpheus is tool
   results (text/JSON) to the MCP client.
2. **Free and open only.** No commercial APIs, no paid models, no accounts, no "free tier with
   signup." Every model and library must be pip/checkpoint-downloadable without auth.
   (One-time checkpoint downloads from HuggingFace/GitHub at install time are fine — that is
   fetching *tools*, not sending *audio*.)
3. **Owned audio only.** Orpheus never fetches, scrapes, or captures streamed audio
   (standing decision, dev-log 2026-07-18). Input is a local file the user has rights to.
4. **MIDI transcription is the LAST resort**, never the default deliverable.
5. **Claude stays the brain.** Orpheus exposes tools; the driving model (any Claude via
   Claude Code / Desktop) does the reasoning. No LLM lives inside Orpheus.
6. **Hardware baseline:** desktop RTX 3080 (10 GB VRAM). Everything in this spec must run on it;
   CPU-only fallback must exist (slower, never broken).

## 2. Scope

**In scope (this spec):**
- Tier 1 — project-side analysis feeding the "why is X better" conversation (the M2 brain).
- Tier 2 — reference-side analysis: recon → routed separation → verified stems → character
  descriptors, for local files the user owns.
- The descriptor-diff → VST/FX recommendation loop, including the curated free-VST catalog and
  the stock-first parameter-apply path.

**Out of scope (explicitly):**
- Anything cloud. (Revisit only if a zero-setup, zero-cost, audio-stays-local option ever exists —
  today none beats local, so the registry simply has no cloud backend in v1.)
- Fetching reference audio from streaming services in any form.
- "40 clean orchestral stems" — dense tutti demixing is an unsolved research problem; the spec's
  escalation ladder (§6) is the honest treatment.
- Style *fingerprints* & `recommend_changes` (M3) — this spec produces their inputs but does not
  redesign them.

## 3. Architecture — the staged pipeline

One new package: `src/orpheus_mcp/analysis/` with submodules `recon.py`, `router.py`,
`separation.py`, `verify.py`, `descriptors.py`, `cachefs.py`, plus `data/model_registry.json`
and `data/vst_catalog.json`. Tools land in a new `reference` category (registry-gated like
`arrange`, present in `default`/`full`).

```
mp3 ──► Stage 0 FINGERPRINT ─► Stage 1 RECON ─► Stage 2 ROUTE ─► Stage 3 EXTRACT
              (hash+cache)        (CPU, sec)      (question-       (GPU, registry
                                      │            driven)          checkpoints)
                                      │                                  │
        answers many questions ◄──────┘                Stage 4 VERIFY ◄──┘
        with zero GPU work                          (gates, §5) pass│fail─► ESCALATE (§6)
                                                                │
                                                    Stage 5 CHARACTERIZE (descriptors, §7)
                                                                │
                                            descriptor diff vs user's rendered track
                                                                │
                                            Stage 6 FX MAPPING (§8: installed VSTs,
                                            catalog suggestions, stock-first apply loop)
```

### 3.1 "Start cheap" — the cost-tier ladder, made concrete

Every operation is assigned a tier. **The router never runs a tier higher than the question
needs**, and every tier's output is cached (§9), so costs are paid at most once per file.
Timings are order-of-magnitude for a 4-minute song on the 3080 box; calibrate in Task 1 of the
implementation plan.

| Tier | What runs | Compute | Typical wall time |
|---|---|---|---|
| T0 | cache lookup by content hash | none | ms |
| T1 | **recon**: decode, LUFS (pyloudnorm), tempo + beat grid, key estimate (chroma + Krumhansl profile w/ confidence + alternates), structure segmentation (self-similarity novelty → section boundaries), band-energy sketch (librosa/numpy) | CPU | 5–20 s |
| T2 | **instrument-activity timeline**: audio tagger (PANNs CNN14-class / Essentia MTG-Jamendo instrument model) on hop windows → per-second instrument probabilities | CPU or tiny GPU | 10–30 s |
| T3 | **targeted separation**: ONE registry checkpoint, cropped to the region of interest ± 5 s context (crop first, separate second — a 20 s crop is ~12× cheaper than the full song) | GPU | 10–60 s |
| T4 | **escalation**: alternate checkpoint, 2-model ensemble (waveform averaging), complement subtraction, or full-song separation | GPU | minutes |
| T5 | **last resort**: polyphonic note transcription (basic-pitch — instrument-agnostic single note stream; MT3-class per-instrument attribution deferred) | GPU/CPU | minutes |

Routing table (question class → max tier): key/tempo/structure/loudness/"is it in tune" → **T1**;
"what instruments, when / arrangement density / what enters at the chorus" → **T2**; "this
specific sound at 1:12 / compare my guitar to theirs" → **T3** (+T4 only on gate failure);
"give me the notes it plays" → **T5** only after T3/T4 stems exist or the user asks for notes
explicitly. The convenience orchestrator tool applies this table; Claude can also call stages
directly and override.

Two standing efficiency rules: **(a) region first** — when the user's question names a moment or
recon/tagging can localize it, never separate the whole song; **(b) exposure check** — if the
tagger shows the target sound is *exposed* (solo/intro/break: normalized non-target activation,
§5.2 definition, < 0.15 in that window), skip separation entirely and characterize the mix region directly. Distinctive
sounds are very often exposed somewhere; the router searches the timeline for the target's most
exposed window before reaching for a separator.

## 4. Model registry (`data/model_registry.json`)

**No model is hard-coded.** Each entry declares: `id`, `task` (multi-stem | single-target |
tagger | dereverb | drumkit-split | transcription), `targets` (e.g. `["vocals"]`,
`["kick","snare","toms","hats","cymbals"]`), `engine`, `checkpoint_source` (URL + sha256, pinned —
same policy as the sound-pack), `vram_gb`, `rt_factor` (× realtime on reference HW), `benchmark`
(published SDR where known), `license`, `rank` (router preference order per target).

**Inference engines (two — both required, verified 2026-07-22):** `python-audio-separator`
(MIT, pip-installable) covers the MDX/VR/Demucs/MDXC families — which includes BS-RoFormer and
Mel-Band RoFormer — but does **not** implement SCNet. Since SCNet is the cross-architecture
arbiter (§5.4) and ensemble partner (§6 R2), vendoring ZFTurbo's
`Music-Source-Separation-Training` inference path for SCNet is a **hard requirement**, not a
contingency. All checkpoints are pre-fetched at install time (§9); tools never hit the network at
call time. Decision point for next session: pin exact versions + hashes of both engines.

**v1 registry seed** (exact checkpoint ids/hashes pinned at implementation time — the *slots* are
the spec):

| Slot | Model family | Notes |
|---|---|---|
| 4-stem A (default) | **BS-RoFormer ep17** (ZFTurbo GitHub release, avg SDR 9.65) | free, no account; runs in audio-separator. NOT the viperx checkpoints — those are 2-stem vocal models |
| 4-stem B (arbiter / ensemble partner) | **SCNet-XL** (ZFTurbo release, avg SDR 9.80) | different architecture → the disagreement signal; actually beats A on avg SDR; needs the MSST engine |
| 4-stem C (CPU fallback) | HTDemucs-FT | runs anywhere, well-understood |
| vocals (dedicated) | Mel-Band RoFormer vocal checkpoint (viperx/KimberleyJensen class) | 2-stem vocal/instrumental models — vocal questions only |
| 6-stem | **BS-Roformer-SW** (jarredou HF mirror) + htdemucs_6s fallback | adds guitar + piano stems — the free piano answer. **No dedicated free piano or bass RoFormer checkpoints exist** (verified 2026-07: MVSEP piano is site/account-gated); bass = the bass stem of 4/6-stem models |
| guitar (dedicated) | becruily Mel-RoFormer guitar (HF, free) | quality varies by mix — always gate (§5) |
| drum-kit split | DrumSep (jarredou release) / LarsNet | kick/snare/hats/cymbals from the drum stem. **LarsNet weights are CC BY-NC 4.0** — record in `license` |
| de-reverb | anvuew de-reverb Mel-RoFormer | **vocal-only by training domain** → reverb estimator for VOCAL stems only (§7); non-vocal stems use the decay heuristic, flagged approximate |
| tagger | PANNs **framewise SED variant** (CNN14-DecisionLevel*-class) | T2 timeline + §5 bleed judge. Plain CNN14 is clip-level — the framewise variant is required. Essentia is OUT: no Windows wheels on PyPI, MTG-Jamendo weights CC BY-NC (this settles §11 Q2) |
| transcription | basic-pitch (Spotify, Apache-2.0) | T5 only. **Instrument-agnostic**: ONE undifferentiated note stream, no per-instrument attribution (that is MT3-class territory, deferred). Pins Python ≤3.11; low-maintenance upstream |

**Known gap, stated honestly:** no strong open checkpoints for strings/winds/brass sections, and
no *dedicated* free piano/bass extractors (6-stem models are the free ceiling there) as of this
writing (commercial-only territory). Router strategy for those targets: complement
subtraction (extract vocals+drums+bass with the strong models; the residual *is* the
section) + exposure search + tagger timeline. Registry slots exist so the moment community
checkpoints appear, they're a JSON entry, not code.

## 5. Verification — "against what?"

There is no ground-truth stem for a commercial mix, so verification is **triangulation from four
independent, cheap signals**. Each produces a number; together they gate.

1. **Mix consistency (physics) — multi-stem outputs ONLY.** Stems from one model must sum back
   to the input: `residual_db = 20·log10(rms(mix − Σstems)/rms(mix))`. Great ≤ −30 dB; suspect
   > −15 dB. Catches over-suppression and model collapse. Explicitly **not applicable to
   single-target extractions** (vocal/guitar/de-reverb models): there the "residual" is the whole
   rest of the song by construction (and defining the complement as `mix − stem` makes it
   identically zero) — single-target runs are judged by gates 2–4 alone. (Mask-based multi-stem
   models pass this easily — necessary, not sufficient; it exists to catch catastrophic failures
   cheaply.)
2. **Bleed score (independent judge).** Run the T2 tagger *on the isolated stem*. The tagger is
   multi-label (independent per-class sigmoids that do NOT sum to 1), so the metric must be
   defined, not hand-waved: `bleed = mean over active frames of [max non-target activation] /
   (target activation + ε)`, clipped to [0, 1]. Pass < 0.20; hard-fail > 0.40. The judge (tagger)
   and the contestant (separator) are different models trained on different tasks — that
   independence is what makes the check meaningful. (§3.1's exposure check uses this same
   normalized definition.)
3. **Silence honesty — with FX-tail tolerance.** In windows where the tagger timeline says the
   target is *not playing*, the stem's energy should be ≈ 0. `false_energy_db` = stem RMS in
   those windows relative to its active-window RMS. Pass ≤ −35 dB. **Carve-out:** a correctly
   separated stem legitimately carries reverb/delay tails past the last played note — exactly the
   character §7 wants to measure — so windows within a tail allowance (default 3 s, config) after
   any active region are excluded from the check, not merely thresholded. Catches "the model put
   a ghost of everything everywhere."
4. **Cross-model agreement (escalation only).** When two architectures (RoFormer vs SCNet) have
   both produced the target: mel-spectrogram correlation between the two stems. High agreement
   (> 0.90) → high confidence, pick the one with the better bleed score. Low agreement (< 0.75)
   → both suspect → ensemble them and re-gate the ensemble. Ensemble method: **waveform
   averaging** (`avg_wave`) as primary — ZFTurbo's own benchmarks show it consistently matching
   or beating magnitude-spectrogram averaging — with spectrogram averaging as the alternate.
   Disagreement between independent systems is the closest available substitute for ground truth.

All thresholds above are **initial values, marked for calibration** against the synthetic-truth
test set (§10) — they are spec'd as config (`analysis/verify.py` constants), not magic numbers
scattered in code. Every gate result (pass/fail + numbers) is returned to Claude in the tool
result's meta, so the model can reason about confidence and tell the user the truth ("the guitar
stem is 88% clean; treat the reverb-tail number as approximate").

## 6. Escalation — "how does it know?"

Deterministic ladder, driven only by gate results and the registry. Each rung logs *why* it fired
(the failing gate + value) into the tool result meta.

```
run rank-1 checkpoint for target, region-cropped
 └─ gates pass → done (Stage 5)
 └─ gate fails →
    R1 rank-2 checkpoint (different architecture)          [T4 stage; ~T3-scale cost]
    R2 2-model ensemble (avg_wave) + re-gate               [T4]
    R3 complement subtraction (extract strong targets,     [T4]
       take residual as the target) + re-gate
    R4 widen region → full-song separation (some models    [T4]
       do better with full context) + re-gate
    R5 exposure fallback: best-available exposed window
       of the raw mix, descriptors flagged "mix-context"   [T1–T2 cost]
    R6 LAST RESORT: transcription-backed analysis (notes/  [T5 — opt-in only]
       rhythm only; timbre descriptors marked unavailable)
```

All escalation rungs are T4-stage by definition (R1's *cost* is merely T3-scale). **R6 never
fires automatically**: it requires explicit opt-in (`max_tier=T5` or the user asking for notes) —
consistent with §3.1's T5 rule — because on gate exhaustion the stems are by definition bad, and
transcription on the raw mix is basic-pitch at its weakest. At default settings the ladder ends
at R5 with confidence flagged low and the gate numbers reported honestly.

Budget guard: the orchestrator takes `max_tier` and `max_seconds` parameters (defaults T4 /
300 s); Claude can raise them when the user says "take your time, get it right." Ladder position,
like every other artifact, is cached — re-asking a question never re-climbs.

## 7. The descriptor schema — "the character sheet"

`descriptors.py` computes a versioned `SoundCharacter` dataclass (JSON-serializable; schema
version field for cache invalidation). Computed identically for reference stems and for renders
of the user's own tracks — the diff of two `SoundCharacter`s is the FX-mapping input.

- **Spectral:** centroid, rolloff, tilt (dB/oct fit), 8-band energy profile, harmonic-vs-
  percussive ratio.
- **Dynamics:** integrated + short-term LUFS, crest factor, attack-time distribution
  (transient sharpness), envelope shape stats.
- **Harmonics/saturation:** odd/even harmonic energy profile on sustained segments,
  inharmonicity estimate → "clean / warm-even / gritty-odd" classification.
- **Stereo:** per-band mid/side ratio (width), inter-channel correlation.
- **Time-based FX (the "hard to describe in words" payload):**
  - delay: envelope autocorrelation → echo time(s) in ms + feedback estimate. **Flagged
    approximate by design:** tempo-synced delays are confounded with musical repetition (an
    ⅛-note echo and an ⅛-note performance pattern autocorrelate identically) — prefer measuring
    on exposed/tail regions, and report confidence;
  - modulation: periodicity of spectral flux → rate (Hz) + depth → chorus/phaser/tremolo/vibrato
    classification;
  - reverb — two paths by stem type: **vocal stems** use the de-reverb model (wet − dry →
    wet/dry ratio, RT60-class decay, early/late ratio; the model is vocal-only by training
    domain, §4); **all other stems** use an envelope decay-slope heuristic on note releases,
    always flagged approximate.
- **Pitch (melodic stems):** f0 track summary, vibrato rate/depth, portamento presence.

## 8. The FX loop — from numbers to knobs

**Inputs:** `SoundCharacter(reference_stem)` vs `SoundCharacter(render_of_user_track)` (rendering
a solo'd track is a bridge verb: `render_track_region` — new, small, needed anyway for M3's
`render_and_audit`).

**Step 1 — diff → effect intents.** Rule table mapping descriptor deltas to intent objects:
`{"intent": "boost", "what": "presence 2–5 kHz", "amount_db": 6}`, `{"intent": "add_delay",
"time_ms": 28, "feedback": 0.25}`, `{"intent": "saturate", "flavor": "even-harmonic"}`. Pure
functions, unit-testable, no REAPER needed.

**Step 2 — intent → plugin, three sources in priority order:**
1. **The user's installed inventory** (`list_installed_fx` — live-verified, 252 plugins on the
   target machine): exact name-matched recommendations first.
2. **REAPER stock + JSFX** (always present): ReaEQ/ReaComp/ReaDelay/ReaVerb/ReaXcomp + the
   bundled JSFX library. These have **stable, documented parameter indices** — they are the
   *apply* path (Step 3). JSFX is a first-class citizen here deliberately: free, scriptable,
   param-addressable, huge community catalog via ReaPack.
3. **The curated free-VST catalog** (§8.1): when neither of the above covers the intent well,
   recommend an install ("grab Valhalla Supermassive — free — for this cavern-style reverb").

**Step 3 — apply (stock-first) + audit.** Un-defer the M1 FX verbs: `add_fx_by_name`
(inventory-validated), `set_fx_param`, `get_fx_params`. v1 ships **param maps only for ReaPlugs +
selected JSFX** (`data/fx_param_maps.json`: plugin → param name → index + unit + range). For
mapped plugins Claude applies the chain, calls `render_track_region`, re-computes descriptors,
and iterates until the diff converges or plateaus (max-iterations guard) — the closed loop.
Unmapped third-party plugins get precise *human* instructions instead ("Serum: unison 4, detune
0.15"); param maps grow plugin-by-plugin as a data contribution, not a code change.

**Honesty clause carried into every tool docstring:** this converges *toward* a sound
("convincingly in the neighborhood"), it does not clone it; anything claiming otherwise is lying.

### 8.1 Curated free-VST catalog (`data/vst_catalog.json` — v1 seed)

Selection bar: genuinely free (OSS or unrestricted freeware), Windows-solid, community-respected
quality, no account walls (exceptions flagged). Each entry: name, vendor, category, license
(OSS/freeware/donationware), `account_required`, url, and which effect-intents it serves.

**Instruments / sound sources**
| Plugin | What / why it's on the list | License |
|---|---|---|
| Surge XT | flagship OSS synth — subtractive/wavetable/FM, enormous range | GPL3 |
| Vital | modern wavetable standard — **flag: free tier needs a vital.audio account**; Vitalium (OSS fork, via DISTRHO-Ports) needs none | freemium / GPL fork |
| Dexed | DX7-class FM, loads original patch banks | GPL3 |
| Odin 2 | 24-voice polysynth | GPL3 |
| OB-Xd | classic analog-poly character — **catalog pins the legacy 2.x line (GPL3, reales/OB-Xd)**; current discoDSP 3.x is freemium with commercial-use restrictions | GPL3 (legacy line) |
| sforzando + **VSCO2 Community Edition** | free SFZ player + CC0 orchestral library — **the concert-band answer** on the synthesis side | free / CC0 |
| Decent Sampler (+ Pianobook libraries) | free sampler with a large free-library ecosystem — **flag: email-gated download** | freeware |
| MT Power Drum Kit 2 | workhorse free acoustic drums | freeware |
| **Neural Amp Modeler (NAM)** | OSS neural guitar-amp modeling + thousands of free community captures at **TONE3000** (ToneHunt's successor; free, no account) — the guitar-tone-replication workhorse | **MIT** |
| Splice INSTRUMENT (free tier) | absorbed the former Spitfire LABS catalog (LABS discontinued late 2025) — **flag: Splice account required** | freeware (account) |

**Effects**
| Plugin | Intents served | License |
|---|---|---|
| **Airwindows Consolidated** | saturation/tape/console/character — hundreds of processors, one plugin | MIT (sources; Consolidated binaries GPL3 via JUCE) |
| Valhalla Supermassive / FreqEcho / SpaceModulator | huge reverbs, freq-shift echo, flange — legendary free tier | freeware |
| TDR Nova | dynamic EQ (surgical + de-harsh intents) | freeware |
| TDR Kotelnikov | mastering-grade compressor | freeware |
| TDR VOS SlickEQ | tone-shaping EQ w/ saturation stages | freeware |
| Melda MFreeFXBundle | 30+ utility FX (chorus, tremolo, autopan, …) | freeware tier |
| Xfer OTT | the modern multiband-upward-comp sound | freeware |
| CHOW Tape Model (+ ChowDSP suite) | physical-model tape saturation/wow/flutter | GPL |
| Dragonfly Reverbs | OSS hall/room/plate family | GPL |
| LSP plugin suite | large OSS FX collection (surgical tools) | LGPL |
| Analog Obsession (catalog) | console/comp/EQ emulations — **flag: distributed via Patreon posts (free Patreon account)** | donationware |
| Voxengo SPAN + Youlean Loudness Meter | analysis/metering (also useful to *show* the user what Orpheus measured) | freeware |
| Polyverse Wider | mono-safe widening — **flag: name+email registration for download** | freeware |

(REAPER stock ReaPlugs + JSFX are implicit — always present, always the apply-path default.)

## 9. Caching, packaging, tool surface

**Cache** (`~/.orpheus_cache/<sha256-of-file>/`): `recon.json`, `timeline.json`,
`stems/<model-id>/<region>/*.flac` + `verify.json` per stem, `descriptors/<stem-or-region>.json`,
`ladder.log`. Content-addressed → renames/moves of the mp3 don't invalidate; descriptor schema
version busts stale entries. A `clear_analysis_cache` tool + size cap (config, default 20 GB, LRU).

**Packaging:** core Orpheus stays lean. New extras: `orpheus-mcp[analysis]` (librosa, pyloudnorm,
tagger — CPU-friendly, enables T1/T2 everywhere incl. the laptop) and `orpheus-mcp[separation]`
(torch + audio-separator + vendored MSST inference — the desktop). Tools register only when
their extra is importable; otherwise they surface a clear "install `orpheus-mcp[separation]` on
this machine" error. Same graceful-degradation pattern already specced for BasicPitch.
**Checkpoint pre-fetch is mandatory:** an `orpheus-mcp fetch-models` install step downloads and
hash-verifies every registry checkpoint up front — `audio-separator`'s default lazy
download-on-first-use is disabled, so analysis tools NEVER touch the network at call time
(keeps §1.2 honest). Note on hints: `readOnlyHint` on analysis tools means "does not modify the
REAPER project or the input file" — they do write to the local cache. basic-pitch's Python ≤3.11
ceiling is a packaging constraint for whichever extra carries it (isolate or sub-process it if
the main env moves past 3.11).

**New tools** (category `reference`, all `readOnlyHint` except the FX-apply verbs):
`analyze_reference(file, question_focus?, region?, max_tier?, max_seconds?)` (the orchestrator),
`recon_reference(file)`, `instrument_timeline(file)`, `separate_region(file, target, region,
model_id?)`, `verify_stem(...)`, `characterize(file_or_stem, region?)`,
`compare_character(a, b)`, `clear_analysis_cache()` — plus bridge-side
`render_track_region(track, region)` and the un-deferred `add_fx_by_name` / `set_fx_param` /
`get_fx_params` in `mix`.
Every heavy tool streams progress + returns gate numbers in meta (Claude must be able to say
"this took the R2 ensemble rung; bleed 0.12").

## 10. Testing strategy — where ground truth DOES exist

Live commercial mixes have no truth; **synthetic mixes do**. Test fixtures: render 8–16-bar
multi-instrument pieces with Orpheus's own compose stack + free SFZ instruments → we possess the
true stems → mix them down → run the pipeline → compute *real* SDR/bleed against truth. This
calibrates the §5 gate thresholds empirically (choose thresholds that separate known-good from
known-bad runs) instead of hand-waving them. Plus: unit tests for every descriptor on
constructed signals (a synthetic 28 ms echo must yield `delay_ms ≈ 28`; a 6 dB/oct tilt must
read back as one); router table tests (question class → tier ceiling, fake registry); ladder
tests (forced gate failures walk R1→R6 in order, with logging); cache tests (hash stability,
schema-version busting); registry tests (checkpoint hash pinning — the `<PIN_BEFORE_SHIP>`
lesson). CI runs CPU-only with a tiny Demucs or stubbed engine; the RoFormer checkpoints are
desktop-verified like everything live-REAPER.

## 11. Open questions for next session (the "discuss more" list)

1. **Engine pin:** BOTH engines are required (§4 — audio-separator lacks SCNet). Verify on the
   desktop that audio-separator runs BS-RoFormer ep17 + the vocal/guitar/6-stem/de-reverb
   checkpoints, that the vendored MSST path runs SCNet-XL, then pin versions + hashes.
2. **Tagger pick — SETTLED 2026-07-22:** PANNs framewise SED variant. Essentia is out: no
   Windows wheels on PyPI (`pip install` fails on the target machine) and the MTG-Jamendo weights
   are CC BY-NC. Remaining sub-question: which SED head (DecisionLevelMax/Avg/Att) gives the best
   per-second precision for the §5 gates.
3. **Chord recognition:** T1 currently yields key + chroma; do we add a dedicated audio
   chord-sequence model (e.g. a CRNN chord estimator) or derive chords from chroma + music21?
   Affects "chord structure" fidelity on dense mixes.
4. **Threshold calibration protocol:** how many synthetic fixtures, which genres, and do we also
   hand-label a handful of *owned* real songs as a sanity panel?
5. **`render_track_region` mechanics:** REAPER render API via bridge (which render preset,
   where do files land, cleanup policy).
6. **Descriptor→intent rule table v1:** which 10–15 deltas ship first (presence, mud, air, width,
   delay, reverb size, saturation flavor, transient snap, …)?
7. **VST catalog delivery:** recommend-only (JSON + docstrings) in v1, or also a
   `suggest_free_vst(intent)` tool that reads the catalog?
8. **VRAM guard:** detect available VRAM at tool call time and auto-degrade (3080 = 10 GB is
   fine, but don't OOM if REAPER + a big project already holds memory)?

## 12. Suggested next-session sequence (at the desktop)

1. Read this spec top to bottom; argue with it; edit inline.
2. Resolve open questions 1–3 (they gate everything else) with quick local experiments:
   install `audio-separator`, run the rank-1 vocal + 4-stem checkpoints on an owned mp3,
   eyeball/ear-check stems, time them on the 3080.
3. Run the tagger candidates on the same file; compare timelines against your ears.
4. Then invoke the planning flow (superpowers `writing-plans`) to cut this into TDD tasks —
   suggested slice order: **(1)** cache+recon (T0/T1), **(2)** tagger timeline (T2),
   **(3)** registry+engine+separate_region (T3), **(4)** verify gates + ladder (T4),
   **(5)** descriptors, **(6)** compare + FX intents, **(7)** FX verbs + apply loop,
   **(8)** orchestrator tool + docs.
