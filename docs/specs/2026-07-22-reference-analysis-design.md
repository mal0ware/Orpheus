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
| T1 | **recon**: decode, LUFS (pyloudnorm), tempo + beat grid **+ downbeat/meter estimate** (bars are measured, never assumed 4/4) **+ beat-grid confidence** (low confidence → §13 falls back to fixed-window mode), **tuning-offset estimate** (`librosa.estimate_tuning`; applied to ALL downstream chroma — 432 Hz bands and detuned tapes must not smear chords — and it IS the routing table's "is it in tune" answer), key estimate (tuning-corrected chroma + Krumhansl profile w/ confidence + alternates), structure segmentation (self-similarity novelty → section boundaries; short files degrade to a single section), band-energy sketch (librosa/numpy) | CPU | 5–20 s |
| T2 | **instrument-activity timeline + chord lane**: framewise PANNs SED tagger → per-second instrument probabilities; beat-synchronous chord recognition (§13 — baseline chord engine is CPU-only and always available; model recognizers upgrade the lane when `[separation]` is installed) | CPU or tiny GPU | 10–40 s |
| T3 | **targeted separation**: ONE registry checkpoint, cropped to the region of interest ± 5 s context (crop first, separate second — a 20 s crop is ~12× cheaper than the full song) | GPU | 10–60 s |
| T4 | **escalation**: alternate checkpoint, 2-model ensemble (waveform averaging), complement subtraction, or full-song separation | GPU | minutes |
| T5 | **last resort**: polyphonic note transcription (basic-pitch — instrument-agnostic single note stream; MT3-class per-instrument attribution deferred) | GPU/CPU | minutes |

Routing table (question class → max tier): key/tempo/structure/loudness/"is it in tune" → **T1**;
"what instruments, when / arrangement density / what enters at the chorus / what are the chords
and why do they feel like that" → **T2**; "this specific sound at 1:12 / compare my guitar to
theirs" → **T3** (+T4 only on gate failure); **lyric-quote navigation** ("what am I hearing
after the line '…'") → the first such question on a song builds the lyric index (§14: vocal
separation + ASR, T3-scale, cached once per file), after which lyric-anchored questions resolve
at **T1–T2**; "give me the notes it plays" → **T5** only after T3/T4 stems exist or the user
asks for notes explicitly. The convenience orchestrator tool applies this table; Claude can also call stages
directly and override.

Four standing rules: **(a) region first** — when the user's question names a moment or
recon/tagging can localize it, never separate the whole song; **(b) exposure check** — if the
tagger shows the target sound is *exposed* (solo/intro/break: normalized non-target activation,
§5.2 definition, < 0.15 in that window), skip separation entirely and characterize the mix
region directly (distinctive sounds are very often exposed somewhere; the router searches the
timeline for the target's most exposed window before reaching for a separator); **(c)
absent-target pre-gate** — if the T2 timeline never shows the target's mapped `tagger_classes`
(§4) above the activity threshold anywhere in the song, return "target not detected in this
song" at T2 cost and STOP: the ladder never spends a GPU-minute extracting an instrument that
isn't there ("compare my guitar to theirs" on a song with no guitar is a T2 answer, not a T4
failure); **(d) crop hygiene** — the ±5 s separation context is trimmed back to the requested
region before any verify/characterize step, and every region clamps to `[0, duration]` (a
"+8 bars" window at the song's end clamps to EOF and says so).

## 4. Model registry (`data/model_registry.json`)

**No model is hard-coded.** Each entry declares: `id`, `task` (multi-stem | single-target |
tagger | dereverb | drumkit-split | transcription), `targets` (e.g. `["vocals"]`,
`["kick","snare","toms","hats","cymbals"]`), `engine`, `checkpoint_source` (URL + sha256, pinned —
same policy as the sound-pack), `vram_gb`, `rt_factor` (× realtime on reference HW), `benchmark`
(published SDR where known), `license`, `rank` (router preference order per target), and
`tagger_classes` — the PANNs/AudioSet class ids that count as *target* for the §5.2 bleed gate
and the §3.1 exposure/absent-target checks. An empty `tagger_classes` list (e.g. "synth pad",
which has no clean AudioSet class) means those gates return **N/A** for that entry rather than
computing garbage against the wrong class.

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
| chord (model lane) | rank-1 **BTC large-voca** (MIT, ckpt in-repo, 170 classes); rank-2 **music-x-lab ISMIR2019** (MIT, extensions + inversions) — §13.1 | complex chords are the point (§13); the in-house template+Viterbi baseline in `[analysis]` is the correctness floor, models are upgrades. madmom EXCLUDED (no py3.12 install; NC-licensed models; majmin-only) |
| lyrics ASR | **faster-whisper** (MIT, large-v3, word timestamps) + **whisperX** wav2vec2 forced alignment (BSD-2, maintained, py3.12 OK) — §14.1 | runs on the ISOLATED VOCAL STEM, never the raw mix; stable-ts (MIT, archived 2026-05) as fallback only |

**Known gap, stated honestly:** no strong open checkpoints for strings/winds/brass sections, and
no *dedicated* free piano/bass extractors (6-stem models are the free ceiling there) as of this
writing (commercial-only territory). Router strategy for those targets: complement
subtraction (extract vocals+drums+bass with the strong models; the residual *is* the
section) + exposure search + tagger timeline. Registry slots exist so the moment community
checkpoints appear, they're a JSON entry, not code.

## 5. Verification — "against what?"

There is no ground-truth stem for a commercial mix, so verification is **triangulation from four
independent, cheap signals**. Each produces a number; together they gate.

**Decision rule (this, precisely, is what makes §6 deterministic):** every gate returns
`pass` / `fail` / `N/A`. The ladder escalates **iff any applicable gate is not `pass`**;
`N/A` gates are excluded from the decision; if ALL gates are N/A the artifact is delivered
flagged `unverifiable` and the ladder does **not** climb (climbing cannot fix
unverifiability). The prose quality bands below ("great … suspect") are calibration targets
for §15.1, not decision inputs — the shipped decision is binary per gate. Alignment
precondition for all gates: separator outputs are length-aligned and resampled to the input
before any arithmetic (separators pad/resample; an unaligned `mix − Σstems` residual is
spuriously huge).

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
   (target activation + ε)`, clipped to [0, 1]. "Target" = the registry entry's
   `tagger_classes` (§4); empty mapping ⇒ **N/A**. **Zero active frames ⇒ N/A** (mean over an
   empty set is not a number — this is also what the §3.1 absent-target pre-gate should have
   caught upstream). Pass < 0.20 (single threshold; the former 0.20–0.40 band is a §15.1
   calibration range only). The judge (tagger) and the contestant (separator) are different
   models trained on different tasks — that independence is what makes the check meaningful.
   Calibration note: PANNs is trained on full mixes, and its activations on dry isolated stems
   are domain-shifted — §15.1 calibrates bleed thresholds on *stems*, not mixes. (§3.1's
   exposure check uses this same normalized definition.)
3. **Silence honesty — with FX-tail tolerance.** In windows where the tagger timeline says the
   target is *not playing*, the stem's energy should be ≈ 0. `false_energy_db` = stem RMS in
   those windows relative to its active-window RMS. Pass ≤ −35 dB. **Carve-out:** a correctly
   separated stem legitimately carries reverb/delay tails past the last played note — exactly the
   character §7 wants to measure — so windows within a tail allowance (default 3 s, config) after
   any active region are excluded from the check, not merely thresholded. **Wall-to-wall
   instruments (a pad or bass that never stops) have zero inactive windows ⇒ N/A** — stated so
   the gate is honestly absent rather than silently vacuous; gating then rests on bleed +
   agreement. Catches "the model put a ghost of everything everywhere."
4. **Cross-model agreement (escalation only).** When two architectures (RoFormer vs SCNet) have
   both produced the target: mel-spectrogram correlation between the two stems. Agreement
   ≥ 0.90 → pass, pick the stem with the better bleed score. **< 0.90 → ensemble them and
   re-gate the ensemble** (no undefined middle band — one threshold, one action). Ensemble method: **waveform
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

**R3 precondition:** complement subtraction is only as good as what's subtracted — the stems
being subtracted must *individually* pass gates 2–4 first; if the ladder reached R3 because
those same stems failed, R3 is skipped (subtracting bad stems yields a worse residual, plus
the residual inherits the reverb/delay tails of everything subtracted — the residual's
descriptors carry a `complement: true` flag so Claude discounts tail-sensitive numbers).

Budget guard: the orchestrator takes `max_tier` and `max_seconds` parameters (defaults T4 /
300 s); Claude can raise them when the user says "take your time, get it right."
**Exhaustion semantics:** if the budget expires mid-ladder, the orchestrator returns the best
gate-scored artifact so far + the ladder log + `budget_capped: true` — never an error, never
a silent partial. The cache records **the cap under which the ladder stopped**; a later call
with a larger budget *resumes the climb from the recorded rung* — the "re-asking never
re-climbs" rule applies only to ladders that ran to completion, otherwise a capped first run
would freeze a low-quality answer forever.

## 7. The descriptor schema — "the character sheet"

`descriptors.py` computes a versioned `SoundCharacter` dataclass (JSON-serializable; schema
version field for cache invalidation). Computed identically for reference stems and for renders
of the user's own tracks — the diff of two `SoundCharacter`s is the FX-mapping input.

`SoundCharacter` carries `source` metadata: `{stem_type (vocals|guitar|…|mix-region|complement),
produced_by (model id or "render"), region, sample_rate, bandwidth_hz, channels}` — §15.3's
vocal-only rules read `stem_type` (the intent engine otherwise cannot know a diff is vocal),
and §8's normalization reads `bandwidth_hz`. **Mono policy:** stereo descriptors (width,
correlation) are N/A on mono sources — never fabricated via upmixing; separators that require
stereo receive dual-mono input, and the output's stereo descriptors stay N/A.

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

**Normalization before ANY diff (mandatory, `compare_character` enforces it):** resample both
sides to a common analysis rate, detect the reference's codec bandwidth ceiling (lossy files
have a spectral cliff, typically 16–20 kHz for mp3), and restrict every spectral descriptor
diff to the common band `min(ref_ceiling, user_ceiling)`. Without this, a full-bandwidth
32-bit render diffed against an mp3 reads systematically as "reference is darker," rules 3/5
fire in the wrong direction, and the closed loop *converges by making the user's track duller
than the codec artifacts* — the flagship feature optimizing toward mp3 damage. The applied
band is recorded in the diff output so Claude can say "compared up to 16 kHz (mp3-limited)."

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

**Input contract** (stated once, here): supported inputs are mp3/wav/flac/m4a(AAC)/ogg/aiff,
decoded via soundfile+ffmpeg; DRM-protected files (`.m4p`, FairPlay AAC) are **rejected with a
clear message**, never worked around; corrupted/truncated files raise a defined `DecodeError`
result (partial decodes are not silently analyzed). The cache key is the **sha256 of the
decoded PCM** (not the file bytes) — so editing an ID3 tag does not orphan minutes of cached
GPU work, and byte-identical audio in different containers shares one cache entry.

**Cache** (`~/.orpheus_cache/<pcm-sha256>/`): `recon.json`, `timeline.json`, `chords.json`,
`lyrics.json`+`lines.json`+`sections.json`, `stems/<model-id>/<region>/*.flac` + `verify.json`
per stem, `descriptors/<stem-or-region>.json`, `ladder.log`. **Every artifact's key includes
the identity of what produced it**: `(producing model id + checkpoint hash, registry version,
verify-config hash, schema version)` — not just the audio hash. Without this, swapping the
SED head after §15.6, upgrading a Whisper checkpoint, or re-tuning §15.1 thresholds serves
stale artifacts forever (and §15.1 *guarantees* those events happen). Ladder runs additionally
record the budget cap they stopped under (§6 exhaustion semantics). Entries in use by an
in-flight pipeline are **pinned against LRU eviction**. A `clear_analysis_cache` tool + size
cap (config, default 20 GB, LRU).

**Packaging — full dependency map (no orphan dependencies):** core Orpheus stays lean.
`orpheus-mcp[analysis]` = librosa, pyloudnorm, PANNs tagger, music21, the §13 in-house chord
baseline — CPU-friendly, enables T1/T2 everywhere incl. the laptop. `orpheus-mcp[separation]`
= the torch stack: audio-separator, vendored MSST inference, BTC + music-x-lab chord models,
faster-whisper + whisperX — the desktop extra (lyrics lives here because it *requires* the
vocal stem). basic-pitch — **decided, not deferred**: its own `orpheus-mcp[transcribe]` extra
running in a dedicated subprocess environment pinned to Python ≤3.11 (uv-managed), invoked
CLI-style — it must not constrain the main env's Python. Tools register only when their extra
is importable; otherwise they surface a clear "install `orpheus-mcp[separation]` on this
machine" error. **Orchestrator behavior when routing demands a missing extra** (laptop asked a
T3 question): answer at the highest available tier, with an explicit
`capped_by_missing_extra: "separation"` note — degrade with a name, never a generic error
when lower tiers can still say something useful.
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
`compare_character(a, b)`, `get_chords(file, region?)` (§13), `transcribe_lyrics(file)` +
`locate_lyric(file, quote, occurrence?)` (§14), `suggest_free_vst(need, limit?)` (§15.4),
`clear_analysis_cache()` — plus bridge-side `render_track_region(track, region)` and the
un-deferred `add_fx_by_name` / `set_fx_param` / `get_fx_params` in `mix`.
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

## 11. Open questions — ALL RESOLVED 2026-07-22 (kept for the record)

Design policy after the second session: **nothing is deferred to worker sessions as "think
about it."** Every former open question now has either a decision (with its section) or a
defined mechanical procedure with an acceptance bar. The only legitimately-open items are
empirical results (hash pins, F1 scores, ear-panel labels) whose procedures are fully specified.

1. **Engine pin:** BOTH engines are required (§4 — audio-separator lacks SCNet). Verify on the
   desktop that audio-separator runs BS-RoFormer ep17 + the vocal/guitar/6-stem/de-reverb
   checkpoints, that the vendored MSST path runs SCNet-XL, then pin versions + hashes.
2. **Tagger pick — SETTLED 2026-07-22:** PANNs framewise SED variant. Essentia is out: no
   Windows wheels on PyPI (`pip install` fails on the target machine) and the MTG-Jamendo weights
   are CC BY-NC. Remaining sub-question: which SED head (DecisionLevelMax/Avg/Att) gives the best
   per-second precision for the §5 gates.
3. **Chord recognition — SETTLED:** dedicated harmony engine, complex chords first-class → §13.
   (Chroma-only was rejected this session: simplified majmin output loses exactly the voicings
   that make progressions land.)
4. **Threshold calibration protocol — SETTLED:** fixed procedure + acceptance bar → §15.1.
5. **`render_track_region` mechanics — SETTLED:** exact bridge sequence, settings snapshot/
   restore, failure modes → §15.2.
6. **Descriptor→intent rule table v1 — SETTLED:** exactly 14 launch rules, enumerated with
   trigger conditions and stock mappings → §15.3.
7. **VST catalog delivery — SETTLED:** `suggest_free_vst` tool ships; recommend-only, Orpheus
   never installs plugins → §15.4.
8. **VRAM guard — SETTLED:** call-time free-VRAM query + registry `vram_gb` filter + CPU
   fallback chain, degradation always reported → §15.5. (SED-head pick mechanized in §15.6.)

## 12. Suggested next-session sequence (at the desktop)

1. Read this spec top to bottom; argue with it; edit inline.
2. Empirical pins, in order (each is run-and-record, not think-and-decide):
   a. both engines: run the 4-stem A/B + vocal + guitar + 6-stem + de-reverb checkpoints on an
      owned mp3; ear-check; time on the 3080; pin versions + sha256.
   b. chord: run the §13 baseline + rank-1 model on 2–3 owned songs whose chords Mal can verify
      by ear (pick ones with real 7ths/slash chords); record exact-label accuracy impressions.
   c. lyrics: vocal stem + faster-whisper on one owned song; test `locate_lyric` matching by
      hand against 3 quotes, including one deliberate misquote.
   d. tagger: the §15.6 SED-head script over the §15.1 fixtures.
3. Then invoke the planning flow (superpowers `writing-plans`) to cut this into TDD tasks —
   suggested slice order: **(1)** cache+recon (T0/T1), **(2)** tagger timeline (T2),
   **(3)** chord baseline + theory lane (§13, CPU parts), **(4)** registry + engines +
   `separate_region` (T3), **(5)** verify gates + ladder (T4), **(6)** descriptors (§7),
   **(7)** lyrics pipeline + `locate_lyric` (§14), **(8)** chord model lane + bass-informed
   refinement (§13 GPU parts), **(9)** compare + FX intents (§15.3), **(10)** FX verbs +
   apply loop, **(11)** orchestrator + `suggest_free_vst` + docs.

---

## 13. Harmony engine — chord structure done properly (closes §11 Q3)

Simplified majmin chord tracking is explicitly NOT acceptable (user requirement, 2026-07-22):
the target output is the chord *as played* — `Fmaj7#11/A`, `B♭m9`, `G7sus4 → G7` — because
voicing and extensions are usually the entire reason a progression lands. Three layers,
cheap → rich, all beat-synchronous on recon's beat grid.

**13.1 Base recognizer** (registry task `chord`) — pinned by the 2026-07-22 verification pass:
- **Rank-1: BTC large-voca** (jayg996/BTC-ISMIR19; MIT; checkpoint committed in-repo, no
  account). Vocabulary: 170 classes = 12 roots × 14 qualities (maj, min, dim, aug, min6, maj6,
  min7, minmaj7, maj7, 7, dim7, hdim7, sus2, sus4) + N/X. **Known limit: NO inversions — the
  label set strips bass distinctions**, which is precisely why §13.2 (bass-informed slash
  detection) is a separate, mandatory layer rather than an optional nicety. 2019-era PyTorch,
  unmaintained — vendor the inference path like MSST.
- **Rank-2: music-x-lab ISMIR2019 Large-Vocabulary Chord Recognition** ("Chord Structure
  Decomposition"; MIT; pretrained models in-repo/Drive). Decomposed output includes
  **extensions AND bass/inversions natively** — richer than rank-1 but older code; use as the
  cross-check/escalation model, same pattern as SCNet vs BS-RoFormer.
- **Considered and excluded:** crema (BSD-2; structured root/bass/pitch-class decode ≈602
  classes incl. inversions — attractive on paper, but TensorFlow ⇒ CPU-only on native Windows
  and Keras-3-fragile on py3.12; keep as a registry candidate, do not build on it). madmom
  (pip install broken on py3.12 — issues #527/#535; models CC BY-NC-SA; chord model
  majmin-only). ChordFormer (2025 paper, **no public checkpoint**). OMAR-RQ (AGPL code +
  NC weights). No 2023+ turnkey large-vocab open model exists as of 2026-07 — verified.
- **Always-available baseline** (no torch, ships in `[analysis]`): in-house beat-synchronous
  template matching over HPSS-harmonic chroma with Viterbi smoothing — vocabulary
  maj/min/7/maj7/min7/sus2/sus4/dim/aug + no-chord. ≈200 LOC, pure numpy/librosa, unit-tested
  against rendered fixtures. The baseline is the correctness floor; models are upgrades, never
  requirements.

**13.2 Bass-informed slash/inversion detection.** Fundamental of the bass lane — bass stem
when a separation exists, else low-passed mix + pYIN — evaluated **per chord segment, not per
beat**: emit a slash chord only when a single non-root bass pitch class is stable across
≥ 60% of the segment's beats (the §13.3 stability rule, same constant) **and** is a chord
tone; a stable non-chord-tone bass is annotated `pedal`, and unstable bass movement is left
alone. (The naive per-beat rule would turn a walking bass under one C-major bar into
`C, C/D, C/E, C/G` — false inversions by design; rejected.) Reuses machinery the pipeline
already has; no new model.

**13.3 Extension refinement.** Per chord segment: harmonic-chroma residual after subtracting
the detected triad template; pitch classes above an energy threshold become candidate
extensions (7, add9, 6, #11, 13, sus tones), accepted only when stable across ≥ 60% of the
segment's beats **and the segment is ≥ 2 beats long** (on a 1-beat segment any single
observation is trivially "100% stable" — noise would upgrade labels) → label upgrade
(`C` → `Cadd9`) with per-extension confidence. This is how the baseline lane also reaches
complex chords, model or no model.

**13.4 Theory layer** (music21). Sliding-window key segments — **window 16 bars, hop 4 bars,
with hysteresis: a modulation is accepted only when the new key wins ≥ 2 consecutive
windows** (otherwise every borrowed-chord passage would read as a key change) → per-chord
Roman numerals against the *local* key; annotate borrowed chords, secondary dominants, modal
mixture, and modulation points. Precedence when lanes disagree: the local key lane is
authoritative for Roman numerals; T1's global Krumhansl key remains the headline "key of the
song" (mode of the lane, weighted by duration). Output BOTH absolute (`Fmaj7#11`) and
functional (`IVmaj7#11`) spellings — Claude explains "why it feels like that" from the
functional lane.

**Beat-grid fallback (rubato / no steady pulse):** when recon's beat-grid confidence is below
threshold (ambient, free-time ballads, drifting live takes), the whole chord lane switches
from beat-synchronous to **fixed 1 s windows**, `chords.json` gains `grid: "fixed"` so every
consumer knows beat/bar arithmetic is unavailable, and bar-based region math (§14) degrades
to seconds. Stated here so no worker session invents it.

**Output schema** (`chords.json`, versioned): `[{start_beat, end_beat, start_s, end_s, root,
bass, quality, extensions[], label, roman, key_context, confidence, produced_by}]` where
`produced_by` records which layer wrote the segment. **Placement:** baseline at T2 (CPU,
seconds, alongside the tagger); model lane + bass refinement re-write segments at T2-GPU/T3
with higher confidence. **Acceptance bar (fixtures, §15.1 set extended with rendered
progressions containing extensions + inversions per genre):** baseline ≥ 90% majmin-correct;
model lane ≥ 80% exact-label. Below bar → thresholds/templates iterate before ship.

## 14. Lyric-anchored navigation — "after the line …" (user requirement 2026-07-22)

People locate song moments by lyric, not bar number. Every reference tool accepts a lyric
quote as a region specifier.

**14.1 Pipeline** (one-time per song, cached):
1. **Vocal stem** — rank-1 vocal model, full song (T3/T4 cost; cached and shared with §7's
   vocal descriptors — this is the same stem, computed once).
2. **Local ASR** — **faster-whisper** (MIT; large-v3; `int8_float16` on the 3080, `int8` on
   CPU) with `word_timestamps=True`, run on the vocal stem; model downloads are ungated
   (pre-fetched per §9). Singing is materially harder than speech — the isolated stem is what
   makes this viable at all, and this is now *evidenced*, not assumed: source-separated vocals
   give consistent WER reductions over full mixes for Whisper-family lyrics transcription
   (arXiv:2506.15514, SOTA open-source on the Jam-ALT benchmark zero-shot; also LyricWhiz,
   ISMIR 2023). Accuracy remains genre-dependent (strong on clear pop vocals, weak on
   screamed/heavily-processed vocals); the system degrades *honestly* — low-confidence words
   carry their confidence, nothing is invented.
   **Hallucination gate (mandatory, runs BEFORE ASR):** energy/VAD gate the vocal stem
   per-window; windows below the vocal-activity threshold are **excluded from transcription
   entirely**, and a file whose stem never crosses it returns `no_vocals` — because
   Whisper-family models famously hallucinate confident text ("Thanks for watching…") on
   silence and instrumental bleed, with plausible timestamps. Without this gate, instrumentals
   grow fabricated lyrics, section naming clusters them, and `locate_lyric` "finds" lines
   nobody sang — the exact failure the honesty rule forbids. `locate_lyric` on a `no_vocals`
   file returns a defined "no vocal content detected" result, not an empty fuzzy match.
3. **Alignment tightening** — **whisperX** (BSD-2-Clause; actively maintained, py3.12 +
   Windows + CUDA verified 2026-07) wav2vec2-CTC forced alignment snapping word boundaries;
   melisma stretches sung words far beyond any speech-timing assumption, so Whisper's own
   timestamps are treated as coarse until aligned. Note: whisperX's *diarization* extra pulls
   HF-gated pyannote models — diarization is irrelevant on a solo vocal stem and MUST stay
   uninstalled (keeps the no-accounts rule intact). Fallback if whisperX breaks: stable-ts
   (MIT — archived upstream 2026-05, fallback only).
4. **Line + section assembly** — words → lines (silence-gap + punctuation heuristics);
   near-identical repeated line blocks clustered → chorus candidates; cross-checked against
   §3.1 recon's structure boundaries → **named, indexed sections** (`verse 1`, `chorus 2`, …).
   Side effect: "the second chorus" now resolves textually too, not just structurally.
   **Through-composed fallback:** no repeated blocks → sections stay structural
   (`section A/B/C` from recon boundaries) — the namer never invents a "chorus."
   **Language:** Whisper's detected language is recorded; the wav2vec2 aligner is selected
   per-language (the default aligner is English-only — aligning Japanese lyrics with it
   fails); if no aligner exists for the detected language, Whisper's own coarser timestamps
   are used and flagged. `locate_lyric`'s phonetic scorer (double-metaphone) is
   English-phonetics — for other languages matching drops to edit-distance-only, stated in
   the result meta.

Cache artifacts: `lyrics.json` `[{word, start_s, end_s, conf}]`, `lines.json`, `sections.json`.

**14.2 `locate_lyric(file, quote, occurrence?)` — the resolver.** Normalize both sides
(case/punctuation/whitespace); slide the quote over the transcript scoring token-level edit
distance + phonetic equivalence (double-metaphone class), because the two failure modes are
symmetric and both guaranteed: the ASR mishears, and the user misremembers. Returns ranked
matches `[{start_s, end_s, matched_text, score, section, occurrence}]` — a repeated chorus
line returns *all* occurrences; `occurrence` indexes **temporal order** (occurrence 2 = the
second time it's sung), independent of score ranking. Quotes may span line boundaries (the
match window slides over the word stream, not line-by-line). Normalization handles Whisper's
asterisk-masked profanity (`f***` ↔ the user's actual word must not tank the edit distance —
masked tokens match any token sharing the visible prefix). Best score < 0.6 → return top-3
candidates + a low-confidence flag; Claude asks the user instead of guessing.

**Router integration:** "what am I hearing after the line ⟨quote⟩" → `locate_lyric` → region
`[match.end_s, +8 bars]` → §3.1 timeline diff (which instrument activations rise vs the
preceding region) → answered at T1–T2. Separation fires only if the follow-up chases a
specific sound's *character*.

**14.3 Policy** (extends the standing copyright decisions): transcripts are locally-derived
navigation metadata from audio the user owns — cached locally, **never fetched from lyric
services** (no Genius/Musixmatch/etc.: network egress + ToS + §1.1), never embedded into
project files. `place_lyric_markers`' original-lyrics-only rule is unchanged and unaffected.

## 15. Closing decisions (nothing left as "think about it")

**15.1 Threshold calibration protocol** (closes Q4). Fixture set: **24 synthetic mixes** =
8 genre profiles × 3 densities (sparse/mid/dense), 8–16 bars, rendered by the compose stack
through free SFZ instruments (VSCO2 CE + free kits); truth stems retained. Plus a **6-song
owned-audio sanity panel**, ear-labeled at the desktop (clean/acceptable/bad per stem).
Procedure: run every registry separator on every fixture; compute true SDR + all §5 gate
metrics; per gate, choose the threshold at the ROC knee separating good (SDR > 7) from bad
(SDR < 4) runs; freeze in `verify.py` config together with the fixture-set hash. Re-runs
automatically (CI, CPU-models-only job) whenever the registry or fixtures change.

**15.2 `render_track_region` bridge mechanics** (closes Q5). One handler, settings always
restored (Lua pcall-wrapped, restore in the error path too), **serialized by a bridge-side
render mutex** (REAPER render settings are global project state — two racing render calls
would snapshot each other's temp values and corrupt the project's render config permanently):
1. Refuse if transport is playing/recording (stop-or-error, configurable; default error).
2. Snapshot `GetSetProjectInfo` `RENDER_*` values + **every track's** solo state + master-FX
   bypass state.
3. **Clear `I_SOLO` on ALL tracks** (a track the user left soloed would otherwise contaminate
   the render), then set the target track `I_SOLO=2` — that is **solo-in-place** (0=off,
   1=solo, 2=SIP; "exclusive solo" is a misnomer for this value): SIP is chosen
   *deliberately* because it keeps sends, so a guitar routed through a reverb bus renders
   WITH its bus FX — which is the sound the user actually hears. Consequence, stated: solo'ing
   a bus renders its whole submix; that is correct behavior, not a bug.
4. **Bypass the master FX chain** (snapshot in step 2, restored after): the user's master
   limiter/EQ must not be baked into per-track renders — the reference stem has no master
   chain on it, and rule 7 (compression) would misfire against limiter squash otherwise.
5. `GetSet_LoopTimeRange(region)` — region arrives in seconds; callers converting from
   beats/bars MUST use the **project tempo map** (`TimeMap2_beatsToTime`), never recon's
   reference-song grid (they are different songs' grids).
6. Set render source = master mix, bounds = time selection, `RENDER_FILE` = cache *directory*
   + `RENDER_PATTERN` = fixed basename (REAPER splits directory and filename across these two
   — using only RENDER_FILE mis-targets), format WAV 32-bit float, `RENDER_SRATE` = project
   rate → `Main_OnCommand(42230)` ("Render project, using the most recent render settings" —
   never 40015, which opens the dialog).
7. Assert the output exists, is non-empty, **and is not digital silence** (an offline-media /
   frozen track renders a valid silent file that passes an existence check; silence ⇒
   `BridgeError("track rendered silent — offline media? muted item?")`).
8. Restore snapshot (render settings, all solos, master-FX bypass). Returns
   `{path, region, sr}`. Cleanup via the §9 cache LRU.

**15.3 Descriptor→intent rule table v1** (closes Q6) — ships with **exactly these 14 rules**,
stored as data (`data/intent_rules.json`), each `{trigger on §7 diff → intent JSON → stock
apply recipe → one-line human explanation template}`, unit-tested on constructed descriptor
pairs:

| # | Trigger (descriptor diff) | Intent | Stock apply path |
|---|---|---|---|
| 1 | 2–5 kHz band Δ > +3 dB | presence boost | ReaEQ bell 3.5 kHz |
| 2 | 200–400 Hz Δ (ref leaner) | mud cut | ReaEQ cut ~300 Hz |
| 3 | 10–16 kHz Δ | air shelf | ReaEQ high shelf 12 kHz |
| 4 | 100–250 Hz Δ (ref fuller) | warmth shelf | ReaEQ low shelf 180 Hz |
| 5 | spectral tilt Δ > 1.5 dB/oct | tilt | ReaEQ two-shelf |
| 6 | attack-time distribution Δ | transient snap ± | JSFX transient controller |
| 7 | crest factor Δ > 4 dB | compression (ratio/attack derived from Δ) | ReaComp |
| 8 | even-harmonic energy Δ | saturation, tape/tube flavor | JSFX saturation / Airwindows |
| 9 | odd-harmonic energy Δ | saturation, drive/clip flavor | same, different flavor param |
| 10 | per-band M/S Δ | widen (band-scoped, mono-safety check) | JSFX stereo width |
| 11 | echo detected in ref, absent in user | delay (time snapped to nearest musical division, feedback, mix) | ReaDelay |
| 12 | wet/dry + decay-class Δ | reverb (size, mix, predelay) | ReaVerb w/ bundled IRs |
| 13 | 5–9 kHz dynamic Δ on vocals | de-ess | ReaXcomp band |
| 14 | LUFS + crest joint Δ | limiting/loudness — **master bus only with explicit user opt-in** | JSFX limiter |

Rule notes (each closes an ambiguity a worker would otherwise guess at): rule 11's musical
division snaps to the **project's** tempo when applying (the user's song is the canvas), while
the reference's raw ms value is reported alongside; rule 10's mono-safety check = post-widen
inter-channel correlation must stay ≥ 0.5 in the widened band (verified in the audit render,
reverted if violated); rule 13 knows a diff is vocal via `SoundCharacter.source.stem_type`
(§7) — vocal-only rules are skipped, with a note, when stem identity is `mix-region`.

Diffs outside these 14: Claude reasons and recommends freely, but Orpheus does **not**
auto-apply — the automation boundary is explicit and honest.

**15.4 VST catalog delivery** (closes Q7): ship `suggest_free_vst(need, limit=3)` reading
`data/vst_catalog.json`; returns entries + URLs + license/account flags. Recommend-only —
Orpheus never downloads or installs plugins (standing decision from the original design).

**15.5 VRAM guard + concurrency** (closes Q8): before every GPU tool call, check
`torch.cuda.is_available()` **first** (`mem_get_info` *raises* with no CUDA device — the
guard itself must not crash on a CPU-only box or after a driver failure; no CUDA ⇒ CPU chain
directly), then query free VRAM; registry entries carry `vram_gb`; the router filters to
models fitting 0.8 × free VRAM, else falls through the CPU chain (HTDemucs-FT-CPU / §13
baseline chord / int8-CPU ASR) and **reports the degradation + reason in tool meta**.
**A process-wide GPU semaphore (1 slot) serializes all model loads/inference** — MCP clients
can issue parallel tool calls, and two calls that both see "9 GB free" then both load 6 GB
models is a textbook time-of-check/time-of-use OOM; the free-VRAM check is only meaningful
under the semaphore. Never OOM, never silently skip, never silently downgrade.

**15.6 PANNs SED-head pick** (closes the Q2 residue): scripted — run DecisionLevelMax /
DecisionLevelAvg / DecisionLevelAtt over the §15.1 fixture set, select highest frame-level
F1; tie breaks to DecisionLevelAtt. Part of the calibration CI job, not a discussion.
