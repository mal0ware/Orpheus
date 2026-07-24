"""The typed contract that ties the analyze → recommend → apply loop together.

`CompositionSpec` is the universal IR: the analyzer targets it (project → Spec),
the recommender diffs against it (Spec(current) vs a style fingerprint → EditPlan),
and the builder consumes it (Spec/EditPlan → bridge calls).

Returning these Pydantic models from tools gives us MCP ``structuredContent`` for
free, so each `EditPlan` is *simultaneously* the machine-applicable payload and the
human-readable list of reasons.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Musical primitives
# --------------------------------------------------------------------------- #

Beats = Annotated[float, Field(ge=0.0, description="Musical time in beats (not PPQ/ticks).")]
Velocity = Annotated[int, Field(ge=0, le=127, description="MIDI velocity 0–127.")]
Pitch = Annotated[int, Field(ge=0, le=127, description="MIDI pitch 0–127 (60 = middle C).")]


class Mode(StrEnum):
    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    LOCRIAN = "locrian"


class TrackRole(StrEnum):
    """Canonical role vocabulary shared across analysis, mixing, and composition."""

    KICK = "kick"
    SNARE = "snare"
    HAT = "hat"
    DRUMS = "drums"
    BASS = "bass"
    SUB_BASS = "sub_bass"
    KEYS = "keys"
    PAD = "pad"
    LEAD = "lead"
    ARP = "arp"
    GUITAR = "guitar"
    STRINGS = "strings"
    VOCAL = "vocal"
    FX = "fx"
    OTHER = "other"


class Note(BaseModel):
    """A single MIDI note, addressed to the model in beats."""

    pitch: Pitch
    start_beat: Beats
    duration_beats: Annotated[float, Field(gt=0.0)]
    velocity: Velocity = 96


# --------------------------------------------------------------------------- #
# Project representation (mirrors dschuler36's typed tree, over the live bridge)
# --------------------------------------------------------------------------- #


class FXSpec(BaseModel):
    name: str
    bypassed: bool = False
    params: dict[str, float] = Field(
        default_factory=dict, description="Decoded by name, not opaque 0-1."
    )


class TrackSpec(BaseModel):
    guid: str | None = Field(
        default=None, description="Stable identity that survives across bridge calls."
    )
    name: str
    role: TrackRole = TrackRole.OTHER
    volume_db: float = 0.0
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    mute: bool = False
    solo: bool = False
    fx_chain: list[FXSpec] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)


class Section(BaseModel):
    name: str = Field(..., description="e.g. 'intro', 'verse', 'drop'.")
    start_bar: int
    length_bars: int


# --------------------------------------------------------------------------- #
# Reference-analysis primitives (§9) — shared by the cache filesystem and every
# downstream analysis module (recon, timeline, separation, lyrics, ...).
# --------------------------------------------------------------------------- #


class Window(BaseModel):
    """A time span in seconds — the region-of-interest primitive threaded through the
    reference-analysis pipeline (stem crops, timeline gaps, lyric line spans, ...)."""

    start_s: float
    end_s: float


class BeatGrid(BaseModel):
    """Beat/downbeat/meter estimate produced by ``analysis/recon.py`` (T1). ``confidence``
    is the mean beat-onset alignment score, normalized [0, 1]; below
    ``CFG.BEAT_GRID_CONFIDENCE_MIN`` downstream consumers switch to a fixed grid — the flag
    rides on this field, this module never switches itself (``kind`` is always ``"beat"``
    coming out of recon). ``meter_confidence`` is a DIFFERENT signal (D10): the margin ratio
    between the winning {3,4}-group correlation and the runner-up, i.e. how decisively the
    meter itself (not the grid alignment) was picked — a track can have a rock-solid grid
    (high ``confidence``) with an ambiguous meter (low ``meter_confidence``), e.g. an
    unaccented uniform click train. Calibration caveat: on real mixes this margin reads
    low (~0.05-0.16 measured on the fixture set) even when the picked meter is correct —
    treat it as a decisiveness signal, never a correctness signal."""

    times_s: list[float]
    downbeats_s: list[float]
    meter: tuple[int, int]
    confidence: float
    meter_confidence: float
    kind: Literal["beat", "fixed"]


class KeyEstimate(BaseModel):
    """One Krumhansl-Schmuckler key reading — ``Recon.key`` is the best correlation,
    ``Recon.key_alternates`` the next two."""

    root: str
    mode: str
    confidence: float


class Recon(BaseModel):
    """The T1 recon artifact (§9 ``recon.json``): decode + LUFS + tempo/beat grid + tuning +
    key + structure + a coarse band-energy sketch. Cached by ``recon_reference``."""

    schema_version: int = 1
    duration_s: float
    sample_rate: int
    lufs_integrated: float | None
    tempo_bpm: float
    grid: BeatGrid
    tuning_offset_cents: float
    key: KeyEstimate
    key_alternates: list[KeyEstimate]
    sections: list[Window]
    band_energy_db: dict[str, float]


class Timeline(BaseModel):
    """The T2 instrument-activity timeline (§9 ``timeline.json``): raw, un-thresholded
    tagger activations, mapped down to the union of ``data/model_registry.json``'s
    ``tagger_classes`` (§9 tool contract) so a 4-minute song stays tool-result-sized
    (~240 x |mapped| floats) instead of the tagger's full ~527-class AudioSet output.

    ``hop_s`` is LOCKED to 1.0: ``build_timeline`` max-pools the tagger's native (~10 ms)
    frames up to a 1 s hop — a spike anywhere within a second makes that second read the
    spike's value, never an average that would wash out a short hit. Every gate/exposure
    computation downstream (``most_exposed_window``, ``target_absent``, the §5.2 bleed
    gate) consumes these same 1 s frames. No threshold is baked in here — CFG.
    TIMELINE_ACTIVITY_THRESHOLD / EXPOSURE_THRESHOLD are applied by consumers, never by
    this model or ``build_timeline`` itself."""

    schema_version: int = 1
    classes: list[str]
    hop_s: float
    frames: list[list[float]]


class ChordSegment(BaseModel):
    """One chord in the §13 baseline lane. ``root``/``quality`` are the base triad reading
    (``quality`` stays the plain triad quality — maj/min/7/maj7/min7/sus2/sus4/dim/aug or
    ``"N"`` for no-chord); ``extensions`` carries §13.3 upgrades (``"add9"``, ``"maj7"``,
    ``"7"``, ``"6"``, ``"#11"``, ``"13"``, ``"pedal"``) and ``bass`` the §13.2 slash bass
    pitch (``None`` in root position). ``label`` is the composed spelling (``"Cadd9"``,
    ``"C/E"``, ``"Em7"``). ``roman``/``key_context`` come from the §13.4 theory layer (both
    ``None`` until it runs). ``start_beat``/``end_beat`` are ``None`` in fixed-grid mode
    (rubato fallback — the lane rides on ``start_s``/``end_s`` then). ``produced_by`` is
    ``"baseline"`` for every segment this lane writes."""

    start_beat: float | None
    end_beat: float | None
    start_s: float
    end_s: float
    root: str
    bass: str | None
    quality: str
    extensions: list[str] = Field(default_factory=list)
    label: str
    roman: str | None = None
    key_context: str | None = None
    confidence: float
    produced_by: str


class ChordsLane(BaseModel):
    """The §13 chord lane (§9 ``chords.json``). ``grid`` is ``"beat"`` on a confident recon
    grid, ``"fixed"`` when recon's grid confidence fell below
    ``CFG.BEAT_GRID_CONFIDENCE_MIN`` and the lane switched to fixed 1 s windows (§13.4
    rubato fallback)."""

    schema_version: int = 1
    grid: Literal["beat", "fixed"]
    segments: list[ChordSegment]


# --------------------------------------------------------------------------- #
# §14 lyric-anchored-navigation primitives — the typed contract analysis/lyrics.py
# produces (VAD-gated ASR → aligned words → lines → named/indexed sections). ``locate_lyric``
# (§14.2) resolves lyric quotes against ``LyricsResult.words``; the section names ("chorus 2",
# "section B") let "the second chorus" resolve textually as well as structurally (§14.1.4).
# --------------------------------------------------------------------------- #


class LyricWord(BaseModel):
    """One ASR-transcribed word with its timing and confidence (§14.1 ``lyrics.json``).
    ``start_s``/``end_s`` are Whisper's own (coarse) timestamps until whisperX forced
    alignment tightens them — the ``LyricsResult.aligned`` flag says which (§14.1.3).
    ``conf`` is the model's word probability [0, 1]; low-confidence words carry their
    confidence rather than being dropped or invented (the §14.1 honesty rule)."""

    word: str
    start_s: float
    end_s: float
    conf: float


class LyricLine(BaseModel):
    """A line of lyrics assembled from consecutive words (§14.1.4): a new line begins on a
    silence gap greater than ``CFG.LYRIC_LINE_GAP_S`` or after sentence-ending punctuation.
    ``text`` is the words joined with single spaces; ``word_indices`` point back into
    ``LyricsResult.words`` (temporal order)."""

    text: str
    start_s: float
    end_s: float
    word_indices: list[int] = Field(default_factory=list)


class LyricSection(BaseModel):
    """A named, indexed lyric section (§14.1.4). ``kind`` is ``"chorus"``/``"verse"`` when
    repeated line-blocks were detected (cross-checked against ``Recon.sections``), or
    ``"section"`` for the through-composed fallback (no repeats → structural ``section A/B/C``
    from recon boundaries, never an invented "chorus"). ``index`` is the 1-based TEMPORAL
    order among sections of this same ``kind`` (so "the second chorus" = ``kind="chorus",
    index=2``), independent of any score ranking. ``line_indices`` point into
    ``LyricsResult.lines``."""

    name: str
    kind: str
    index: int
    start_s: float
    end_s: float
    line_indices: list[int] = Field(default_factory=list)


class LyricsResult(BaseModel):
    """The §14.1 lyric index (``lyrics.json`` + ``lines.json`` + ``sections.json``, one
    artifact here). ``status`` is ``"no_vocals"`` when the vocal stem never crosses the
    §14.1 VAD hallucination gate — ASR never ran, and every list is empty (Whisper-family
    models hallucinate confident text on silence, so the gate runs BEFORE ASR, not after).
    ``language`` is Whisper's detected language (``None`` on ``no_vocals``). ``aligned`` is
    ``False`` when no wav2vec2 aligner exists for the detected language (or none is installed)
    and the coarse Whisper timestamps were kept (§14.1.3)."""

    schema_version: int = 1
    status: Literal["ok", "no_vocals"]
    language: str | None
    aligned: bool = False
    words: list[LyricWord] = Field(default_factory=list)
    lines: list[LyricLine] = Field(default_factory=list)
    sections: list[LyricSection] = Field(default_factory=list)


class LocateMatch(BaseModel):
    """One §14.2 ``locate_lyric`` candidate match. ``matched_text`` is the ACTUAL transcript
    words spanned by the match window (not the user's query quote) — what was really sung.
    ``section`` is the containing :class:`LyricSection`'s ``name`` (e.g. ``"chorus 2"``) when
    the match window's start falls inside one, else ``None``. ``occurrence`` is the 1-based
    TEMPORAL rank (sung order, not score order) among the candidates in the same
    :class:`LocateResult` — a repeated chorus line's second occurrence always carries
    ``occurrence=2``, even if it scored lower than occurrence 1."""

    start_s: float
    end_s: float
    matched_text: str
    score: float
    section: str | None
    occurrence: int


class LocateResult(BaseModel):
    """The §14.2 ``locate_lyric`` result. ``status="no_vocals"`` is the DEFINED result for a
    file whose vocal stem never crossed the §14.1 VAD gate — ``matches`` stays empty, and this
    is deliberately distinguishable from an ``"ok"`` result that merely found nothing (the
    honesty rule: a no-vocal file must never look like an ordinary empty match list).
    ``low_confidence`` is set when the best candidate scored below
    ``CFG.LYRIC_MATCH_MIN_SCORE`` — ``matches`` then holds the top-3 candidates instead of the
    (empty) passing set, and callers should ask the user rather than guess. ``notes`` carries
    honesty strings about the scoring itself — e.g. a non-English transcript disables the
    phonetic-equivalence leg of token matching (§14.2 fix wave 1: accented Latin tokens could
    otherwise collide under the English-only phonetic reduction), so matching degrades to plain
    edit distance and that degradation is stated here, not silently applied."""

    schema_version: int = 1
    status: Literal["ok", "no_vocals"]
    matches: list[LocateMatch] = Field(default_factory=list)
    low_confidence: bool = False
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Separation + verification primitives (§5/§9) — the typed contract separation.py
# (engines, ``separate_region``) produces and verify.py (Task 4.2, gates) fills in.
# --------------------------------------------------------------------------- #


class GateResult(BaseModel):
    """One §5 verification gate outcome. ``verdict`` is ``"na"`` when the gate does not
    apply to this stem (single-target has no mix-consistency, a wall-to-wall pad has no
    silent window to score, ...); ``value`` is the measured number (dB residual, bleed
    ratio, ...) or ``None`` when N/A. The gate *logic* lands in Task 4.2's ``verify.py`` —
    this is the shared typed shape both tasks address."""

    verdict: Literal["pass", "fail", "na"]
    value: float | None


class GateSet(BaseModel):
    """The four §5 gates for a stem plus the composed ``overall`` verdict (§5 decision
    rule: ``fail`` if any applicable gate fails, ``unverifiable`` iff every gate is N/A,
    else ``pass``)."""

    mix_consistency: GateResult
    bleed: GateResult
    silence_honesty: GateResult
    agreement: GateResult
    overall: Literal["pass", "fail", "unverifiable"]


class StemResult(BaseModel):
    """One separated stem, region-cropped and length-aligned to the input (§3.1d/§5). The
    engine adapters (Task 4.1) produce everything except ``gates`` — the §5 gate results
    are stamped by ``separate_region``'s Task 4.2 tool wrapper. ``region_trimmed`` is the
    EXACT region the stem covers after the ±context-pad crop was trimmed back and clamped
    to ``[0, duration]`` (a window past EOF clamps and this field says so). ``notes`` carries
    the §15.5 VRAM-degradation / mono-policy reason strings for tool meta (a fallback to a
    lower-VRAM or CPU model, or dual-mono fed to a stereo-only engine, is surfaced here, never
    hidden)."""

    stem_path: str
    model_id: str
    target: str
    region_trimmed: Window
    sample_rate: int
    gates: GateSet | None = None
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# §7 SoundCharacter primitives — the per-source descriptor contract the reference
# analysis engine's ``characterize`` tool (Task 6.1) produces. ``SourceMeta`` rides on
# every ``SoundCharacter`` so §8 knows exactly WHAT was measured (which stem, produced how,
# over which region, mono-vs-stereo, complement-residual-or-not); the descriptor blocks are
# plain dicts (not sub-models) because their KEY NAMES are the shared contract with
# ``data/intent_rules.json`` (§15.3) and Task 10.1 cross-checks them at load time — a typed
# sub-model per block would fossilize that contract in two places.
# --------------------------------------------------------------------------- #


class SourceMeta(BaseModel):
    """What a :class:`SoundCharacter` was measured FROM (§7/§8). ``stem_type`` is the
    separation target (``"vocals"``/``"drums"``/…) or ``"mix-region"`` for a literal-path
    source; ``produced_by`` is the producing model id or ``"file"``. ``bandwidth_hz`` is
    stamped by ``characterize`` (the §7 spectral-cliff measurement) and is ``None`` until
    measured. ``channels`` is the HONEST channel count of the actual audio — a dual-mono
    stem (two identical channels) reports ``1`` so the §7 mono policy fires (stereo
    descriptors stay ``None``, never fabricated from a fake stereo field). ``complement`` is
    ``True`` for a §6 R3 residual stem (it inherits the tails of everything subtracted, so
    §8 discounts tail-sensitive descriptors)."""

    stem_type: str
    produced_by: str
    region: Window | None = None
    sample_rate: int
    bandwidth_hz: float | None = None
    channels: int
    complement: bool = False


class StemHandle(BaseModel):
    """A reference to an already-produced/cached stem, carrying everything
    ``resolve_source`` needs to build honest :class:`SourceMeta` without re-deriving it: the
    on-disk audio path, the stem's type (its separation target), what produced it, the
    region it covers, and the producer's ``notes`` (the §6 R3 ``complement=True`` flag and
    §7 mono-policy strings ride here)."""

    stem_path: str
    stem_type: str
    produced_by: str
    region: Window | None = None
    notes: list[str] = Field(default_factory=list)


class SoundCharacter(BaseModel):
    """The §7 post-FX descriptor bundle for one source. Each block is a flat dict whose keys
    are the locked §15.3 contract (see ``analysis/descriptors.py`` for the exact key set per
    block). ``stereo`` values are ALL ``None`` when ``source.channels == 1`` (mono policy —
    never upmixed); ``pitch`` is ``None`` wholesale for non-melodic stems. ``notes`` carries
    honesty strings (approximation caveats, the dereverb-degraded path, …)."""

    source: SourceMeta
    spectral: dict[str, float | None] = Field(default_factory=dict)
    dynamics: dict[str, float | None] = Field(default_factory=dict)
    harmonics: dict[str, float | str | None] = Field(default_factory=dict)
    stereo: dict[str, float | None] = Field(default_factory=dict)
    time_fx: dict[str, float | str | bool | None] = Field(default_factory=dict)
    pitch: dict[str, float | str | None] | None = None
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# §15.3 descriptor→intent rules — the typed contract ``analysis/intents.py`` (Task 10.1)
# produces and ``tools/reference.py::compare_character`` returns. Intent generation lives
# in the rules engine ONLY (§9): Claude reads these, never computes them.
# --------------------------------------------------------------------------- #


class Intent(BaseModel):
    """One fired §15.3 rule. ``apply_path`` is the stock (ReaPlugs/JSFX) apply recipe from
    ``data/intent_rules.json``'s ``"apply"`` block, verbatim, or ``None`` for a recommend-only
    intent — either because the rule itself never auto-applies (rule 14: master-bus loudness
    needs explicit user opt-in) or because a §6 R3 complement/residual source made a
    tail-sensitive rule (11, 12) recommend-only for this particular diff (the discount note
    rides on :class:`CharacterDiff.notes`, not here). ``vocal_only_skipped`` is reserved for a
    partially-evaluated vocal-only rule; Task 10.1's engine instead skips vocal-only rules
    entirely (with a note) when the stem isn't vocals on both sides, so it stays ``False`` in
    practice today. ``installed_matches`` (§8 step 2.1) are the caller's own
    ``list_installed_fx`` names whose text substring-matched (case-insensitive) this rule's
    ``inventory_keywords`` — empty when no ``inventory`` was passed to ``compare_character``."""

    rule_id: int
    intent: str
    params: dict[str, float | str] = Field(default_factory=dict)
    apply_path: dict[str, str | int | float] | None
    explanation: str
    vocal_only_skipped: bool = False
    installed_matches: list[str] = Field(default_factory=list)


class CharacterDiff(BaseModel):
    """The §9 ``compare_character`` result: ``diffs`` maps every numeric descriptor present on
    both sides (excluding string/bool-typed fields like ``flavor``/``mod_class``/
    ``reverb_rt60_class``/``delay_approx``/``reverb_approx``, which the fixed §15.3 rules read
    directly off the source :class:`SoundCharacter` objects instead) to ``{"a": ..., "b": ...,
    "delta": a - b, "unit": ...}`` (§9 contract) — spectral (``band_*_db``) keys whose upper
    edge exceeds ``band_limit_hz`` are excluded entirely (§8 mandatory normalization: comparing
    a full-bandwidth render against a codec-limited reference outside their common band reads
    the render as systematically 'darker', misfiring rules 3/5). ``unit`` is a fixed string
    (``"db"``/``"ms"``/``"hz"``/``"ratio"``/``"unitless"``/``"db_per_oct"``) looked up per
    descriptor by ``analysis.intents``'s module-level unit map — never inferred from the value.
    ``intents`` are every §15.3 rule that fired, in rule-id order. ``band_limit_hz`` is
    ``min(a.source.bandwidth_hz, b.source.bandwidth_hz)`` (or whichever side measured a
    bandwidth, or the Nyquist ceiling if neither did) — always recorded, per §8, so Claude can
    say 'compared up to 16 kHz (mp3-limited)'. ``notes`` carries the band-limit statement plus
    any vocal-only-skip / complement-discount honesty strings the engine emitted."""

    diffs: dict[str, dict[str, float | str | None]] = Field(default_factory=dict)
    intents: list[Intent] = Field(default_factory=list)
    band_limit_hz: float
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Analysis outputs
# --------------------------------------------------------------------------- #


class HarmonyAnalysis(BaseModel):
    key_root: str | None = None
    mode: Mode | None = None
    key_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="music21 K-S confidence - surface it, never hide it.",
    )
    alternative_keys: list[str] = Field(default_factory=list)
    roman_numerals: list[str] = Field(default_factory=list)
    cadences: list[str] = Field(default_factory=list)
    note: str = Field(
        default="",
        description="Hedging note when confidence is low or the project lacks chordal MIDI.",
    )


class GrooveAnalysis(BaseModel):
    swing_pct: float | None = Field(
        default=None,
        description="0 = straight 8ths, 1 = full triplet swing (offbeats at 2/3 of the beat).",
    )
    tightness: float | None = Field(
        default=None,
        description="1 = grid-perfect, 0 = onsets a full half-grid off. From beat onsets.",
    )
    velocity_mean: float | None = Field(default=None, description="Mean note velocity (0-127).")
    velocity_stddev: float | None = Field(
        default=None, description="Velocity spread — 0 means machine-flat dynamics."
    )
    density_notes_per_beat: float | None = Field(
        default=None, description="Note onsets per beat over the analyzed span."
    )
    feel: str = Field(
        default="", description="Human-readable quantization feel, e.g. 'hard-quantized, straight'."
    )


class AudioCharacter(BaseModel):
    """Objective post-FX DSP features — the 'sonic signature' vector."""

    low_energy_db: float | None = None
    mid_energy_db: float | None = None
    high_energy_db: float | None = None
    spectral_centroid_hz: float | None = None
    lufs_integrated: float | None = None
    true_peak_db: float | None = None
    crest_factor_db: float | None = None
    stereo_width: float | None = None


class CompositionSpec(BaseModel):
    """The universal IR. A snapshot of a project's musical + sonic identity."""

    tempo_bpm: float
    time_signature: tuple[int, int] = (4, 4)
    harmony: HarmonyAnalysis = Field(default_factory=HarmonyAnalysis)
    groove: GrooveAnalysis = Field(default_factory=GrooveAnalysis)
    audio: AudioCharacter = Field(default_factory=AudioCharacter)
    tracks: list[TrackSpec] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)


class FeatureDelta(BaseModel):
    """One measured-vs-fingerprint comparison — the atomic unit of 'not a vibe'."""

    feature: str = Field(..., description="Dimension name, e.g. 'tempo', 'swing'.")
    measured: float | str | None = Field(
        default=None, description="What the project actually measures."
    )
    target: float | str | None = Field(default=None, description="What the fingerprint expects.")
    matches: bool
    explanation: str = Field(..., description="The delta in plain English, numbers included.")


class StyleExplanation(BaseModel):
    """'Here's why it does/doesn't sound like X' — every claim backed by a FeatureDelta."""

    style: str
    verdict: str = Field(..., description="e.g. 'sounds like classical (5/6 dimensions match)'.")
    score: float = Field(..., ge=0.0, le=1.0, description="Matched / evaluated dimensions.")
    deltas: list[FeatureDelta] = Field(default_factory=list)
    caveats: list[str] = Field(
        default_factory=list,
        description="Unmeasured dimensions and low-confidence detections, declared not hidden.",
    )


class MusicReport(BaseModel):
    """The 'LLM-Readable Music Report' — structured musical *facts*, not raw numbers."""

    spec: CompositionSpec
    summary: str = Field(..., description="Plain-English summary of the track's sound.")
    observations: list[str] = Field(
        default_factory=list, description="Human-readable findings, e.g. 'drum-forward'."
    )


# --------------------------------------------------------------------------- #
# The recommend → apply contract
# --------------------------------------------------------------------------- #


class EditAction(StrEnum):
    SET_TEMPO = "set_tempo"
    TRANSPOSE = "transpose"
    REWRITE_PROGRESSION = "rewrite_progression"
    ADD_TRACK = "add_track"
    ADD_FX = "add_fx"
    SET_FX_PARAM = "set_fx_param"
    APPLY_GROOVE = "apply_groove"
    MASTER_MATCH = "master_match"


class ProposedEdit(BaseModel):
    """One change in an EditPlan: simultaneously machine-applicable and self-explaining."""

    action: EditAction
    target: str = Field(..., description="Target: a track GUID/name, 'project', or 'master'.")
    reason: str = Field(
        ..., description="The human-readable 'why' - a delta vs the reference, shown for approval."
    )
    params: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class EditPlan(BaseModel):
    """The output of `recommend_changes` (read-only) and the input to `apply_changes` (gated)."""

    intent: str = Field(..., description='e.g. "make this sound Classical".')
    target_style: str | None = None
    edits: list[ProposedEdit] = Field(default_factory=list)
    caveats: list[str] = Field(
        default_factory=list,
        description="Honest limits — low detection confidence, taste-based choices, etc.",
    )


# --------------------------------------------------------------------------- #
# §9 answer envelope — the ``analyze_reference`` orchestrator's return shape (Task 12.1).
# The one tool-result that ties the whole reference-analysis engine together: it routes a
# question class to its §3.1 tier ceiling, runs only the stages that class needs, and packages
# WHATEVER it could produce (with every degradation named) into one honest bundle.
# --------------------------------------------------------------------------- #


class AnswerEnvelope(BaseModel):
    """The §9 answer envelope produced by ``analyze_reference``. ``question_class`` is the
    resolved (validated) question class (``None`` focus ⇒ ``"recon"``). ``tier_reached`` is the
    HIGHEST §3.1 cost tier actually run (``"T1"``..``"T5"``) — capped by the class ceiling,
    ``max_tier``, the budget, and any missing extra, and honestly lower than the ceiling when a
    stage was skipped/degraded. ``recon_summary`` is the compact T1 fact bundle (tempo/key/grid/
    loudness) present on every envelope.

    ``artifacts`` holds ONLY the groups the question class needed (§9 packaging note): a
    ``"recon"`` envelope has none; ``"timeline"``/``"chords"`` populate their one key;
    ``"sound_at"``/``"compare_track"`` populate ``stems`` + ``descriptors``; ``"lyric_navigate"``
    populates ``timeline`` + the region/activation-diff of the §14.2 window; ``"notes"`` populates
    ``notes`` (the T5 note-event stream). ``gates`` maps each delivered stem's target to its §5
    :class:`GateSet`. ``ladder_log`` is the §6 escalation transition record (empty when no ladder
    ran). ``budget_capped`` is ``True`` when ``max_seconds`` expired mid-ladder;
    ``capped_by_missing_extra`` names the extra (``"separation"``/``"analysis"``) whose absence
    forced an answer at a lower tier (§9 degrade-with-a-name rule); ``notes`` carries the
    cache-hit/miss log, the follow-up-call plan (``compare_track``), and every other honesty
    string."""

    schema_version: int = 1
    question_class: str
    tier_reached: str
    recon_summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    gates: dict[str, GateSet] = Field(default_factory=dict)
    ladder_log: list[str] = Field(default_factory=list)
    budget_capped: bool = False
    capped_by_missing_extra: str | None = None
    notes: list[str] = Field(default_factory=list)


class VstEntry(BaseModel):
    """One row of the §15.4 curated free-VST catalog (``data/vst_catalog.json``, §8.1's
    instruments + effects tables transcribed verbatim). ``category`` is ``"instrument"`` or
    ``"effect"``. ``account_required`` flags a plugin whose free download sits behind an
    account/email/Patreon wall (the catalog's exceptions, called out per-row in §8.1) —
    ``url`` and ``license`` are always populated regardless. ``intents`` is the keyword set
    ``suggest_free_vst`` matches a ``need`` string against (free-text terms describing what
    the plugin is for — NOT the §15.3 rule-table intent names, a different vocabulary).
    ``notes`` carries the row's flagged nuance (account-gate reason, legacy-line pin,
    license caveat, ...) verbatim from the spec."""

    name: str
    vendor: str
    category: Literal["instrument", "effect"]
    license: str
    account_required: bool
    url: str
    intents: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CacheClearResult(BaseModel):
    """``clear_analysis_cache``'s result: how much the §9 analysis cache
    (``~/.orpheus_cache``) held before it was wiped. ``bytes_freed`` is the total on-disk
    size of every cached pcm-hash directory (``CacheFS.size_bytes()``, measured BEFORE
    ``CacheFS.clear()`` runs); ``entries_cleared`` is the number of top-level pcm-hash
    directories removed. Both read ``0`` on an already-empty cache — not an error."""

    bytes_freed: int
    entries_cleared: int
