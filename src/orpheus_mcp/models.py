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
from typing import Annotated, Any

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
    guid: str | None = Field(None, description="Stable identity that survives across bridge calls.")
    name: str
    role: TrackRole = TrackRole.OTHER
    volume_db: float = 0.0
    pan: float = Field(0.0, ge=-1.0, le=1.0)
    mute: bool = False
    solo: bool = False
    fx_chain: list[FXSpec] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)


class Section(BaseModel):
    name: str = Field(..., description="e.g. 'intro', 'verse', 'drop'.")
    start_bar: int
    length_bars: int


# --------------------------------------------------------------------------- #
# Analysis outputs
# --------------------------------------------------------------------------- #


class HarmonyAnalysis(BaseModel):
    key_root: str | None = None
    mode: Mode | None = None
    key_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="music21 K-S confidence - surface it, never hide it."
    )
    alternative_keys: list[str] = Field(default_factory=list)
    roman_numerals: list[str] = Field(default_factory=list)
    cadences: list[str] = Field(default_factory=list)
    note: str = Field(
        "",
        description="Hedging note when confidence is low or the project lacks chordal MIDI.",
    )


class GrooveAnalysis(BaseModel):
    swing_pct: float | None = Field(
        None,
        description="0 = straight 8ths, 1 = full triplet swing (offbeats at 2/3 of the beat).",
    )
    tightness: float | None = Field(
        None, description="1 = grid-perfect, 0 = onsets a full half-grid off. From beat onsets."
    )
    velocity_mean: float | None = Field(None, description="Mean note velocity (0-127).")
    velocity_stddev: float | None = Field(
        None, description="Velocity spread — 0 means machine-flat dynamics."
    )
    density_notes_per_beat: float | None = Field(
        None, description="Note onsets per beat over the analyzed span."
    )
    feel: str = Field(
        "", description="Human-readable quantization feel, e.g. 'hard-quantized, straight'."
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
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class EditPlan(BaseModel):
    """The output of `recommend_changes` (read-only) and the input to `apply_changes` (gated)."""

    intent: str = Field(..., description="e.g. \"make this sound Classical\".")
    target_style: str | None = None
    edits: list[ProposedEdit] = Field(default_factory=list)
    caveats: list[str] = Field(
        default_factory=list,
        description="Honest limits — low detection confidence, taste-based choices, etc.",
    )
