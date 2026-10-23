"""``analysis/harmony.py`` (Task 3.1, §13): the baseline chord lane — templates + Viterbi,
§13.3 extensions, §13.2 slash detection, §13.4 theory layer.

The recognizer core is a pure function of a ``(beats, 12)`` chroma matrix, so the extension /
slash / theory cases feed constructed chroma and constructed lanes straight in (no librosa).
The ``>= 90%`` majmin bar (the §12 slice-3 acceptance bar) runs on a *synthesized* clean
progression per the brief ("synthesized C-Am-F-G ... asserted numerically"): the committed
``ci_sparse`` FULL-MIX render is deliberately messy voicings and is exercised only as a
non-crash smoke case (its C/E span reads as D — oracle top-5 chord recall ~0.70 — so the
dense render cannot reach the bar with template matching; see
``.superpowers/sdd/task-3.1-report.md``). The bar is additionally enforced (D11) on
``ci_sparse``'s isolated chords STEM — beat-gridded from the mix's recon — which is not
chroma-smeared and reaches the bar honestly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from orpheus_mcp.analysis import harmony
from orpheus_mcp.analysis.cachefs import decode
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.harmony import (
    _ROOT_INDEX,
    bass_pc_lane,
    chord_lane_baseline,
    detect_slash,
    refine_extensions,
    theory_layer,
)
from orpheus_mcp.analysis.recon import run_recon
from orpheus_mcp.models import BeatGrid, ChordSegment, ChordsLane

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "audio"

# Triad fundamentals (Hz) for the synthesized-progression cases.
_TRIADS: dict[str, list[float]] = {
    "G": [196.00, 246.94, 293.66],
    "C": [261.63, 329.63, 392.00],
    "D": [293.66, 369.99, 440.00],
    "Em": [164.81, 196.00, 246.94],
    "Am": [220.00, 261.63, 329.63],
    "F": [174.61, 220.00, 261.63],
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_progression(
    path: Path,
    symbols: list[str],
    *,
    bpm: float = 120.0,
    beats_per_chord: int = 4,
    sr: int = 22050,
) -> Path:
    """Render a chord progression as re-struck sine triads — one decaying strike per beat so
    the beat tracker finds a real grid (chords held with no onsets yield ~1 beat per chord,
    which starves Viterbi of the within-chord evidence it needs to smooth)."""
    beat_len = 60.0 / bpm
    n = int(beat_len * sr)
    t = np.arange(n) / sr
    env = np.exp(-3.0 * t).astype(np.float32)
    out: list[np.ndarray] = []
    for sym in symbols:
        for _ in range(beats_per_chord):
            strike = np.zeros(n, dtype=np.float32)
            for freq in _TRIADS[sym]:
                strike += (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            out.append(strike * env)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.concatenate(out), sr, subtype="PCM_16")
    return path


def _majmin(quality: str) -> str:
    return "min" if quality in {"min", "min7", "dim"} else "maj"


def _parse_symbol(sym: str) -> tuple[int, str]:
    m = re.match(r"([A-G]#?)(.*)", sym)
    assert m is not None
    root, rest = m.group(1), m.group(2)
    quality = "min" if rest.startswith("m") and not rest.startswith("maj") else "maj"
    return _ROOT_INDEX[root], quality


def _segment(
    root: str,
    quality: str,
    start_s: float,
    end_s: float,
    *,
    start_beat: float | None = None,
    end_beat: float | None = None,
) -> ChordSegment:
    return ChordSegment(
        start_beat=start_beat,
        end_beat=end_beat,
        start_s=start_s,
        end_s=end_s,
        root=root,
        bass=None,
        quality=quality,
        extensions=[],
        label=harmony._compose_label(root, quality, [], None),
        confidence=0.9,
        produced_by="baseline",
    )


def _beat_grid(times: list[float], downbeats: list[float]) -> BeatGrid:
    return BeatGrid(
        times_s=times,
        downbeats_s=downbeats,
        meter=(4, 4),
        confidence=0.9,
        meter_confidence=0.5,
        kind="beat",
    )


def _c_major_bar_lane() -> ChordsLane:
    """A one-segment lane: a C-major bar spanning 4 beats — the shared fixture for the
    §13.3 extension and §13.2 slash cases."""
    seg = _segment("C", "maj", 0.0, 2.0, start_beat=0, end_beat=4)
    return ChordsLane(grid="beat", segments=[seg])


def _score_majmin(
    lane: ChordsLane, symbols: list[str], beat_times: list[float], dur: float
) -> float:
    """Beat-level majmin accuracy of a lane against a synthesized progression expanded onto
    equal-duration spans (one span per symbol)."""
    span = dur / len(symbols)
    correct = 0
    for t in beat_times:
        idx = min(len(symbols) - 1, int(t / span))
        truth_root, truth_q = _parse_symbol(symbols[idx])
        matches = [s for s in lane.segments if s.start_s <= t < s.end_s] or [lane.segments[-1]]
        seg = matches[0]
        if (
            seg.quality != "N"
            and _ROOT_INDEX.get(seg.root, -1) == truth_root
            and _majmin(seg.quality) == _majmin(truth_q)
        ):
            correct += 1
    return correct / len(beat_times)


# --------------------------------------------------------------------------- #
# Vocabulary + templates sanity
# --------------------------------------------------------------------------- #


def test_vocabulary_is_109_binary_l2_normalized_templates():
    assert harmony._N_STATES == 109
    assert harmony._TEMPLATES.shape == (109, 12)
    norms = np.linalg.norm(harmony._TEMPLATES, axis=1)
    assert np.allclose(norms, 1.0)  # every template L2-normalized
    # the no-chord row is the flat 1/12 template (uniform before normalization)
    flat = harmony._TEMPLATES[harmony._N_STATE]
    assert np.allclose(flat, flat[0])


# --------------------------------------------------------------------------- #
# The §12 slice-3 acceptance bar — >= 90% majmin on a synthesized progression.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "symbols",
    [["G", "C", "D", "Em", "G", "C", "D", "Em"], ["C", "Am", "F", "G", "C", "Am", "F", "G"]],
    ids=["G-C-D-Em(ci_sparse)", "C-Am-F-G"],
)
def test_synthesized_progression_majmin_at_least_90pct(tmp_path, symbols):
    path = _make_progression(tmp_path / "prog.wav", symbols)
    pcm = decode(str(path))
    recon = run_recon(pcm)
    lane = chord_lane_baseline(pcm, recon)

    accuracy = _score_majmin(lane, symbols, list(recon.grid.times_s), pcm.duration_s)
    assert accuracy >= 0.90, f"majmin accuracy {accuracy:.3f} below the 0.90 bar"


def test_ci_sparse_render_produces_a_sane_lane():
    """The committed render is intentionally messy-voiced (its C/E span reads as D even
    with whole-span aggregation — oracle top-5 chord recall ~0.70), so it is a non-crash
    smoke case, not a 90% target. What it must do: run, stay beat-gridded, and label chords
    (not collapse to a single no-chord run)."""
    pcm = decode(str(FIXTURES / "ci_sparse" / "mix.flac"))
    recon = run_recon(pcm)
    lane = chord_lane_baseline(pcm, recon)

    assert lane.grid == "beat"
    assert len(lane.segments) >= 1
    assert any(s.quality != "N" for s in lane.segments)
    assert all(s.produced_by == "baseline" for s in lane.segments)


def test_ci_sparse_chords_stem_beat_level_majmin_at_least_90pct():
    """The honest fixture form of the §13/§12 slice-3 bar: the rendered ci_sparse FULL MIX
    is dense-render chroma-smeared (the baseline scores ~0.20 there; oracle template ceiling
    ~0.70 — see ``.superpowers/sdd/task-3.1-report.md``), but the fixture's isolated chords
    STEM carries the harmonic content the bar is meant to certify. Run the baseline lane on
    the stem's chroma with the beat grid from the MIX's recon (real usage — a router aligns
    every lane onto one recon grid), then score beat-level majmin accuracy against
    ``truth.json``'s progression (Gmaj7,C/E,D,Em7 @ 116 bpm, 2 bars/chord = 8 bars total)
    expanded onto that grid."""
    truth = json.loads((FIXTURES / "ci_sparse" / "truth.json").read_text())
    symbols = truth["progression"].split(",")
    bpm = float(truth["bpm"])
    beats_per_chord = 2 * 4  # 2 bars/chord, 4/4
    dur = len(symbols) * beats_per_chord * 60.0 / bpm

    mix_pcm = decode(str(FIXTURES / "ci_sparse" / "mix.flac"))
    recon = run_recon(mix_pcm)
    chords_pcm = decode(str(FIXTURES / "ci_sparse" / "stems" / "chords.flac"))
    lane = chord_lane_baseline(chords_pcm, recon)

    accuracy = _score_majmin(lane, symbols, list(recon.grid.times_s), dur)
    assert accuracy >= 0.90, f"stem majmin accuracy {accuracy:.3f} below the 0.90 bar"


# --------------------------------------------------------------------------- #
# §13.3 extension refinement
# --------------------------------------------------------------------------- #


def test_added_d_over_c_triad_becomes_cadd9():
    chroma = np.zeros((4, 12))
    for pc in (0, 4, 7, 2):  # C, E, G triad + a strong added D
        chroma[:, pc] = 1.0
    lane = _c_major_bar_lane()
    grid = _beat_grid([0.0, 0.5, 1.0, 1.5], [0.0])

    out = refine_extensions(lane, chroma, grid)
    assert out.segments[0].label == "Cadd9"
    assert out.segments[0].extensions == ["add9"]
    assert out.segments[0].quality == "maj"  # quality stays the base triad; ext carries the 9


def test_one_beat_segment_never_upgrades():
    """§13.3 min-beats rule: on a 1-beat segment any single observation is trivially 100%
    stable, so noise must not upgrade the label."""
    chroma = np.zeros((4, 12))
    for pc in (0, 4, 7, 2):
        chroma[:, pc] = 1.0
    seg = _segment("C", "maj", 0.0, 0.5, start_beat=0, end_beat=1)
    lane = ChordsLane(grid="beat", segments=[seg])
    grid = _beat_grid([0.0, 0.5, 1.0, 1.5], [0.0])

    out = refine_extensions(lane, chroma, grid)
    assert out.segments[0].label == "C"
    assert out.segments[0].extensions == []


def test_maj7_with_9th_and_flat7_residual_composes_add9_not_double_seven():
    """Task 9.1 review fix: a maj7-quality segment with a genuine 9th present AND residual
    energy at the flat/dominant-7 pitch class (the kind of harmonic bleed real audio produces)
    must never grow a redundant/contradictory "7" extension on top of its own maj7 seventh —
    this is exactly how the review's live-captured "Gmaj77add9"-style garbled labels were
    produced. Only the genuine 9th becomes an extension."""
    chroma = np.zeros((4, 12))
    for pc in (7, 11, 2, 6):  # G maj7 triad tones: G, B, D, F#
        chroma[:, pc] = 1.0
    chroma[:, 9] = 1.0  # add9 (A) -- genuinely present
    chroma[:, 5] = 1.0  # flat-7 residual (F) -- must NOT become a "7" extension
    seg = _segment("G", "maj7", 0.0, 2.0, start_beat=0, end_beat=4)
    lane = ChordsLane(grid="beat", segments=[seg])
    grid = _beat_grid([0.0, 0.5, 1.0, 1.5], [0.0])

    out = refine_extensions(lane, chroma, grid)
    assert out.segments[0].extensions == ["add9"]
    assert out.segments[0].label == "Gmaj7add9"


def test_maj7_strong_major_seventh_residual_never_appends_maj7_extension():
    """A plain maj7 triad (its own major-7th tone is a chord tone, not a residual) must never
    grow a spurious "maj7" extension on top of the base quality's own suffix — locking the
    "interval already in triad" skip that keeps "Gmaj7" from ever becoming "Gmaj7maj7"."""
    chroma = np.zeros((4, 12))
    for pc in (7, 11, 2, 6):  # G maj7 triad tones: G, B, D, F#
        chroma[:, pc] = 1.0
    seg = _segment("G", "maj7", 0.0, 2.0, start_beat=0, end_beat=4)
    lane = ChordsLane(grid="beat", segments=[seg])
    grid = _beat_grid([0.0, 0.5, 1.0, 1.5], [0.0])

    out = refine_extensions(lane, chroma, grid)
    assert out.segments[0].extensions == []
    assert out.segments[0].label == "Gmaj7"


def test_plain_triad_gains_maj7_extension_from_genuine_residual():
    """The other side of the seventh-guard: a PLAIN major triad with genuine, stable major-7th
    residual energy must still be upgraded — the quality-aware guard only fires when the base
    quality itself already encodes a seventh, never for plain triads (9.1 review item c)."""
    chroma = np.zeros((4, 12))
    for pc in (0, 4, 7):  # C major triad tones
        chroma[:, pc] = 1.0
    chroma[:, 11] = 0.5  # strong, stable B residual (major 7th of C)
    seg = _segment("C", "maj", 0.0, 2.0, start_beat=0, end_beat=4)
    lane = ChordsLane(grid="beat", segments=[seg])
    grid = _beat_grid([0.0, 0.5, 1.0, 1.5], [0.0])

    out = refine_extensions(lane, chroma, grid)
    assert "maj7" in out.segments[0].extensions
    assert out.segments[0].label == "Cmaj7"


# --------------------------------------------------------------------------- #
# §13.2 bass-informed slash detection
# --------------------------------------------------------------------------- #


def test_walking_bass_under_c_major_stays_plain_c():
    """The mandatory regression: a walking bass C-D-E-G under one C-major bar must NOT emit
    C, C/D, C/E, C/G — no single non-root bass pc is stable, so the segment is untouched."""
    lane = _c_major_bar_lane()
    out = detect_slash(lane, [0, 2, 4, 7])  # C, D, E, G bass — one per beat
    assert out.segments[0].label == "C"
    assert out.segments[0].bass is None


def test_stable_chord_tone_bass_emits_slash():
    lane = _c_major_bar_lane()
    out = detect_slash(lane, [4, 4, 4, 4])  # E (a chord tone) stable across the whole segment
    assert out.segments[0].label == "C/E"
    assert out.segments[0].bass == "E"


def test_stable_non_chord_tone_bass_is_annotated_pedal():
    lane = _c_major_bar_lane()
    out = detect_slash(lane, [2, 2, 2, 2])  # D is stable but not a C-major chord tone
    assert out.segments[0].bass is None
    assert "pedal" in out.segments[0].extensions


# --------------------------------------------------------------------------- #
# §13.4 rubato fixed-grid fallback
# --------------------------------------------------------------------------- #


def test_low_grid_confidence_switches_lane_to_fixed_one_second_windows(tmp_path):
    path = _make_progression(tmp_path / "prog.wav", ["C", "F", "G", "C"])
    pcm = decode(str(path))
    recon = run_recon(pcm)
    # Force the rubato branch: grid confidence below BEAT_GRID_CONFIDENCE_MIN.
    recon.grid.confidence = 0.3
    assert recon.grid.confidence < CFG.BEAT_GRID_CONFIDENCE_MIN

    lane = chord_lane_baseline(pcm, recon)
    assert lane.grid == "fixed"
    # Segments carve at whole-second boundaries (1 s windows) and drop the beat indices.
    for seg in lane.segments:
        assert seg.start_beat is None and seg.end_beat is None
        assert seg.start_s == pytest.approx(round(seg.start_s), abs=1e-6)


# --------------------------------------------------------------------------- #
# §13.4 theory layer — hysteresis + Roman numerals
# --------------------------------------------------------------------------- #


def test_modulation_accepted_only_after_two_windows(monkeypatch):
    """§13.4 hysteresis: a 32-bar lane, first 16 bars diatonic C major, last 16 diatonic E
    major. With 4-bar non-overlapping windows the votes are [C,C,C,C,E,E,E,E]; a 2-window
    hysteresis must keep the bar-16 window on C (a single E vote is not enough) and only
    flip at the bar-20 window."""
    monkeypatch.setattr(CFG, "KEY_WINDOW_BARS", 4)
    monkeypatch.setattr(CFG, "KEY_HOP_BARS", 4)
    monkeypatch.setattr(CFG, "KEY_HYSTERESIS_WINDOWS", 2)

    c_major = ["C", "F", "G", "Am"]
    e_major = ["E", "A", "B", "C#m"]
    segs: list[ChordSegment] = []
    for bar in range(32):
        sym = (c_major if bar < 16 else e_major)[bar % 4]
        root, quality = _parse_symbol(sym)
        segs.append(_segment(harmony._PITCH_CLASSES[root], quality, bar * 2.0, bar * 2.0 + 2.0))
    lane = ChordsLane(grid="beat", segments=segs)
    grid = _beat_grid([float(i) for i in range(64)], [float(2 * b) for b in range(32)])

    out = theory_layer(lane, grid)
    keys = [s.key_context for s in out.segments]
    assert keys[15].startswith("C ")  # deep in the C region
    assert keys[16].startswith("C ")  # first E-voting window — hysteresis holds C
    assert keys[19].startswith("C ")  # still the bar-16 window
    assert keys[20].startswith("E ")  # second E window — modulation accepted


def test_roman_f_in_c_major_is_IV():
    segs = [
        _segment("C", "maj", 0.0, 2.0),
        _segment("F", "maj", 2.0, 4.0),
        _segment("G", "maj", 4.0, 6.0),
        _segment("A", "min", 6.0, 8.0),
    ]
    lane = ChordsLane(grid="beat", segments=segs)
    grid = _beat_grid([float(i) for i in range(8)], [0.0, 2.0, 4.0, 6.0])

    out = theory_layer(lane, grid)
    by_root = {s.root: s for s in out.segments}
    assert by_root["C"].key_context == "C major"
    assert by_root["F"].roman == "IV"
    assert by_root["G"].roman == "V"


# --------------------------------------------------------------------------- #
# bass_pc_lane (audio path)
# --------------------------------------------------------------------------- #


def test_bass_pc_lane_tracks_a_low_note(tmp_path):
    """A steady low C (65.4 Hz, C2) under a beat grid reads back as pitch class 0."""
    sr = 22050
    dur = 4.0
    t = np.arange(int(dur * sr)) / sr
    audio = (0.4 * np.sin(2 * np.pi * 65.41 * t)).astype(np.float32)
    path = tmp_path / "bass.wav"
    sf.write(str(path), audio, sr, subtype="PCM_16")
    pcm = decode(str(path))
    grid = _beat_grid([0.5, 1.5, 2.5, 3.5], [0.5])

    lane = bass_pc_lane(pcm, grid)
    assert len(lane) == 4
    voiced = [pc for pc in lane if pc is not None]
    assert voiced, "pYIN found no voiced bass frames"
    assert all(pc == 0 for pc in voiced)  # every voiced beat is pitch class C
