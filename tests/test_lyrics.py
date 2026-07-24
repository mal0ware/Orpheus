"""``analysis/lyrics.py`` (Task 8.1, §14): the VAD hallucination gate, the ASR contract,
whisperX alignment, and line/section assembly.

Every CI test here is model-free: a :class:`StubAsr` scripts ``(language, words)`` and
counts its calls, and vocal audio is synthesized in-numpy so the VAD threshold math is
exercised deterministically against ``CFG.VAD_VOCAL_ACTIVE_DBFS`` (−45 dBFS). The real
faster-whisper end-to-end smoke is ``@pytest.mark.desktop @pytest.mark.gpu`` (bottom).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from orpheus_mcp.analysis.cachefs import Pcm, decode
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.lyrics import (
    align_words,
    assemble,
    build_lyrics,
    vad_windows,
)
from orpheus_mcp.models import BeatGrid, KeyEstimate, LyricWord, Recon, Window
from stub_engines import StubAsr

SR = 22050


# --------------------------------------------------------------------------- #
# Signal + fixture builders
# --------------------------------------------------------------------------- #


def _tone(dbfs: float, seconds: float, *, sr: int = SR) -> np.ndarray:
    """A 220 Hz sine whose full-scale RMS lands at exactly ``dbfs`` (sine RMS = A/√2)."""
    amp = float(10.0 ** (dbfs / 20.0) * math.sqrt(2.0))
    t = np.arange(int(round(seconds * sr))) / sr
    return (amp * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)


def _silence(seconds: float, *, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(seconds * sr)), dtype=np.float32)


def _pcm(mono: np.ndarray, *, sr: int = SR, path: str = "") -> Pcm:
    sig = np.ascontiguousarray(mono.reshape(-1, 1), dtype=np.float32)
    return Pcm(
        samples=sig,
        sample_rate=sr,
        path=path,
        pcm_hash="test",
        duration_s=sig.shape[0] / sr,
        decoder="test",
    )


def _word(text: str, start: float, end: float, conf: float = 0.9) -> LyricWord:
    return LyricWord(word=text, start_s=start, end_s=end, conf=conf)


def _recon(sections: list[Window], duration: float = 20.0) -> Recon:
    return Recon(
        duration_s=duration,
        sample_rate=SR,
        lufs_integrated=-14.0,
        tempo_bpm=120.0,
        grid=BeatGrid(
            times_s=[0.0],
            downbeats_s=[0.0],
            meter=(4, 4),
            confidence=0.9,
            meter_confidence=0.5,
            kind="beat",
        ),
        tuning_offset_cents=0.0,
        key=KeyEstimate(root="C", mode="major", confidence=0.9),
        key_alternates=[],
        sections=sections,
        band_energy_db={},
    )


# --------------------------------------------------------------------------- #
# The hallucination gate (the load-bearing assertion): silent stem → no_vocals AND
# ASR is NEVER called (call-counted).
# --------------------------------------------------------------------------- #


def test_silent_stem_returns_no_vocals_and_asr_never_runs():
    pcm = _pcm(_silence(3.0))
    asr = StubAsr(words=[_word("ghost", 0.5, 1.0)])

    result = build_lyrics(pcm, _recon([]), asr)

    assert result.status == "no_vocals"
    assert result.language is None
    assert result.words == []
    assert result.lines == []
    assert result.sections == []
    assert asr.calls == 0  # the load-bearing assertion: ASR never ran on silence


# --------------------------------------------------------------------------- #
# VAD threshold math — both sides of CFG.VAD_VOCAL_ACTIVE_DBFS.
# --------------------------------------------------------------------------- #


def test_vad_threshold_both_sides_of_config():
    thr = CFG.VAD_VOCAL_ACTIVE_DBFS  # −45 dBFS
    loud = _tone(thr + 5.0, 1.0)  # −40 dBFS → active
    quiet = _tone(thr - 5.0, 1.0)  # −50 dBFS → inactive
    mono = np.concatenate([loud, quiet])

    windows = vad_windows(mono.reshape(-1, 1), SR)

    assert len(windows) == 1
    assert windows[0].start_s < 0.15
    assert 0.85 <= windows[0].end_s <= 1.15  # covers the loud second, stops before the quiet


def test_vad_all_below_threshold_yields_no_windows():
    quiet = _tone(CFG.VAD_VOCAL_ACTIVE_DBFS - 8.0, 2.0)
    assert vad_windows(quiet.reshape(-1, 1), SR) == []


def test_vad_windows_merges_small_gap_but_splits_large_gap():
    """Two active bursts separated by a gap < ``CFG.VAD_MERGE_GAP_S`` (0.20 s) bridge into ONE
    window (a brief dip mid-phrase must not shred a line); separated by 0.5 s they stay TWO."""
    burst = _tone(CFG.VAD_VOCAL_ACTIVE_DBFS + 8.0, 0.3)

    small_gap = _silence(CFG.VAD_MERGE_GAP_S / 2.0)  # well under the merge threshold
    merged = np.concatenate([burst, small_gap, burst])
    assert len(vad_windows(merged.reshape(-1, 1), SR)) == 1

    large_gap = _silence(0.5)  # well over the merge threshold
    split = np.concatenate([burst, large_gap, burst])
    assert len(vad_windows(split.reshape(-1, 1), SR)) == 2


def test_vad_windows_drops_burst_shorter_than_min_window(monkeypatch):
    """An active burst shorter than ``CFG.VAD_MIN_WINDOW_S`` is dropped as noise — proven by
    raising the threshold above a burst that survives at the default (sanity-checked first)."""
    burst = _tone(CFG.VAD_VOCAL_ACTIVE_DBFS + 8.0, 0.3)
    mono = np.concatenate([_silence(0.5), burst, _silence(0.5)])

    assert len(vad_windows(mono.reshape(-1, 1), SR)) == 1  # survives at the default min-window

    monkeypatch.setattr(CFG, "VAD_MIN_WINDOW_S", 1.0)  # now longer than the 0.3 s burst
    assert vad_windows(mono.reshape(-1, 1), SR) == []


# --------------------------------------------------------------------------- #
# ASR receives ONLY the VAD-active windows.
# --------------------------------------------------------------------------- #


def test_asr_receives_only_active_windows():
    # active only in the middle second of a 3 s stem
    mono = np.concatenate(
        [_silence(1.0), _tone(CFG.VAD_VOCAL_ACTIVE_DBFS + 6.0, 1.0), _silence(1.0)]
    )
    pcm = _pcm(mono)
    # language "zz" keeps this test model-free — no wav2vec2 aligner import is triggered (the
    # real whisperX/stable-ts alignment path is desktop-only); window gating is the subject.
    asr = StubAsr(language="zz", words=[_word("hi", 1.2, 1.4)])

    expected = vad_windows(mono.reshape(-1, 1), SR)
    build_lyrics(pcm, _recon([]), asr)

    assert asr.calls == 1
    assert asr.received_windows is not None
    assert len(asr.received_windows) == len(expected) == 1
    # the one window sits in the middle, NOT [0, 3] (the whole file)
    w = asr.received_windows[0]
    assert 0.9 < w.start_s < 1.15
    assert 1.9 < w.end_s < 2.15


# --------------------------------------------------------------------------- #
# Line splitting — silence gaps and sentence punctuation.
# --------------------------------------------------------------------------- #


def test_line_splitting_on_gap_and_punctuation():
    words = [
        _word("hold", 0.0, 0.3),
        _word("the", 0.3, 0.5),
        _word("line.", 0.5, 0.8),  # sentence punctuation → line ends here
        _word("and", 0.9, 1.1),  # small gap, same line
        _word("carry", 1.1, 1.4),
        # gap > LYRIC_LINE_GAP_S (1.0 s) → new line
        _word("through", 3.0, 3.4),
        _word("again", 3.4, 3.8),
    ]
    lines, _sections = assemble(words, _recon([]))

    assert len(lines) == 3
    assert lines[0].text == "hold the line."
    assert lines[1].text == "and carry"
    assert lines[2].text == "through again"
    assert lines[0].start_s == pytest.approx(0.0)
    assert lines[0].end_s == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Chorus clustering — repeated line blocks → chorus 1/2, TEMPORAL index.
# --------------------------------------------------------------------------- #


def _line_words(text: str, start: float) -> list[LyricWord]:
    """Lay a line's tokens end-to-end starting at ``start``, the last token punctuated so
    the line closes cleanly (0.4 s/word, no intra-line gap)."""
    toks = text.split()
    out: list[LyricWord] = []
    t = start
    for i, tok in enumerate(toks):
        w = tok + ("." if i == len(toks) - 1 else "")
        out.append(_word(w, t, t + 0.35))
        t += 0.4
    return out


def test_repeated_block_clusters_into_indexed_choruses():
    chorus = "sing it out loud"
    verse1 = "morning over the hill"
    verse2 = "walking the open ground"
    words: list[LyricWord] = []
    words += _line_words(verse1, 0.0)
    words += _line_words(chorus, 3.0)  # chorus, 1st time
    words += _line_words(verse2, 6.0)
    words += _line_words(chorus, 9.0)  # chorus, 2nd time

    _lines, sections = assemble(words, _recon([Window(start_s=0, end_s=12)], duration=12))

    choruses = [s for s in sections if s.kind == "chorus"]
    assert len(choruses) == 2
    assert [s.index for s in choruses] == [1, 2]  # temporal order
    assert [s.name for s in choruses] == ["chorus 1", "chorus 2"]
    # the 2nd chorus comes strictly after the 1st in time
    assert choruses[0].start_s < choruses[1].start_s
    # a repeated block must never be named a structural "section"
    assert all(s.kind != "section" for s in sections)


# --------------------------------------------------------------------------- #
# Through-composed — NO repeats → structural sections only (section A/B/C), no "chorus".
# --------------------------------------------------------------------------- #


def test_through_composed_yields_structural_sections_only():
    words: list[LyricWord] = []
    words += _line_words("first unique idea here", 0.0)
    words += _line_words("second different thought now", 3.0)
    words += _line_words("third distinct notion again", 6.0)
    recon = _recon(
        [
            Window(start_s=0.0, end_s=3.0),
            Window(start_s=3.0, end_s=6.0),
            Window(start_s=6.0, end_s=9.0),
        ],
        duration=9.0,
    )

    _lines, sections = assemble(words, recon)

    assert len(sections) == 3
    assert all(s.kind == "section" for s in sections)
    assert [s.name for s in sections] == ["section A", "section B", "section C"]
    assert not any("chorus" in s.name for s in sections)  # namer never invents a chorus


# --------------------------------------------------------------------------- #
# Language with no wav2vec2 aligner → original (coarse) timestamps, aligned flag False.
# --------------------------------------------------------------------------- #


def test_no_aligner_language_keeps_coarse_timestamps():
    words = [_word("kore", 0.0, 0.4), _word("wa", 0.4, 0.7)]
    aligned_words, flag = align_words(words, "does-not-matter.wav", "zz")

    assert flag is False  # no aligner for language "zz"
    assert [w.start_s for w in aligned_words] == [0.0, 0.4]  # untouched
    assert [w.end_s for w in aligned_words] == [0.4, 0.7]


def test_align_words_none_language_keeps_coarse_timestamps():
    """The ``language is None`` branch (undetectable language) — same coarse-timestamp,
    aligned_flag=False contract as an unsupported language, exercised directly."""
    words = [_word("kore", 0.0, 0.4), _word("wa.", 0.4, 0.7)]
    aligned_words, flag = align_words(words, "does-not-matter.wav", None)

    assert flag is False  # no language detected -> no aligner to select
    assert [w.word for w in aligned_words] == ["kore", "wa."]
    assert [w.start_s for w in aligned_words] == [0.0, 0.4]  # untouched
    assert [w.end_s for w in aligned_words] == [0.4, 0.7]


def test_build_lyrics_flags_coarse_for_unaligned_language():
    mono = _tone(CFG.VAD_VOCAL_ACTIVE_DBFS + 6.0, 1.0)
    pcm = _pcm(mono)
    asr = StubAsr(language="zz", words=[_word("kore", 0.1, 0.4), _word("wa.", 0.4, 0.7)])

    result = build_lyrics(pcm, _recon([Window(start_s=0, end_s=1)], duration=1), asr)

    assert result.status == "ok"
    assert result.language == "zz"
    assert result.aligned is False


# --------------------------------------------------------------------------- #
# _snap_to_recon — edges snap to a DIFFERING recon boundary within CFG.LYRIC_SECTION_SNAP_S.
# --------------------------------------------------------------------------- #


def test_snap_to_recon_reads_lyric_section_snap_s(monkeypatch):
    """``_snap_to_recon`` must read ``CFG.LYRIC_SECTION_SNAP_S`` (its own dedicated tolerance,
    split off from the conflated ``CFG.LYRIC_LINE_GAP_S``), not just happen to work because the
    two constants share a value. Raising SNAP_S to 5.0 while leaving LINE_GAP_S at its 1.0
    default, then testing a 2.0 s raw-to-edge distance, only snaps if the call site reads
    SNAP_S — this fails both if _snap_to_recon becomes a no-op and if it silently reverts to
    reading LINE_GAP_S."""
    from orpheus_mcp.analysis.lyrics import _snap_to_recon

    assert CFG.LYRIC_LINE_GAP_S < 2.0  # sanity: the old conflated constant would NOT cover this
    monkeypatch.setattr(CFG, "LYRIC_SECTION_SNAP_S", 5.0)

    recon = _recon([Window(start_s=12.0, end_s=20.0)])
    start, end = _snap_to_recon(10.0, 18.0, recon)  # each 2.0 s from a differing recon edge

    assert start == pytest.approx(12.0)  # snapped to the recon edge, not left at raw 10.0
    assert end == pytest.approx(20.0)  # snapped to the recon edge, not left at raw 18.0


# --------------------------------------------------------------------------- #
# Real faster-whisper end-to-end smoke (desktop + GPU) — the §14.1 vocal-stem ASR path.
# --------------------------------------------------------------------------- #

_CI_VOCAL_DIR = Path(__file__).resolve().parent / "fixtures" / "audio" / "ci_vocal"


def _norm_tokens(text: str) -> list[str]:
    keep = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    return keep.split()


def _overlap(truth_line: str, transcript: str) -> float:
    truth = _norm_tokens(truth_line)
    trans = set(_norm_tokens(transcript))
    if not truth:
        return 0.0
    return sum(1 for t in truth if t in trans) / len(truth)


@pytest.mark.desktop
@pytest.mark.gpu
def test_real_faster_whisper_recovers_truth_lines():
    """Run the REAL FasterWhisperAsr on the ci_vocal vocal STEM (SAPI TTS reading 6 known
    lines from truth.json). Assert status ok, English, and ≥4 of 6 truth lines fuzzy-
    recoverable (normalized token overlap ≥ 0.6) — TTS through separation is imperfect, so
    the per-line scores are reported honestly, not asserted line-for-line."""
    import json

    pytest.importorskip("faster_whisper")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA for the real faster-whisper path")

    stem = _CI_VOCAL_DIR / "stems" / "vocals.flac"
    if not stem.is_file():
        pytest.skip("ci_vocal vocal stem missing")
    truth = json.loads((_CI_VOCAL_DIR / "truth.json").read_text(encoding="utf-8"))
    truth_lines = truth["lyrics"]["lines"]

    from orpheus_mcp.analysis.lyrics import FasterWhisperAsr

    pcm = decode(str(stem))
    result = build_lyrics(pcm, _recon([Window(start_s=0, end_s=pcm.duration_s)],
                                      duration=pcm.duration_s), FasterWhisperAsr())

    assert result.status == "ok"
    assert result.language == "en"
    transcript = " ".join(w.word for w in result.words)
    scores = [(_overlap(line, transcript), line) for line in truth_lines]
    recovered = sum(1 for s, _ in scores if s >= 0.6)
    detail = "\n".join(f"  {s:.2f}  {line}" for s, line in scores)
    assert recovered >= 4, f"only {recovered}/6 lines recovered:\n{detail}\nASR: {transcript}"
