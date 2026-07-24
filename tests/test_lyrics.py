"""``analysis/lyrics.py`` (Task 8.1, §14): the VAD hallucination gate, the ASR contract,
whisperX alignment, and line/section assembly.

Every CI test here is model-free: a :class:`StubAsr` scripts ``(language, words)`` and
counts its calls, and vocal audio is synthesized in-numpy so the VAD threshold math is
exercised deterministically against ``CFG.VAD_VOCAL_ACTIVE_DBFS`` (−45 dBFS). The real
faster-whisper end-to-end smoke is ``@pytest.mark.desktop @pytest.mark.gpu`` (bottom).
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from orpheus_mcp.analysis.cachefs import Pcm, decode
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.lyrics import (
    align_words,
    assemble,
    build_lyrics,
    locate_lyric,
    vad_windows,
)
from orpheus_mcp.models import (
    BeatGrid,
    KeyEstimate,
    LyricsResult,
    LyricWord,
    Recon,
    Window,
)
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
# locate_lyric (Task 8.2, §14.2) — the pure fuzzy resolver. Every case here is a plain
# LyricsResult built by hand: no audio, no cache, no engines.
# --------------------------------------------------------------------------- #


def _transcript(*lines_at: tuple[str, float]) -> LyricsResult:
    """Build a bare-bones ``LyricsResult`` (``status="ok"``, no sections) from
    ``(text, start_s)`` pairs, each line laid out via ``_line_words`` (0.4 s/word)."""
    words: list[LyricWord] = []
    for text, start in lines_at:
        words += _line_words(text, start)
    return LyricsResult(
        status="ok", language="en", aligned=True, words=words, lines=[], sections=[]
    )


def test_locate_lyric_exact_quote_scores_1_0_with_right_timestamps():
    lyrics = _transcript(("hold the line and carry through", 0.0))

    result = locate_lyric(lyrics, "the line and carry")

    assert result.status == "ok"
    assert len(result.matches) == 1
    m = result.matches[0]
    assert m.score == pytest.approx(1.0)
    assert m.matched_text == "the line and carry"
    assert m.start_s == pytest.approx(0.4)  # "the" is the 2nd word (0.4 s/word)
    assert m.end_s == pytest.approx(1.95)  # "carry" is the 5th word, ends at 1.6 + 0.35


def test_locate_lyric_misquote_still_top_match_via_metaphone():
    """'the son also rises' (misheard/misremembered) vs the ACTUAL transcript 'the sun also
    rises' — 'son'/'sun' collapse to the same phonetic code, so the mismatch costs nothing
    and the match still scores (and ranks) as the top result."""
    lyrics = _transcript(("the sun also rises over the hill", 0.0))

    result = locate_lyric(lyrics, "the son also rises")

    assert result.low_confidence is False
    assert result.matches
    best = result.matches[0]
    assert best.score == pytest.approx(1.0)
    assert best.matched_text == "the sun also rises"


def test_locate_lyric_repeated_chorus_all_occurrences_temporal_order_independent_of_score():
    """The 2nd sung instance is a PERFECT match (score 1.0); the 1st is slightly garbled
    ('proud' for 'loud', still >= LYRIC_MATCH_MIN_SCORE). Sorted by score, occurrence 2 would
    lead — but each match's ``occurrence`` field must still reflect SUNG (temporal) order."""
    lyrics = _transcript(
        ("verse one opens quiet", 0.0),
        ("sing it out proud", 3.0),  # occurrence 1 (garbled: proud, not loud)
        ("verse two carries on", 6.0),
        ("sing it out loud", 9.0),  # occurrence 2 (exact)
    )

    result = locate_lyric(lyrics, "sing it out loud")

    assert result.low_confidence is False
    assert len(result.matches) == 2
    by_occurrence = {m.occurrence: m for m in result.matches}
    assert set(by_occurrence) == {1, 2}
    assert by_occurrence[1].start_s < by_occurrence[2].start_s  # occurrence 1 sung FIRST
    assert by_occurrence[2].score == pytest.approx(1.0)
    assert by_occurrence[1].score < by_occurrence[2].score  # occurrence 1 IS the lower score
    # score-ranked list order: the higher-scoring (2nd sung) match leads the returned list
    assert result.matches[0].occurrence == 2


def test_locate_lyric_quote_spans_line_boundary():
    """The window slides over the flat WORD stream, not line-by-line — a quote whose words
    straddle two ``LyricLine``s (irrelevant here since this LyricsResult carries none) must
    still match."""
    lyrics = _transcript(("hold the line and carry the", 0.0), ("weight home tonight", 3.0))

    result = locate_lyric(lyrics, "carry the weight home")

    assert result.low_confidence is False
    assert len(result.matches) == 1
    m = result.matches[0]
    assert m.score == pytest.approx(1.0)
    assert m.matched_text == "carry the. weight home"  # raw transcript text (punctuation kept)
    assert m.start_s < 3.0 < m.end_s  # genuinely straddles the line boundary


def test_locate_lyric_masked_profanity_prefix_match():
    """Whisper's asterisk-masked profanity ('f***') must match the user's actual word via the
    shared visible prefix, not tank the edit distance."""
    lyrics = _transcript(("this f*** rocks", 0.0))

    result = locate_lyric(lyrics, "this fudge rocks")

    assert result.low_confidence is False
    best = result.matches[0]
    assert best.score == pytest.approx(1.0)
    assert best.matched_text == "this f*** rocks."  # last token in the line carries the period


def test_locate_lyric_low_confidence_returns_top_3_candidates_flagged():
    lyrics = _transcript(
        ("morning light comes over the hill", 0.0),
        ("walking on the open ground", 3.0),
        ("hold the line and carry through", 6.0),
        ("simple words will see us new", 9.0),
    )

    result = locate_lyric(lyrics, "completely unrelated phrase nobody sang")

    assert result.status == "ok"
    assert result.low_confidence is True
    assert len(result.matches) == CFG.LYRIC_MATCH_TOP_N == 3
    assert all(m.score < CFG.LYRIC_MATCH_MIN_SCORE for m in result.matches)


def test_locate_lyric_on_no_vocals_file_is_defined_result_not_empty_list():
    """A ``no_vocals`` ``LyricsResult`` must produce the DEFINED no-vocal ``LocateResult``
    (``status="no_vocals"``) — never look like an ordinary "found nothing" ``"ok"`` result."""
    no_vocals = LyricsResult(
        status="no_vocals", language=None, aligned=False, words=[], lines=[], sections=[]
    )

    result = locate_lyric(no_vocals, "any quote at all")

    assert result.status == "no_vocals"
    assert result.matches == []
    assert result.low_confidence is False
    # distinguishable from a "found nothing" ok-result by STATUS, not by matches being empty
    assert result.status != "ok"


def test_locate_lyric_occurrence_filter_selects_one_match():
    lyrics = _transcript(
        ("sing it out loud", 0.0),
        ("something else entirely", 3.0),
        ("sing it out loud", 6.0),
    )

    only_second = locate_lyric(lyrics, "sing it out loud", occurrence=2)

    assert len(only_second.matches) == 1
    assert only_second.matches[0].occurrence == 2
    assert only_second.matches[0].start_s == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# Windows cuDNN DLL search-path registration (infra fix wave 1). CI-safe: sys.platform and
# os.add_dll_directory are both mocked, so this never depends on a real Windows DLL load.
# --------------------------------------------------------------------------- #


def test_register_torch_cuda_dll_dir_calls_add_dll_directory_on_win32(monkeypatch):
    torch = pytest.importorskip("torch")
    from orpheus_mcp.analysis import lyrics

    calls: list[str] = []
    monkeypatch.setattr(lyrics.sys, "platform", "win32")
    monkeypatch.setattr(lyrics.os, "add_dll_directory", calls.append, raising=False)

    lyrics._register_torch_cuda_dll_dir()

    expected = str(Path(torch.__file__).parent / "lib")
    assert calls == [expected]


def test_register_torch_cuda_dll_dir_noop_off_windows(monkeypatch):
    pytest.importorskip("torch")
    from orpheus_mcp.analysis import lyrics

    calls: list[str] = []
    monkeypatch.setattr(lyrics.sys, "platform", "linux")
    monkeypatch.setattr(lyrics.os, "add_dll_directory", calls.append, raising=False)

    lyrics._register_torch_cuda_dll_dir()

    assert calls == []  # non-Windows: never touches the DLL search path


def test_faster_whisper_asr_load_registers_dll_dir_before_model_on_cuda(monkeypatch, tmp_path):
    """The actual §infra-fix-wave-1 wiring: ``FasterWhisperAsr._load`` must call the DLL-dir
    registration BEFORE constructing ``WhisperModel`` whenever torch reports CUDA available
    (``device == "cuda"``) — not just the helper in isolation."""
    torch = pytest.importorskip("torch")
    import orpheus_mcp.analysis.lyrics as lyrics_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    calls: list[str] = []
    monkeypatch.setattr(
        lyrics_mod, "_register_torch_cuda_dll_dir", lambda: calls.append("called")
    )

    class _FakeModel:
        def __init__(self, model_dir, device, compute_type):
            self.model_dir = model_dir
            self.device = device
            self.compute_type = compute_type

    fake_module = types.SimpleNamespace(WhisperModel=_FakeModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    asr = lyrics_mod.FasterWhisperAsr()
    monkeypatch.setattr(asr, "_checkpoint_dir", lambda: str(tmp_path))

    asr._load()

    assert calls == ["called"]  # registered
    assert asr._device == "cuda"
    assert isinstance(asr._model, _FakeModel)


def test_faster_whisper_asr_load_skips_dll_registration_on_cpu(monkeypatch, tmp_path):
    """No CUDA -> no DLL registration needed (device == "cpu"): the fix must not fire
    unconditionally on every load, only on the CUDA path that actually needs the DLLs."""
    torch = pytest.importorskip("torch")
    import orpheus_mcp.analysis.lyrics as lyrics_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    calls: list[str] = []
    monkeypatch.setattr(
        lyrics_mod, "_register_torch_cuda_dll_dir", lambda: calls.append("called")
    )

    class _FakeModel:
        def __init__(self, model_dir, device, compute_type):
            pass

    fake_module = types.SimpleNamespace(WhisperModel=_FakeModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    asr = lyrics_mod.FasterWhisperAsr()
    monkeypatch.setattr(asr, "_checkpoint_dir", lambda: str(tmp_path))

    asr._load()

    assert calls == []
    assert asr._device == "cpu"


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
