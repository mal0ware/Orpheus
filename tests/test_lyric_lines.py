"""Lyric line splitting on SUNG transcripts (bug found on the first real-music run, 2026-07-25).

Whisper punctuates sung lyrics sparsely: the owner's 200 s reference song came back with one
full stop and 26 commas in 327 words, and only 3 pauses over 1.0 s. Splitting only on > 1.0 s
pauses and sentence punctuation produced 5 "lines" (one of 183 words), so repeated chorus lines
could never match and the namer called almost everything "verse 1". Lines now also break at
commas and breaths, and over-long lines split at their widest pause.

All text here is made up; only the structure mirrors the real song.
"""

from __future__ import annotations

from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.lyrics import assemble
from orpheus_mcp.models import BeatGrid, KeyEstimate, LyricWord, Recon, Window


def _w(word: str, start: float, dur: float = 0.25) -> LyricWord:
    return LyricWord(word=word, start_s=start, end_s=start + dur, conf=0.9)


def _phrase(text: str, start: float, *, step: float = 0.3, tail: str = ",") -> list[LyricWord]:
    """Words 0.05 s apart (sung legato), the last one ending in ``tail``."""
    toks = text.split()
    last = len(toks) - 1
    return [_w(t + (tail if i == last else ""), start + i * step) for i, t in enumerate(toks)]


def _recon(duration: float) -> Recon:
    return Recon(
        duration_s=duration,
        sample_rate=44100,
        lufs_integrated=-14.0,
        tempo_bpm=120.0,
        grid=BeatGrid(times_s=[0.0], downbeats_s=[0.0], meter=(4, 4), confidence=0.9,
                      meter_confidence=0.5, kind="beat"),
        tuning_offset_cents=0.0,
        key=KeyEstimate(root="C", mode="major", confidence=0.9),
        key_alternates=[],
        sections=[Window(start_s=0.0, end_s=duration)],
        band_energy_db={},
    )


def _texts(words: list[LyricWord]) -> list[str]:
    lines, _ = assemble(words, _recon(200.0))
    return [ln.text for ln in lines]


def test_comma_ends_a_line():
    words = _phrase("paper lanterns over water", 0.0)
    words += _phrase("candles on the stair", 1.4, tail="")
    assert _texts(words) == ["paper lanterns over water,", "candles on the stair"]


def test_breath_pause_ends_a_line_but_a_short_gap_does_not():
    a = _phrase("counting all the cars", 0.0, tail="")
    b = _phrase("drifting down the road", a[-1].end_s + CFG.LYRIC_LINE_BREATH_S + 0.05, tail="")
    c = _phrase("under neon light", b[-1].end_s + CFG.LYRIC_LINE_BREATH_S - 0.2, tail="")
    assert _texts(a + b + c) == ["counting all the cars", "drifting down the road under neon light"]


def test_soft_breaks_never_make_lines_shorter_than_the_minimum():
    words = _phrase("oh", 0.0) + _phrase("we keep on running home", 0.4, tail="")
    assert _texts(words) == ["oh, we keep on running home"]


def test_overlong_line_splits_at_its_widest_pause():
    words = [_w(f"w{i}", i * 0.3) for i in range(30)]
    for later in words[17:]:  # one slightly wider (0.15 s) pause after word 17
        later.start_s += 0.1
        later.end_s += 0.1
    lines = _texts(words)
    assert all(len(t.split()) <= CFG.LYRIC_LINE_MAX_WORDS for t in lines)
    assert sum(len(t.split()) for t in lines) == 30
    assert any(t.split()[-1] == "w16" for t in lines)  # the cut sits on the widest pause


def test_sparsely_punctuated_song_gets_verses_and_choruses():
    """Verse / chorus / verse / chorus, sung with commas and short breaths, one long pause."""
    chorus = ["shine a little longer", "hold the evening still", "shine a little longer"]
    verse1 = ["morning on the platform", "coffee going cold", "every train is leaving"]
    verse2 = ["letters in the drawer", "pictures turning gold", "every door is closing"]
    t, words = 0.0, []
    for block in (verse1, chorus, verse2, chorus):
        for line in block:
            ph = _phrase(line, t)
            words += ph
            t = ph[-1].end_s + 0.3  # a short breath: under the old 1.0 s rule, no break
        t += 0.2

    _, sections = assemble(words, _recon(t))

    assert [s.name for s in sections] == ["verse 1", "chorus 1", "verse 2", "chorus 2"]
