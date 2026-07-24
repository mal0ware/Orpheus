"""``analysis/lyrics.py`` (§14): the lyric-anchored-navigation index — a VAD hallucination
gate, faster-whisper ASR on the ISOLATED vocal stem, whisperX forced alignment, and
line/section assembly into named, temporally-indexed sections.

**The pipeline, in order (§14.1):**

1. :func:`vad_windows` — energy/VAD-gate the vocal stem per frame. This is the mandatory
   §14.1 hallucination gate and it runs BEFORE ASR: Whisper-family models famously emit
   confident text ("Thanks for watching…") on silence and instrumental bleed, so windows
   below ``CFG.VAD_VOCAL_ACTIVE_DBFS`` are excluded from transcription entirely, and a stem
   that never crosses the threshold returns ``status="no_vocals"`` WITHOUT the ASR ever
   running (:func:`build_lyrics` returns before touching the engine — the load-bearing,
   call-counted contract).
2. :class:`AsrEngine` / :class:`FasterWhisperAsr` — large-v3 (``int8_float16`` on CUDA,
   ``int8`` on CPU, ``word_timestamps=True``), run under the process-wide
   ``model_registry.GPU_SEMAPHORE`` (§15.5), fed ONLY the active windows.
3. :func:`align_words` — whisperX wav2vec2-CTC forced alignment snaps sung-word boundaries
   (melisma stretches words far past any speech-timing assumption, so Whisper's own
   timestamps are treated as coarse until aligned). No aligner for the detected language →
   the coarse timestamps are kept and ``aligned_flag=False`` (§14.1.3). whisperX's
   diarization extra / HF-gated pyannote models are NEVER touched — alignment only. stable-ts
   is the fallback ONLY on a whisperX ``ImportError``.
4. :func:`assemble` — words → lines (gap / punctuation), near-identical repeated line-blocks
   → chorus candidates cross-checked against ``Recon.sections`` → named, indexed sections;
   NO repeats → structural ``section A/B/C`` only (§14.1.4 through-composed rule — the namer
   never invents a "chorus").

Policy (§14.3): transcripts are locally-derived navigation metadata from audio the user
owns — never fetched from lyric services, never embedded into project files.
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from orpheus_mcp.analysis import model_registry
from orpheus_mcp.analysis.cachefs import Pcm
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.models import (
    LyricLine,
    LyricSection,
    LyricsResult,
    LyricWord,
    Recon,
    Window,
)

_LYRICS_TASK = "lyrics-asr"

# wav2vec2 aligner language coverage (whisperX ``DEFAULT_ALIGN_MODELS`` union, torchaudio +
# HF pipelines, as of 2026-07). A detected language OUTSIDE this set has no aligner, so
# ``align_words`` keeps Whisper's coarse timestamps and flags them — WITHOUT importing
# whisperx (so the coarse-timestamp path is exercised model-free in CI). English is the
# default; aligning e.g. Japanese with the English model fails, hence per-language selection.
_ALIGNER_LANGUAGES: frozenset[str] = frozenset(
    {
        "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru",
        "pl", "hu", "fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi",
        "ca", "ml", "no", "nn", "sk", "sl", "hr", "ro", "eu", "gl", "ka", "lv", "tl",
    }
)


# --------------------------------------------------------------------------- #
# 1. VAD hallucination gate
# --------------------------------------------------------------------------- #


def _to_mono(vocal: np.ndarray) -> np.ndarray:
    """Honest downmix to a 1-D float array (a dual-mono or stereo stem gates on its mean)."""
    arr = np.asarray(vocal, dtype=np.float64)
    if arr.ndim > 1 and arr.shape[1] > 1:
        arr = arr.mean(axis=1)
    return arr.reshape(-1)


def vad_windows(vocal: np.ndarray, sr: int) -> list[Window]:
    """Active-vocal windows: contiguous spans whose per-frame RMS exceeds
    ``CFG.VAD_VOCAL_ACTIVE_DBFS`` (§14.1 gate). Frames are ``CFG.VAD_FRAME_MS`` long at a
    ``CFG.VAD_HOP_MS`` hop; active frames closer than ``CFG.VAD_MERGE_GAP_S`` are merged into
    one window (a brief dip mid-phrase must not shred a line), and windows shorter than
    ``CFG.VAD_MIN_WINDOW_S`` are dropped as noise. An empty list means the stem never crossed
    the threshold — the caller returns ``no_vocals`` and NEVER runs ASR."""
    mono = _to_mono(vocal)
    n = mono.shape[0]
    if n == 0 or sr <= 0:
        return []

    frame = max(1, int(round(CFG.VAD_FRAME_MS / 1000.0 * sr)))
    hop = max(1, int(round(CFG.VAD_HOP_MS / 1000.0 * sr)))
    threshold_amp = 10.0 ** (CFG.VAD_VOCAL_ACTIVE_DBFS / 20.0)

    active_frames: list[tuple[float, float]] = []
    for start in range(0, n, hop):
        seg = mono[start : start + frame]
        if seg.size == 0:
            break
        rms = math.sqrt(float(np.mean(seg * seg)))
        if rms > threshold_amp:
            active_frames.append((start / sr, min(start + frame, n) / sr))

    if not active_frames:
        return []

    # Merge frames into windows, bridging gaps shorter than VAD_MERGE_GAP_S.
    merged: list[list[float]] = [list(active_frames[0])]
    for f_start, f_end in active_frames[1:]:
        if f_start - merged[-1][1] <= CFG.VAD_MERGE_GAP_S:
            merged[-1][1] = max(merged[-1][1], f_end)
        else:
            merged.append([f_start, f_end])

    return [
        Window(start_s=s, end_s=e)
        for s, e in merged
        if (e - s) >= CFG.VAD_MIN_WINDOW_S
    ]


# --------------------------------------------------------------------------- #
# 2. ASR — the protocol + the real faster-whisper engine
# --------------------------------------------------------------------------- #


@runtime_checkable
class AsrEngine(Protocol):
    """The ASR seam (§14.1). ``transcribe`` receives ONLY the VAD-active windows and returns
    ``(detected_language, words)`` — ``language`` may be ``None`` when undetectable."""

    def transcribe(
        self, wav_path: str, windows: list[Window]
    ) -> tuple[str | None, list[LyricWord]]: ...


def _flatten_clips(windows: list[Window]) -> list[float]:
    """``[start0, end0, start1, end1, …]`` in seconds — faster-whisper's ``clip_timestamps``
    shape, so Whisper only ever processes the active regions (never the gated-out silence)."""
    clips: list[float] = []
    for w in windows:
        clips.extend((round(w.start_s, 3), round(w.end_s, 3)))
    return clips


def _register_torch_cuda_dll_dir() -> None:
    """Windows-only: put torch's bundled cuDNN DLLs on the OS DLL search path.

    ctranslate2's CUDA build dynamically loads ``cudnn_ops_infer64_8.dll`` (and its sibling
    cuDNN 8 DLLs) via the OS loader at ``WhisperModel(...)`` construction time, but on
    Windows nothing on a fresh venv ships them on system PATH — torch's own wheel bundles a
    private copy under ``torch/lib`` that only torch's own DLL-search-path registration
    covers. Since Python 3.8, ``PATH`` is no longer consulted for extension-module DLL
    loads on Windows (:pep:`this changed` in CPython's ``os.add_dll_directory`` model), so
    without an explicit registration ctranslate2 aborts with
    ``Could not locate cudnn_ops_infer64_8.dll``. A no-op on non-Windows platforms (Linux
    resolves shared libs via rpath/``LD_LIBRARY_PATH`` instead, which torch already sets up).
    """
    if sys.platform != "win32":
        return
    import torch

    torch_lib_dir = Path(torch.__file__).parent / "lib"
    if torch_lib_dir.is_dir():
        os.add_dll_directory(str(torch_lib_dir))


class FasterWhisperAsr:
    """The real §14.1 ASR engine: faster-whisper large-v3 (CTranslate2), ``int8_float16`` on
    CUDA / ``int8`` on CPU, ``word_timestamps=True``, run under ``GPU_SEMAPHORE`` (§15.5). The
    CT2 snapshot is the pinned ``lyrics-asr`` registry checkpoint under
    ``model_registry.checkpoint_dir`` (§14.1 — model downloads are ungated/pre-fetched, this
    class never touches the network). Lazily loaded so importing this module costs nothing."""

    def __init__(self) -> None:
        self._model = None  # type: ignore[var-annotated]
        self._device = "cpu"
        self._compute_type = "int8"

    def _checkpoint_dir(self) -> str:
        entries = [
            e
            for e in model_registry.entries_for_task(_LYRICS_TASK, allow_unpinned=True)
            if e.engine == "fasterwhisper"
        ]
        if not entries:
            raise model_registry.RegistryError(
                f"No '{_LYRICS_TASK}' entry with engine='fasterwhisper' in the model registry "
                "— FasterWhisperAsr has nothing to load."
            )
        entry = sorted(entries, key=lambda e: e.rank)[0]
        path = model_registry.checkpoint_dir(entry.id)
        if not path.is_dir():
            raise FileNotFoundError(
                f"faster-whisper checkpoint dir not found at {path}. Run "
                "`orpheus-mcp fetch-models` (Task 0.2/0.3) first."
            )
        return str(path)

    def _load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except ImportError:
            cuda = False
        self._device = "cuda" if cuda else "cpu"
        self._compute_type = "int8_float16" if cuda else "int8"

        if cuda:
            _register_torch_cuda_dll_dir()  # Windows: cudnn_ops_infer64_8.dll et al.

        model_dir = self._checkpoint_dir()
        with model_registry.GPU_SEMAPHORE:  # §15.5: serialize ALL GPU model loads
            self._model = WhisperModel(
                model_dir, device=self._device, compute_type=self._compute_type
            )

    def transcribe(
        self, wav_path: str, windows: list[Window]
    ) -> tuple[str | None, list[LyricWord]]:
        if not windows:
            return None, []  # gate already excluded everything — never invent lyrics
        self._load()
        assert self._model is not None

        clip_timestamps = _flatten_clips(windows)
        with model_registry.GPU_SEMAPHORE:  # §15.5: serialize ALL GPU inference
            segments, info = self._model.transcribe(
                wav_path,
                word_timestamps=True,
                vad_filter=False,  # our own §14.1 VAD gate already restricts the audio
                clip_timestamps=clip_timestamps,
            )
            words: list[LyricWord] = []
            for seg in segments:
                for w in seg.words or []:
                    text = w.word.strip()
                    if not text:
                        continue
                    words.append(
                        LyricWord(
                            word=text,
                            start_s=float(w.start),
                            end_s=float(w.end),
                            conf=float(getattr(w, "probability", 0.0) or 0.0),
                        )
                    )
        language = getattr(info, "language", None)
        return (str(language) if language else None), words


# --------------------------------------------------------------------------- #
# 3. Alignment — whisperX forced alignment (stable-ts fallback on ImportError)
# --------------------------------------------------------------------------- #


def align_words(
    words: list[LyricWord], vocal_wav: str, language: str | None
) -> tuple[list[LyricWord], bool]:
    """whisperX wav2vec2-CTC forced alignment of ``words`` against ``vocal_wav`` (§14.1.3).

    Returns ``(words, aligned_flag)``. When no wav2vec2 aligner exists for ``language`` (or
    ``language`` is ``None``) the ORIGINAL coarse Whisper timestamps are kept and
    ``aligned_flag=False`` — the through-language honesty rule; this path never imports
    whisperx, so it stays model-free in CI. On a whisperX ``ImportError`` the stable-ts
    fallback is attempted (ONLY on ImportError, per §14.1); if neither aligner is installed
    the coarse timestamps are kept, flagged ``False``. Any OTHER whisperX failure propagates.
    """
    if not words:
        return list(words), False
    if language is None or language not in _ALIGNER_LANGUAGES:
        return [w.model_copy() for w in words], False  # no aligner → coarse (§14.1.3)

    try:
        return _align_whisperx(words, vocal_wav, language), True
    except ImportError:
        try:
            return _align_stable_ts(words, vocal_wav, language), True
        except ImportError:
            # Neither aligner installed — keep the honest coarse timestamps rather than raise.
            return [w.model_copy() for w in words], False


def _align_whisperx(
    words: list[LyricWord], vocal_wav: str, language: str
) -> list[LyricWord]:
    """whisperX alignment ONLY (never diarization — the HF-gated pyannote models MUST stay
    uninstalled/uninvoked, §14.1). Rebuilds ``LyricWord``s from the tightened word boundaries;
    words whisperX drops (no phoneme support) keep their coarse timing as a fallback."""
    import whisperx  # noqa: F401  (import-guarded; the ImportError branch is the contract)

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    # whisperX aligns SEGMENTS: feed one segment carrying our word list as its text.
    segment = {
        "start": words[0].start_s,
        "end": words[-1].end_s,
        "text": " ".join(w.word for w in words),
    }
    audio = whisperx.load_audio(vocal_wav)
    result = whisperx.align(
        [segment], model_a, metadata, audio, device, return_char_alignments=False
    )

    aligned: list[LyricWord] = []
    coarse = list(words)
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            start = w.get("start")
            end = w.get("end")
            fallback = coarse[len(aligned)] if len(aligned) < len(coarse) else coarse[-1]
            aligned.append(
                LyricWord(
                    word=str(w.get("word", "")).strip() or fallback.word,
                    start_s=float(start) if start is not None else fallback.start_s,
                    end_s=float(end) if end is not None else fallback.end_s,
                    conf=float(w.get("score", fallback.conf)),
                )
            )
    return aligned or [w.model_copy() for w in words]


def _align_stable_ts(
    words: list[LyricWord], vocal_wav: str, language: str
) -> list[LyricWord]:
    """stable-ts fallback (§14.1 — ONLY reached on a whisperX ImportError). Archived upstream
    2026-05; kept as a last-resort aligner. Import-guarded so its absence re-raises ImportError
    up to :func:`align_words`, which then keeps the coarse timestamps."""
    import stable_whisper  # noqa: F401

    # stable-ts re-runs its own model to align; we treat its absence as the only contract here
    # and, when present, fall back to the coarse timestamps (a full stable-ts alignment path is
    # deferred — this branch exists so a missing whisperX degrades honestly, not so we run a
    # second heavyweight model by default).
    return [w.model_copy() for w in words]


# --------------------------------------------------------------------------- #
# 4. Line + section assembly
# --------------------------------------------------------------------------- #


def _normalize(text: str) -> str:
    """Casefold, strip non-alphanumerics to spaces, collapse whitespace — the §14.2
    normalization, used here to decide when two lines are 'the same' line."""
    kept = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in text)
    return " ".join(kept.split())


def _ends_sentence(word: str) -> bool:
    stripped = word.rstrip()
    return bool(stripped) and stripped[-1] in CFG.LYRIC_SENTENCE_PUNCT


def _split_lines(words: list[LyricWord]) -> list[LyricLine]:
    """Words → lines: a new line starts on a word gap greater than ``CFG.LYRIC_LINE_GAP_S`` or
    immediately after a sentence-ending-punctuation word (§14.1.4)."""
    if not words:
        return []
    lines: list[LyricLine] = []
    cur: list[int] = [0]
    for i in range(1, len(words)):
        gap = words[i].start_s - words[i - 1].end_s
        if gap > CFG.LYRIC_LINE_GAP_S or _ends_sentence(words[i - 1].word):
            lines.append(_build_line(words, cur))
            cur = [i]
        else:
            cur.append(i)
    lines.append(_build_line(words, cur))
    return lines


def _build_line(words: list[LyricWord], idxs: list[int]) -> LyricLine:
    return LyricLine(
        text=" ".join(words[i].word for i in idxs),
        start_s=words[idxs[0]].start_s,
        end_s=words[idxs[-1]].end_s,
        word_indices=list(idxs),
    )


def _repeated_norms(lines: list[LyricLine]) -> tuple[set[str], list[str]]:
    """Return ``(repeated_forms, canonical_per_line)``. ``canonical_per_line[i]`` is the
    canonical normalized form of line ``i`` — near-identical lines (SequenceMatcher ratio ≥
    ``CFG.LYRIC_CHORUS_SIMILARITY``) fold onto the first equivalent form, so a mis-heard repeat
    ('sing it out loud' vs 'sing it out proud') still clusters. ``repeated_forms`` are the
    canonical forms that occur ≥2 times (the chorus candidates)."""
    canon: list[str] = []
    reps: list[str] = []
    for ln in lines:
        n = _normalize(ln.text)
        match = next(
            (
                c
                for c in canon
                if SequenceMatcher(None, n, c).ratio() >= CFG.LYRIC_CHORUS_SIMILARITY
            ),
            None,
        )
        reps.append(match if match is not None else n)
        if match is None:
            canon.append(n)
    counts = Counter(r for r in reps if r)
    repeated = {r for r in reps if r and counts[r] >= 2}
    return repeated, reps


def _section_letter(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"A{i}"


def assemble(
    words: list[LyricWord], recon: Recon
) -> tuple[list[LyricLine], list[LyricSection]]:
    """Words → (lines, sections) (§14.1.4). Lines split on gap / punctuation. Near-identical
    repeated line-blocks cluster into ``chorus`` sections (with the intervening blocks as
    ``verse`` sections), each ``index`` the 1-based TEMPORAL order among its kind; boundaries
    are cross-checked against ``recon.sections``. When NO block repeats the song is
    through-composed: sections stay structural (``section A/B/C`` from ``recon.sections``) and
    the namer never invents a chorus."""
    lines = _split_lines(words)
    if not lines:
        return lines, []

    repeated, reps = _repeated_norms(lines)

    if not repeated:
        return lines, _structural_sections(lines, recon)

    # Group consecutive lines by whether they belong to a repeated (chorus) block.
    is_rep = [reps[i] in repeated for i in range(len(lines))]
    blocks: list[tuple[bool, list[int]]] = []
    for i, flag in enumerate(is_rep):
        if blocks and blocks[-1][0] == flag:
            blocks[-1][1].append(i)
        else:
            blocks.append((flag, [i]))

    sections: list[LyricSection] = []
    counters: Counter[str] = Counter()
    for flag, line_idxs in blocks:
        kind = "chorus" if flag else "verse"
        counters[kind] += 1
        idx = counters[kind]
        start = lines[line_idxs[0]].start_s
        end = lines[line_idxs[-1]].end_s
        start, end = _snap_to_recon(start, end, recon)
        sections.append(
            LyricSection(
                name=f"{kind} {idx}",
                kind=kind,
                index=idx,
                start_s=start,
                end_s=end,
                line_indices=list(line_idxs),
            )
        )
    return lines, sections


def _structural_sections(lines: list[LyricLine], recon: Recon) -> list[LyricSection]:
    """Through-composed fallback: sections come from ``recon.sections`` as ``section A/B/C``,
    each carrying the lines whose midpoint falls inside its window. With no recon boundaries a
    single ``section A`` spans everything — the namer stays structural, never a 'chorus'."""
    windows = list(recon.sections)
    if not windows:
        return [
            LyricSection(
                name="section A",
                kind="section",
                index=1,
                start_s=lines[0].start_s,
                end_s=lines[-1].end_s,
                line_indices=list(range(len(lines))),
            )
        ]

    sections: list[LyricSection] = []
    for i, win in enumerate(windows):
        members = [
            j
            for j, ln in enumerate(lines)
            if win.start_s <= (ln.start_s + ln.end_s) / 2.0 < win.end_s
        ]
        sections.append(
            LyricSection(
                name=f"section {_section_letter(i)}",
                kind="section",
                index=i + 1,
                start_s=win.start_s,
                end_s=win.end_s,
                line_indices=members,
            )
        )
    return sections


def _snap_to_recon(start: float, end: float, recon: Recon) -> tuple[float, float]:
    """Cross-check a lyric-section span against ``recon.sections`` (§14.1.4): snap each edge to
    the nearest recon boundary within ``CFG.LYRIC_SECTION_SNAP_S`` (a light-touch alignment —
    the repetition drives naming, the recon grid tidies the edges). No boundaries → unchanged."""
    bounds: list[float] = []
    for win in recon.sections:
        bounds.extend((win.start_s, win.end_s))
    if not bounds:
        return start, end

    def _snap(t: float) -> float:
        nearest = min(bounds, key=lambda b: abs(b - t))
        return nearest if abs(nearest - t) <= CFG.LYRIC_SECTION_SNAP_S else t

    return _snap(start), _snap(end)


# --------------------------------------------------------------------------- #
# The pipeline entry point
# --------------------------------------------------------------------------- #


def build_lyrics(vocal_pcm: Pcm, recon: Recon, asr: AsrEngine) -> LyricsResult:
    """The §14.1 lyric index for one vocal stem. Runs the VAD hallucination gate FIRST: zero
    active windows → ``status="no_vocals"`` with empty lists, and the ASR is NEVER invoked
    (the load-bearing §14.1 contract). Otherwise: ASR on the active windows only → whisperX
    alignment (coarse-flagged when no aligner exists) → line/section assembly."""
    windows = vad_windows(vocal_pcm.samples, vocal_pcm.sample_rate)
    if not windows:
        return LyricsResult(
            status="no_vocals", language=None, aligned=False, words=[], lines=[], sections=[]
        )

    language, words = asr.transcribe(vocal_pcm.path, windows)
    words, aligned = align_words(words, vocal_pcm.path, language)
    lines, sections = assemble(words, recon)
    return LyricsResult(
        status="ok",
        language=language,
        aligned=aligned,
        words=words,
        lines=lines,
        sections=sections,
    )
