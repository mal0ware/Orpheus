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

import logging
import math
import os
import re
import subprocess
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from orpheus_mcp.analysis import model_registry
from orpheus_mcp.analysis.cachefs import Pcm
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.models import (
    LocateMatch,
    LocateResult,
    LyricLine,
    LyricSection,
    LyricsResult,
    LyricWord,
    Recon,
    Window,
)

_LYRICS_TASK = "lyrics-asr"

logger = logging.getLogger(__name__)


class LyricsAsrEnvMissing(Exception):
    """The faster-whisper ASR stack isn't installed. Raised in place of a raw ``ImportError``
    escaping a tool call (§9 ``capped_by_missing_extra`` pattern) — mirrors
    ``analysis.separation.SeparatorEnvMissing`` for the ``[separation]`` extra's ASR half
    (faster-whisper ships in the SAME extra as audio-separator, per ``pyproject.toml``). A
    future orchestrator (§9 ``analyze_reference``) catches this exact type and records
    ``capped_by_missing_extra="separation"`` in the answer envelope; this task's tools just
    need to raise it honestly instead of a bare ``ImportError``."""

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


# --- ASR device choice ------------------------------------------------------ #
#
# torch can see the GPU while ctranslate2 still cannot run on it: on the owner's desktop
# ctranslate2 4.4 (whisperx 3.3.1 pins it below 4.5) links cuDNN 8, torch bundles cuDNN 9,
# WhisperModel loads fine, and the first transcribe aborts the WHOLE process natively
# ("Could not locate cudnn_ops_infer64_8.dll", exit 127). Nothing in-process can catch that, so
# "auto" proves CUDA in a throwaway child before the server process ever touches it.

ASR_DEVICE_ENV = "ORPHEUS_ASR_DEVICE"
_ASR_DEVICE_CHOICES = ("auto", "cpu", "cuda")
_CUDA_PROBE_TIMEOUT_S = 300.0
_CUDA_PROBE_RESULT: bool | None = None  # per process: probed at most once
_CUDA_PROBE_LOCK = threading.Lock()

# A real (tiny) CUDA transcription: loading alone succeeds even without cuDNN ops; only
# inference trips the abort. Runs with ORPHEUS_ASR_DEVICE=cuda so it never probes again.
_CUDA_PROBE_CODE = """import os, tempfile
import numpy as np, soundfile as sf
from orpheus_mcp.analysis.lyrics import FasterWhisperAsr
from orpheus_mcp.models import Window
t = np.arange(16000) / 16000.0
wav = os.path.join(tempfile.mkdtemp(), "probe.wav")
sf.write(wav, (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype("float32"), 16000)
asr = FasterWhisperAsr()
asr.transcribe(wav, [Window(start_s=0.0, end_s=1.0)])
print("OK", asr._device)
"""


def _probe_cuda_asr_in_child() -> bool:
    """True only if a child process completed a CUDA transcription. A native abort, a timeout or
    a CPU fallback in the child all count as "CUDA is not usable here"."""
    env = {**os.environ, ASR_DEVICE_ENV: "cuda", "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CUDA_PROBE_CODE],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_CUDA_PROBE_TIMEOUT_S,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0 and "OK cuda" in (proc.stdout or "")


def resolve_asr_device() -> str:
    """``"cpu"`` or ``"cuda"`` for lyric ASR.

    ``ORPHEUS_ASR_DEVICE``: ``cpu`` forces CPU; ``cuda`` forces CUDA without the safety probe
    (and is what the probe child runs with); ``auto`` (default) uses CUDA only if torch sees a
    GPU AND a child process proved a CUDA transcription works, probed once per process.
    """
    raw = os.environ.get(ASR_DEVICE_ENV, "auto")
    choice = raw.strip().lower() or "auto"
    if choice not in _ASR_DEVICE_CHOICES:
        raise ValueError(
            f"{ASR_DEVICE_ENV}={raw!r} is not one of {', '.join(_ASR_DEVICE_CHOICES)}."
        )
    if choice != "auto":
        return choice

    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except ImportError:
        cuda = False
    if not cuda:
        return "cpu"

    global _CUDA_PROBE_RESULT
    with _CUDA_PROBE_LOCK:
        if _CUDA_PROBE_RESULT is None:
            with model_registry.GPU_SEMAPHORE:  # the child uses the GPU: nothing else may
                _CUDA_PROBE_RESULT = _probe_cuda_asr_in_child()
            if not _CUDA_PROBE_RESULT:
                logger.warning(
                    "Lyric ASR: CUDA is visible but a test transcription failed in a child "
                    "process (usually a cuDNN mismatch), so ASR runs on CPU. Set %s=cpu to "
                    "skip this probe.",
                    ASR_DEVICE_ENV,
                )
        return "cuda" if _CUDA_PROBE_RESULT else "cpu"


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
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise LyricsAsrEnvMissing(
                "faster-whisper is not installed — install the `[separation]` extra "
                "(`pip install 'orpheus-mcp[separation]'`) to run lyric transcription."
            ) from exc

        self._device = resolve_asr_device()
        cuda = self._device == "cuda"
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


# --------------------------------------------------------------------------- #
# 5. locate_lyric -- the section 14.2 fuzzy resolver (pure: no audio, no cache, no I/O)
# --------------------------------------------------------------------------- #

_TOKEN_STRIP_RE = re.compile(r"^[^\w*]+|[^\w*]+$", re.UNICODE)


def _norm_token(word: str) -> str:
    """Casefold + strip surrounding punctuation, KEEPING internal ``*`` (Whisper's
    profanity-masking character -- ``_masked_prefix_match`` needs it) and internal
    apostrophes/letters. Empty for a token that was pure punctuation."""
    return _TOKEN_STRIP_RE.sub("", word.casefold())


# A deliberately small (~40 LOC), pure-python, ENGLISH-phonetics-only reduction -- NOT the full
# Lawrence Philips Double Metaphone algorithm (that runs to hundreds of lines). Good enough to
# equate the common ASR-mishear / misremembered-lyric class the section 14.2 scorer targets
# ("son" vs "sun", "their" vs "there") by collapsing to a consonant skeleton with vowels dropped
# (except a single initial-vowel marker) -- classical Metaphone's core trick. CORRECTED (fix
# wave 1): a non-English token does NOT reliably "simply never phonetic-match" on its own --
# accented Latin tokens can still collide under this reduction (e.g. "café"/"résumé" both drop
# their accents via ``isalpha``-filtering and can land on the same consonant skeleton), so the
# non-English degradation is ENFORCED by the caller (``locate_lyric`` gates the phonetic leg on
# ``lyrics.language`` being English, threading a ``phonetic`` flag down through
# ``_tokens_match``/``_token_edit_distance``/``_slide_candidates``) rather than assumed to fall
# out for free -- and the caller states the degradation in ``LocateResult.notes``.
_DIGRAPHS: list[tuple[str, str]] = [
    ("SCH", "SK"), ("PH", "F"), ("GH", "K"), ("CK", "K"), ("SH", "X"), ("CH", "X"),
    ("TH", "0"), ("WH", "W"), ("QU", "K"), ("KN", "N"), ("GN", "N"), ("WR", "R"), ("PS", "S"),
]
_CONSONANT_MAP: dict[str, str] = {"C": "K", "Z": "S", "V": "F", "Q": "K"}
_VOWELS = frozenset("AEIOU")


def _phonetic_code(token: str) -> str:
    s = "".join(c for c in token.upper() if c.isalpha())
    if not s:
        return ""
    for a, b in _DIGRAPHS:
        s = s.replace(a, b)
    out: list[str] = []
    prev = ""
    for i, ch in enumerate(s):
        if ch in _VOWELS:
            if i == 0:
                out.append("A")  # single initial-vowel marker, then dropped
            prev = ""
            continue
        code = _CONSONANT_MAP.get(ch, ch)
        if code != prev:
            out.append(code)
        prev = code
    return "".join(out)


def _masked_prefix_match(a: str, b: str) -> bool:
    """Section 14.2 masked-profanity rule: if either normalized token is asterisk-masked
    (Whisper's ``f***``), it matches any token sharing its visible prefix -- Whisper's
    censorship must not tank the edit distance against the actual word on either side."""
    if "*" not in a and "*" not in b:
        return False
    masked, other = (a, b) if "*" in a else (b, a)
    prefix = masked.split("*", 1)[0]
    return bool(prefix) and other.startswith(prefix)


def _tokens_match(raw_a: str, raw_b: str, *, phonetic: bool = True) -> bool:
    """Per-token section 14.2 match: exact (normalized) OR masked-profanity-prefix OR, when
    ``phonetic`` is True, double-metaphone-equal (English phonetics ONLY). ``phonetic=False``
    (fix wave 1: set by ``locate_lyric`` whenever ``lyrics.language`` isn't English) disables
    the phonetic leg entirely -- the reduction is English-only and does NOT reliably degrade to
    inequality on its own for non-English tokens (accented Latin tokens can still collide under
    it, e.g. "café"/"résumé"), so the caller must gate it rather than rely on a natural miss."""
    a, b = _norm_token(raw_a), _norm_token(raw_b)
    if not a or not b:
        return a == b
    if a == b:
        return True
    if _masked_prefix_match(a, b):
        return True
    if not phonetic:
        return False
    ca, cb = _phonetic_code(a), _phonetic_code(b)
    return bool(ca) and ca == cb


def _token_edit_distance(a: list[str], b: list[str], *, phonetic: bool = True) -> int:
    """Levenshtein distance over TOKEN sequences (insert/delete/substitute, unit cost),
    substitution cost 0 when ``_tokens_match`` -- the section 14.2 window score's numerator.
    ``phonetic`` threads straight through to ``_tokens_match`` (fix wave 1: the English-only
    gate)."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            sub_cost = 0 if _tokens_match(a[i - 1], b[j - 1], phonetic=phonetic) else 1
            cur[j] = min(
                prev[j] + 1,  # delete from a
                cur[j - 1] + 1,  # insert into a
                prev[j - 1] + sub_cost,  # substitute (or match)
            )
        prev = cur
    return prev[m]


@dataclass
class _Candidate:
    start: int  # word index, inclusive
    end: int  # word index, exclusive
    score: float


def _slide_candidates(
    words: list[LyricWord], quote_tokens: list[str], *, phonetic: bool = True
) -> list[_Candidate]:
    """Slide every window length in ``[len(quote) - SLOP, len(quote) + SLOP]`` (min 1) over the
    FLAT word stream (never per-line -- a quote may span a line boundary, section 14.2) and
    score each via ``1 - token_edit_distance / max(len(quote), len(window))``. ``phonetic``
    threads through to ``_token_edit_distance`` (fix wave 1: ``locate_lyric`` disables it for a
    non-English transcript)."""
    qlen = len(quote_tokens)
    if qlen == 0 or not words:
        return []
    slop = CFG.LYRIC_MATCH_WINDOW_SLOP
    lengths = [n for n in range(max(1, qlen - slop), qlen + slop + 1) if n <= len(words)]

    candidates: list[_Candidate] = []
    word_texts = [w.word for w in words]
    for length in lengths:
        for start in range(0, len(words) - length + 1):
            window = word_texts[start : start + length]
            dist = _token_edit_distance(quote_tokens, window, phonetic=phonetic)
            score = 1.0 - dist / max(qlen, length)
            candidates.append(_Candidate(start, start + length, score))
    return candidates


def _merge_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    """Overlapping candidate windows collapse to ONE representative, keeping the best score --
    classic non-max suppression, NOT a transitive interval merge. A transitive merge (grow an
    envelope through every pairwise overlap) is wrong here: because window length ranges over
    +/- ``CFG.LYRIC_MATCH_WINDOW_SLOP`` at every start position, adjacent sliding windows
    ALWAYS overlap by construction, so a transitive merge would chain-collapse the entire
    candidate set -- across the WHOLE transcript -- into one cluster, silently discarding every
    occurrence but one (the load-bearing bug this function exists to avoid: a repeated chorus
    line must return ALL its occurrences, section 14.2). NMS instead repeatedly takes the
    best-scoring remaining candidate, keeps it, and discards every OTHER candidate whose
    word-index span overlaps it at all (two windows describing the same sung moment always
    share word indices; two genuinely distinct occurrences of a repeated line never do)."""
    remaining = sorted(candidates, key=lambda c: -c.score)
    kept: list[_Candidate] = []
    for c in remaining:
        if any(c.start < k.end and c.end > k.start for k in kept):
            continue
        kept.append(c)
    return kept


def _section_for(sections: list[LyricSection], t: float) -> str | None:
    for sec in sections:
        if sec.start_s <= t < sec.end_s:
            return sec.name
    return None


def _to_match(words: list[LyricWord], sections: list[LyricSection], c: _Candidate) -> LocateMatch:
    span = words[c.start : c.end]
    return LocateMatch(
        start_s=span[0].start_s,
        end_s=span[-1].end_s,
        matched_text=" ".join(w.word for w in span),
        score=c.score,
        section=_section_for(sections, span[0].start_s),
        occurrence=0,  # filled in below once the returned set is fixed (temporal order)
    )


def _phonetic_enabled(language: str | None) -> bool:
    """Fix wave 1 (Critical review finding): the ~40-LOC ``_phonetic_code`` reduction is
    ENGLISH-only and accented/non-English tokens can still collide under it (e.g.
    "café"/"résumé" both drop their diacritics and can land on the same consonant skeleton) --
    it does NOT reliably degrade to inequality on its own. So the phonetic leg of
    ``_tokens_match`` is explicitly gated here on the transcript's detected language being
    English ("en" or an "en-*" regional tag); ``language is None`` (undetectable) counts as
    non-English -- the honest default when the ASR couldn't even tell us."""
    if language is None:
        return False
    folded = language.lower()
    return folded == "en" or folded.startswith("en-")


def locate_lyric(
    lyrics: LyricsResult, quote: str, occurrence: int | None = None
) -> LocateResult:
    """The section 14.2 fuzzy resolver -- PURE (no file I/O, no cache; the tool layer,
    ``tools.reference.locate_lyric``, owns fetching/caching the :class:`LyricsResult`).

    ``lyrics.status == "no_vocals"`` returns the DEFINED no-vocal :class:`LocateResult`
    (``status="no_vocals"``, empty matches) -- never an ordinary empty-match "ok" result, so a
    caller can tell "nothing sung here" apart from "nothing matched" (section 14.1).

    Otherwise: normalize both sides, slide the quote over the flat word stream (line
    boundaries are irrelevant -- a quote may span one), merge overlapping windows keeping the
    best, then either return every match scoring >= ``CFG.LYRIC_MATCH_MIN_SCORE`` (temporal
    ``occurrence`` order, list itself ranked best-score-first) or, if the best candidate falls
    short of that bar, the top ``CFG.LYRIC_MATCH_TOP_N`` candidates with ``low_confidence=True``
    (section 14.2: Claude asks the user instead of guessing). ``occurrence`` is always the
    1-based temporal rank WITHIN the returned set -- the one definition that covers both the
    normal and the low-confidence path. ``occurrence`` (the parameter) filters the final set
    down to a single requested occurrence when given.

    Fix wave 1 (Critical): the phonetic-equivalence leg of token matching is gated on
    ``lyrics.language`` being English (:func:`_phonetic_enabled`) -- a non-English transcript
    degrades to plain edit distance ONLY, and that degradation is stated in
    ``LocateResult.notes``, never silently applied or silently assumed."""
    if lyrics.status == "no_vocals":
        return LocateResult(status="no_vocals", matches=[], low_confidence=False)

    phonetic = _phonetic_enabled(lyrics.language)
    notes: list[str] = (
        []
        if phonetic
        else ["non-English transcript: phonetic matching disabled (edit-distance only)"]
    )

    quote_tokens = [t for t in (_norm_token(tok) for tok in quote.split()) if t]
    candidates = _merge_candidates(
        _slide_candidates(lyrics.words, quote_tokens, phonetic=phonetic)
    )
    if not candidates:
        return LocateResult(status="ok", matches=[], low_confidence=False, notes=notes)

    best_score = max(c.score for c in candidates)
    low_confidence = best_score < CFG.LYRIC_MATCH_MIN_SCORE
    if low_confidence:
        chosen = sorted(candidates, key=lambda c: (-c.score, c.start))[: CFG.LYRIC_MATCH_TOP_N]
    else:
        chosen = [c for c in candidates if c.score >= CFG.LYRIC_MATCH_MIN_SCORE]

    chosen_temporal = sorted(chosen, key=lambda c: c.start)
    matches = [_to_match(lyrics.words, lyrics.sections, c) for c in chosen_temporal]
    for idx, m in enumerate(matches, start=1):
        m.occurrence = idx
    matches.sort(key=lambda m: -m.score)  # ranked best-first; occurrence stays temporal

    if occurrence is not None:
        matches = [m for m in matches if m.occurrence == occurrence]

    return LocateResult(status="ok", matches=matches, low_confidence=low_confidence, notes=notes)
