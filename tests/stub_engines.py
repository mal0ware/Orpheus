"""Deterministic stub engines for reference-analysis tests — no real model weights, no
network I/O, no GPU. ``StubTagger`` (Task 2.1, T2) is the first resident of this module;
Task 4.1 extends it with ``StubEngine`` (a ``SeparatorEngine`` double) and friends as
later phases need engine doubles that satisfy the same Protocol contracts as their real
counterparts (``analysis/timeline.py::TaggerEngine``, ``analysis/separation.py::
SeparatorEngine``, ...).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from orpheus_mcp.analysis.cachefs import Pcm
from orpheus_mcp.models import GateResult, GateSet, StemResult

if TYPE_CHECKING:
    from orpheus_mcp.analysis.model_registry import ModelEntry

# Drawn from the REAL registry's tagger_classes union (model_registry.mapped_tagger_classes())
# so build_timeline's registry-driven subsetting keeps them — a stub class name absent from
# every registry entry's tagger_classes would silently vanish from the built Timeline, which
# is not what these tests want to exercise.
STUB_CLASSES = [
    "Singing",
    "Speech",
    "Drum kit",
    "Snare drum",
    "Bass guitar",
    "Guitar",
    "Electric guitar",
    "Piano",
]


class StubTagger:
    """A deterministic, no-dependency ``TaggerEngine`` double.

    Two modes: pass ``activations=<ndarray>`` for an exact, fixed ``(frames, classes)``
    matrix (precise control over individual frame values — the locked numeric test
    cases); omit it for an all-zero matrix shaped from ``ceil(pcm.duration_s / hop_s)``
    frames (frame-alignment / round-trip tests against a real decoded ``Pcm``).
    """

    def __init__(
        self,
        classes: Sequence[str] | None = None,
        *,
        hop_s: float = 0.01,
        activations: np.ndarray | Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.classes: list[str] = list(classes) if classes is not None else list(STUB_CLASSES)
        self.hop_s = hop_s
        self._fixed: np.ndarray | None = (
            None if activations is None else np.asarray(activations, dtype=np.float64)
        )

    def activations(self, pcm: Pcm) -> np.ndarray:
        if self._fixed is not None:
            return self._fixed
        n_frames = max(1, math.ceil(pcm.duration_s / self.hop_s))
        return np.zeros((n_frames, len(self.classes)), dtype=np.float64)


class StubEngine:
    """A deterministic, no-dependency ``SeparatorEngine`` double (Task 4.1).

    It performs *identity* separation: every requested stem is a verbatim copy of the audio
    it was handed (optionally with ``extra_pad_samples`` zeros appended to the tail — the
    "separator returned a slightly-too-long stem" case ``separate_region`` must length-align
    away). Because the stem equals the engine's input, a caller can assert sample-accurately
    that ``separate_region`` cropped to ``region ± pad`` and trimmed back to exactly ``region``
    (the returned samples must equal the original PCM over the requested window).

    Every ``separate`` call is recorded in ``self.calls`` (duration/sample-count/sr/channels of
    the wav actually received) so the pad-math assertions can read back what the engine saw.
    ``barrier_hook`` is an optional callable invoked mid-``separate`` (used by the semaphore
    test to detect overlap). ``requires_stereo`` mirrors the real stereo-only engines so the
    §7 mono→dual-mono path can be exercised.
    """

    def __init__(
        self,
        targets: Sequence[str] | None = None,
        *,
        extra_pad_samples: int = 0,
        requires_stereo: bool = False,
        barrier_hook: Callable[[], None] | None = None,
    ) -> None:
        self.targets: list[str] | None = list(targets) if targets is not None else None
        self.extra_pad_samples = extra_pad_samples
        self.requires_stereo = requires_stereo
        self._barrier_hook = barrier_hook
        self.calls: list[dict[str, float | int | str]] = []

    def separate(self, wav_path: str, entry: ModelEntry, out_dir: Path) -> dict[str, str]:
        data, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
        self.calls.append(
            {
                "wav_path": str(wav_path),
                "sr": int(sr),
                "n_samples": int(data.shape[0]),
                "channels": int(data.shape[1]),
                "duration_s": data.shape[0] / sr,
            }
        )
        if self._barrier_hook is not None:
            self._barrier_hook()

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        targets = self.targets if self.targets is not None else list(entry.targets)
        result: dict[str, str] = {}
        for tgt in targets:
            stem = data.copy()
            if self.extra_pad_samples:
                pad = np.zeros((self.extra_pad_samples, stem.shape[1]), dtype=stem.dtype)
                stem = np.concatenate([stem, pad], axis=0)
            dest = out / f"{tgt}.flac"
            sf.write(str(dest), stem, sr, subtype="PCM_24")  # 24-bit: sample-accurate enough
            result[tgt] = str(dest)
        return result

    @property
    def last_call(self) -> dict[str, float | int | str]:
        return self.calls[-1]


# --------------------------------------------------------------------------- #
# Ladder (Task 5.2) test seams: an engine factory that remembers its engines (so
# separate() calls can be counted across rungs) and a scripted GateSet source (so a
# test drives FAILING/PASSING gates per rung deterministically without a real model).
# --------------------------------------------------------------------------- #


class EngineHub:
    """A ``climb`` engine factory (``Callable[[ModelEntry], SeparatorEngine]``) that hands
    out one identity ``StubEngine`` per registry entry id and remembers them, so a ladder
    test can read back the total number of ``separate()`` calls (``total_calls``) — the
    "no re-run of R1/R2 on resume" and "completed ladder never re-climbs" assertions."""

    def __init__(self, *, requires_stereo: bool = False) -> None:
        self.engines: dict[str, StubEngine] = {}
        self._requires_stereo = requires_stereo

    def __call__(self, entry: ModelEntry) -> StubEngine:
        eng = self.engines.get(entry.id)
        if eng is None:
            eng = StubEngine(list(entry.targets), requires_stereo=self._requires_stereo)
            self.engines[entry.id] = eng
        return eng

    @property
    def total_calls(self) -> int:
        return sum(len(e.calls) for e in self.engines.values())

    def calls_for(self, model_id: str) -> int:
        eng = self.engines.get(model_id)
        return len(eng.calls) if eng is not None else 0


def gateset(
    overall: str = "pass", *, fail_gate: str = "bleed", value: float = 0.42
) -> GateSet:
    """Build a :class:`GateSet` with a chosen ``overall`` verdict for scripted ladder tests.

    ``overall="fail"`` marks ``fail_gate`` failing with ``value`` (so the ladder's transition
    log can read ``<rung>: <gate>=<value>``); ``"unverifiable"`` makes every gate N/A;
    ``"pass"`` passes all applicable gates (agreement stays N/A, matching the single-take case).
    """
    fields = ["mix_consistency", "bleed", "silence_honesty", "agreement"]
    verdicts: dict[str, str] = {
        "mix_consistency": "pass",
        "bleed": "pass",
        "silence_honesty": "pass",
        "agreement": "na",
    }
    if overall == "fail":
        verdicts[fail_gate] = "fail"
    elif overall == "unverifiable":
        verdicts = {f: "na" for f in fields}

    def _res(field: str) -> GateResult:
        v = verdicts[field]
        if v == "fail":
            return GateResult(verdict="fail", value=value)
        if v == "pass":
            return GateResult(verdict="pass", value=0.0)
        return GateResult(verdict="na", value=None)

    return GateSet(
        mix_consistency=_res("mix_consistency"),
        bleed=_res("bleed"),
        silence_honesty=_res("silence_honesty"),
        agreement=_res("agreement"),
        overall=overall,  # type: ignore[arg-type]
    )


class ScriptedGates:
    """A deterministic ``verify`` seam for ladder tests: returns pre-scripted GateSets in
    order, one per call, and records every call ``(stem targets, target, other_stem_path)``
    so a test can assert what the ladder verified and in what sequence. A string item is
    expanded via :func:`gateset`; a :class:`GateSet` item is returned verbatim. When the
    queue empties it keeps returning an all-pass GateSet."""

    def __init__(self, script: Sequence[str | GateSet]) -> None:
        self._queue: list[str | GateSet] = list(script)
        self.calls: list[tuple[frozenset[str], str, str | None]] = []

    def __call__(
        self,
        stems: dict[str, StemResult],
        target: str,
        other_stem_path: str | None = None,
    ) -> GateSet:
        self.calls.append((frozenset(stems), target, other_stem_path))
        item = self._queue.pop(0) if self._queue else "pass"
        return item if isinstance(item, GateSet) else gateset(item)
