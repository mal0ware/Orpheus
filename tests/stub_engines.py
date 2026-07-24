"""Deterministic stub engines for reference-analysis tests — no real model weights, no
network I/O, no GPU. ``StubTagger`` (Task 2.1, T2) is the first resident of this module;
Task 4.1 extends it with ``StubEngine`` (a ``SeparatorEngine`` double) and friends as
later phases need engine doubles that satisfy the same Protocol contracts as their real
counterparts (``analysis/timeline.py::TaggerEngine``, ``analysis/separation.py::
SeparatorEngine``, ...).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from orpheus_mcp.analysis.cachefs import Pcm

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
