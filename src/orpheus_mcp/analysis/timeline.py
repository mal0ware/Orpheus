"""``analysis/timeline.py`` (T2, §3.1b/c, §9): the framewise PANNs SED instrument-activity
timeline, plus the two pure decision helpers that read it — ``most_exposed_window`` (§3.1b:
"is the target sound exposed somewhere, skip separation entirely") and ``target_absent``
(§3.1c: "the target never appears, stop at T2 cost"). Both consume the SAME normalized
ratio (§5.2's bleed-gate formula) so the exposure check and the separation-verification
bleed gate can never silently disagree about what "clean" means.

No threshold is baked into ``build_timeline`` or the ``Timeline`` model it returns — every
constant (`CFG.TIMELINE_ACTIVITY_THRESHOLD`, `CFG.EXPOSURE_THRESHOLD`,
`CFG.EXPOSURE_MIN_WINDOW_S`) is applied here, in the two consumer functions, never at the
tagger or the cached artifact (§9 tool contract).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from orpheus_mcp.analysis import model_registry
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.models import Timeline, Window

if TYPE_CHECKING:
    from orpheus_mcp.analysis.cachefs import Pcm

# The Timeline's locked output hop (§9: "hop_s is LOCKED to 1.0"). A tagger's native frame
# rate (PANNs: ~10 ms) is max-pooled UP to this — never averaged, so a short spike anywhere
# within a second survives into that second's reading.
_LOCKED_HOP_S = 1.0

# §5.2's bleed-gate formula: max(non-target activation) / (target activation + eps). eps only
# guards the (target == 0) division; it is small enough that a 0.05/0.9 frame reads 0.056 and
# a 0.2/0.9 frame reads 0.222 (the brief's two locked numeric cases), not perturbed by it.
_EPS = 1e-6

# PANNs' native output resolution: hop_size=320 samples at its fixed 32 kHz input rate.
_PANNS_SR = 32000
_PANNS_NATIVE_HOP_S = 320 / _PANNS_SR


class TaggerEngine(Protocol):
    """Structural contract any tagger (PANNs, a test double, a future alternative) must
    satisfy — ``build_timeline`` only ever calls through this, never a concrete class.

    ``classes`` is declared as a read-only property (not a plain attribute) so both a
    lazily-loaded ``@property`` implementation (``PannsTagger``, which only knows its
    label list after the checkpoint loads) and a plain instance-attribute implementation
    (``StubTagger``) satisfy this Protocol structurally — a plain-attribute Protocol
    member requires read+write, which a property-backed class can never match."""

    hop_s: float  # native frame hop, in seconds, of `activations`' first axis

    @property
    def classes(self) -> list[str]:
        """The full AudioSet (or engine-specific) label list, in column order."""
        ...

    def activations(self, pcm: Pcm) -> np.ndarray:
        """``(frames, len(self.classes))`` sigmoid activations at ``hop_s`` resolution."""
        ...


class PannsTagger:
    """The real T2 tagger: PANNs' framewise Sound Event Detection CNN (Cnn14_DecisionLevel*
    family), lazily loaded — torch/panns_inference/librosa are imported inside methods only,
    never at module import time, so importing this module never requires the [analysis]
    extra to be installed.

    ``head`` names the SED head variant (§15.6's Task 2.2 pin run selects among Max/Avg/Att
    by frame-level F1 on the fixture set and records the winner in the registry's ``head``
    field). Known, honestly-stated scope limit: the ``panns-inference`` PyPI package (the
    free/open, pip-installable distribution §1.2 requires) ships ONLY the
    ``Cnn14_DecisionLevelMax`` architecture — the Avg/Att variants exist in the upstream
    ``qiuqiangkong/audioset_tagging_cnn`` GitHub repo but are not part of this package, and
    the registry's pinned checkpoint (`tagger/panns-sed`, Task 0.2/0.3) is the Max-variant
    weights. ``head`` is therefore accepted (interface parity with the field Task 2.2 will
    write) and folded into ``model_id`` (so a future head change busts the cache), but the
    ONLY architecture ever loaded today is DecisionLevelMax — this docstring is that fact
    stated plainly, not silently faked support.
    """

    def __init__(self, head: str = "DecisionLevelAtt") -> None:
        self.head = head
        self.model_id = f"tagger/panns-sed:{head}"
        self.hop_s = _PANNS_NATIVE_HOP_S
        self._classes: list[str] | None = None
        self._sed = None

    @property
    def classes(self) -> list[str]:
        self._load()
        assert self._classes is not None
        return self._classes

    def activations(self, pcm: Pcm) -> np.ndarray:
        self._load()
        assert self._sed is not None

        samples = pcm.samples
        is_multichannel = samples.ndim > 1 and samples.shape[1] > 1
        mono = samples.mean(axis=1) if is_multichannel else samples.reshape(-1)
        mono = mono.astype(np.float32)

        if pcm.sample_rate != _PANNS_SR:
            import librosa

            mono = librosa.resample(mono, orig_sr=pcm.sample_rate, target_sr=_PANNS_SR)

        audio_batch = mono[np.newaxis, :].astype(np.float32)
        with model_registry.GPU_SEMAPHORE:  # §15.5: serialize ALL GPU inference calls
            framewise_output = self._sed.inference(audio_batch)  # (1, frames, classes)
        return np.asarray(framewise_output[0], dtype=np.float64)

    def _load(self) -> None:
        if self._sed is not None:
            return

        _ensure_audioset_labels_csv()  # MUST happen before panns_inference is ever imported

        import panns_inference

        checkpoint_path = _panns_checkpoint_path()
        with model_registry.GPU_SEMAPHORE:  # §15.5: serialize ALL GPU model loads
            self._sed = panns_inference.SoundEventDetection(
                checkpoint_path=str(checkpoint_path), device="cuda"
            )
        self._classes = list(panns_inference.config.labels)


def _panns_checkpoint_path() -> Path:
    """The pinned SED checkpoint from ``model_registry.checkpoint_dir()`` (NEVER
    ``~/panns_data`` — that hardcoded path is panns_inference's own default download
    target, which Task 0.2's ``fetch-models`` deliberately bypasses by pre-fetching into
    our own checkpoint cache instead)."""
    from orpheus_mcp.analysis.fetch_models import _filename_for_url

    entries = [
        e
        for e in model_registry.entries_for_task("tagger", allow_unpinned=True)
        if e.engine == "panns"
    ]
    if not entries:
        raise model_registry.RegistryError(
            "No 'tagger' task entry with engine='panns' in the model registry — "
            "PannsTagger has nothing to load."
        )
    entry = sorted(entries, key=lambda e: e.rank)[0]
    if not isinstance(entry.checkpoint_url, str):
        raise model_registry.RegistryError(
            f"{entry.id}: expected a single checkpoint_url (str), got {entry.checkpoint_url!r}"
        )
    filename = _filename_for_url(entry.checkpoint_url, entry.id)
    path = model_registry.checkpoint_dir(entry.id) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"PANNs SED checkpoint not found at {path}. Run `orpheus-mcp fetch-models` "
            "(Task 0.2) first."
        )
    return path


def _ensure_audioset_labels_csv() -> None:
    """``panns_inference``'s package ``__init__`` unconditionally imports ``.inference`` ->
    ``.config``, which — unless a labels CSV already sits at
    ``~/panns_data/class_labels_indices.csv`` — shells out to ``wget`` against a Google
    Storage URL at IMPORT TIME. That is both a Windows breakage (no ``wget`` on PATH by
    default) and a "local only" (§1.1) violation: the mere act of importing the library
    would perform network I/O. This pre-seeds the CSV from our own bundled copy
    (``data/audioset_class_labels.csv`` — the standard, static 527-class AudioSet
    ontology) before ``panns_inference`` is ever imported, so its own `os.path.isfile`
    guard finds the file already present and the ``wget`` branch never runs."""
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_bundled_labels_csv_path().read_bytes())


def _bundled_labels_csv_path() -> Path:
    """Locate ``data/audioset_class_labels.csv`` in both layouts (installed wheel vs. dev
    checkout) — same two-layout probe as ``model_registry.registry_path()``."""
    package_path = Path(__file__).resolve().parents[1] / "data" / "audioset_class_labels.csv"
    if package_path.is_file():
        return package_path
    return Path(__file__).resolve().parents[3] / "data" / "audioset_class_labels.csv"


# --------------------------------------------------------------------------- #
# build_timeline — decode -> tagger.activations -> max-pool to the locked 1 s hop
# --------------------------------------------------------------------------- #


def build_timeline(pcm: Pcm, tagger: TaggerEngine) -> Timeline:
    """Run ``tagger`` over ``pcm`` and max-pool its native-hop activations up to the
    LOCKED 1 s hop, keeping only the columns in the registry's mapped-subset class union
    (``model_registry.mapped_tagger_classes`` — §9 tool contract: a 4-minute song's
    timeline stays ~240 x |mapped| floats, never the tagger's full ~527-class output).

    ``len(Timeline.frames) == ceil(pcm.duration_s / 1.0)`` — the OUTPUT frame count is
    driven by the audio's actual duration, not by how many native frames the tagger
    happened to emit (a tagger's windowing can fall a little short of the nominal length).
    """
    raw = np.asarray(tagger.activations(pcm), dtype=np.float64)
    native_hop = tagger.hop_s
    tagger_classes = list(tagger.classes)

    mapped = set(model_registry.mapped_tagger_classes(allow_unpinned=True))
    keep_idx = [i for i, c in enumerate(tagger_classes) if c in mapped]
    keep_names = [tagger_classes[i] for i in keep_idx]

    n_native = raw.shape[0] if raw.ndim == 2 else 0
    n_out = math.ceil(pcm.duration_s / _LOCKED_HOP_S) if pcm.duration_s > 0 else 0

    frames: list[list[float]] = []
    for i in range(n_out):
        t0 = i * _LOCKED_HOP_S
        t1 = (i + 1) * _LOCKED_HOP_S
        start_idx = max(0, min(int(math.floor(t0 / native_hop + 1e-6)), n_native))
        end_idx = max(start_idx, min(int(math.ceil(t1 / native_hop - 1e-6)), n_native))

        if end_idx > start_idx and keep_idx:
            pooled = raw[start_idx:end_idx][:, keep_idx].max(axis=0).tolist()
        else:
            pooled = [0.0] * len(keep_idx)
        frames.append(pooled)

    return Timeline(classes=keep_names, hop_s=_LOCKED_HOP_S, frames=frames)


# --------------------------------------------------------------------------- #
# most_exposed_window / target_absent — §3.1b/c, sharing the §5.2 ratio formula
# --------------------------------------------------------------------------- #


def _target_and_nontarget_indices(
    tl: Timeline, target_classes: Sequence[str]
) -> tuple[list[int], list[int]] | None:
    """Column indices for ``target_classes`` within ``tl.classes``, and everything else.

    Returns ``None`` for the two N/A cases both consumer functions treat identically: an
    EMPTY ``target_classes`` (the registry entry has no tagger mapping — §4's "empty
    tagger_classes => gates N/A") and a non-empty ``target_classes`` none of which this
    particular Timeline's mapped-subset columns actually cover.
    """
    if not target_classes:
        return None
    target_idx = [tl.classes.index(c) for c in target_classes if c in tl.classes]
    if not target_idx:
        return None
    nontarget_idx = [i for i in range(len(tl.classes)) if i not in target_idx]
    return target_idx, nontarget_idx


def most_exposed_window(
    tl: Timeline,
    target_classes: list[str],
    min_len_s: float = CFG.EXPOSURE_MIN_WINDOW_S,
) -> Window | None:
    """§3.1b: the MOST exposed window (lowest mean normalized non-target ratio) of at
    least ``min_len_s`` where, in every frame, ``max(non-target) / (target + eps) <
    CFG.EXPOSURE_THRESHOLD`` AND ``target > CFG.TIMELINE_ACTIVITY_THRESHOLD`` — or
    ``None`` if no such window exists (including the N/A empty/unmapped-target case)."""
    indices = _target_and_nontarget_indices(tl, target_classes)
    if indices is None:
        return None
    target_idx, nontarget_idx = indices

    hop = tl.hop_s
    min_frames = max(1, math.ceil(min_len_s / hop)) if hop > 0 else 1

    ratios: list[float] = []
    exposed: list[bool] = []
    for frame in tl.frames:
        target_val = max((frame[j] for j in target_idx), default=0.0)
        nontarget_val = max((frame[j] for j in nontarget_idx), default=0.0)
        ratio = nontarget_val / (target_val + _EPS)
        ratios.append(ratio)
        is_exposed = ratio < CFG.EXPOSURE_THRESHOLD and target_val > CFG.TIMELINE_ACTIVITY_THRESHOLD
        exposed.append(is_exposed)

    best_span: tuple[int, int] | None = None
    best_mean: float | None = None
    i = 0
    n = len(exposed)
    while i < n:
        if not exposed[i]:
            i += 1
            continue
        j = i
        while j < n and exposed[j]:
            j += 1
        if j - i >= min_frames:
            mean_ratio = sum(ratios[i:j]) / (j - i)
            if best_mean is None or mean_ratio < best_mean:
                best_mean = mean_ratio
                best_span = (i, j)
        i = j

    if best_span is None:
        return None
    return Window(start_s=best_span[0] * hop, end_s=best_span[1] * hop)


def target_absent(tl: Timeline, target_classes: list[str]) -> bool:
    """§3.1c: ``True`` iff ``target_classes`` never exceeds
    ``CFG.TIMELINE_ACTIVITY_THRESHOLD`` in any frame of ``tl``. ``False`` (never ``True``)
    for the N/A case — empty ``target_classes``, or none of them mapped into this
    Timeline's columns — the caller must treat that as "unknown", not "confirmed absent"."""
    indices = _target_and_nontarget_indices(tl, target_classes)
    if indices is None:
        return False
    target_idx, _nontarget_idx = indices

    for frame in tl.frames:
        target_val = max(frame[j] for j in target_idx)
        if target_val > CFG.TIMELINE_ACTIVITY_THRESHOLD:
            return False
    return True
