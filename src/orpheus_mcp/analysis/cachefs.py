"""``analysis/cachefs.py`` (§9): decode, canonical-PCM hashing, and the on-disk artifact
cache every downstream analysis module reads/writes through.

**Canonical PCM hashing** (locked, do not re-decide): the cache key is
``sha256(sample_rate || channel_count || int16-canonicalized samples)`` — native sample
rate and channel count, float32 samples converted to int16 (round-half-away-from-zero,
clipped), never resampled or downmixed. Float LSBs differ across decoder builds (and even
between soundfile/ffmpeg versions), so hashing the raw float32 bytes would make the cache
key non-portable; the int16 canonicalization makes it stable for lossless paths and
stable-per-ffmpeg-build for lossy ones. An ffmpeg upgrade that changes lossy decode output
naturally produces a different ``pcm_hash`` for the same file — that surfaces as an
ordinary cache MISS (a fresh cache entry gets built), never as corruption of an old one.

**The memo file** (``<root>/hash_memo.json``, ``sha256(file bytes) -> pcm_hash``) exists so
repeat lookups of the *same file* never re-decode (§3.1 T0) — the first ``pcm_hash()`` call
on a path pays for one full decode; every later call for that exact file is a dict lookup.

**The meta-sidecar contract**: every cached JSON artifact gets a ``<kind>.meta.json``
sidecar next to ``<kind>.json``. ``read_json`` compares the sidecar to the caller's ``meta``
dict field-for-field; ANY mismatch is treated as a cache miss (the stale artifact is left in
place for LRU, never trusted or silently repaired). Callers own their own meta shape — this
module only stores/compares whatever dict they pass. The two hash ingredients most callers
need (``registry_version``, ``verify_config_hash``) are computed here so every later task
shares one implementation.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.fetch_models import ffmpeg_on_path
from orpheus_mcp.models import Window

# --------------------------------------------------------------------------- #
# Pcm — the decoded-audio payload. A plain dataclass (not pydantic): it carries an
# ndarray and never crosses a tool boundary (tools return pydantic models built FROM it).
# --------------------------------------------------------------------------- #


@dataclass
class Pcm:
    samples: np.ndarray  # float32, shape (frames, channels) — native sample rate, unmixed
    sample_rate: int
    path: str
    pcm_hash: str
    duration_s: float
    decoder: str  # "soundfile" or "ffmpeg <version>" — which decode path produced this


class DecodeError(Exception):
    """Audio could not be decoded — the message always says why (corrupt, truncated, DRM)."""


_NATIVE_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
# .mp4: music videos — ffmpeg (already invoked with -vn) extracts the audio stream.
_FFMPEG_EXTS = {".mp3", ".m4a", ".mp4"}
_DRM_EXTS = {".m4p"}

_STDERR_TAIL_LINES = 15


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #


def decode(path: str) -> Pcm:
    """Decode ``path`` to canonical Pcm. soundfile handles wav/flac/ogg/aiff natively;
    mp3/m4a/mp4 (incl. music videos — the audio stream is extracted, video discarded) go
    through an ffmpeg subprocess (list argv, no shell) to a temp wav first.

    Raises :class:`DecodeError` — never a raw soundfile/subprocess exception — for DRM
    (``.m4p``, checked before any subprocess), a missing/corrupt/truncated file, a missing
    ffmpeg, or a nonzero ffmpeg exit (message includes the stderr tail).
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in _DRM_EXTS:
        raise DecodeError(
            f"DRM-protected file cannot be decoded: {p}. FairPlay/.m4p audio is rejected "
            "outright — Orpheus only ever analyzes audio you can already play back freely."
        )

    if not p.is_file():
        raise DecodeError(f"No such audio file: {p}")

    if ext in _FFMPEG_EXTS:
        samples, sample_rate, decoder = _decode_via_ffmpeg(p)
    else:
        samples, sample_rate = _decode_via_soundfile(p)
        decoder = "soundfile"

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    frames = samples.shape[0]
    duration_s = frames / sample_rate if sample_rate else 0.0

    return Pcm(
        samples=samples,
        sample_rate=sample_rate,
        path=str(p),
        pcm_hash=_canonical_pcm_hash(samples, sample_rate),
        duration_s=duration_s,
        decoder=decoder,
    )


def _decode_via_soundfile(p: Path) -> tuple[np.ndarray, int]:
    if p.suffix.lower() == ".wav":
        _check_wav_not_truncated(p)
    import soundfile as sf

    try:
        data, sample_rate = sf.read(str(p), always_2d=True, dtype="float32")
    except Exception as exc:  # noqa: BLE001 - normalize every decode failure to DecodeError
        raise DecodeError(f"Could not decode {p}: {exc}") from exc
    return data, sample_rate


def _check_wav_not_truncated(p: Path) -> None:
    """WAV's RIFF header states the file's total byte size; libsndfile tolerates a
    truncated data chunk by silently reading whatever bytes ARE present (i.e. it would
    return partial audio, not raise) — so this checks the declared size against the file's
    actual size ourselves, ahead of ever handing the file to soundfile."""
    with p.open("rb") as fh:
        header = fh.read(8)
    if len(header) < 8 or header[:4] != b"RIFF":
        return  # not a RIFF file in a way we can sanity-check here; let soundfile decide
    (riff_size,) = struct.unpack("<I", header[4:8])
    expected_total = riff_size + 8
    actual_total = p.stat().st_size
    if actual_total < expected_total:
        raise DecodeError(
            f"Truncated WAV file: {p} declares {expected_total} bytes but only "
            f"{actual_total} are present on disk."
        )


def _decode_via_ffmpeg(p: Path) -> tuple[np.ndarray, int, str]:
    if not ffmpeg_on_path():
        raise DecodeError(
            f"ffmpeg was not found on PATH — required to decode {p.suffix} files. Install "
            "it (Windows: `winget install ffmpeg`; macOS: `brew install ffmpeg`; Linux: "
            "your package manager) and retry."
        )

    import soundfile as sf

    with tempfile.TemporaryDirectory() as td:
        tmp_wav = Path(td) / "decoded.wav"
        argv = ["ffmpeg", "-y", "-i", str(p), "-vn", str(tmp_wav)]
        # UTF-8, not the Windows locale: ffmpeg prints tags and paths as UTF-8, and a cp1252
        # reader thread dies on bytes like 0x9D and hands back stderr=None.
        proc = subprocess.run(
            argv, capture_output=True, encoding="utf-8", errors="replace", check=False
        )
        if proc.returncode != 0:
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-_STDERR_TAIL_LINES:])
            raise DecodeError(
                f"ffmpeg failed to decode {p} (exit {proc.returncode}):\n{stderr_tail}"
            )
        try:
            data, sample_rate = sf.read(str(tmp_wav), always_2d=True, dtype="float32")
        except Exception as exc:  # noqa: BLE001
            raise DecodeError(f"ffmpeg produced an unreadable wav for {p}: {exc}") from exc

    return data, sample_rate, f"ffmpeg {_ffmpeg_version()}"


def _ffmpeg_version() -> str:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, encoding="utf-8", errors="replace",
            check=False,
        )
        first_line = proc.stdout.splitlines()[0] if proc.stdout else ""
        parts = first_line.split()
        if len(parts) >= 3 and parts[0] == "ffmpeg" and parts[1] == "version":
            return parts[2]
    except OSError:
        pass
    return "unknown"


def decoder_for(path: str) -> str:
    """The ``decoder`` tag ``decode(path)`` will report, WITHOUT decoding — deterministic
    from the extension alone (native formats -> ``"soundfile"``, ffmpeg-only formats ->
    ``"ffmpeg <version>"``), so a cache-artifact meta dict can be built (and probed for a
    HIT) before ever paying for a decode (§9 T0: cache lookups cost ms after the first
    decode, and building meta must not itself force one)."""
    ext = Path(path).suffix.lower()
    if ext in _FFMPEG_EXTS:
        return f"ffmpeg {_ffmpeg_version()}"
    return "soundfile"


# --------------------------------------------------------------------------- #
# Canonical PCM hash + file-bytes memo
# --------------------------------------------------------------------------- #


def _float_to_int16(samples: np.ndarray) -> np.ndarray:
    """Round-half-away-from-zero, then clip to int16 range — deterministic regardless of
    numpy's (banker's-rounding) default, so the same float32 always canonicalizes the same
    way across numpy versions."""
    scaled = samples.astype(np.float64) * 32768.0
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    clipped = np.clip(rounded, -32768, 32767)
    return clipped.astype(np.int16)


def _canonical_pcm_hash(samples: np.ndarray, sample_rate: int) -> str:
    int16 = _float_to_int16(samples)
    channels = int16.shape[1] if int16.ndim > 1 else 1
    digest = hashlib.sha256()
    digest.update(struct.pack("<II", sample_rate, channels))
    digest.update(np.ascontiguousarray(int16).tobytes())
    return digest.hexdigest()


_MEMO_CHUNK_SIZE = 1024 * 1024


def _file_bytes_hash(p: Path) -> str:
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_MEMO_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_cache_root() -> Path:
    """``~/.orpheus_cache`` unless overridden via ``CFG.CACHE_DIR_ENV`` — the same
    override hook ``model_registry.checkpoint_dir`` uses, so tests never touch a real
    home directory and both cache roots move together under one env var."""
    override = os.environ.get(CFG.CACHE_DIR_ENV)
    return Path(override) if override else Path.home() / ".orpheus_cache"


def _memo_path() -> Path:
    return _default_cache_root() / "hash_memo.json"


def _load_memo() -> dict[str, str]:
    path = _memo_path()
    if not path.is_file():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_memo(memo: dict[str, str]) -> None:
    _atomic_write_json(_memo_path(), memo)


def pcm_hash(path: str) -> str:
    """sha256 of the canonical PCM for ``path``, memoized by ``sha256(file bytes)`` so a
    repeat lookup of the same file never re-decodes (§3.1 T0)."""
    p = Path(path)
    file_hash = _file_bytes_hash(p)
    memo = _load_memo()
    cached = memo.get(file_hash)
    if cached is not None:
        return cached

    pcm = decode(str(p))
    memo[file_hash] = pcm.pcm_hash
    _save_memo(memo)
    return pcm.pcm_hash


# --------------------------------------------------------------------------- #
# Shared meta-hash helpers (§9) — every later task shares these two, per the brief.
# --------------------------------------------------------------------------- #

_CAL_COMMENT = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*:.*#\s*cal\b", re.MULTILINE)


def _cal_field_names() -> list[str]:
    """Names of every §16 tunable marked ``# cal`` in ``config.py`` — read from the live
    source via ``inspect.getsource`` so this can never drift out of sync with config.py's
    own comments (the source of truth `calibrate_gates.py` re-fits)."""
    from orpheus_mcp.analysis.config import AnalysisConfig

    source = inspect.getsource(AnalysisConfig)
    return sorted(set(_CAL_COMMENT.findall(source)))


def verify_config_hash() -> str:
    """sha256 of the sorted cal-marked CFG fields, AS EFFECTIVE at read time (D7): once
    ``data/calibrated_thresholds.json`` overrides load into CFG (Task 5.3), this naturally
    reflects them — a re-calibration correctly busts stale gate-artifact cache entries."""
    fields = _cal_field_names()
    payload = json.dumps({name: getattr(CFG, name) for name in fields}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def registry_version() -> str:
    """sha256 of ``data/model_registry.json`` bytes — busts artifacts when the registry
    entry that produced them (checkpoint id, hash, rank, ...) changes."""
    from orpheus_mcp.analysis.model_registry import registry_path

    return hashlib.sha256(registry_path().read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Atomic JSON I/O (Windows-safe: os.replace can transiently PermissionError while another
# handle stats the destination — battle-tested retry pattern from tests/fake_reaper.py).
# --------------------------------------------------------------------------- #

_REPLACE_RETRIES = 25
_REPLACE_RETRY_DELAY_S = 0.002


def _atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
        last_exc: OSError | None = None
        for _ in range(_REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                return
            except OSError as exc:
                last_exc = exc
                time.sleep(_REPLACE_RETRY_DELAY_S)
        assert last_exc is not None
        raise last_exc
    finally:
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()


def _rmtree_retry(path: Path) -> None:
    """``shutil.rmtree`` with the same Windows transient-lock tolerance as the atomic
    JSON writer — a concurrent reader can briefly hold a handle open under ``path``."""
    last_exc: OSError | None = None
    for _ in range(_REPLACE_RETRIES):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(_REPLACE_RETRY_DELAY_S)
    if last_exc is not None:
        raise last_exc


# --------------------------------------------------------------------------- #
# CacheFS — the on-disk artifact store
# --------------------------------------------------------------------------- #

_RESERVED_TOP_LEVEL = frozenset({"checkpoints"})
"""Top-level cache-root subdirectories §9's ``clear``/``evict_lru`` (and anything built on
:meth:`CacheFS.entry_dirs`) must NEVER touch. ``checkpoints/`` is where
``model_registry.checkpoint_dir()`` stores downloaded model weights (frequently ~10 GB) —
an install-time TOOL artifact that happens to live under the same cache root as §9's
analysis artifacts, not one of them. ``clear_analysis_cache``/LRU eviction govern ANALYSIS
artifacts only; a model checkpoint is never "stale" the way a cached recon/timeline/chords
artifact is, and re-downloading ~10 GB because a user asked to free a few MB of analysis
cache would be a silent, expensive surprise. NOTE: ``renders/`` (§7.1) is deliberately NOT in
this set — render outputs are analysis-side artifacts §9's LRU already treats as ordinary,
evictable/clearable entries, and must stay that way.

``hash_memo.json`` (a FILE, not a directory) survives ``clear``/``evict_lru`` without being
listed here: :meth:`CacheFS.entry_dirs` filters ``self.root.iterdir()`` down to ``d.is_dir()``
entries only, so the memo file is never a candidate in the first place — its survival rides
that ``is_dir()`` filter, not membership in this set. Any FUTURE reserved top-level cache-root
entry that IS a directory (not a lone file like the memo) must be added here explicitly, or
``entry_dirs``/``clear``/``evict_lru`` will silently treat it as an ordinary evictable/clearable
analysis-cache entry."""


class CacheFS:
    """The §9 cache filesystem, rooted at ``<root>/<pcm-sha256>/``.

    Layout per pcm-hash directory: ``<kind>.json`` + sidecar ``<kind>.meta.json`` for
    every cached artifact (recon, timeline, chords, ladder, ...); ``stems/<model-id>/
    <start_ms>-<end_ms>/`` for separated-stem regions (``stem_dir``). LRU eviction operates
    on whole pcm-hash directories, keyed by that directory's mtime; pinned hashes are
    skipped entirely (in-flight-pipeline protection — an in-process set, not persisted).
    ``checkpoints/`` (``model_registry.checkpoint_dir()``'s downloaded model weights) is
    NEVER a candidate for ``clear``/``evict_lru``/``size_bytes`` — see
    :data:`_RESERVED_TOP_LEVEL` and :meth:`entry_dirs`.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else _default_cache_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self._pinned: set[str] = set()

    # -- paths -------------------------------------------------------------- #

    def artifact_dir(self, pcm_hash: str) -> Path:
        """The per-pcm-hash directory, created if missing."""
        d = self.root / pcm_hash
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _dir_path(self, pcm_hash: str) -> Path:
        """Non-creating path lookup — reads must never fabricate an empty directory."""
        return self.root / pcm_hash

    def _artifact_path(self, pcm_hash: str, kind: str) -> Path:
        return self._dir_path(pcm_hash) / f"{kind}.json"

    def _meta_path(self, pcm_hash: str, kind: str) -> Path:
        return self._dir_path(pcm_hash) / f"{kind}.meta.json"

    def stem_dir(self, pcm_hash: str, model_id: str, region: Window) -> Path:
        """``stems/<model-id>/<start_ms>-<end_ms>/`` under this hash's artifact dir (§9).
        ``model_id`` containing ``/`` (bag-of-models ids) nests into subdirectories, the
        same convention ``model_registry.checkpoint_dir`` uses."""
        start_ms = round(region.start_s * 1000)
        end_ms = round(region.end_s * 1000)
        d = (
            self.artifact_dir(pcm_hash)
            / "stems"
            / Path(*model_id.split("/"))
            / f"{start_ms}-{end_ms}"
        )
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- JSON artifacts ------------------------------------------------------ #

    def read_json(self, pcm_hash: str, kind: str, meta: dict) -> dict | None:
        """``None`` on a cache miss OR a meta mismatch — the stale artifact (if any) is
        left on disk for LRU, never trusted or repaired here."""
        artifact_path = self._artifact_path(pcm_hash, kind)
        meta_path = self._meta_path(pcm_hash, kind)
        if not artifact_path.is_file() or not meta_path.is_file():
            return None

        try:
            stored_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if stored_meta != meta:
            return None

        try:
            obj = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        self._touch(pcm_hash)  # a hit keeps this pcm-hash's whole dir fresh for LRU
        return obj

    def write_json(self, pcm_hash: str, kind: str, obj: dict, meta: dict) -> Path:
        artifact_path = self._artifact_path(pcm_hash, kind)
        meta_path = self._meta_path(pcm_hash, kind)
        self.artifact_dir(pcm_hash)  # ensure the directory exists
        _atomic_write_json(artifact_path, obj)
        _atomic_write_json(meta_path, meta)
        return artifact_path

    def _touch(self, pcm_hash: str) -> None:
        d = self._dir_path(pcm_hash)
        now = time.time()
        with contextlib.suppress(OSError):
            os.utime(d, (now, now))

    # -- pinning -------------------------------------------------------------- #

    def pin(self, pcm_hash: str) -> None:
        """Protect ``pcm_hash`` against eviction — an in-flight pipeline is using it."""
        self._pinned.add(pcm_hash)

    def unpin(self, pcm_hash: str) -> None:
        self._pinned.discard(pcm_hash)

    # -- size + eviction -------------------------------------------------------- #

    def size_bytes(self) -> int:
        """Total on-disk size of every cached pcm-hash directory (excludes the memo AND
        ``checkpoints/`` — see :data:`_RESERVED_TOP_LEVEL`)."""
        return sum(_dir_size(d) for d in self.entry_dirs())

    def entry_dirs(self) -> list[Path]:
        """Every top-level directory under the cache root that §9's clear/LRU are allowed to
        touch — every entry EXCEPT :data:`_RESERVED_TOP_LEVEL` (currently just
        ``checkpoints/``). This is the single source of truth: ``size_bytes``, ``evict_lru``,
        and ``clear`` all filter through this method, and
        ``tools.reference.clear_analysis_cache`` counts ``entries_cleared`` from it too, so
        the "what counts as a clearable/evictable entry" definition never drifts between the
        two call sites."""
        if not self.root.is_dir():
            return []
        return [
            d
            for d in self.root.iterdir()
            if d.is_dir() and d.name not in _RESERVED_TOP_LEVEL
        ]

    def entry_count(self) -> int:
        """``len(self.entry_dirs())`` — the count ``clear_analysis_cache``'s
        ``entries_cleared`` reports, measured BEFORE ``clear()`` runs."""
        return len(self.entry_dirs())

    def evict_lru(self) -> int:
        """Evict whole pcm-hash directories, oldest-mtime first, until under
        ``CFG.CACHE_SIZE_CAP_GB`` — pinned hashes are never touched (and may leave the
        cache over cap if pinned data alone exceeds it); ``checkpoints/`` is never a
        candidate either (excluded by :meth:`entry_dirs`, see :data:`_RESERVED_TOP_LEVEL`).
        Returns bytes freed."""
        cap_bytes = int(CFG.CACHE_SIZE_CAP_GB * (1024**3))
        entries = [
            (d.stat().st_mtime, d) for d in self.entry_dirs() if d.name not in self._pinned
        ]
        total = self.size_bytes()
        if total <= cap_bytes:
            return 0

        entries.sort(key=lambda item: item[0])  # oldest first
        freed = 0
        for _, d in entries:
            if total - freed <= cap_bytes:
                break
            freed += _dir_size(d)
            _rmtree_retry(d)
        return freed

    def clear(self) -> None:
        """Wipe every cached ANALYSIS artifact AND the file-bytes memo — a full reset of §9's
        cache. Deliberately does NOT touch ``checkpoints/`` (see :data:`_RESERVED_TOP_LEVEL`):
        model weights under there are install-time tool artifacts, not analysis cache."""
        for d in self.entry_dirs():
            _rmtree_retry(d)
        memo = self.root / "hash_memo.json"
        with contextlib.suppress(FileNotFoundError):
            memo.unlink()


def _dir_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
