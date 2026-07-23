"""The model registry (§4): every downloadable checkpoint Orpheus can route to, as data.

No model is hard-coded at a call site. `data/model_registry.json` is the single source of
truth for what checkpoints exist, what they cost (VRAM), how good they are (rank, benchmark),
and what they're licensed under. `resolve()` is the router's only view into that data — it
never sees a model id that isn't in the registry.

Checkpoints themselves are pre-fetched at install time by ``orpheus-mcp fetch-models`` (the
ONLY code path, besides ``install_sound_pack``, allowed to touch the network — §1.1/§9). This
module never downloads anything; it only reads the JSON and answers routing questions.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from orpheus_mcp.analysis.config import CFG

Task = Literal[
    "multi-stem",
    "single-target",
    "tagger",
    "dereverb",
    "drumkit-split",
    "transcription",
    "chord",
    "lyrics-asr",
]
Engine = Literal[
    "audio-separator",
    "msst",
    "panns",
    "btc",
    "musicxlab",
    "fasterwhisper",
    "basicpitch",
    "internal",
]

# A process-wide GPU semaphore (§15.5): MCP clients can issue parallel tool calls, and two
# calls that both see "9 GB free" then both load 6 GB models is a textbook TOCTOU OOM. Every
# GPU model load/inference call must hold this single slot.
GPU_SEMAPHORE = threading.Semaphore(1)

_NC_TOKEN = re.compile(r"\bNC\b", re.IGNORECASE)


class RegistryError(Exception):
    """The registry JSON is missing, malformed, or has unpinned checkpoints outside pin mode."""


class ModelEntry(BaseModel):
    id: str
    task: Task
    targets: list[str]
    engine: Engine
    checkpoint_url: str | None
    sha256: str | None  # None until pinned; loader REJECTS None outside pin mode
    vram_gb: float
    rt_factor: float | None
    benchmark: str | None
    license: str
    rank: int
    tagger_classes: list[str]  # empty => bleed/exposure gates N/A for this entry (§4)


def registry_path() -> Path:
    """Locate data/model_registry.json in both layouts.

    Installed wheel: pyproject force-includes data/ INSIDE the package (orpheus_mcp/data).
    Dev checkout: data/ sits at the repo root next to src/.
    """
    package_path = Path(__file__).resolve().parents[1] / "data" / "model_registry.json"
    if package_path.is_file():
        return package_path
    return Path(__file__).resolve().parents[3] / "data" / "model_registry.json"


def _is_nc_licensed(license_text: str) -> bool:
    """Whether a license string carries a non-commercial restriction (e.g. 'CC BY-NC 4.0')."""
    return _NC_TOKEN.search(license_text) is not None


def load_registry(*, allow_unpinned: bool = False) -> list[ModelEntry]:
    """Parse data/model_registry.json into ModelEntry objects.

    Entries that declare a real ``checkpoint_url`` MUST also carry a ``sha256`` unless
    ``allow_unpinned=True`` (the Task 0.3 pin run fills these in) — a None hash on a
    downloadable checkpoint means "not yet verified", and tools must never trust an
    unverified checkpoint outside the explicit pin workflow. Entries with NO checkpoint_url
    (e.g. an in-repo ``internal`` baseline with nothing to download) have nothing to pin and
    are exempt — there is no hash to be missing.
    """
    path = registry_path()
    if not path.is_file():
        raise RegistryError(f"No model registry found at {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Malformed model registry JSON at {path}: {exc}") from exc

    entries = [ModelEntry.model_validate(item) for item in raw]

    if not allow_unpinned:
        unpinned = [e.id for e in entries if e.checkpoint_url is not None and e.sha256 is None]
        if unpinned:
            raise RegistryError(
                "Model registry has unpinned checkpoints (sha256 is None): "
                f"{unpinned}. Run `orpheus-mcp fetch-models --pin` (Task 0.3) first, or pass "
                "allow_unpinned=True for routing-only reads."
            )

    return entries


def resolve(
    task: str,
    target: str,
    *,
    vram_free_gb: float | None,
    allow_nc: bool = False,
) -> list[ModelEntry]:
    """Candidate entries for (task, target), in rank order, filtered by VRAM + license (§15.5).

    - Hashes need not be pinned yet to resolve a routing candidate (the hash only matters at
      download/verify time), so this reads the registry with ``allow_unpinned=True``.
    - NC-licensed entries (e.g. LarsNet, CC BY-NC 4.0) are excluded unless the caller opts in
      via ``allow_nc=True`` (Global Constraints §1.2, mechanically enforced here).
    - ``vram_free_gb=None`` (no CUDA) keeps only CPU-capable entries — by registry convention
      that's ``vram_gb == 0.0`` (runs with no GPU requirement at all). Otherwise entries are
      kept when they fit ``CFG.VRAM_FIT_FRACTION`` of the free VRAM (§15.5); a CPU-capable
      entry always fits too.
    """
    entries = load_registry(allow_unpinned=True)

    candidates = [e for e in entries if e.task == task and target in e.targets]

    if not allow_nc:
        candidates = [e for e in candidates if not _is_nc_licensed(e.license)]

    if vram_free_gb is None:
        candidates = [e for e in candidates if e.vram_gb <= 0.0]
    else:
        threshold = CFG.VRAM_FIT_FRACTION * vram_free_gb
        candidates = [e for e in candidates if e.vram_gb <= threshold]

    return sorted(candidates, key=lambda e: e.rank)


def free_vram_gb() -> float | None:
    """Free VRAM on the default CUDA device, in GiB — or None if there is no CUDA to ask.

    §15.5's guard, precisely: check ``torch.cuda.is_available()`` FIRST. Calling
    ``torch.cuda.mem_get_info()`` with no CUDA device raises, so the availability check must
    happen before it — this function must never raise, on a CPU-only box or after a driver
    failure, it just reports "nothing to report".
    """
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return free_bytes / (1024**3)


def checkpoint_dir(model_id: str | None = None) -> Path:
    """The checkpoint cache root, or a specific model's subdirectory within it.

    ``~/.orpheus_cache/checkpoints/`` by default, overridable via ``CFG.CACHE_DIR_ENV``
    (``ORPHEUS_CACHE_DIR``) the same way the rest of the cache filesystem is (§9) — tests
    never write into a real home directory. Passing ``model_id`` returns
    ``<root>/checkpoints/<model_id>/``; a bare id containing a "/" (e.g. "4stem-a/bsroformer")
    is split into nested subdirectories, which also keeps two entries with the same short
    name from colliding.
    """
    override = os.environ.get(CFG.CACHE_DIR_ENV)
    root = Path(override) if override else Path.home() / ".orpheus_cache"
    base = root / "checkpoints"
    if model_id is None:
        return base
    return base.joinpath(*model_id.split("/"))
