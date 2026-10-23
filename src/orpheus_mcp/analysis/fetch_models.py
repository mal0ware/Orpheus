"""``orpheus-mcp fetch-models`` — the checkpoint pre-fetch install step (§1.1/§9).

This is the ONLY new code in Task 0.2 allowed to touch the network (the pre-existing,
consent-gated ``install_sound_pack`` is the other standing exception). Every analysis tool
call must be able to assume its checkpoints already exist locally and are hash-verified —
``audio-separator``'s default lazy download-on-first-use is exactly what this disables.

Two modes:
- Normal (default): download each registry checkpoint, sha256-VERIFY it against the
  registry's pinned hash, and refuse (raise) on a mismatch or a missing pin.
- Pin mode (``--pin``, Task 0.3's workflow): download each checkpoint and sha256-RECORD it
  instead of verifying — used exactly once, by ``scripts/pin_models.py``, to populate the
  registry's hashes in the first place. Never used by a normal install.

Stdlib-only (``urllib.request`` + ``hashlib``) — no new dependency for something this small,
and it keeps this module importable without any analysis extra installed.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path
from typing import TextIO
from urllib.parse import urlparse
from urllib.request import urlopen

from orpheus_mcp.analysis.model_registry import checkpoint_dir, load_registry

_CHUNK_SIZE = 1024 * 1024


class FfmpegMissing(Exception):
    """ffmpeg is not on PATH — required for every audio decode step downstream."""


class FetchModelsError(Exception):
    """A checkpoint could not be fetched or failed hash verification."""


def ffmpeg_on_path() -> bool:
    """Whether the ``ffmpeg`` executable is discoverable on PATH."""
    return shutil.which("ffmpeg") is not None


def require_ffmpeg() -> None:
    """Raise an actionable :class:`FfmpegMissing` if ffmpeg is not on PATH."""
    if not ffmpeg_on_path():
        raise FfmpegMissing(
            "ffmpeg was not found on PATH. Every Orpheus analysis tool decodes audio through "
            "it. Install it first (Windows: `winget install ffmpeg`; macOS: `brew install "
            "ffmpeg`; Linux: your package manager), then re-run `orpheus-mcp fetch-models`."
        )


def _filename_for_url(url: str, entry_id: str) -> str:
    """The local filename for a single checkpoint URL - the URL PATH's basename, query
    strings stripped (a raw ``rsplit("/")`` on a Google Drive ``?id=...&export=download``
    URL, or on an HF repo-root URL with no path segment, produces a Windows-illegal or
    empty filename)."""
    path = urlparse(url).path
    name = path.rstrip("/").rsplit("/", 1)[-1]
    return name or f"{entry_id.replace('/', '_')}.bin"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` atomically (download to a .part file, then rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urlopen(url) as response, tmp.open("wb") as fh:  # noqa: S310 (fixed install-time step)
        shutil.copyfileobj(response, fh)
    tmp.replace(dest)


def fetch_models(
    *,
    pin: bool = False,
    out: TextIO = sys.stdout,
    downloader=download_checkpoint,
) -> dict[str, str | list[str]]:
    """Download + verify (or, in pin mode, record) every registry checkpoint.

    Returns ``{entry.id: status}`` where status is ``"skipped (no checkpoint_url)"``,
    ``"verified"``, or — in pin mode — the freshly computed sha256 hex digest (a ``str`` for a
    single-file entry, a ``list[str]`` for a bag-of-models entry, positionally matching the
    entry's ``checkpoint_url`` list). ``downloader`` is injectable so tests never hit the
    network (default streams for real).

    Scope: every registry checkpoint, not just separator engines — PANNs SED weights, the
    BTC + musicxlab chord checkpoints, and the faster-whisper large-v3 snapshot are fetched
    here too, all landing under ``checkpoint_dir()``, never inside the repo.
    """
    require_ffmpeg()

    entries = load_registry(allow_unpinned=True)
    results: dict[str, str | list[str]] = {}

    for entry in entries:
        if entry.checkpoint_url is None:
            results[entry.id] = "skipped (no checkpoint_url)"
            continue

        if not pin and entry.sha256 is None:
            raise FetchModelsError(
                f"{entry.id}: checkpoint_url is set but sha256 is not pinned, and this is "
                "not a --pin run. Run `orpheus-mcp fetch-models --pin` (Task 0.3's pin "
                "workflow) to populate hashes first."
            )

        urls = (
            entry.checkpoint_url
            if isinstance(entry.checkpoint_url, list)
            else [entry.checkpoint_url]
        )
        expected = entry.sha256 if isinstance(entry.sha256, list) else [entry.sha256]

        digests: list[str] = []
        for url in urls:
            dest = checkpoint_dir(entry.id) / _filename_for_url(url, entry.id)
            print(f"Fetching {entry.id} -> {dest}", file=out)
            downloader(url, dest)
            digests.append(_sha256_file(dest))

        if pin:
            # Preserve the entry's shape: str hash for a str url, list for a list.
            results[entry.id] = digests if isinstance(entry.checkpoint_url, list) else digests[0]
            continue

        for url, digest, want in zip(urls, digests, expected, strict=True):
            if digest != want:
                raise FetchModelsError(
                    f"{entry.id}: sha256 mismatch for {url}. Expected {want}, got {digest}. "
                    "The checkpoint may be corrupted or the upstream file changed — delete it "
                    "and re-run, or re-pin if the upstream release genuinely moved."
                )
        results[entry.id] = "verified"

    return results
