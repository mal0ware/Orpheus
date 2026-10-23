# src/orpheus_mcp/soundpack.py
"""Optional default sound pack: a few CC0 FreePats instruments played by REAPER's own sampler.

Opt-in only (``install_sound_pack(confirm=True)``). Files come from FreePats' GitHub repos at
pinned commits, one sha256 per file, listed in ``data/sound_pack.json`` (written by
``scripts/pin_sound_pack.py``). They land under ``~/.orpheus/soundpack/<instrument>/`` exactly
as the repo lays them out, so each instrument's SFZ finds its samples by relative path. Loading
turns the SFZ into RS5k zones (``orpheus_mcp.sfz``) for the bridge's ``add_sampler_zones``.
No third-party plugin, no installer, no plugin rescan. Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from orpheus_mcp.sfz import parse_sfz, rs5k_zones

PACK_DIR = Path.home() / ".orpheus" / "soundpack"
PACK_DIR_ENV = "ORPHEUS_SOUNDPACK_DIR"
_RETRY_STATUS = {429, 500, 502, 503, 504}
_RETRIES = 4


def pack_dir() -> Path:
    """Where the pack lives: ``ORPHEUS_SOUNDPACK_DIR`` if set, else ``~/.orpheus/soundpack``.
    Read at call time so tests (and users) can point it elsewhere."""
    env = os.environ.get(PACK_DIR_ENV)
    return Path(env) if env else PACK_DIR


def manifest_path() -> Path:
    package_path = Path(__file__).resolve().parent / "data" / "sound_pack.json"
    if package_path.exists():
        return package_path
    return Path(__file__).resolve().parents[2] / "data" / "sound_pack.json"


def load_manifest() -> dict:
    return json.loads(manifest_path().read_text(encoding="utf-8"))


def raw_url(repo: str, commit: str, path: str) -> str:
    """GitHub raw URL at a fixed commit. Spaces and ``#`` in paths must be escaped: a raw ``#``
    would end the URL path as a fragment (``samples/pick/F#.flac``)."""
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(path, safe='/')}"


def _http_get(url: str) -> bytes:
    for attempt in range(_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (pinned https)
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRY_STATUS or attempt == _RETRIES - 1:
                raise
            time.sleep(2**attempt)  # GitHub rate-limits unauthenticated raw downloads
    raise AssertionError("unreachable")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_ok(path: Path, spec: dict, *, full_check: bool) -> bool:
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    return _sha256_file(path) == spec["sha256"] if full_check else True


def install_sound_pack(
    root: Path | None = None,
    fetch: Callable[[str], bytes] | None = None,
    manifest: dict | None = None,
) -> dict:
    """Download and verify every pinned file. A file already present with the right hash is
    kept. Raises ``ValueError`` on a hash mismatch, and writes nothing for that file."""
    manifest = manifest if manifest is not None else load_manifest()
    fetch = fetch or _http_get
    root = Path(root) if root is not None else pack_dir()
    downloaded = kept = 0
    for inst in manifest["instruments"]:
        base = root / inst["id"]
        for spec in inst["files"]:
            dest = base / spec["path"]
            if _file_ok(dest, spec, full_check=True):
                kept += 1
                continue
            data = fetch(raw_url(inst["repo"], inst["commit"], spec["path"]))
            digest = hashlib.sha256(data).hexdigest()
            if digest != spec["sha256"]:
                raise ValueError(
                    f"checksum mismatch for {inst['id']}/{spec['path']}: got {digest}, "
                    f"want {spec['sha256']}"
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            downloaded += 1
    return {
        "installed": True,
        "dir": str(root),
        "instruments": [i["id"] for i in manifest["instruments"]],
        "downloaded_files": downloaded,
        "already_present": kept,
    }


def installed_instruments(root: Path | None = None, manifest: dict | None = None) -> list[dict]:
    """Manifest entries whose files are all present at the pinned sizes (a fast check; install
    verifies the hashes)."""
    manifest = manifest if manifest is not None else load_manifest()
    root = Path(root) if root is not None else pack_dir()
    return [
        inst
        for inst in manifest["instruments"]
        if all(_file_ok(root / inst["id"] / s["path"], s, full_check=False) for s in inst["files"])
    ]


def instrument_for_role(
    role: str, root: Path | None = None, manifest: dict | None = None
) -> dict | None:
    """The installed pack instrument for ``role`` ('keys', 'bass', 'guitar'), or None."""
    return next((i for i in installed_instruments(root, manifest) if i["role"] == role), None)


def zones_for(
    instrument_id: str, root: Path | None = None, manifest: dict | None = None
) -> list[dict]:
    """RS5k zones for an installed instrument, with absolute sample paths."""
    manifest = manifest if manifest is not None else load_manifest()
    inst = next((i for i in manifest["instruments"] if i["id"] == instrument_id), None)
    if inst is None:
        known = ", ".join(i["id"] for i in manifest["instruments"])
        raise ValueError(f"unknown sound-pack instrument {instrument_id!r} (have: {known})")
    base = (Path(root) if root is not None else pack_dir()) / inst["id"]
    sfz = base / inst["sfz"]
    if not sfz.is_file():
        raise FileNotFoundError(
            f"sound pack not installed: {sfz} is missing. Run install_sound_pack(confirm=True)."
        )
    widen = tuple(inst["widen"]) if inst.get("widen") else None
    return rs5k_zones(
        parse_sfz(sfz.read_text(encoding="utf-8")),
        base_dir=base,
        velocity=int(inst.get("velocity", 100)),
        widen=widen,  # type: ignore[arg-type]
    )
