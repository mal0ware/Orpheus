# src/orpheus_mcp/soundpack.py
"""Optional curated sound pack: ONE BSD-licensed sfizz .vst3 + a CC0 SFZ patch, pinned by
URL + sha256, placed into a user-writable folder. Never runs automatically. No installer
.exe is executed; no licensed software is redistributed."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

# Pinned artifacts. Fill sha256 from the exact pinned release before shipping (Task 15).
PACK: list[dict] = [
    {
        "name": "sfizz",
        "filename": "sfizz.vst3",
        "url": "https://github.com/sfztools/sfizz/releases/download/PINNED/sfizz.vst3",
        "sha256": "<PIN_BEFORE_SHIP>",
        "license": "BSD-2-Clause",
    },
    {
        "name": "cc0-gm-patch",
        "filename": "orpheus_gm.sfz",
        "url": "https://example.org/cc0/orpheus_gm.sfz",  # CC0 patch, pinned in Task 15
        "sha256": "<PIN_BEFORE_SHIP>",
        "license": "CC0-1.0",
    },
]


def install_sound_pack(
    dest_dir: Path, fetch: Callable[[str], bytes] | None = None
) -> dict:
    """Download each pinned artifact, verify its sha256, and place it in dest_dir (which the
    caller has pointed REAPER at). Raises ValueError on any checksum mismatch — nothing is
    written for a mismatched item. `fetch` is injectable for tests."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if fetch is None:
        fetch = _http_get

    placed: list[str] = []
    for item in PACK:
        data = fetch(item["url"])
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(
                f"checksum mismatch for {item['name']}: got {digest}, want {item['sha256']}"
            )
        (dest_dir / item["filename"]).write_bytes(data)
        placed.append(item["filename"])
    return {"installed": True, "placed": placed, "dir": str(dest_dir)}


def _http_get(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (pinned https)
        return resp.read()
