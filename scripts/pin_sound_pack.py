"""Pin the FreePats sound pack: download each file once, record size + sha256 in
data/sound_pack.json. Run by a maintainer, not by users:

    uv run --no-sync python scripts/pin_sound_pack.py <scratch-dir>

Which files get pinned is decided by each instrument's own SFZ: the samples of the chosen
velocity layer, plus the SFZ, licence and readme. Every download is cross-checked against the
git blob SHA-1 from GitHub's tree API, an independent second hash.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orpheus_mcp.sfz import parse_sfz, select_layer  # noqa: E402
from orpheus_mcp.soundpack import _http_get, raw_url  # noqa: E402

PINS = [
    {
        "id": "guitar",
        "role": "guitar",
        "name": "Spanish Classical Guitar (FreePats, 2019-06-18)",
        "repo": "freepats/spanish-classical-guitar",
        "commit": "6f4eb1b092acc88f5448cea1a0001bd07b971af8",
        "sfz": "SpanishClassicalGuitar-20190618.sfz",
        "velocity": 100,
        "widen": None,
        "extra": ["LICENSE", "README.md"],
    },
    {
        "id": "bass",
        "role": "bass",
        "name": "Picked Electric Bass YR (FreePats, 2019-09-30)",
        "repo": "freepats/electric-bass-YR",
        "commit": "8dcb7ea9116f417273ef8c030d15e7b3aa654301",
        "sfz": "PickedBassYR 20190930.sfz",
        "velocity": 100,
        "widen": [24, 55],  # recorded E1..A#2 (26-46); stretch to a 4-string bass's range
        "extra": ["LICENSE.txt", "README.txt"],
    },
    {
        "id": "piano",
        "role": "keys",
        "name": "Upright Piano KW, loud layer (FreePats, 2022-02-21)",
        "repo": "freepats/upright-piano-KW",
        "commit": "570f6c60ed2eff67accad3b85d5b452e57a3ad28",
        "sfz": "UprightPianoKW-20220221.sfz",
        "velocity": 100,  # lovel=81 groups: the loud layer only (RS5k zones have no velocity)
        "widen": None,
        "extra": ["LICENSE", "README.md"],
    },
]


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 (git id)


def _tree(repo: str, commit: str) -> dict[str, str]:
    out = subprocess.run(
        ["gh", "api", f"repos/{repo}/git/trees/{commit}?recursive=1",
         "--jq", r'.tree[] | select(.type=="blob") | "\(.sha) \(.path)"'],
        capture_output=True, encoding="utf-8", check=True,
    ).stdout
    return {line[41:]: line[:40] for line in out.splitlines() if line}


def main(scratch: Path) -> None:
    instruments = []
    for pin in PINS:
        tree = _tree(pin["repo"], pin["commit"])
        sfz_text = _http_get(raw_url(pin["repo"], pin["commit"], pin["sfz"])).decode("utf-8")
        samples = sorted({r.sample for r in select_layer(parse_sfz(sfz_text), pin["velocity"])})
        files = []
        for path in [pin["sfz"], *pin["extra"], *samples]:
            data = _http_get(raw_url(pin["repo"], pin["commit"], path))
            if _git_blob_sha1(data) != tree[path]:
                raise SystemExit(f"{pin['repo']}:{path}: git blob hash mismatch")
            dest = scratch / pin["id"] / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            files.append({"path": path, "size": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
        entry = {k: v for k, v in pin.items() if k != "extra"}
        entry["files"] = files
        instruments.append(entry)
        total = sum(f["size"] for f in files)
        print(f"{pin['id']}: {len(files)} files, {total:,} bytes", flush=True)

    manifest = {
        "schema_version": 1,
        "source": "FreePats, https://freepats.zenvoid.org (GitHub mirrors at pinned commits)",
        "license": "CC0-1.0",
        "license_note": "Every instrument here is published under the Creative Commons CC0 1.0 "
                        "public domain dedication.",
        "instruments": instruments,
    }
    out = REPO / "data" / "sound_pack.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
