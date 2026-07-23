"""``scripts/pin_models.py`` — model registry checkpoint pin run.

Task 0.2 (this task) ships the SKELETON: pin every checkpoint's sha256 by downloading
through :func:`orpheus_mcp.analysis.fetch_models.fetch_models` in ``--pin`` mode and
writing the recorded hashes back into ``data/model_registry.json``. That is the whole
hash-pinning half of the job, and it needs no GPU, no separation engine, and no owned
audio — it works today.

Task 0.3 FINISHES this script: for every separation/dereverb registry entry, additionally
run it against an owned audio file (``--audio``), write stems to ``docs/pins/audio_out/``
(gitignored), and record wall time + rt_factor (on the reference 3080) into both the
registry JSON and a markdown pin report (``docs/pins/2026-07-model-pins.md``) with a notes
column for the human ear-check (§12 step 3). That half requires the ``[separation]`` extra
and the vendored MSST SCNet path (Task 0.3 Step 1) — deliberately not built here.

Usage today:      ``uv run python scripts/pin_models.py``
Usage (Task 0.3): ``uv run --extra separation python scripts/pin_models.py --audio <owned.mp3>``
"""

from __future__ import annotations

import argparse
import json

from orpheus_mcp.analysis.fetch_models import fetch_models
from orpheus_mcp.analysis.model_registry import registry_path

# Statuses fetch_models() can return for an entry that isn't a real downloaded hash.
_NON_HASH_STATUSES = {"skipped (no checkpoint_url)"}


def pin_hashes() -> dict[str, str]:
    """Download every registry checkpoint and record its sha256 into the registry JSON.

    Returns the id -> sha256 map that was written (entries with no checkpoint_url, which
    have nothing to pin, are omitted).
    """
    results = fetch_models(pin=True)
    hashes = {
        model_id: digest
        for model_id, digest in results.items()
        if digest not in _NON_HASH_STATUSES
    }

    path = registry_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    for entry in raw:
        digest = hashes.get(entry["id"])
        if digest is not None:
            entry["sha256"] = digest
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        default=None,
        help="Owned audio file for the separation/dereverb pin run (Task 0.3; not yet "
        "implemented here).",
    )
    args = parser.parse_args()

    hashes = pin_hashes()
    for model_id, digest in hashes.items():
        print(f"{model_id}: {digest}")
    print(f"Pinned {len(hashes)} checkpoint hashes into {registry_path()}.")

    if args.audio is not None:
        raise NotImplementedError(
            "The separation/dereverb pin run (wall time, rt_factor, ear-check, pin report) "
            "is Task 0.3 — this skeleton only pins checkpoint hashes."
        )


if __name__ == "__main__":
    main()
