"""Fixture-forge tests (Task 0.4).

CI-safe (no audio libs, no sfizz, no network):
  * every committed CI file's sha256 matches ``manifest.json``;
  * ``fixture_set_hash`` recomputes from the file hashes (the §15.1/§9 calibration identity);
  * both CI fixtures have the required layout + ``truth.json`` keys (role map / bpm / key /
    progression; ``ci_vocal`` also lyric lines).

Desktop-only:
  * re-rendering ``ci_sparse``'s instrument stems reproduces the committed hashes bit-for-bit
    (sfizz_render + FLAC are deterministic). The SAPI vocal lane (D8) is never re-rendered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

AUDIO_ROOT = Path(__file__).resolve().parent / "fixtures" / "audio"
MANIFEST = AUDIO_ROOT / "manifest.json"
CI_FIXTURES = ("ci_sparse", "ci_vocal")
REQUIRED_TRUTH_KEYS = {"roles", "bpm", "key", "progression"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text("utf-8"))


def test_manifest_exists_and_covers_both_fixtures() -> None:
    manifest = _load_manifest()
    covered = {rel.split("/", 1)[0] for rel in manifest["files"]}
    assert set(CI_FIXTURES) <= covered


def test_manifest_hashes_match_committed_files() -> None:
    manifest = _load_manifest()
    assert manifest["files"], "manifest lists no files"
    for rel, want in manifest["files"].items():
        path = AUDIO_ROOT / rel
        assert path.exists(), f"manifest references missing file: {rel}"
        assert _sha256(path) == want, f"sha256 drift for {rel}"


def test_no_uncommitted_extra_audio_files() -> None:
    manifest = _load_manifest()
    listed = set(manifest["files"])
    on_disk = {
        p.relative_to(AUDIO_ROOT).as_posix()
        for p in AUDIO_ROOT.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    assert on_disk == listed


def test_fixture_set_hash_recomputes() -> None:
    manifest = _load_manifest()
    recomputed = hashlib.sha256(
        "".join(sorted(manifest["files"].values())).encode("utf-8")
    ).hexdigest()
    assert recomputed == manifest["fixture_set_hash"]


@pytest.mark.parametrize("fixture", CI_FIXTURES)
def test_fixture_schema(fixture: str) -> None:
    fixture_dir = AUDIO_ROOT / fixture
    assert (fixture_dir / "mix.flac").is_file()
    stems = fixture_dir / "stems"
    assert stems.is_dir()
    stem_files = list(stems.glob("*.flac"))
    assert stem_files, f"{fixture} has no stems"

    truth = json.loads((fixture_dir / "truth.json").read_text("utf-8"))
    assert set(truth) >= REQUIRED_TRUTH_KEYS
    assert {"root", "mode"} <= set(truth["key"])
    # Every declared role maps to a stem file that exists.
    for role, rel in truth["roles"].items():
        assert (fixture_dir / rel).is_file(), f"{fixture} role {role} -> missing {rel}"


def test_ci_vocal_carries_lyric_lines() -> None:
    truth = json.loads((AUDIO_ROOT / "ci_vocal" / "truth.json").read_text("utf-8"))
    lyrics = truth["lyrics"]
    assert len(lyrics["lines"]) == 6
    assert all(isinstance(line, str) and line.strip() for line in lyrics["lines"])
    assert "vocals" in truth["roles"]
    # Placements are on the bar grid and ordered.
    starts = [p["start_s"] for p in lyrics["placements"]]
    assert starts == sorted(starts)


def test_ci_sparse_has_no_vocal_lane() -> None:
    truth = json.loads((AUDIO_ROOT / "ci_sparse" / "truth.json").read_text("utf-8"))
    assert "vocals" not in truth["roles"]
    assert "lyrics" not in truth


@pytest.mark.desktop
def test_ci_sparse_rerender_is_deterministic(tmp_path: Path) -> None:
    """Re-render ci_sparse's instrument stems and compare to the committed manifest hashes."""
    import scripts.make_fixtures as mf

    manifest = _load_manifest()
    ctx = mf.make_render_ctx(mf.ASSETS_DIR / "scratch")
    mf.build_ci_instrumental(ctx, tmp_path)
    for rel, want in manifest["files"].items():
        if not rel.startswith("ci_sparse/") or rel.endswith("truth.json"):
            continue  # truth.json + vocal lane are not part of the determinism contract
        assert _sha256(tmp_path / rel) == want, f"non-deterministic re-render: {rel}"
