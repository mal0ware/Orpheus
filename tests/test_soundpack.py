# tests/test_soundpack.py
"""The opt-in FreePats sound pack: manifest integrity, verified install, zones from the SFZ.

No network: ``fetch`` is injected, and the fake serves bytes whose sha256 a test-local manifest
records. The real manifest is checked for shape and licence only.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import pytest

from orpheus_mcp import soundpack

FIX = Path(__file__).resolve().parent / "fixtures" / "sfz"


# --------------------------------------------------------------------------- real manifest
def test_real_manifest_is_cc0_pinned_and_complete():
    m = soundpack.load_manifest()
    assert m["license"] == "CC0-1.0"
    assert {i["role"] for i in m["instruments"]} == {"guitar", "bass", "keys"}
    for inst in m["instruments"]:
        assert re.fullmatch(r"freepats/[\w-]+", inst["repo"])
        assert re.fullmatch(r"[0-9a-f]{40}", inst["commit"])  # a commit, never a branch
        paths = [f["path"] for f in inst["files"]]
        assert inst["sfz"] in paths and any(p.startswith("LICENSE") for p in paths)
        for f in inst["files"]:
            assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] > 0
    total = sum(f["size"] for i in m["instruments"] for f in i["files"])
    assert 20e6 < total < 30e6  # the approved ~24.5 MB pack


def test_real_manifest_pins_every_sample_its_sfz_needs():
    m = soundpack.load_manifest()
    for inst in m["instruments"]:
        sfz_file = FIX / inst["sfz"]
        zones = soundpack.rs5k_zones(
            soundpack.parse_sfz(sfz_file.read_text(encoding="utf-8")),
            base_dir=Path("."), velocity=inst["velocity"],
        )
        pinned = {str(Path(f["path"])) for f in inst["files"]}
        assert {str(Path(z["file"])) for z in zones} <= pinned


def test_raw_url_escapes_spaces_and_hash():
    sha = "a" * 40
    url = soundpack.raw_url("freepats/electric-bass-YR", sha, "samples/pick/F#.flac")
    assert url == (
        f"https://raw.githubusercontent.com/freepats/electric-bass-YR/{sha}/samples/pick/F%23.flac"
    )
    spaced = soundpack.raw_url("r/x", "c", "PickedBassYR 20190930.sfz")
    assert spaced.endswith("/PickedBassYR%2020190930.sfz")


# --------------------------------------------------------------------------- install
def _pack(tmp_files: dict[str, bytes]) -> tuple[dict, dict[str, bytes]]:
    """A one-instrument manifest over the real bass SFZ plus fake sample bytes."""
    sfz = (FIX / "PickedBassYR 20190930.sfz").read_bytes()
    files = {"PickedBassYR 20190930.sfz": sfz, **tmp_files}
    manifest = {
        "instruments": [{
            "id": "bass", "role": "bass", "name": "test bass", "repo": "freepats/x",
            "commit": "c" * 40, "sfz": "PickedBassYR 20190930.sfz", "velocity": 100,
            "widen": [24, 55],
            "files": [{"path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                      for p, b in files.items()],
        }]
    }
    served = {soundpack.raw_url("freepats/x", "c" * 40, p): b for p, b in files.items()}
    return manifest, served


def test_install_verifies_then_skips_what_is_already_there(tmp_path):
    manifest, served = _pack({"samples/pick/E.flac": b"E", "samples/pick/F#.flac": b"F#"})
    calls: list[str] = []

    def fetch(url):
        calls.append(url)
        return served[url]

    got = soundpack.install_sound_pack(tmp_path, fetch=fetch, manifest=manifest)
    assert got["downloaded_files"] == 3 and got["already_present"] == 0
    assert (tmp_path / "bass" / "samples" / "pick" / "F#.flac").read_bytes() == b"F#"

    again = soundpack.install_sound_pack(tmp_path, fetch=fetch, manifest=manifest)
    assert again["downloaded_files"] == 0 and again["already_present"] == 3
    assert len(calls) == 3
    assert [i["id"] for i in soundpack.installed_instruments(tmp_path, manifest)] == ["bass"]


def test_install_rejects_a_tampered_file_and_writes_nothing_for_it(tmp_path):
    manifest, served = _pack({"samples/pick/E.flac": b"E"})

    def fetch(url):
        return b"tampered" if url.endswith("E.flac") else served[url]

    with pytest.raises(ValueError, match="checksum mismatch for bass/samples/pick/E.flac"):
        soundpack.install_sound_pack(tmp_path, fetch=fetch, manifest=manifest)
    assert not (tmp_path / "bass" / "samples" / "pick" / "E.flac").exists()
    assert soundpack.installed_instruments(tmp_path, manifest) == []


# --------------------------------------------------------------------------- zones
def test_zones_for_an_installed_instrument_use_absolute_paths_and_widen(tmp_path):
    manifest, _ = _pack({})
    dest = tmp_path / "bass"
    dest.mkdir()
    shutil.copy(FIX / "PickedBassYR 20190930.sfz", dest / "PickedBassYR 20190930.sfz")
    zones = soundpack.zones_for("bass", root=tmp_path, manifest=manifest)
    assert len(zones) == 13
    assert zones[0]["note_lo"] == 24 and zones[-1]["note_hi"] == 55
    assert Path(zones[0]["file"]) == dest / "samples" / "pick" / "E.flac"


def test_zones_for_explains_a_missing_install(tmp_path):
    manifest, _ = _pack({})
    with pytest.raises(FileNotFoundError, match="install_sound_pack"):
        soundpack.zones_for("bass", root=tmp_path, manifest=manifest)
    with pytest.raises(ValueError, match="unknown sound-pack instrument"):
        soundpack.zones_for("tuba", root=tmp_path, manifest=manifest)


def test_pack_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ORPHEUS_SOUNDPACK_DIR", str(tmp_path))
    assert soundpack.pack_dir() == tmp_path
    monkeypatch.delenv("ORPHEUS_SOUNDPACK_DIR")
    assert soundpack.pack_dir() == Path.home() / ".orpheus" / "soundpack"
