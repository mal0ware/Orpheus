# tests/test_soundpack.py
"""install_sound_pack: pinned + checksum-verified placement into a user-writable dir."""
from __future__ import annotations

import hashlib

import pytest

from orpheus_mcp.soundpack import PACK, install_sound_pack


def _content_for(item: dict) -> bytes:
    return b"CONTENT:" + item["filename"].encode()


def _fake_fetch(payloads):
    def fetch(url):
        return payloads[url]
    return fetch


@pytest.fixture(autouse=True)
def _pin_pack_checksums_to_fixture(monkeypatch):
    """PACK ships with `<PIN_BEFORE_SHIP>` placeholder checksums (Task 15 pins the real
    release values). This unit test must not depend on real bytes or the network, so for
    the duration of each test we pin every PACK entry's sha256 to match the deterministic
    fixture content used below — the point under test is the verify-then-place logic, not
    the real artifact bytes. monkeypatch.setitem restores the placeholder after each test.
    """
    for item in PACK:
        monkeypatch.setitem(item, "sha256", hashlib.sha256(_content_for(item)).hexdigest())


def test_installs_when_checksums_match(tmp_path):
    payloads = {item["url"]: _content_for(item) for item in PACK}
    got = install_sound_pack(tmp_path, fetch=_fake_fetch(payloads))
    assert got["installed"] is True
    for item in PACK:
        dest = tmp_path / item["filename"]
        assert dest.exists()
        assert dest.read_bytes() == _content_for(item)


def test_rejects_on_checksum_mismatch(tmp_path):
    bad = {item["url"]: b"tampered" for item in PACK}
    with pytest.raises(ValueError, match="checksum"):
        install_sound_pack(tmp_path, fetch=_fake_fetch(bad))
    for item in PACK:
        assert not (tmp_path / item["filename"]).exists()
