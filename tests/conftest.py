"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_sound_pack(tmp_path_factory, monkeypatch):
    """Never let a real ~/.orpheus/soundpack on the dev machine change what the composers
    load: every test sees an empty pack folder unless it installs into one itself."""
    monkeypatch.setenv("ORPHEUS_SOUNDPACK_DIR", str(tmp_path_factory.mktemp("soundpack")))
