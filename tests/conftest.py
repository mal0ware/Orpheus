"""Shared test setup."""

from __future__ import annotations

import importlib.util

import pytest

_HAS_LIBROSA = importlib.util.find_spec("librosa") is not None


def pytest_collection_modifyitems(config, items):
    """Tests marked ``analysis`` need librosa. On a base install (CI's dev-extra-only job,
    which proves the server degrades cleanly) they skip instead of failing."""
    if _HAS_LIBROSA:
        return
    skip = pytest.mark.skip(reason="needs the [analysis] extra (librosa)")
    for item in items:
        if item.get_closest_marker("analysis") is not None:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_sound_pack(tmp_path_factory, monkeypatch):
    """Never let a real ~/.orpheus/soundpack on the dev machine change what the composers
    load: every test sees an empty pack folder unless it installs into one itself."""
    monkeypatch.setenv("ORPHEUS_SOUNDPACK_DIR", str(tmp_path_factory.mktemp("soundpack")))


@pytest.fixture(autouse=True)
def _isolated_nc_backends(tmp_path_factory, monkeypatch):
    """A real ORPHEUS_NC_BACKENDS, YuE2 install or sketch folder on the dev machine never
    changes the tool surface or receives test renders."""
    monkeypatch.delenv("ORPHEUS_NC_BACKENDS", raising=False)
    monkeypatch.setenv("ORPHEUS_YUE2_DIR", str(tmp_path_factory.mktemp("yue2")))
    monkeypatch.setenv("ORPHEUS_SKETCH_DIR", str(tmp_path_factory.mktemp("sketches")))


@pytest.fixture(autouse=True)
def _no_bridge_autostart(monkeypatch):
    """Tests never ask the real REAPER on this machine to start a bridge: automatic starts are
    off unless a test switches them on with its own stand-ins."""
    monkeypatch.setenv("ORPHEUS_AUTOSTART_BRIDGE", "0")
    from orpheus_mcp.bridge import launch

    launch.reset_backoff()
