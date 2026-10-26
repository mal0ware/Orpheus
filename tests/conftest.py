"""Shared test setup."""

from __future__ import annotations

import pytest


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
