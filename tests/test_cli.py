"""CLI entrypoint tests for `orpheus-mcp install-bridge` and `orpheus-mcp fetch-models`.

Regression: on Windows, a non-interactive console defaults to a legacy codepage
(cp1252) that cannot encode the "✓"/"→" in the success message — the install
succeeded but the CLI crashed with UnicodeEncodeError afterwards. The output must
degrade gracefully instead of crashing. `fetch-models` reuses the same degradation
pattern and dispatch style, so its tests live here too.
"""

from __future__ import annotations

import io
import sys

import pytest


def test_install_bridge_cli_survives_cp1252_console(tmp_path, monkeypatch):
    from orpheus_mcp import server

    resource = tmp_path / "REAPER"
    resource.mkdir()
    monkeypatch.setattr(
        "orpheus_mcp.install.find_reaper_resource_dir", lambda: resource
    )
    monkeypatch.setattr(sys, "argv", ["orpheus-mcp", "install-bridge"])

    buf = io.BytesIO()
    cp1252_stdout = io.TextIOWrapper(buf, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", cp1252_stdout)

    server.main()  # must not raise UnicodeEncodeError

    cp1252_stdout.flush()
    out = buf.getvalue().decode("cp1252")
    assert "Installed Orpheus bridge" in out
    assert (resource / "Scripts" / "orpheus" / "orpheus_bridge.lua").is_file()


def test_fetch_models_cli_dispatches_and_reports(monkeypatch):
    from orpheus_mcp import server

    calls: dict[str, object] = {}

    def fake_fetch_models(*, pin: bool):
        calls["pin"] = pin
        return {"fake/model": "verified"}

    monkeypatch.setattr(
        "orpheus_mcp.analysis.fetch_models.fetch_models", fake_fetch_models
    )
    monkeypatch.setattr(sys, "argv", ["orpheus-mcp", "fetch-models"])

    buf = io.BytesIO()
    cp1252_stdout = io.TextIOWrapper(buf, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", cp1252_stdout)

    server.main()  # must not raise UnicodeEncodeError

    cp1252_stdout.flush()
    out = buf.getvalue().decode("cp1252")
    assert calls["pin"] is False
    assert "fake/model: verified" in out
    assert "fetch-models complete" in out


def test_fetch_models_cli_pin_flag_forwarded(monkeypatch):
    from orpheus_mcp import server

    calls: dict[str, object] = {}

    def fake_fetch_models(*, pin: bool):
        calls["pin"] = pin
        return {}

    monkeypatch.setattr(
        "orpheus_mcp.analysis.fetch_models.fetch_models", fake_fetch_models
    )
    monkeypatch.setattr(sys, "argv", ["orpheus-mcp", "fetch-models", "--pin"])

    server.main()
    assert calls["pin"] is True


def test_fetch_models_cli_reports_ffmpeg_missing_and_exits(monkeypatch):
    from orpheus_mcp import server
    from orpheus_mcp.analysis.fetch_models import FfmpegMissing

    def fake_fetch_models(*, pin: bool):
        raise FfmpegMissing("ffmpeg not found; winget install ffmpeg")

    monkeypatch.setattr(
        "orpheus_mcp.analysis.fetch_models.fetch_models", fake_fetch_models
    )
    monkeypatch.setattr(sys, "argv", ["orpheus-mcp", "fetch-models"])

    with pytest.raises(SystemExit) as exc_info:
        server.main()
    assert exc_info.value.code == 1


# -- auto-start commands (2026-10-25) --------------------------------------------------- #

def _cli(monkeypatch, *argv):
    from orpheus_mcp import server

    monkeypatch.setattr(sys, "argv", ["orpheus-mcp", *argv])
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = 0
    try:
        server.main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    return code, buf.getvalue()


@pytest.fixture
def resource(tmp_path, monkeypatch):
    res = tmp_path / "REAPER"
    res.mkdir()
    monkeypatch.setattr("orpheus_mcp.install.find_reaper_resource_dir", lambda: res)
    return res


def test_install_bridge_sets_up_autostart(monkeypatch, resource):
    code, out = _cli(monkeypatch, "install-bridge")
    assert code == 0 and "__startup.lua" in out and "starts with REAPER" in out
    assert (resource / "Scripts" / "__startup.lua").is_file()


def test_install_bridge_no_autostart(monkeypatch, resource):
    code, out = _cli(monkeypatch, "install-bridge", "--no-autostart")
    assert code == 0 and not (resource / "Scripts" / "__startup.lua").exists()


def test_remove_autostart_command(monkeypatch, resource):
    _cli(monkeypatch, "install-bridge")
    code, out = _cli(monkeypatch, "remove-autostart")
    assert code == 0 and "Removed" in out
    assert not (resource / "Scripts" / "__startup.lua").exists()
    code, out = _cli(monkeypatch, "remove-autostart")
    assert code == 0 and "nothing to remove" in out


@pytest.mark.parametrize(("status", "code"), [("started", 0), ("already_running", 0),
                                              ("reaper_not_open", 1), ("no_heartbeat", 1)])
def test_start_bridge_command_reports_and_exits(monkeypatch, status, code):
    monkeypatch.setattr("orpheus_mcp.bridge.launch.start_bridge",
                        lambda *a, **k: {"status": status, "message": f"msg {status}"})
    got, out = _cli(monkeypatch, "start-bridge")
    assert got == code and f"msg {status}" in out
