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
