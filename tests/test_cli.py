"""CLI entrypoint tests for `orpheus-mcp install-bridge`.

Regression: on Windows, a non-interactive console defaults to a legacy codepage
(cp1252) that cannot encode the "✓"/"→" in the success message — the install
succeeded but the CLI crashed with UnicodeEncodeError afterwards. The output must
degrade gracefully instead of crashing.
"""

from __future__ import annotations

import io
import sys


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
