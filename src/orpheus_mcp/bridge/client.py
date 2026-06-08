"""Python side of the file-based JSON IPC bridge to REAPER.

Protocol (validated by shiehn/total-reaper-mcp and xDarkzx/Reaper-MCP):

    1. Assign a sequential request_id.
    2. Write <bridge_dir>/request_<id>.json   — ATOMICALLY (write .tmp then os.replace),
       so the in-REAPER poller never reads a half-written file. This is THE critical
       correctness rule of the whole bridge.
    3. Poll for <bridge_dir>/response_<id>.json, parse {ok, result, error}.
    4. Delete the response file.

Identity that crosses calls is always a stable GUID or index — NEVER a live REAPER
userdata pointer (it can't be serialized and goes stale).

See docs/architecture.md for the full rationale and the hardening checklist.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_BRIDGE_DIR = Path(os.environ.get("REAPER_MCP_BRIDGE_DIR", Path(tempfile.gettempdir()) / "orpheus_bridge"))
HEARTBEAT_FILE = "heartbeat.lock"
DEFAULT_TIMEOUT_S = 2.0
POLL_INTERVAL_S = 0.02

# Keep every single call bounded so REAPER's single-threaded audio engine never stalls.
MAX_NOTES_PER_CALL = 512


class BridgeError(RuntimeError):
    """A command reached REAPER but failed there (returned ok=false)."""


class BridgeTimeout(BridgeError):
    """REAPER/the bridge did not answer in time — usually a closed REAPER or dead script."""


class BridgeClient:
    """Talks to the in-REAPER Lua loop over the JSON file channel.

    Example
    -------
    >>> bridge = BridgeClient()
    >>> bridge.is_alive()
    True
    >>> bridge.call("get_project_info")          # doctest: +SKIP
    {'tempo': 120.0, 'time_signature': [4, 4], ...}
    """

    def __init__(self, bridge_dir: Path | None = None, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.bridge_dir = Path(bridge_dir) if bridge_dir else DEFAULT_BRIDGE_DIR
        self.timeout_s = timeout_s
        self._request_id = 0
        self.bridge_dir.mkdir(parents=True, exist_ok=True)

    # -- health ----------------------------------------------------------- #

    def is_alive(self, max_age_s: float = 3.0) -> bool:
        """True if the in-REAPER loop has touched the heartbeat file recently."""
        hb = self.bridge_dir / HEARTBEAT_FILE
        if not hb.exists():
            return False
        return (time.time() - hb.stat().st_mtime) <= max_age_s

    # -- core call -------------------------------------------------------- #

    def call(self, fn: str, **params: Any) -> Any:
        """Invoke a bridge function and return its result, or raise BridgeError/Timeout.

        TODO(M0): implement the atomic write + poll loop. Sketch:

            self._request_id += 1
            rid = self._request_id
            payload = {"id": rid, "fn": fn, "params": params}
            self._atomic_write(self.bridge_dir / f"request_{rid}.json", payload)
            resp = self._await_response(rid)          # poll, honor self.timeout_s
            if not resp["ok"]:
                raise BridgeError(resp.get("error", "unknown bridge error"))
            return resp["result"]
        """
        raise NotImplementedError("BridgeClient.call lands in M0 — see docs/roadmap.md")

    def batch(self, calls: list[dict[str, Any]]) -> list[Any]:
        """Bundle several primitive ops into ONE round-trip (defeats the ~10 ops/sec ceiling).

        One MCP round-trip should equal one musical intent — e.g. create a track and
        write 64 notes in a single bridge call rather than 65.
        """
        raise NotImplementedError("BridgeClient.batch lands in M0 — see docs/roadmap.md")

    # -- internals -------------------------------------------------------- #

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        """Write JSON via a temp file + os.replace so readers never see a partial file."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)  # atomic on POSIX; replace() on Windows

    def _await_response(self, request_id: int) -> dict[str, Any]:
        raise NotImplementedError("lands in M0")
