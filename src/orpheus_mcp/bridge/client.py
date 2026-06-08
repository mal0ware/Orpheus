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
import time
from pathlib import Path
from typing import Any

# Default lives under $HOME (stable across the two processes) rather than $TMPDIR, which
# can differ between the Python server and REAPER and silently break the rendezvous.
DEFAULT_BRIDGE_DIR = Path(
    os.environ.get("REAPER_MCP_BRIDGE_DIR", Path.home() / ".orpheus_bridge")
)
HEARTBEAT_FILE = "heartbeat.lock"
DEFAULT_TIMEOUT_S = 2.0
POLL_INTERVAL_S = 0.01
HEARTBEAT_MAX_AGE_S = 3.0

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
    >>> bridge.is_alive()                          # doctest: +SKIP
    True
    >>> bridge.call("get_project_info")            # doctest: +SKIP
    {'tempo': 120.0, 'time_signature': [4, 4], ...}
    """

    def __init__(
        self, bridge_dir: Path | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> None:
        self.bridge_dir = Path(bridge_dir) if bridge_dir else DEFAULT_BRIDGE_DIR
        self.timeout_s = timeout_s
        self._request_id = 0
        self.bridge_dir.mkdir(parents=True, exist_ok=True)

    # -- health ----------------------------------------------------------- #

    def is_alive(self, max_age_s: float = HEARTBEAT_MAX_AGE_S) -> bool:
        """True if the in-REAPER loop has touched the heartbeat file recently."""
        try:
            age = time.time() - (self.bridge_dir / HEARTBEAT_FILE).stat().st_mtime
        except FileNotFoundError:
            return False
        return age <= max_age_s

    # -- core call -------------------------------------------------------- #

    def call(self, fn: str, **params: Any) -> Any:
        """Invoke a bridge function and return its result, or raise BridgeError/Timeout."""
        return self._dispatch(fn, params)

    def batch(self, calls: list[dict[str, Any]]) -> list[Any]:
        """Bundle several primitive ops into ONE round-trip (defeats the ~10 ops/sec ceiling).

        ``calls`` is a list of ``{"fn": ..., "params": {...}}``. Returns the list of results
        in order. One MCP round-trip should equal one musical intent.
        """
        return self._dispatch("__batch__", {"calls": calls})

    # -- internals -------------------------------------------------------- #

    def _dispatch(self, fn: str, params: dict[str, Any]) -> Any:
        if not self.is_alive():
            raise BridgeTimeout(
                f"REAPER bridge not running — no heartbeat in {self.bridge_dir}. "
                "Run orpheus_bridge.lua inside REAPER."
            )
        self._request_id += 1
        rid = self._request_id
        self._atomic_write(
            self.bridge_dir / f"request_{rid}.json", {"id": rid, "fn": fn, "params": params}
        )
        resp = self._await_response(rid)
        if not resp.get("ok"):
            raise BridgeError(resp.get("error", "unknown bridge error"))
        return resp.get("result")

    def _await_response(self, request_id: int) -> dict[str, Any]:
        path = self.bridge_dir / f"response_{request_id}.json"
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                except (json.JSONDecodeError, OSError):
                    # Caught the file mid-write — retry on the next tick.
                    time.sleep(POLL_INTERVAL_S)
                    continue
                path.unlink(missing_ok=True)
                return data
            time.sleep(POLL_INTERVAL_S)
        # Clean up our orphaned request so a late bridge doesn't act on a stale command.
        (self.bridge_dir / f"request_{request_id}.json").unlink(missing_ok=True)
        raise BridgeTimeout(
            f"No response to request {request_id} ('{path.name}') within {self.timeout_s}s"
        )

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        """Write JSON via a temp file + os.replace so readers never see a partial file.

        Safe because each path has a single writer with a unique id (request_<id> from the
        client, response_<id> from the bridge), so temp names never collide in practice.
        """
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)  # atomic on POSIX; replace() on Windows
