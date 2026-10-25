"""Python side of the file-based JSON IPC bridge to REAPER.

Protocol (validated by shiehn/total-reaper-mcp and xDarkzx/Reaper-MCP):

    1. Assign a request_id that is unique across the whole process (see _next_request_id).
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
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

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

_ID_LOCK = threading.Lock()
_last_request_id = 0


def _next_request_id() -> int:
    """A request id no other caller in this process has used, and in practice no other process.

    Every tool call builds its own BridgeClient and FastMCP runs sync tools on a thread pool, so
    a per-client counter made parallel calls collide on ``request_1.json``. Ids are therefore
    process-wide and lock-protected, and they follow the wall clock in microseconds (about
    1.8e15, well under 2**53, so JSON and Lua read them as exact integers). A process started
    later starts higher than any id an earlier one has handed out, since the bridge can't take
    a call per microsecond. Still plain digits: the Lua loop scans ``^request_%d+%.json$``.
    """
    global _last_request_id
    with _ID_LOCK:
        _last_request_id = max(_last_request_id + 1, time.time_ns() // 1_000)
        return _last_request_id


# The bridge rewrites heartbeat.lock every tick by swapping in a new file. On Windows a read or
# stat that lands mid-swap fails for a moment: PermissionError while the replace runs, and
# FileNotFoundError between the Lua writer's remove and rename. Retry briefly before believing it.
_SWAP_ATTEMPTS = 5
_SWAP_PAUSE_S = 0.002
T = TypeVar("T")


def _through_swap(op: Callable[[], T]) -> T:
    for _ in range(_SWAP_ATTEMPTS - 1):
        try:
            return op()
        except (PermissionError, FileNotFoundError):
            time.sleep(_SWAP_PAUSE_S)
    return op()


def read_heartbeat(bridge_dir: Path | None = None) -> dict[str, Any] | None:
    """The heartbeat file's content. v2 (spec 2026-10-23 REAPER sync §2.4) is a JSON object with
    ``v``, ``t``, ``reaper_version``, ``project_name``, ``project_path``, ``playing`` and
    ``state_change_count``; the old bridge wrote a bare timestamp, returned as ``{"v": 1, "t": …}``.
    None when the file is missing or unreadable. Liveness is NOT decided here: that is the
    file's modification time (``BridgeClient.is_alive``)."""
    hb = Path(bridge_dir or DEFAULT_BRIDGE_DIR) / HEARTBEAT_FILE
    try:
        text = _through_swap(lambda: hb.read_text(encoding="utf-8")).strip()
    except OSError:
        return None
    try:
        return {"v": 1, "t": float(text)}
    except ValueError:
        pass
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


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
        self._request_id = 0  # the last id THIS client used (ids come from _next_request_id)
        self.bridge_dir.mkdir(parents=True, exist_ok=True)

    # -- health ----------------------------------------------------------- #

    def is_alive(self, max_age_s: float = HEARTBEAT_MAX_AGE_S) -> bool:
        """True if the in-REAPER loop has touched the heartbeat file recently."""
        try:
            hb = self.bridge_dir / HEARTBEAT_FILE
            age = time.time() - _through_swap(hb.stat).st_mtime
        except (FileNotFoundError, PermissionError):
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
        rid = _next_request_id()
        self._request_id = rid
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
                    data = json.loads(path.read_text(encoding="utf-8"))
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
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)  # atomic on POSIX; replace() on Windows
