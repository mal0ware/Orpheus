"""Run the Lua-side bridge handler suites in-process via lupa — no lua interpreter needed.

This is how the dev-log verified the Lua half on machines without a standalone `lua`
(see docs/dev-log.md, 2026-07-14): the suites are plain Lua 5.4 files, and lupa embeds a
real Lua runtime inside Python. Two interceptions make that safe:

- ``os.exit`` would kill the HOST Python process; we replace it with a recorder (each
  suite calls it exactly once, at the end, with its pass/fail code).
- ``arg[0]`` (the script path) is normally set by the lua CLI; the suites use it to
  locate orpheus_bridge.lua relative to themselves, so we provide it.

test_bridge.lua is excluded here on EVERY platform: it drives its temp bridge dir
through ``io.popen`` (rm/ls), which lupa's embedded runtime does not support (proved
by the first ubuntu CI run: 'popen' not supported). That suite needs a standalone lua
interpreter; the file-IPC loop it exercises is already covered end-to-end by the
Python integration tests against the behavioural fake and by the 2026-07-15 live
REAPER smoke (dev-log). The self-contained M1 handler suite runs everywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LUA_DIR = REPO / "tests" / "lua"

SUITES = ["test_m1_handlers.lua"]


def run_suite(suite_path: Path) -> int:
    from lupa import LuaRuntime

    lua = LuaRuntime(unpack_returned_tuples=True)
    exit_codes: list[int] = []
    lua_globals = lua.globals()
    lua_globals.arg = lua.table_from({0: str(suite_path)})
    lua_globals.os.exit = lambda code=0: exit_codes.append(int(code))
    lua.execute(suite_path.read_text(encoding="utf-8"))
    return exit_codes[-1] if exit_codes else 0


def main() -> int:
    worst = 0
    for name in SUITES:
        print(f"== {name} (via lupa) ==", flush=True)
        code = run_suite(LUA_DIR / name)
        if code != 0:
            print(f"FAILED: {name} exited {code}", file=sys.stderr)
        worst = max(worst, code)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
