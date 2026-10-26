-- Lua-side tests for the bridge's lifecycle (auto-start, 2026-10-25): the loop finds its own
-- file even when REAPER's __startup.lua started it (get_action_context then names
-- __startup.lua), and when a second copy starts, the older one stops, so requests are never
-- handled twice.
-- Run: lua tests/lua/test_lifecycle.lua (or scripts/run_lua_tests.py, via lupa).

local passed, failed = 0, 0
local function ok(cond, msg)
  if cond then passed = passed + 1
  else failed = failed + 1; io.stderr:write("FAIL: " .. tostring(msg) .. "\n") end
end
local function eq(a, b, msg)
  ok(a == b, (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end

local TMPDIR = (os.getenv("TEMP") or os.getenv("TMPDIR") or "/tmp"):gsub("\\", "/")
local ext = { orpheus = { bridge_dir = TMPDIR } }
local console, deferred = {}, {}
local now = 100.0

reaper = {
  GetExtState = function(section, key) return (ext[section] or {})[key] or "" end,
  SetExtState = function(section, key, value, persist)
    ext[section] = ext[section] or {}
    ext[section][key] = value
    ext.persisted = persist
  end,
  time_precise = function() now = now + 0.001; return now end,
  GetAppVersion = function() return "7.0/luatest" end,
  RecursiveCreateDirectory = function() end,
  ShowConsoleMsg = function(s) console[#console + 1] = s end,
  defer = function(fn) deferred[#deferred + 1] = fn end,
  EnumerateFiles = function() return nil end,
  -- As REAPER reports it when __startup.lua dofile()s the bridge: the startup script's path.
  get_action_context = function()
    return true, "C:/Users/x/AppData/Roaming/REAPER/Scripts/__startup.lua", 0, 0, 0, 0, 0
  end,
  GetProjectName = function() return "" end,
  EnumProjects = function() return "PROJ", "" end,
  GetProjectLength = function() return 0 end,
  GetPlayState = function() return 0 end,
}

local here = (arg and arg[0] or ""):match("^(.*[/\\])") or "./"
local BRIDGE = here .. "../../src/orpheus_mcp/bridge/lua/orpheus_bridge.lua"
local function load_bridge(autorun)
  _G.ORPHEUS_NO_AUTORUN = not autorun
  return assert(loadfile(BRIDGE))()
end
local function console_has(text)
  for _, line in ipairs(console) do if line:find(text, 1, true) then return true end end
  return false
end

-- 1. The bridge's own path comes from its chunk, not from the action that ran it.
local M = load_bridge(false)
local self_path = M.resolve_self_path()
ok(self_path and self_path:find("orpheus_bridge.lua", 1, true), "own path: " .. tostring(self_path))
ok(not tostring(self_path):find("__startup", 1, true), "never the startup script's path")

-- 2. Started (autorun): claims the instance, arms hot-reload on its own file, starts the loop.
console, deferred = {}, {}
local A = load_bridge(true)
local token_a = ext.orpheus.bridge_instance
ok(token_a and token_a ~= "", "the first copy claims the instance")
eq(ext.persisted, false, "the claim is not saved across REAPER restarts")
eq(#deferred, 1, "the first copy's loop is running")
ok(console_has("hot-reload armed"), "hot-reload armed on the bridge file")

-- 3. A second copy starts: it takes over, and the first stops at its next tick.
local B = load_bridge(true)
local token_b = ext.orpheus.bridge_instance
ok(token_b ~= token_a, "the second copy claims a new instance")
eq(#deferred, 2, "the second copy's loop is running")
local first_loop = deferred[1]
local before = #deferred
first_loop()
eq(#deferred, before, "the older loop does not schedule itself again")
ok(console_has("newer copy"), "the older copy says why it stopped")
deferred[2]()
eq(#deferred, before + 1, "the newer loop keeps running")

-- 4. A module loaded for hot reload (no autorun) never claims or stops anything.
load_bridge(false)
eq(ext.orpheus.bridge_instance, token_b, "a hot-reload load leaves the claim alone")

_G.ORPHEUS_NO_AUTORUN = true
os.remove(TMPDIR .. "/heartbeat.lock")
print(string.format("lua lifecycle: %d passed, %d failed", passed, failed))
os.exit(failed == 0 and 0 or 1)
