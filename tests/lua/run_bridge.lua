-- Headless driver for integration tests: runs the REAL orpheus_bridge.lua against a
-- stubbed `reaper` global, polling for ~8s. Usage: lua run_bridge.lua <bridge_dir>
-- (Outside REAPER, so reaper.* is stubbed; the bridge logic/JSON/IO is the real thing.)

local dir = assert(arg[1], "usage: run_bridge.lua <bridge_dir>")

local function list_dir(d)
  local files, p = {}, io.popen('ls -1 "' .. d .. '" 2>/dev/null')
  if p then for line in p:lines() do files[#files + 1] = line end; p:close() end
  return files
end

local _l = {}
reaper = {
  GetExtState = function(a, b) if a == "orpheus" and b == "bridge_dir" then return dir end return "" end,
  GetAppVersion = function() return "7.0/integration" end,
  RecursiveCreateDirectory = function() end,
  ShowConsoleMsg = function() end,
  defer = function() end,
  EnumerateFiles = function(d, i)
    if i <= 0 then _l = list_dir(d) end
    if i < 0 then return nil end
    return _l[i + 1]
  end,
}

_G.ORPHEUS_NO_AUTORUN = true
local here = arg[0]:match("(.*/)") or "./"
local M = assert(loadfile(here .. "../../src/orpheus_mcp/bridge/lua/orpheus_bridge.lua"))()

local deadline = os.time() + 8
while os.time() < deadline do
  M.tick()
  os.execute("sleep 0.02")
end
