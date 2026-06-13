-- Headless driver for integration tests: runs the REAL orpheus_bridge.lua against a
-- stubbed `reaper` global, polling for ~8s. Usage: lua run_bridge.lua <bridge_dir>
-- (Outside REAPER, so reaper.* is stubbed; the bridge logic/JSON/IO is the real thing.)

local dir = assert(arg[1], "usage: run_bridge.lua <bridge_dir>")

local IS_WINDOWS = package.config:sub(1, 1) == "\\"

local function list_dir(d)
  -- Portable directory listing for the EnumerateFiles stub (REAPER provides the real one).
  local cmd = IS_WINDOWS and ('dir /b "' .. d .. '" 2>nul') or ('ls -1 "' .. d .. '" 2>/dev/null')
  local files, p = {}, io.popen(cmd)
  if p then for line in p:lines() do files[#files + 1] = line end; p:close() end
  return files
end

local function nap()
  os.execute(IS_WINDOWS and "ping -n 1 -w 20 127.0.0.1 >nul" or "sleep 0.02")
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
local here = arg[0]:match("(.*[/\\])") or ("." .. package.config:sub(1, 1))
local M = assert(loadfile(here .. "../../src/orpheus_mcp/bridge/lua/orpheus_bridge.lua"))()

local deadline = os.time() + 8
while os.time() < deadline do
  M.tick()
  nap()
end
