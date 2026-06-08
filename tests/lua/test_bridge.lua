-- Lua-side tests for orpheus_bridge.lua. Run: lua tests/lua/test_bridge.lua
-- Stubs the `reaper` global and exercises JSON + the real request/dispatch/response cycle.

local passed, failed = 0, 0
local function ok(cond, msg)
  if cond then passed = passed + 1
  else failed = failed + 1; io.stderr:write("FAIL: " .. msg .. "\n") end
end
local function eq(a, b, msg) ok(a == b, (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")") end

-- temp bridge dir
local TMP = (os.getenv("TMPDIR") or "/tmp"):gsub("/$", "") .. "/orpheus_lua_test"
os.execute('rm -rf "' .. TMP .. '" && mkdir -p "' .. TMP .. '"')

-- file listing for the EnumerateFiles stub
local function list_dir(dir)
  local files, p = {}, io.popen('ls -1 "' .. dir .. '" 2>/dev/null')
  if p then for line in p:lines() do files[#files + 1] = line end; p:close() end
  return files
end

local _listing = {}
reaper = {
  GetExtState = function(a, b) if a == "orpheus" and b == "bridge_dir" then return TMP end return "" end,
  GetAppVersion = function() return "7.0/test" end,
  RecursiveCreateDirectory = function() end,
  ShowConsoleMsg = function() end,
  defer = function() end,
  EnumerateFiles = function(dir, idx)
    if idx <= 0 then _listing = list_dir(dir) end
    if idx < 0 then return nil end
    return _listing[idx + 1]
  end,
}

-- load the bridge WITHOUT starting the defer loop
_G.ORPHEUS_NO_AUTORUN = true
local here = arg[0]:match("(.*/)") or "./"
local M = assert(loadfile(here .. "../../src/orpheus_mcp/bridge/lua/orpheus_bridge.lua"))()
local json = M.json

local function write(name, text)
  local f = assert(io.open(TMP .. "/" .. name, "w")); f:write(text); f:close()
end
local function read(name)
  local f = io.open(TMP .. "/" .. name, "r"); if not f then return nil end
  local s = f:read("*a"); f:close(); return s
end
local function exists(name) local f = io.open(TMP .. "/" .. name, "r"); if f then f:close(); return true end return false end

-- 1. bridge dir resolved from ExtState
eq(M.bridge_dir, TMP, "bridge_dir resolves from GetExtState")

-- 2. JSON round-trips the exact request shape Python writes
local req = json.decode('{"id":1,"fn":"get_connection_status","params":{}}')
eq(req.id, 1, "decoded id is integer 1")
ok(math.type(req.id) == "integer", "id decodes as an INTEGER (not 1.0)")
eq(req.fn, "get_connection_status", "decoded fn")
ok(type(req.params) == "table", "decoded params is a table")

-- 3. JSON round-trips nested arrays/objects/values
local rt = json.decode(json.encode({ a = 1, b = { 2, 3 }, c = "x\"y", d = true, e = json.null }))
eq(rt.a, 1, "rt number"); eq(rt.b[2], 3, "rt nested array"); eq(rt.c, 'x"y', "rt escaped string"); eq(rt.d, true, "rt bool")

-- 4. full cycle: plant a request, tick once, read the response
write("request_1.json", '{"id":1,"fn":"get_connection_status","params":{}}')
M.tick()
ok(exists("response_1.json"), "response_1.json written (integer-id filename, NOT response_1.0.json)")
ok(not exists("request_1.json"), "request consumed")
ok(exists("heartbeat.lock"), "heartbeat touched")
local resp = json.decode(read("response_1.json"))
eq(resp.ok, true, "response ok")
eq(resp.result.reaper_version, "7.0/test", "response carries reaper version")
os.remove(TMP .. "/response_1.json")

-- 5. unknown fn -> ok=false
write("request_7.json", '{"id":7,"fn":"does_not_exist","params":{}}')
M.tick()
local r7 = json.decode(read("response_7.json"))
eq(r7.ok, false, "unknown fn -> ok=false")
ok(r7.error:match("unknown fn"), "unknown fn error message")
os.remove(TMP .. "/response_7.json")

-- 6. __batch__ runs each call and returns a list
write("request_9.json", '{"id":9,"fn":"__batch__","params":{"calls":[{"fn":"get_connection_status"},{"fn":"get_connection_status"}]}}')
M.tick()
local r9 = json.decode(read("response_9.json"))
eq(r9.ok, true, "batch ok")
eq(#r9.result, 2, "batch returns 2 results")
eq(r9.result[1].reaper_version, "7.0/test", "batch result[1]")

os.execute('rm -rf "' .. TMP .. '"')
print(string.format("lua bridge: %d passed, %d failed", passed, failed))
os.exit(failed == 0 and 0 or 1)
