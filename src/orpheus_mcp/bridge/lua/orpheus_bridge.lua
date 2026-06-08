--[[
  orpheus_bridge.lua — the in-REAPER half of the Orpheus bridge.

  Run this once per REAPER session (Actions -> Show action list -> Run ReaScript, or a
  toolbar button). It installs a persistent background poll loop via reaper.defer() that:

    1. Touches a heartbeat lock-file so the Python side knows REAPER is alive.
    2. Scans <bridge_dir> for request_N.json (cache-invalidated listing).
    3. Dispatches the requested function through a STATIC table (no loadstring/eval).
    4. Writes response_N.json ATOMICALLY (tmp then rename) and deletes the request.

  Hardening (docs/architecture.md): static dispatch, atomic writes, EnumerateFiles cache
  invalidation, per-call work bounded, stable GUID/index identity (never live pointers).

  Testable: set _G.ORPHEUS_NO_AUTORUN = true before loading to get the module table back
  without starting the defer loop (see tests/lua/test_bridge.lua).
--]]

local M = {}

-- --------------------------------------------------------------------------- --
-- Minimal JSON (encode/decode) — REAPER's Lua ships no JSON library.
-- --------------------------------------------------------------------------- --

local json = {}

local ESCAPES = { ['"'] = '\\"', ['\\'] = '\\\\', ['\n'] = '\\n', ['\r'] = '\\r',
                  ['\t'] = '\\t', ['\b'] = '\\b', ['\f'] = '\\f' }

local function encode_str(s)
  return '"' .. s:gsub('[%z\1-\31\\"]', function(c)
    return ESCAPES[c] or string.format('\\u%04x', c:byte())
  end) .. '"'
end

local function is_array(t)
  local n = 0
  for k in pairs(t) do
    if type(k) ~= "number" then return false end
    n = n + 1
  end
  return n == #t
end

local function encode_value(v)
  local tv = type(v)
  if v == nil or v == json.null then
    return "null"
  elseif tv == "boolean" then
    return tostring(v)
  elseif tv == "number" then
    if math.type and math.type(v) == "integer" then return string.format("%d", v) end
    return string.format("%.14g", v)
  elseif tv == "string" then
    return encode_str(v)
  elseif tv == "table" then
    local parts = {}
    if next(v) == nil then return "{}" end  -- empty table -> object
    if is_array(v) then
      for i = 1, #v do parts[i] = encode_value(v[i]) end
      return "[" .. table.concat(parts, ",") .. "]"
    end
    for k, val in pairs(v) do
      parts[#parts + 1] = encode_str(tostring(k)) .. ":" .. encode_value(val)
    end
    return "{" .. table.concat(parts, ",") .. "}"
  end
  error("cannot encode type: " .. tv)
end

json.null = setmetatable({}, { __tostring = function() return "null" end })
function json.encode(v) return encode_value(v) end

-- Recursive-descent decoder.
local decode_value  -- forward decl

local function skip_ws(s, i)
  local _, j = s:find("^[ \t\r\n]*", i)
  return (j or i - 1) + 1
end

local function decode_str(s, i)
  i = i + 1  -- skip opening quote
  local buf = {}
  while i <= #s do
    local c = s:sub(i, i)
    if c == '"' then
      return table.concat(buf), i + 1
    elseif c == "\\" then
      local e = s:sub(i + 1, i + 1)
      local map = { ['"'] = '"', ["\\"] = "\\", ["/"] = "/", n = "\n", r = "\r",
                    t = "\t", b = "\b", f = "\f" }
      if e == "u" then
        local hex = s:sub(i + 2, i + 5)
        buf[#buf + 1] = utf8 and utf8.char(tonumber(hex, 16)) or string.char(tonumber(hex, 16) % 256)
        i = i + 6
      else
        buf[#buf + 1] = map[e] or e
        i = i + 2
      end
    else
      buf[#buf + 1] = c
      i = i + 1
    end
  end
  error("unterminated string")
end

local function decode_object(s, i)
  local obj, first = {}, true
  i = skip_ws(s, i + 1)
  if s:sub(i, i) == "}" then return obj, i + 1 end
  while true do
    i = skip_ws(s, i)
    local key; key, i = decode_str(s, i)
    i = skip_ws(s, i)
    assert(s:sub(i, i) == ":", "expected ':'")
    i = skip_ws(s, i + 1)
    local val; val, i = decode_value(s, i)
    obj[key] = val
    i = skip_ws(s, i)
    local c = s:sub(i, i)
    if c == "}" then return obj, i + 1 end
    assert(c == ",", "expected ',' or '}'")
    i = i + 1
    first = first
  end
end

local function decode_array(s, i)
  local arr = {}
  i = skip_ws(s, i + 1)
  if s:sub(i, i) == "]" then return arr, i + 1 end
  while true do
    local val; val, i = decode_value(s, skip_ws(s, i))
    arr[#arr + 1] = val
    i = skip_ws(s, i)
    local c = s:sub(i, i)
    if c == "]" then return arr, i + 1 end
    assert(c == ",", "expected ',' or ']'")
    i = i + 1
  end
end

decode_value = function(s, i)
  i = skip_ws(s, i)
  local c = s:sub(i, i)
  if c == "{" then return decode_object(s, i)
  elseif c == "[" then return decode_array(s, i)
  elseif c == '"' then return decode_str(s, i)
  elseif c == "t" then return true, i + 4
  elseif c == "f" then return false, i + 5
  elseif c == "n" then return json.null, i + 4
  else
    local num = s:match("^%-?%d+%.?%d*[eE]?[%+%-]?%d*", i)
    assert(num and num ~= "", "unexpected char at " .. i .. ": " .. c)
    return tonumber(num), i + #num
  end
end

function json.decode(s) local v = decode_value(s, 1); return v end

M.json = json

-- --------------------------------------------------------------------------- --
-- Paths
-- --------------------------------------------------------------------------- --

local SEP = package.config:sub(1, 1)

local function default_bridge_dir()
  -- Stable across processes (REAPER and the Python server) — unlike $TMPDIR.
  local home = os.getenv("HOME") or os.getenv("USERPROFILE") or "."
  return home .. SEP .. ".orpheus_bridge"
end

local function resolve_bridge_dir()
  local ext = reaper.GetExtState("orpheus", "bridge_dir")
  if ext and ext ~= "" then return ext end
  local env = os.getenv("REAPER_MCP_BRIDGE_DIR")
  if env and env ~= "" then return env end
  return default_bridge_dir()
end

local BRIDGE_DIR = resolve_bridge_dir()
M.bridge_dir = BRIDGE_DIR

local function path(name) return BRIDGE_DIR .. SEP .. name end

local function id_to_filename(id)
  -- Integer id MUST format as "1", never "1.0", or Python's response poll never matches.
  if math.type and math.type(id) == "integer" then return string.format("%d", id) end
  return string.format("%.0f", id)
end

-- --------------------------------------------------------------------------- --
-- IO helpers
-- --------------------------------------------------------------------------- --

local function write_atomic(p, text)
  local tmp = p .. ".tmp"
  local f, err = io.open(tmp, "w")
  if not f then return false, err end
  f:write(text)
  f:close()
  os.remove(p)        -- Windows os.rename fails if target exists
  return os.rename(tmp, p)
end
M.write_atomic = write_atomic

local function read_file(p)
  local f = io.open(p, "r")
  if not f then return nil end
  local s = f:read("*a")
  f:close()
  return s
end

local function touch_heartbeat()
  write_atomic(path("heartbeat.lock"), tostring(os.time()))
end
M.touch_heartbeat = touch_heartbeat

local function ensure_dir()
  if reaper and reaper.RecursiveCreateDirectory then
    reaper.RecursiveCreateDirectory(BRIDGE_DIR, 0)
  end
end

-- --------------------------------------------------------------------------- --
-- Dispatch — STATIC table, no eval. Composite helpers live here so one request =
-- one musical intent. M0 ships the heartbeat/health handler; M1+ add the rest.
-- --------------------------------------------------------------------------- --

local HANDLERS = {}

HANDLERS.get_connection_status = function(_)
  return {
    ok = true,
    reaper_version = reaper.GetAppVersion and reaper.GetAppVersion() or "unknown",
    bridge_dir = BRIDGE_DIR,
  }
end

local function dispatch(fn, params)
  if fn == "__batch__" then
    local results = {}
    for idx, c in ipairs(params.calls) do
      local r = dispatch(c.fn, c.params or {})
      if not r.ok then return r end
      results[idx] = r.result
    end
    return { ok = true, result = results }
  end
  local handler = HANDLERS[fn]
  if not handler then return { ok = false, error = "unknown fn: " .. tostring(fn) } end
  local ok, res = pcall(handler, params or {})
  if ok then return { ok = true, result = res } end
  return { ok = false, error = tostring(res) }
end
M.dispatch = dispatch
M.handlers = HANDLERS

local function handle_request(name)
  local raw = read_file(path(name))
  if not raw then return end
  local ok, req = pcall(json.decode, raw)
  if not ok or type(req) ~= "table" or req.id == nil then
    os.remove(path(name))
    return
  end
  local result = dispatch(req.fn, req.params or {})
  write_atomic(path("response_" .. id_to_filename(req.id) .. ".json"), json.encode(result))
  os.remove(path(name))
end
M.handle_request = handle_request

local function scan_requests()
  reaper.EnumerateFiles(BRIDGE_DIR, -1)  -- invalidate REAPER's directory cache
  local i, names = 0, {}
  while true do
    local name = reaper.EnumerateFiles(BRIDGE_DIR, i)
    if not name then break end
    if name:match("^request_%d+%.json$") then names[#names + 1] = name end
    i = i + 1
  end
  return names
end

local function tick()
  touch_heartbeat()
  for _, name in ipairs(scan_requests()) do
    pcall(handle_request, name)
  end
end
M.tick = tick

local function loop()
  tick()
  reaper.defer(loop)
end
M.loop = loop

-- --------------------------------------------------------------------------- --
-- Autostart (skipped in tests)
-- --------------------------------------------------------------------------- --

if not _G.ORPHEUS_NO_AUTORUN then
  ensure_dir()
  if reaper and reaper.ShowConsoleMsg then
    reaper.ShowConsoleMsg("Orpheus bridge listening in: " .. BRIDGE_DIR .. "\n")
  end
  loop()
end

return M
