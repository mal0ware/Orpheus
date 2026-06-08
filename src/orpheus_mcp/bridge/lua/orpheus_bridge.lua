--[[
  orpheus_bridge.lua — the in-REAPER half of the Orpheus bridge.

  Run this once per REAPER session (Actions → Run ReaScript, or a toolbar button).
  It installs a persistent background poll loop via reaper.defer() that:

    1. Touches a heartbeat lock-file so the Python side knows REAPER is alive.
    2. Scans <bridge_dir> for request_N.json (cache-invalidated listing).
    3. Dispatches the requested function through a STATIC table (no loadstring/eval).
    4. Writes response_N.json ATOMICALLY (tmp then rename) and deletes the request.

  Hardening (see docs/architecture.md):
    * static dispatch only — never execute arbitrary strings off the channel
    * atomic temp-then-rename writes both directions
    * EnumerateFiles cache invalidation with fileindex = -1
    * per-call note/track caps so each command finishes < ~2s and never blocks audio
    * stable GUID/index identity returned to callers, never live pointers

  STATUS: skeleton. The dispatch table is wired in M0–M1 alongside BridgeClient.
--]]

local SEP = package.config:sub(1, 1)
local BRIDGE_DIR = (reaper.GetExtState("orpheus", "bridge_dir") ~= "" and
                    reaper.GetExtState("orpheus", "bridge_dir"))
                   or (os.getenv("REAPER_MCP_BRIDGE_DIR"))
                   or (os.getenv("TMPDIR") or os.getenv("TEMP") or "/tmp") .. SEP .. "orpheus_bridge"

local POLL_INTERVAL = 0.1   -- seconds; REAPER caps defer at ~30/sec regardless
local HEARTBEAT_FILE = BRIDGE_DIR .. SEP .. "heartbeat.lock"

-- Static dispatch table: fn name -> handler(params) -> result table.
-- TODO(M0–M1): populate. Composite helpers (e.g. create_track_with_notes) live here
-- so one request = one musical intent.
local HANDLERS = {
  get_connection_status = function(_)
    return { ok = true, reaper_version = reaper.GetAppVersion() }
  end,
  -- get_project_info  = function(p) ... end,
  -- create_track      = function(p) ... end,
  -- insert_midi_notes = function(p) ... end,  -- the load-bearing primitive
}

local function ensure_dir()
  reaper.RecursiveCreateDirectory(BRIDGE_DIR, 0)
end

local function write_atomic(path, text)
  local tmp = path .. ".tmp"
  local f = io.open(tmp, "w")
  if not f then return false end
  f:write(text)
  f:close()
  os.rename(tmp, path)  -- atomic enough for our single-writer-per-file protocol
  return true
end

local function touch_heartbeat()
  write_atomic(HEARTBEAT_FILE, tostring(os.time()))
end

-- TODO(M0): JSON encode/decode. Vendor a small pure-Lua JSON lib (no deps) or use
-- REAPER's built-in where available. Kept abstract here so the skeleton stays readable.
local function decode(_) error("JSON decode not yet wired — M0") end
local function encode(_) error("JSON encode not yet wired — M0") end

local function scan_requests()
  -- IMPORTANT: invalidate REAPER's directory listing cache with fileindex = -1,
  -- otherwise EnumerateFiles never sees newly-arrived requests.
  reaper.EnumerateFiles(BRIDGE_DIR, -1)
  local i, names = 0, {}
  while true do
    local name = reaper.EnumerateFiles(BRIDGE_DIR, i)
    if not name then break end
    if name:match("^request_%d+%.json$") then names[#names + 1] = name end
    i = i + 1
  end
  return names
end

local function handle(name)
  local path = BRIDGE_DIR .. SEP .. name
  local rf = io.open(path, "r")
  if not rf then return end
  local raw = rf:read("*a"); rf:close()

  local ok, req = pcall(decode, raw)
  if not ok or type(req) ~= "table" then os.remove(path); return end

  local handler = HANDLERS[req.fn]
  local result
  if handler then
    local hok, hres = pcall(handler, req.params or {})
    result = hok and { ok = true, result = hres } or { ok = false, error = tostring(hres) }
  else
    result = { ok = false, error = "unknown fn: " .. tostring(req.fn) }
  end

  write_atomic(BRIDGE_DIR .. SEP .. ("response_%s.json"):format(req.id), encode(result))
  os.remove(path)
end

local function loop()
  touch_heartbeat()
  for _, name in ipairs(scan_requests()) do
    pcall(handle, name)
  end
  reaper.defer(loop)
end

ensure_dir()
reaper.ShowConsoleMsg("Orpheus bridge listening in: " .. BRIDGE_DIR .. "\n")
loop()
