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

-- --------------------------------------------------------------------------- --
-- REAPER helpers — stable identity (GUID/index, never live pointers) and the
-- beats <-> PPQ math. THE INVARIANT: the model speaks beats; every tick/PPQ
-- conversion happens HERE, inside the bridge, never in the Python tools.
-- --------------------------------------------------------------------------- --

-- Resolve a track from a stable reference: a "{GUID}" string, a 1-based index
-- ("3"), or an exact track name. Returns the MediaTrack* (a transient pointer we
-- only ever use within this single call) or nil + an error message.
local function resolve_track(ref)
  if ref == nil or ref == "" then return nil, "no track reference given" end
  ref = tostring(ref)
  local count = reaper.CountTracks(0)
  -- GUID form: "{...}"
  if ref:sub(1, 1) == "{" then
    for i = 0, count - 1 do
      local tr = reaper.GetTrack(0, i)
      if reaper.GetTrackGUID(tr) == ref then return tr end
    end
    return nil, "no track with GUID " .. ref
  end
  -- 1-based numeric index
  local idx = tonumber(ref)
  if idx and math.floor(idx) == idx then
    local tr = reaper.GetTrack(0, idx - 1)
    if tr then return tr end
    return nil, "track index out of range: " .. ref
  end
  -- exact name match
  for i = 0, count - 1 do
    local tr = reaper.GetTrack(0, i)
    local _, name = reaper.GetTrackName(tr)
    if name == ref then return tr end
  end
  return nil, "no track named '" .. ref .. "'"
end

-- The effective time signature at the project start as (numerator, denominator).
-- TimeMap_GetTimeSigAtTime returns BOTH (unlike GetProjectTimeSignature2, which only
-- yields bpm + the numerator); we need the denominator for the bar<->QN math.
local function project_time_sig()
  local num, den = reaper.TimeMap_GetTimeSigAtTime(0, 0.0)
  num = (num and num > 0) and num or 4
  den = (den and den > 0) and den or 4
  return num, den
end

-- Project quarter-notes per bar from the (numerator/denominator) time signature.
-- A "beat" in Orpheus is one quarter note; a bar is numerator * (4/denominator) QN.
local function qn_per_bar()
  local num, den = project_time_sig()
  return num * (4.0 / den)
end

-- The project quarter-note position where bar `bar` (1-based) begins.
local function bar_start_qn(bar)
  bar = (bar and bar >= 1) and bar or 1
  return (bar - 1) * qn_per_bar()
end

-- The active take of a track's first MIDI item, or nil. Stable by track ref.
local function first_take(tr)
  local item = reaper.GetTrackMediaItem(0, tr, 0)
  if not item then return nil end
  return reaper.GetActiveTake(item)
end

-- Each call is bounded so REAPER's single-threaded audio path never stalls.
local MAX_NOTES = 512

HANDLERS.get_project_info = function(_)
  local num, den = project_time_sig()
  return {
    tempo = reaper.Master_GetTempo(),
    time_signature = { num, den },
    length = reaper.GetProjectLength(0),
    num_tracks = reaper.CountTracks(0),
    playing = (reaper.GetPlayState() & 1) == 1,
  }
end

HANDLERS.list_tracks = function(_)
  local out = {}
  for i = 0, reaper.CountTracks(0) - 1 do
    local tr = reaper.GetTrack(0, i)
    local _, name = reaper.GetTrackName(tr)
    local vol = reaper.GetMediaTrackInfo_Value(tr, "D_VOL")
    local pan = reaper.GetMediaTrackInfo_Value(tr, "D_PAN")
    -- D_VOL is a linear gain; the model wants dB. 0 gain -> -inf, clamp to -150.
    local vol_db = (vol > 0) and (20.0 * math.log(vol, 10)) or -150.0
    out[#out + 1] = {
      index = i + 1,
      guid = reaper.GetTrackGUID(tr),
      name = name or "",
      volume_db = vol_db,
      pan = pan,
      mute = reaper.GetMediaTrackInfo_Value(tr, "B_MUTE") == 1,
      solo = reaper.GetMediaTrackInfo_Value(tr, "I_SOLO") ~= 0,
      num_items = reaper.CountTrackMediaItems(tr),
    }
  end
  return out
end

HANDLERS.set_tempo = function(p)
  reaper.SetCurrentBPM(0, p.bpm, true)
  return { tempo = reaper.Master_GetTempo() }
end

HANDLERS.set_time_signature = function(p)
  -- Set the time signature at the project start via a tempo/timesig marker.
  -- SetTempoTimeSigMarker(proj, ptidx, timepos, measurepos, beatpos, bpm, num, den, linear)
  reaper.SetTempoTimeSigMarker(0, -1, 0.0, -1, -1, reaper.Master_GetTempo(),
    p.numerator, p.denominator, false)
  local num, den = project_time_sig()
  return { time_signature = { num, den } }
end

HANDLERS.play_stop_record = function(p)
  -- Transport via the documented Main action IDs (1007 play, 1016 stop, 1013 record).
  local ids = { play = 1007, stop = 1016, record = 1013 }
  local id = ids[p.command]
  if not id then error("unknown transport command: " .. tostring(p.command)) end
  reaper.Main_OnCommand(id, 0)
  return { command = p.command, play_state = reaper.GetPlayState() }
end

HANDLERS.create_track = function(p)
  local count = reaper.CountTracks(0)
  local at = p.index and (p.index - 1) or count   -- 1-based in, 0-based insert
  if at < 0 then at = 0 elseif at > count then at = count end
  reaper.InsertTrackAtIndex(at, true)
  local tr = reaper.GetTrack(0, at)
  if p.name and p.name ~= "" then
    reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", p.name, true)
  end
  return {
    guid = reaper.GetTrackGUID(tr),
    index = at + 1,
    name = p.name or "",
  }
end

-- Create an empty MIDI item over [start_bar, start_bar+length_bars) and return a
-- stable handle: the owning track GUID + the item index on that track.
HANDLERS.create_midi_item = function(p)
  local tr, err = resolve_track(p.track)
  if not tr then error(err) end
  local qnpb = qn_per_bar()
  local start_qn = bar_start_qn(p.start_bar or 1)
  local end_qn = start_qn + (p.length_bars or 1) * qnpb
  local start_t = reaper.TimeMap2_QNToTime(0, start_qn)
  local end_t = reaper.TimeMap2_QNToTime(0, end_qn)
  local item = reaper.CreateNewMIDIItemInProj(tr, start_t, end_t, false)
  local take = reaper.GetActiveTake(item)
  return {
    track = reaper.GetTrackGUID(tr),
    item_index = reaper.GetMediaItemInfo_Value(item, "IP_ITEMNUMBER"),
    start_bar = p.start_bar or 1,
    length_bars = p.length_bars or 1,
    start_qn = start_qn,
  }
end

-- THE load-bearing primitive. Notes arrive in BEATS; this converts beats -> the
-- take's PPQ via the project tempo/QN map and writes them in one batched, sorted
-- pass. If the track has no MIDI item yet, one is created to span the notes.
HANDLERS.insert_midi_notes = function(p)
  local tr, err = resolve_track(p.track)
  if not tr then error(err) end
  local notes = p.notes or {}
  if #notes > MAX_NOTES then
    error("too many notes in one call: " .. #notes .. " > " .. MAX_NOTES)
  end
  local at_bar = p.at_bar or 1
  local base_qn = bar_start_qn(at_bar)   -- beat 0 of the notes == start of at_bar

  local take = first_take(tr)
  if not take then
    -- Span an item large enough for the notes (rounded up to whole bars).
    local max_end_beat = 0.0
    for _, n in ipairs(notes) do
      local e = n.start_beat + n.duration_beats
      if e > max_end_beat then max_end_beat = e end
    end
    local qnpb = qn_per_bar()
    local bars = math.max(1, math.ceil(max_end_beat / qnpb))
    local start_t = reaper.TimeMap2_QNToTime(0, base_qn)
    local end_t = reaper.TimeMap2_QNToTime(0, base_qn + bars * qnpb)
    local item = reaper.CreateNewMIDIItemInProj(tr, start_t, end_t, false)
    take = reaper.GetActiveTake(item)
  end

  for _, n in ipairs(notes) do
    -- beats -> project QN -> take PPQ. One quarter note == one beat.
    local start_ppq = reaper.MIDI_GetPPQPosFromProjQN(take, base_qn + n.start_beat)
    local end_ppq = reaper.MIDI_GetPPQPosFromProjQN(take,
      base_qn + n.start_beat + n.duration_beats)
    reaper.MIDI_InsertNote(take, false, false, start_ppq, end_ppq,
      n.channel or 0, n.pitch, n.velocity or 96, true)  -- noSortIn=true; sort once below
  end
  reaper.MIDI_Sort(take)
  return { track = reaper.GetTrackGUID(tr), inserted = #notes, at_bar = at_bar }
end

-- Read a take's notes back in BEATS (the inverse of insert_midi_notes' math), so
-- a note written at beat B reads back at beat B. base_qn anchors beat 0 to at_bar.
HANDLERS.get_track_midi = function(p)
  local tr, err = resolve_track(p.track)
  if not tr then error(err) end
  local take = first_take(tr)
  if not take then return { track = reaper.GetTrackGUID(tr), notes = {} } end
  local base_qn = bar_start_qn(p.at_bar or 1)
  local _, note_count = reaper.MIDI_CountEvts(take)
  local notes = {}
  for i = 0, note_count - 1 do
    local _, _, _, start_ppq, end_ppq, chan, pitch, vel = reaper.MIDI_GetNote(take, i)
    local start_qn = reaper.MIDI_GetProjQNFromPPQPos(take, start_ppq)
    local end_qn = reaper.MIDI_GetProjQNFromPPQPos(take, end_ppq)
    notes[#notes + 1] = {
      pitch = pitch,
      start_beat = start_qn - base_qn,
      duration_beats = end_qn - start_qn,
      velocity = vel,
      channel = chan,
    }
  end
  return { track = reaper.GetTrackGUID(tr), notes = notes }
end

HANDLERS.transpose_notes = function(p)
  local tr, err = resolve_track(p.track)
  if not tr then error(err) end
  local take = first_take(tr)
  if not take then error("track has no MIDI take to transpose") end
  local semis = p.semitones or 0
  local _, note_count = reaper.MIDI_CountEvts(take)
  local moved = 0
  for i = 0, note_count - 1 do
    local _, sel, muted, sppq, eppq, chan, pitch, vel = reaper.MIDI_GetNote(take, i)
    local np = pitch + semis
    if np >= 0 and np <= 127 then
      reaper.MIDI_SetNote(take, i, sel, muted, sppq, eppq, chan, np, vel, true)
      moved = moved + 1
    end
  end
  reaper.MIDI_Sort(take)
  return { track = reaper.GetTrackGUID(tr), transposed = moved, semitones = semis }
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
