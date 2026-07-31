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
  -- NB: GetTrackMediaItem(tr, itemidx) takes NO project arg (unlike GetTrack).
  local item = reaper.GetTrackMediaItem(tr, 0)
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

  -- Hoisted so BOTH branches below (no-take / take-exists) can see it: a shared track
  -- gets one item across multiple calls (e.g. arrange_song writing section 2 at a
  -- higher at_bar), and that item must cover every call's notes, not just the first.
  local max_end_beat = 0.0
  for _, n in ipairs(notes) do
    local e = n.start_beat + n.duration_beats
    if e > max_end_beat then max_end_beat = e end
  end

  local take = first_take(tr)
  if not take then
    -- Span an item large enough for the notes (rounded up to whole bars).
    local qnpb = qn_per_bar()
    local bars = math.max(1, math.ceil(max_end_beat / qnpb))
    local start_t = reaper.TimeMap2_QNToTime(0, base_qn)
    local end_t = reaper.TimeMap2_QNToTime(0, base_qn + bars * qnpb)
    local item = reaper.CreateNewMIDIItemInProj(tr, start_t, end_t, false)
    take = reaper.GetActiveTake(item)
  else
    -- The item already exists (a prior call on this same track created it) — grow it
    -- if this call's notes reach past its current right edge. Without this, notes
    -- inserted beyond the item's end exist in the MIDI take but REAPER won't play them.
    -- NOTE (verify live): confirm this D_LENGTH resize behavior in the live smoke.
    local item = reaper.GetMediaItemTake_Item(take)
    local item_pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
    local item_len = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
    local needed_end_t = reaper.TimeMap2_QNToTime(0, base_qn + max_end_beat)
    if needed_end_t > item_pos + item_len then
      reaper.SetMediaItemInfo_Value(item, "D_LENGTH", needed_end_t - item_pos)
    end
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

-- Delete every note in a track's first take (so humanize can rewrite in place).
HANDLERS.clear_track_midi = function(p)
  local tr, err = resolve_track(p.track)
  if not tr then error(err) end
  local take = first_take(tr)
  if not take then return { track = reaper.GetTrackGUID(tr), cleared = 0 } end
  local _, note_count = reaper.MIDI_CountEvts(take)
  for i = note_count - 1, 0, -1 do
    reaper.MIDI_DeleteNote(take, i)
  end
  reaper.MIDI_Sort(take)
  return { track = reaper.GetTrackGUID(tr), cleared = note_count }
end

-- Enumerate installed FX/instruments. EnumInstalledFX(index) -> ret, name, ident.
-- Read-only; installs nothing. Cached by REAPER internally.
HANDLERS.list_installed_fx = function(_)
  local out = {}
  local i = 0
  while true do
    local ok, name = reaper.EnumInstalledFX(i)
    if not ok then break end
    out[#out + 1] = name
    i = i + 1
    if i > 20000 then break end  -- hard safety cap
  end
  return { fx = out }
end
-- NOTE (verify live): reaper.EnumInstalledFX returns (retval, name, ident) in REAPER 7.x.
-- If unavailable on the target build, fall back to reading reaper-vstplugins*.ini.
-- The fake + Lua-stub tests do not depend on the real API; the live smoke (Task 15) confirms it.

-- Load an instrument on a track. kind="named" adds one FX by name (idempotent);
-- kind="drumkit" adds one ReaSamplOmatic5000 per sample, note-filtered to its GM note.
HANDLERS.add_instrument = function(p)
  local tr, err = resolve_track(p.track)
  if not tr then error(err) end

  if p.kind == "drumkit" then
    -- idempotent: don't stack duplicate voices if a kit is already on this track.
    local existing = reaper.TrackFX_AddByName(tr, "ReaSamplOmatic5000", false, 0)  -- 0 = find only
    if existing >= 0 then
      return { track = reaper.GetTrackGUID(tr), loaded = "drumkit", already_present = true }
    end
    local GM = { kick = 36, snare = 38, hat = 42 }
    for voice, file in pairs(p.samples or {}) do
      local fx = reaper.TrackFX_AddByName(tr, "ReaSamplOmatic5000", false, -1)
      if fx >= 0 then
        -- Load the sample and pin it to one MIDI note. NOTE (verify live): the config
        -- parm for the sample file is "FILE0"; note-range params are indices 3 (min) and
        -- 4 (max) as 0..1 normalized pitch (n/127). Confirm on REAPER 7.x in Task 15.
        reaper.TrackFX_SetNamedConfigParm(tr, fx, "FILE0", file)
        local n = GM[voice] or 36
        reaper.TrackFX_SetParamNormalized(tr, fx, 3, n / 127.0)
        reaper.TrackFX_SetParamNormalized(tr, fx, 4, n / 127.0)
      end
    end
    return { track = reaper.GetTrackGUID(tr), loaded = "drumkit", already_present = false }
  end

  -- named synth: idempotent (don't stack duplicates).
  local existing = reaper.TrackFX_AddByName(tr, p.name, false, 0)  -- 0 = find only
  if existing >= 0 then
    return { track = reaper.GetTrackGUID(tr), loaded = p.name, already_present = true }
  end
  local added = reaper.TrackFX_AddByName(tr, p.name, false, -1)  -- -1 = add
  if added < 0 then error("could not add instrument: " .. tostring(p.name)) end
  return { track = reaper.GetTrackGUID(tr), loaded = p.name, already_present = false }
end

-- --------------------------------------------------------------------------- --
-- FX verbs (§8 step 3, Task 11.1) — add_fx_by_name / set_fx_param / get_fx_params.
-- Locked contracts (docs/plans/2026-07-23-reference-analysis.md Task 11.1): unit/range
-- mapping lives PYTHON-side (tools/mix.py + data/fx_param_maps.json); these handlers only
-- speak REAPER's native 0..1 normalized param space.
-- --------------------------------------------------------------------------- --

-- Add an FX to a track BY NAME, probe-first (idempotent) — the exact same
-- find-then-add pattern as add_instrument's drumkit fix above, generalized to any FX.
HANDLERS.add_fx_by_name = function(p)
  local tr, err = resolve_track(p.track)
  if not tr then error(err) end
  local existing = reaper.TrackFX_AddByName(tr, p.name, false, 0)  -- 0 = find only
  if existing >= 0 then
    return { track = reaper.GetTrackGUID(tr), fx_index = existing, already_present = true }
  end
  local added = reaper.TrackFX_AddByName(tr, p.name, false, -1)  -- -1 = add
  if added < 0 then error("could not add fx: " .. tostring(p.name)) end
  return { track = reaper.GetTrackGUID(tr), fx_index = added, already_present = false }
end

-- Set one FX param, NORMALIZED 0..1 (REAPER native). All dB/Hz/ms/ratio translation
-- happens python-side, via data/fx_param_maps.json — this handler never sees real units.
HANDLERS.set_fx_param = function(p)
  local tr, err = resolve_track(p.track)
  if not tr then error(err) end
  local ok = reaper.TrackFX_SetParamNormalized(tr, p.fx_index, p.param_index, p.value)
  if ok == false then
    error("could not set fx " .. tostring(p.fx_index) .. " param " .. tostring(p.param_index))
  end
  return { ok = true }
end

-- Read back an FX's full param list, NORMALIZED 0..1 — name + value only; the
-- unit/min/max/native-value mapping is applied python-side against the param map.
HANDLERS.get_fx_params = function(p)
  local tr, err = resolve_track(p.track)
  if not tr then error(err) end
  local _, fx_name = reaper.TrackFX_GetFXName(tr, p.fx_index, "")
  local n = reaper.TrackFX_GetNumParams(tr, p.fx_index)
  local params = {}
  for i = 0, n - 1 do
    local _, pname = reaper.TrackFX_GetParamName(tr, p.fx_index, i, "")
    local val = reaper.TrackFX_GetParamNormalized(tr, p.fx_index, i)
    params[#params + 1] = { index = i, name = pname, value = val }
  end
  return { fx_name = fx_name or "", params = params }
end

-- Place a named project marker at the start of a bar.
HANDLERS.add_marker = function(p)
  -- NOTE (verify live): AddProjectMarker2(proj, isrgn, pos, rgnend, name, wantidx, color)
  -- returns the created marker index on REAPER 7.x. The fake + lua-stub tests do not
  -- depend on the real API; confirmed by the live smoke.
  local qn = bar_start_qn(p.bar or 1)
  local pos = reaper.TimeMap2_QNToTime(0, qn)
  local idx = reaper.AddProjectMarker2(0, false, pos, 0, p.name or "", -1, 0)
  return { name = p.name or "", bar = p.bar or 1, index = idx }
end

-- --------------------------------------------------------------------------- --
-- render_track_region (§15.2) — the "bridge verb" that bounces ONE track's region
-- to a WAV for the audio analyzer. It mutates GLOBAL project state (render settings,
-- every track's solo, master-FX enable) and MUST leave every bit of it exactly as it
-- found it. The §15.2 failure this guards against: corrupting the user's render
-- settings or solo states permanently. Hence: a non-reentrant bridge-side mutex, a
-- FULL snapshot before any mutation, and restore in BOTH the success and error paths.
-- --------------------------------------------------------------------------- --

-- Bridge-side, non-reentrant. A render is a long, global-state-mutating operation;
-- a second render arriving mid-flight must be refused, never interleaved. Cleared in
-- EVERY path by the handler below (the pcall's `false` assignment runs unconditionally).
local RENDER_MUTEX_HELD = false

-- The last full snapshot taken by snapshot_render_state(). Held at module scope (not just
-- in the handler's pcall closure) so restore_render_state_best_effort() — invoked from the
-- error path OUTSIDE that closure, where the closure's `snap` local is invisible — can
-- still find it. Consumed (set nil) by a restore so a later best-effort restore no-ops
-- rather than re-applying a stale snapshot.
local LAST_RENDER_SNAPSHOT = nil

-- Signal a handler-level error through the static dispatcher. dispatch() pcalls each
-- handler and reports a raise as { ok = false, error = <msg> }; err() raises at level 0
-- so the clean message (matched by the Python tests, e.g. "render busy"/"silent") is not
-- prefixed with this file:line.
local function err(msg) error(msg, 0) end

-- Numeric + string RENDER_* project keys we snapshot and restore. Numeric keys go through
-- GetSetProjectInfo; string keys (paths, the format blob) through GetSetProjectInfo_String.
-- The set covers every field set_render_target() touches PLUS the custom-bounds positions,
-- so nothing the user configured leaks out of a render.
local RENDER_NUM_KEYS = {
  "RENDER_SRATE", "RENDER_CHANNELS", "RENDER_SETTINGS", "RENDER_BOUNDSFLAG",
  "RENDER_TAILFLAG", "RENDER_TAILMS", "RENDER_ADDTOPROJ", "RENDER_STARTPOS", "RENDER_ENDPOS",
}
local RENDER_STR_KEYS = { "RENDER_FILE", "RENDER_PATTERN", "RENDER_FORMAT" }

-- WAV, 32-bit float. RENDER_FORMAT is a base64 "evaw" (='wave' reversed) config blob; the
-- bytes below encode the WAV sink at 32-bit float.
-- NOTE (verify live, Task 11): this base64 MUST be confirmed by copying it out of a real
-- REAPER "WAV / 32 bit FP" render preset (Preferences never exposes the raw blob) — the
-- fake + lua-stub tests do NOT decode it (they only prove it is snapshotted/restored and
-- that a .wav is produced), so a wrong blob here would pass CI yet render the wrong depth.
local WAV_32F_RENDER_FORMAT = "ZXZhdyAAAAA="  -- 'evaw' + depth byte 32: REAPER's minimal
-- WAV config, depth 32 == 32-bit FP. LIVE-VERIFIED 7.78 (the previous 36-byte blob was a
-- fabricated guess that rendered PCM_24 -- caught by the 2026-07-25 live smoke).

-- The project's audio sample rate. NOTE (verify live, Task 11): PROJECT_SRATE only reflects
-- a project-forced rate when PROJECT_SRATE_USE is on; otherwise REAPER renders at the audio
-- device rate. We fall back to 44100 so a render never requests srate 0.
local function get_project_srate()
  local sr = reaper.GetSetProjectInfo(0, "PROJECT_SRATE", 0, false)
  if not sr or sr <= 0 then sr = 44100 end
  return math.floor(sr)
end

-- Master-FX enable toggle. I_FXEN on the master track is REAPER's master-FX-chain bypass
-- (0 = bypassed, !0 = active). NOTE (verify live, Task 11): confirm on the target build that
-- I_FXEN on GetMasterTrack bypasses the master chain for the render (vs. only the UI).
local function set_master_fx_bypass(bypass)
  local master = reaper.GetMasterTrack(0)
  reaper.SetMediaTrackInfo_Value(master, "I_FXEN", bypass and 0 or 1)
end

-- Snapshot EVERYTHING a render mutates: the RENDER_* keys, every track's I_SOLO, the master
-- I_FXEN, and the current time selection (which render bounds=time-sel will overwrite).
local function snapshot_render_state()
  local snap = { num = {}, str = {}, solos = {}, master_fxen = nil, loop_s = nil, loop_e = nil }
  for _, k in ipairs(RENDER_NUM_KEYS) do
    snap.num[k] = reaper.GetSetProjectInfo(0, k, 0, false)
  end
  for _, k in ipairs(RENDER_STR_KEYS) do
    local _, v = reaper.GetSetProjectInfo_String(0, k, "", false)
    snap.str[k] = v
  end
  for i = 0, reaper.CountTracks(0) - 1 do
    snap.solos[i] = reaper.GetMediaTrackInfo_Value(reaper.GetTrack(0, i), "I_SOLO")
  end
  snap.master_fxen = reaper.GetMediaTrackInfo_Value(reaper.GetMasterTrack(0), "I_FXEN")
  local s, e = reaper.GetSet_LoopTimeRange(false, false, 0, 0, false)  -- isLoop=false => time selection
  snap.loop_s, snap.loop_e = s, e
  LAST_RENDER_SNAPSHOT = snap
  return snap
end

-- Restore a snapshot field-for-field. Nil-safe (a nil snap — e.g. an error before the
-- snapshot was taken — is a no-op) and idempotent; consumes LAST_RENDER_SNAPSHOT so a
-- follow-up best-effort restore does not re-apply a now-stale snapshot.
local function restore_render_state(snap)
  if not snap then return end
  for _, k in ipairs(RENDER_NUM_KEYS) do
    if snap.num[k] ~= nil then reaper.GetSetProjectInfo(0, k, snap.num[k], true) end
  end
  for _, k in ipairs(RENDER_STR_KEYS) do
    if snap.str[k] ~= nil then reaper.GetSetProjectInfo_String(0, k, snap.str[k], true) end
  end
  for i = 0, reaper.CountTracks(0) - 1 do
    if snap.solos[i] ~= nil then
      reaper.SetMediaTrackInfo_Value(reaper.GetTrack(0, i), "I_SOLO", snap.solos[i])
    end
  end
  if snap.master_fxen ~= nil then
    reaper.SetMediaTrackInfo_Value(reaper.GetMasterTrack(0), "I_FXEN", snap.master_fxen)
  end
  if snap.loop_s ~= nil then
    reaper.GetSet_LoopTimeRange(true, false, snap.loop_s, snap.loop_e, false)
  end
  LAST_RENDER_SNAPSHOT = nil
end

-- Error-path restore: never raises (wrapped in pcall) — a failure here must not mask the
-- original render error the caller needs to see.
local function restore_render_state_best_effort()
  pcall(restore_render_state, LAST_RENDER_SNAPSHOT)
end

-- Point the render at out_dir/basename.wav, 32-bit-float WAV, project sample rate, master-mix
-- source, bounds = the time selection we set in step 5.
local function set_render_target(out_dir, basename)
  reaper.GetSetProjectInfo_String(0, "RENDER_FILE", out_dir, true)
  reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", basename, true)
  reaper.GetSetProjectInfo_String(0, "RENDER_FORMAT", WAV_32F_RENDER_FORMAT, true)
  reaper.GetSetProjectInfo(0, "RENDER_SRATE", get_project_srate(), true)
  reaper.GetSetProjectInfo(0, "RENDER_CHANNELS", 2, true)
  reaper.GetSetProjectInfo(0, "RENDER_SETTINGS", 0, true)    -- 0 = master mix (no stems)
  reaper.GetSetProjectInfo(0, "RENDER_BOUNDSFLAG", 2, true)  -- 2 = time selection
  reaper.GetSetProjectInfo(0, "RENDER_TAILFLAG", 0, true)    -- no render tail
  reaper.GetSetProjectInfo(0, "RENDER_ADDTOPROJ", 0, true)   -- do NOT import the result into the project
end

-- Step 7: the render must have produced a real, non-silent file. Digital silence
-- (all-zero samples) is the tell-tale of a render that captured nothing — the exact
-- symptom of a botched solo/routing setup, which is why §15.2 demands this assertion.
--
-- Data-chunk-aware, NOT a fixed 44-byte header skip: a canonical 16-byte-fmt PCM WAV
-- happens to put "data" right at byte 45, but REAPER's real 32-bit-float render carries
-- an extra "fact" chunk (and often a larger fmt chunk) BEFORE "data", pushing the payload
-- further out. A fixed skip would then scan into tail bytes of the header/fact chunk as
-- if it were audio — which could read a genuinely silent render as non-silent and defeat
-- this exact assertion. So we scan for the "data" marker itself, read its 4-byte
-- little-endian size, and only inspect that declared payload.
-- verify live: confirm this chunk walk against a real REAPER "WAV / 32 bit FP" render —
-- the fake/lupa tests exercise the parser logic but never touch a real REAPER file.
local function assert_rendered_nonsilent(path)
  local f = io.open(path, "rb")
  if not f then err("render produced no file: " .. path) end
  local data = f:read("*a")
  f:close()
  if not data or #data == 0 then err("render produced an empty file: " .. path) end
  -- Search from byte 13 — past the fixed 12-byte "RIFF"<size>"WAVE" preamble — so we don't
  -- match a stray "data" substring inside that fixed region (there isn't one, but be precise).
  local marker_pos = data:find("data", 13, true)
  if not marker_pos then
    err("unrecognized WAV layout (no 'data' chunk found): " .. path)
  end
  local size_pos = marker_pos + 4
  local b1, b2, b3, b4 = data:byte(size_pos, size_pos + 3)
  if not (b1 and b2 and b3 and b4) then
    err("unrecognized WAV layout (truncated 'data' chunk size): " .. path)
  end
  local data_size = b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
  local payload_start = size_pos + 4
  local payload = data:sub(payload_start, payload_start + data_size - 1)
  if #payload == 0 then err("render is silent (no audio payload): " .. path) end
  for i = 1, #payload do
    if payload:byte(i) ~= 0 then return end
  end
  err("render is silent (digital silence): " .. path)
end

-- §15.2 steps 1–8, in the locked order. See the module header for why each guard exists.
HANDLERS.render_track_region = function(p)
  if RENDER_MUTEX_HELD then return err("render busy") end          -- §15.2 serialization
  RENDER_MUTEX_HELD = true
  local ok, result = pcall(function()
    if reaper.GetPlayState() ~= 0 then error("transport is playing/recording", 0) end  -- step 1
    local snap = snapshot_render_state()   -- RENDER_* via GetSetProjectInfo(+_String), ALL track I_SOLO, master FX bypass
    local tr, rerr = resolve_track(p.track)                         -- fail BEFORE any state mutation below
    if not tr then error(rerr, 0) end
    for i = 0, reaper.CountTracks(0) - 1 do                         -- step 3: clear ALL solos first
      reaper.SetMediaTrackInfo_Value(reaper.GetTrack(0, i), "I_SOLO", 0)
    end
    reaper.SetMediaTrackInfo_Value(tr, "I_SOLO", 2)                 -- SIP (§15.2: keeps sends, deliberate)
    set_master_fx_bypass(true)                                      -- step 4
    reaper.GetSet_LoopTimeRange(true, false, p.start_s, p.end_s, false)  -- step 5 (seconds; caller converted via project tempo map)
    set_render_target(p.out_dir, p.basename)                        -- step 6: RENDER_FILE=dir, RENDER_PATTERN=basename,
                                                                    --   WAV 32f, RENDER_SRATE=project, source=master, bounds=time sel
    reaper.Main_OnCommand(42230, 0)                                 -- step 6: render using most recent settings (NEVER 40015)
    local path = p.out_dir .. "/" .. p.basename .. ".wav"
    assert_rendered_nonsilent(path)                                 -- step 7: exists, non-empty, not digital silence
    restore_render_state(snap)                                      -- step 8
    return { path = path, start_s = p.start_s, end_s = p.end_s, sr = get_project_srate() }
  end)
  RENDER_MUTEX_HELD = false
  if not ok then pcall(restore_render_state_best_effort); return err(result) end   -- restore in error path too (§15.2)
  return result
end

-- --------------------------------------------------------------------------- --
-- import_audio_files — the OPT-IN local-audio import verb: the programmatic
-- equivalent of dragging files the USER owns onto the timeline (motivating case:
-- laying Orpheus-separated stems onto tracks for A/B).
--
-- Boundary (§1 "owned audio only" / §14.3): this handler inserts exactly the paths the
-- Python tool was given — it never discovers, fetches, or embeds anything on its own.
--
-- The WHOLE import is ONE undo block: one Ctrl+Z removes every track/item it created
-- (the README's product promise). A per-file failure is RECORDED, never silent: the
-- structured result reports {imported, failed} and the undo block is still closed
-- properly, so the project is never left half-imported with the block hanging open.
-- --------------------------------------------------------------------------- --

-- "C:\stems\vocals.wav" / "/stems/vocals.wav" -> "vocals". Handles both separators
-- because a REAPER session on Windows happily accepts either in a path.
local function basename_sans_ext(p)
  local base = tostring(p):match("([^/\\]+)$") or tostring(p)
  return (base:gsub("%.[^%.]+$", ""))   -- parens: drop gsub's replacement count
end

-- Import ONE file as a new appended track carrying one media item at at_s. Raises (via
-- err) on ANY failure so the caller records it under `failed` and continues with the
-- remaining files — a nil/zero-length source is a reported error, never a silent skip.
local function import_one_audio_file(file, at_s, prefix)
  -- Create + validate the source BEFORE touching the project, so an unreadable path
  -- leaves NO stray empty track behind (same "fail before any state mutation" rule
  -- render_track_region follows). Note this reorders the create-track/create-source
  -- steps relative to a naive drag-and-drop recipe, deliberately.
  local src = reaper.PCM_Source_CreateFromFile(file)
  if not src then err("could not open audio file: " .. tostring(file)) end
  -- verify live: GetMediaSourceLength(src) returns (length, lengthIsQN). For a PCM/audio
  -- source the units are SECONDS and lengthIsQN is false; only MIDI sources report QN.
  -- If a live source ever reports lengthIsQN=true here, D_LENGTH would need
  -- TimeMap2_QNToTime conversion instead of the raw value. The stub cannot prove this.
  local len, len_is_qn = reaper.GetMediaSourceLength(src)
  if len_is_qn then
    if reaper.PCM_Source_Destroy then reaper.PCM_Source_Destroy(src) end
    err("source reports musical (QN) length — unsupported for audio import: " .. tostring(file))
  end
  if not len or len <= 0 then
    if reaper.PCM_Source_Destroy then reaper.PCM_Source_Destroy(src) end
    err("audio file is empty or not a readable audio source: " .. tostring(file))
  end

  local count = reaper.CountTracks(0)
  reaper.InsertTrackAtIndex(count, true)   -- append at the end
  local tr = reaper.GetTrack(0, count)
  if not tr then
    if reaper.PCM_Source_Destroy then reaper.PCM_Source_Destroy(src) end
    err("could not create a track for: " .. tostring(file))
  end
  local name = tostring(prefix or "") .. basename_sans_ext(file)
  reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", name, true)

  local item = reaper.AddMediaItemToTrack(tr)
  if not item then
    -- The track we just created is still empty and would otherwise be stranded (§ "no
    -- empty track for a failed file" invariant) — and src is still ours to release since
    -- SetMediaItemTake_Source below is what hands ownership to a take.
    if reaper.PCM_Source_Destroy then reaper.PCM_Source_Destroy(src) end
    if reaper.DeleteTrack then reaper.DeleteTrack(tr) end
    err("could not create a media item for: " .. tostring(file))
  end
  local take = reaper.AddTakeToMediaItem(item)
  if not take then
    if reaper.PCM_Source_Destroy then reaper.PCM_Source_Destroy(src) end
    if reaper.DeleteTrack then reaper.DeleteTrack(tr) end
    err("could not create a take for: " .. tostring(file))
  end
  -- The take takes OWNERSHIP of src from here on — do not PCM_Source_Destroy it below.
  reaper.SetMediaItemTake_Source(take, src)
  reaper.SetMediaItemInfo_Value(item, "D_POSITION", at_s)
  reaper.SetMediaItemInfo_Value(item, "D_LENGTH", len)
  -- Pin the item to TIME base (0 = time; -1 would be project default — that was the bug:
  -- a beats-timebase item warps under a later set_tempo). Imported reference audio must
  -- never tempo-warp; a later set_tempo stretched real stems at half speed (live incident
  -- 2026-07-25). verify live: 0=time per the ReaScript docs for C_BEATATTACHMODE.
  reaper.SetMediaItemInfo_Value(item, "C_BEATATTACHMODE", 0)
  -- Belt-and-suspenders against the same class of bug: an imported reference take must
  -- always play back at its native rate, never inherit a project/take playrate that would
  -- also stretch it under a tempo change.
  reaper.SetMediaItemTakeInfo_Value(take, "D_PLAYRATE", 1.0)
  -- verify live: the take-name setter is GetSetMediaItemTakeInfo_String(take, "P_NAME",
  -- name, true) on REAPER 7.x. The stub only proves we call it with those arguments; if
  -- the live build ignores it, the item will still be correct but unnamed.
  reaper.GetSetMediaItemTakeInfo_String(take, "P_NAME", name, true)

  return { file = tostring(file), track_guid = reaper.GetTrackGUID(tr), length_s = len }
end

HANDLERS.import_audio_files = function(p)
  local files = p.files or {}
  if #files == 0 then err("import_audio_files: no files given") end
  local at_s = tonumber(p.at_s) or 0.0
  local prefix = p.name_prefix or ""
  local desc = string.format("Orpheus: import %d audio files", #files)

  -- verify live: Undo_BeginBlock/Undo_EndBlock(desc, -1) nesting. The bridge dispatches
  -- from a defer() tick, so this block may open inside whatever undo scope REAPER is in;
  -- the live smoke must confirm ONE Ctrl+Z reverses the whole import (all N tracks and
  -- items) rather than one track at a time. -1 == "all undo flags" (full state capture).
  reaper.Undo_BeginBlock()
  local imported, failed = {}, {}
  -- The loop runs inside a pcall so NO raise — expected or not — can escape with the
  -- undo block still open. Per-file failures are caught one level in and recorded.
  local ok_all, loop_err = pcall(function()
    for _, file in ipairs(files) do
      local one_ok, one = pcall(import_one_audio_file, file, at_s, prefix)
      if one_ok then
        imported[#imported + 1] = one
      else
        failed[#failed + 1] = { file = tostring(file), error = tostring(one) }
      end
    end
    reaper.UpdateArrange()   -- ONCE, after every file (not per item)
  end)
  reaper.Undo_EndBlock(desc, -1)
  if not ok_all then err(tostring(loop_err)) end

  return { imported = imported, failed = failed }
end

-- --------------------------------------------------------------------------- --
-- clone_track_items — clone every AUDIO item on a source track onto ONE new
-- appended track (motivating case: artificial double-tracking — clone the user's
-- recorded take, then detune/nudge/pan the clone). Each clone re-opens the SAME
-- source file the original take plays; nothing is copied on disk and nothing is
-- fetched — the §1 owned-audio boundary import_audio_files draws applies here too.
--
-- Same promises as import_audio_files: the WHOLE clone is ONE undo block; a
-- per-item failure is RECORDED under `failed`, never silent; and every created
-- item is pinned to TIME base at native playrate (the f4394e9 tempo-immunity fix —
-- a clone of reference audio must never warp under a later set_tempo either).
-- MIDI takes are not errors: they are counted under `skipped_midi` and left alone
-- (the verb's contract is audio items only).
-- --------------------------------------------------------------------------- --

-- Clone ONE audio item onto dest_tr. Raises (via err) on ANY failure so the caller
-- records it under `failed` and continues with the remaining items.
local function clone_one_item(src_item, src_take, dest_tr, offset_s, take_vol)
  if not src_take then err("item has no active take") end
  local src = reaper.GetMediaItemTake_Source(src_take)
  if not src then err("take has no PCM source") end
  local filename = reaper.GetMediaSourceFileName(src, "")
  if not filename or filename == "" then
    err("source has no filename (in-project/section source)")
  end
  local new_src = reaper.PCM_Source_CreateFromFile(filename)
  if not new_src then err("could not re-open source file: " .. tostring(filename)) end
  local item = reaper.AddMediaItemToTrack(dest_tr)
  if not item then
    -- new_src is still ours to release — SetMediaItemTake_Source below is what
    -- hands ownership to a take (same rule as import_one_audio_file).
    if reaper.PCM_Source_Destroy then reaper.PCM_Source_Destroy(new_src) end
    err("could not create a media item on the clone track")
  end
  local take = reaper.AddTakeToMediaItem(item)
  if not take then
    if reaper.PCM_Source_Destroy then reaper.PCM_Source_Destroy(new_src) end
    err("could not create a take on the clone track")
  end
  -- The take takes OWNERSHIP of new_src from here on.
  reaper.SetMediaItemTake_Source(take, new_src)
  reaper.SetMediaItemInfo_Value(item, "D_POSITION",
    reaper.GetMediaItemInfo_Value(src_item, "D_POSITION") + offset_s)
  reaper.SetMediaItemInfo_Value(item, "D_LENGTH",
    reaper.GetMediaItemInfo_Value(src_item, "D_LENGTH"))
  reaper.SetMediaItemTakeInfo_Value(take, "D_STARTOFFS",
    reaper.GetMediaItemTakeInfo_Value(src_take, "D_STARTOFFS"))
  -- Tempo-immunity pins (f4394e9): TIME base + native rate, same as imported items.
  reaper.SetMediaItemInfo_Value(item, "C_BEATATTACHMODE", 0)
  reaper.SetMediaItemTakeInfo_Value(take, "D_PLAYRATE", 1.0)
  reaper.SetMediaItemTakeInfo_Value(take, "D_VOL", take_vol)
end

HANDLERS.clone_track_items = function(p)
  local tr, rerr = resolve_track(p.src_track)
  if not tr then err(rerr) end
  local new_name = p.new_track_name
  if not new_name or new_name == "" then
    err("clone_track_items: new_track_name is required")
  end
  local n = reaper.CountTrackMediaItems(tr)
  if n == 0 then err("no items to clone on track: " .. tostring(p.src_track)) end
  local offset_s = tonumber(p.offset_s) or 0.0
  local take_vol = 10.0 ^ ((tonumber(p.item_volume_db) or 0.0) / 20.0)
  local desc = string.format("Orpheus: clone %d items to %s", n, tostring(new_name))

  reaper.Undo_BeginBlock()
  local dest = nil
  local cloned, skipped_midi, failed = 0, 0, {}
  -- Same shape as import_audio_files: the loop runs inside a pcall so NO raise can
  -- escape with the undo block still open; per-item failures are caught one level in.
  local ok_all, loop_err = pcall(function()
    local count = reaper.CountTracks(0)
    reaper.InsertTrackAtIndex(count, true)   -- ONE new track, appended at the end
    dest = reaper.GetTrack(0, count)
    if not dest then error("could not create the clone track: " .. tostring(new_name), 0) end
    reaper.GetSetMediaTrackInfo_String(dest, "P_NAME", new_name, true)
    for i = 0, n - 1 do
      local src_item = reaper.GetTrackMediaItem(tr, i)
      local src_take = reaper.GetActiveTake(src_item)
      if src_take and reaper.TakeIsMIDI(src_take) then
        skipped_midi = skipped_midi + 1
      else
        local one_ok, one = pcall(clone_one_item, src_item, src_take, dest, offset_s, take_vol)
        if one_ok then
          cloned = cloned + 1
        else
          failed[#failed + 1] = { item_index = i + 1, error = tostring(one) }
        end
      end
    end
    reaper.UpdateArrange()   -- ONCE, after every item (not per clone)
  end)
  reaper.Undo_EndBlock(desc, -1)
  if not ok_all then err(tostring(loop_err)) end

  return {
    track_guid = reaper.GetTrackGUID(dest),
    track_name = tostring(new_name),
    cloned = cloned,
    skipped_midi = skipped_midi,
    failed = failed,
  }
end

-- set_track_params — set a track's fader volume (dB) and/or pan (-1..1, clamped),
-- then read BOTH back so the result reports what the track now actually holds.
HANDLERS.set_track_params = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local vdb = tonumber(p.volume_db)
  if p.volume_db ~= nil and vdb then
    reaper.SetMediaTrackInfo_Value(tr, "D_VOL", 10.0 ^ (vdb / 20.0))
  end
  local pan = tonumber(p.pan)
  if p.pan ~= nil and pan then
    if pan < -1.0 then pan = -1.0 elseif pan > 1.0 then pan = 1.0 end
    reaper.SetMediaTrackInfo_Value(tr, "D_PAN", pan)
  end
  -- D_VOL is a linear gain; 0 -> -inf dB, clamped to -150 (the same floor list_tracks
  -- uses) so the JSON result never carries a non-finite number.
  local vol = reaper.GetMediaTrackInfo_Value(tr, "D_VOL")
  local vol_db = (vol and vol > 0) and (20.0 * math.log(vol, 10)) or -150.0
  return {
    track_guid = reaper.GetTrackGUID(tr),
    volume_db = vol_db,
    pan = reaper.GetMediaTrackInfo_Value(tr, "D_PAN"),
  }
end

-- trim_track_items — clip every AUDIO item on a track to the window [start_s, end_s]
-- (either bound optional; motivating case: cutting the main vocal stack's tail so a
-- cloned, differently-processed copy — e.g. a telephone-EQ ending — can take over).
-- A left trim advances D_POSITION and D_STARTOFFS together so the surviving audio
-- keeps playing at the same timeline moment. An item that falls ENTIRELY outside the
-- window cannot be deleted by this verb (no delete by design) — it is collapsed to
-- 1 ms at the window edge and COUNTED under `collapsed`, never silently skipped.
-- MIDI takes are counted and untouched. ONE undo block.
HANDLERS.trim_track_items = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local w0 = tonumber(p.start_s)
  local w1 = tonumber(p.end_s)
  if w0 == nil and w1 == nil then
    err("trim_track_items: at least one of start_s / end_s is required")
  end
  if w0 and w1 and w1 <= w0 then err("trim_track_items: end_s must be > start_s") end
  local n = reaper.CountTrackMediaItems(tr)
  if n == 0 then err("no items to trim on track: " .. tostring(p.track)) end

  reaper.Undo_BeginBlock()
  local trimmed, skipped_midi, collapsed, untouched = 0, 0, 0, 0
  local ok_all, loop_err = pcall(function()
    for i = 0, n - 1 do
      local item = reaper.GetTrackMediaItem(tr, i)
      local take = reaper.GetActiveTake(item)
      if take and reaper.TakeIsMIDI(take) then
        skipped_midi = skipped_midi + 1
      elseif take then
        local pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
        local len = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
        local lo = w0 or pos
        local hi = w1 or (pos + len)
        local new_pos = math.max(pos, lo)
        local new_end = math.min(pos + len, hi)
        if new_end <= new_pos then
          -- Entirely outside: collapse to 1 ms at the nearest window edge.
          local edge = (pos + len <= lo) and lo or hi
          reaper.SetMediaItemInfo_Value(item, "D_POSITION", edge)
          reaper.SetMediaItemInfo_Value(item, "D_LENGTH", 0.001)
          collapsed = collapsed + 1
        elseif new_pos == pos and new_end == pos + len then
          untouched = untouched + 1
        else
          if new_pos > pos then
            local offs = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
            reaper.SetMediaItemTakeInfo_Value(take, "D_STARTOFFS", offs + (new_pos - pos))
            reaper.SetMediaItemInfo_Value(item, "D_POSITION", new_pos)
          end
          reaper.SetMediaItemInfo_Value(item, "D_LENGTH", new_end - new_pos)
          trimmed = trimmed + 1
        end
      end
    end
    reaper.UpdateArrange()
  end)
  reaper.Undo_EndBlock(string.format("Orpheus: trim items to [%s, %s]",
    w0 and string.format("%.2f", w0) or "-", w1 and string.format("%.2f", w1) or "-"), -1)
  if not ok_all then err(tostring(loop_err)) end
  return {
    track_guid = reaper.GetTrackGUID(tr),
    trimmed = trimmed,
    untouched = untouched,
    collapsed = collapsed,
    skipped_midi = skipped_midi,
  }
end

-- --------------------------------------------------------------------------- --
-- Automation envelopes — set_envelope_points / get_envelope_points (motivating
-- case: risers, filter sweeps, fade automation — the "small stuff" a static FX
-- chain cannot express). Value semantics at the wire: volume points arrive in dB
-- (converted to the envelope's native scale via the Scale*EnvelopeMode API pair),
-- pan points in -1..1, FX-param points NORMALIZED 0..1 (unit translation stays
-- python-side, the exact same split set_fx_param uses).
-- --------------------------------------------------------------------------- --

local MAX_ENV_POINTS = 512

-- Resolve an envelope spec to (env, kind, desc). kind is "volume"/"pan"/"fx".
-- spec: the string "volume" or "pan", or a table {fx_index, param_index}.
-- create_fx: passed to GetFXEnvelope (get=false so a read never creates).
local function resolve_envelope(tr, spec, create_fx)
  if spec == "volume" or spec == "pan" then
    local name = spec == "volume" and "Volume" or "Pan"
    local env = reaper.GetTrackEnvelopeByName(tr, name)
    if not env and reaper.GetTrackEnvelopeByChunkName then
      -- verify live: chunk names for the track volume/pan envelopes on REAPER 7.x.
      -- GetTrackEnvelopeByName only finds VISIBLE envelopes; the chunk route can find
      -- one that exists but is hidden. Neither CREATES a volume/pan envelope.
      env = reaper.GetTrackEnvelopeByChunkName(tr, spec == "volume" and "VOLENV2" or "PANENV2")
    end
    return env, spec, spec
  end
  if type(spec) == "table" and spec.fx_index ~= nil and spec.param_index ~= nil then
    local fi = math.floor(tonumber(spec.fx_index) or -1)
    local pi = math.floor(tonumber(spec.param_index) or -1)
    if fi < 0 or pi < 0 then return nil, nil, nil end
    local env = reaper.GetFXEnvelope(tr, fi, pi, create_fx and true or false)
    return env, "fx", string.format("fx %d param %d", fi, pi)
  end
  return nil, nil, nil
end

-- dB -> the volume envelope's native scale (fader scaling aware), and back.
local function vol_db_to_native(env, db)
  local mode = reaper.GetEnvelopeScalingMode(env)
  return reaper.ScaleToEnvelopeMode(mode, 10.0 ^ (db / 20.0))
end

local function vol_native_to_db(env, native)
  local mode = reaper.GetEnvelopeScalingMode(env)
  local linear = reaper.ScaleFromEnvelopeMode(mode, native)
  -- Same -150 dB floor set_track_params uses: the JSON result never carries -inf.
  return (linear and linear > 0) and (20.0 * math.log(linear, 10)) or -150.0
end

HANDLERS.set_envelope_points = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local points = p.points
  if type(points) ~= "table" or #points == 0 then
    err("set_envelope_points: points is required and must be non-empty")
  end
  if #points > MAX_ENV_POINTS then
    err("too many envelope points in one call: " .. #points .. " > " .. MAX_ENV_POINTS)
  end
  local env, kind, desc = resolve_envelope(tr, p.envelope, true)
  if not kind then
    err("set_envelope_points: envelope must be 'volume', 'pan', or {fx_index, param_index}")
  end
  if not env then
    -- No API-side way to CREATE a track volume/pan envelope without touching the
    -- user's selection state — the human path is explicit instead of a silent no-op.
    err(desc .. " envelope not visible on track: " .. tostring(p.track) ..
      " — open it once in REAPER (track envelopes button, enable " .. desc .. " trim/read)")
  end
  local clear = p.clear
  if clear == nil then clear = true end

  reaper.Undo_BeginBlock()
  local inserted = 0
  local ok_all, loop_err = pcall(function()
    if clear then
      local tmin, tmax = math.huge, -math.huge
      for _, pt in ipairs(points) do
        local t = tonumber(pt.time_s) or 0.0
        if t < tmin then tmin = t end
        if t > tmax then tmax = t end
      end
      -- verify live: DeleteEnvelopePointRange treats the range as [start, end); the
      -- epsilon makes a pre-existing point at exactly tmax get replaced, not doubled.
      reaper.DeleteEnvelopePointRange(env, tmin, tmax + 1e-6)
    end
    for _, pt in ipairs(points) do
      local t = tonumber(pt.time_s) or 0.0
      local v = tonumber(pt.value) or 0.0
      if kind == "volume" then
        v = vol_db_to_native(env, v)
      elseif kind == "pan" then
        -- verify live: caller semantics match set_track_params' D_PAN (-1 left, +1
        -- right); confirm the pan ENVELOPE stores the same sign on REAPER 7.x.
        if v < -1.0 then v = -1.0 elseif v > 1.0 then v = 1.0 end
      else
        if v < 0.0 then v = 0.0 elseif v > 1.0 then v = 1.0 end
      end
      reaper.InsertEnvelopePoint(env, t, v, 0, 0, false, true)   -- noSortIn=true
      inserted = inserted + 1
    end
    reaper.Envelope_SortPoints(env)
  end)
  reaper.Undo_EndBlock(string.format("Orpheus: set %d envelope points (%s)", #points, desc), -1)
  if not ok_all then err(tostring(loop_err)) end
  return {
    track_guid = reaper.GetTrackGUID(tr),
    envelope = desc,
    inserted = inserted,
    cleared = clear and true or false,
  }
end

HANDLERS.get_envelope_points = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local env, kind, desc = resolve_envelope(tr, p.envelope, false)
  if not kind then
    err("get_envelope_points: envelope must be 'volume', 'pan', or {fx_index, param_index}")
  end
  local points = {}
  if env then
    local n = reaper.CountEnvelopePoints(env)
    for i = 0, n - 1 do
      local ok, t, v = reaper.GetEnvelopePoint(env, i)
      if ok then
        if kind == "volume" then v = vol_native_to_db(env, v) end
        points[#points + 1] = { time_s = t, value = v }
      end
    end
  end
  -- A missing envelope reads as EMPTY, never an error — "no automation" is a normal
  -- answer for the analysis loop, unlike a write that would silently go nowhere.
  return { track_guid = reaper.GetTrackGUID(tr), envelope = desc, points = points }
end

-- clone_item_region — clone a TIME REGION of a track's audio onto ONE new appended
-- track at a chosen destination position (motivating case: re-arranging pieces of a
-- recorded take — e.g. layering a background chant sung in section N over the lyrics
-- sung in section N+1). Region semantics: the FIRST audio item overlapping
-- [src_start_s, src_end_s) is windowed — the region is clipped to that item's bounds,
-- the clip's take start-offset advances by the in-item skip, and the new item lands at
-- dest_s shifted by the same clip amount, so source audio at src time T plays at
-- exactly dest_s + (T - src_start_s). Same promises as clone_track_items: re-opens the
-- take's own source file (owned-audio boundary), ONE undo block, f4394e9 tempo pins.
HANDLERS.clone_item_region = function(p)
  local tr, rerr = resolve_track(p.src_track)
  if not tr then err(rerr) end
  local new_name = p.new_track_name
  if not new_name or new_name == "" then err("clone_item_region: new_track_name is required") end
  local s0 = tonumber(p.src_start_s)
  local s1 = tonumber(p.src_end_s)
  local dest = tonumber(p.dest_s)
  if not (s0 and s1 and dest) then
    err("clone_item_region: src_start_s, src_end_s and dest_s are required")
  end
  if s1 <= s0 then err("clone_item_region: src_end_s must be > src_start_s") end
  local take_vol = 10.0 ^ ((tonumber(p.item_volume_db) or 0.0) / 20.0)

  -- Find the first AUDIO item overlapping the region.
  local src_item, src_take = nil, nil
  for i = 0, reaper.CountTrackMediaItems(tr) - 1 do
    local it = reaper.GetTrackMediaItem(tr, i)
    local tk = reaper.GetActiveTake(it)
    if tk and not reaper.TakeIsMIDI(tk) then
      local pos = reaper.GetMediaItemInfo_Value(it, "D_POSITION")
      local len = reaper.GetMediaItemInfo_Value(it, "D_LENGTH")
      if pos < s1 and (pos + len) > s0 then src_item, src_take = it, tk; break end
    end
  end
  if not src_item then
    err(string.format("no audio item overlaps [%.3f, %.3f) on track: %s",
      s0, s1, tostring(p.src_track)))
  end

  local pos = reaper.GetMediaItemInfo_Value(src_item, "D_POSITION")
  local len = reaper.GetMediaItemInfo_Value(src_item, "D_LENGTH")
  local clip0 = math.max(s0, pos)
  local clip1 = math.min(s1, pos + len)
  local src = reaper.GetMediaItemTake_Source(src_take)
  if not src then err("take has no PCM source") end
  local filename = reaper.GetMediaSourceFileName(src, "")
  if not filename or filename == "" then
    err("source has no filename (in-project/section source)")
  end

  reaper.Undo_BeginBlock()
  local result = nil
  local ok_all, loop_err = pcall(function()
    local new_src = reaper.PCM_Source_CreateFromFile(filename)
    if not new_src then error("could not re-open source file: " .. tostring(filename), 0) end
    local count = reaper.CountTracks(0)
    reaper.InsertTrackAtIndex(count, true)
    local dtr = reaper.GetTrack(0, count)
    if not dtr then
      if reaper.PCM_Source_Destroy then reaper.PCM_Source_Destroy(new_src) end
      error("could not create the clone track: " .. tostring(new_name), 0)
    end
    reaper.GetSetMediaTrackInfo_String(dtr, "P_NAME", new_name, true)
    local item = reaper.AddMediaItemToTrack(dtr)
    if not item then
      if reaper.PCM_Source_Destroy then reaper.PCM_Source_Destroy(new_src) end
      error("could not create a media item on the clone track", 0)
    end
    local take = reaper.AddTakeToMediaItem(item)
    if not take then
      if reaper.PCM_Source_Destroy then reaper.PCM_Source_Destroy(new_src) end
      error("could not create a take on the clone track", 0)
    end
    reaper.SetMediaItemTake_Source(take, new_src)   -- take owns new_src from here
    reaper.SetMediaItemInfo_Value(item, "D_POSITION", dest + (clip0 - s0))
    reaper.SetMediaItemInfo_Value(item, "D_LENGTH", clip1 - clip0)
    reaper.SetMediaItemTakeInfo_Value(take, "D_STARTOFFS",
      reaper.GetMediaItemTakeInfo_Value(src_take, "D_STARTOFFS") + (clip0 - pos))
    reaper.SetMediaItemInfo_Value(item, "C_BEATATTACHMODE", 0)   -- f4394e9 pins
    reaper.SetMediaItemTakeInfo_Value(take, "D_PLAYRATE", 1.0)
    reaper.SetMediaItemTakeInfo_Value(take, "D_VOL", take_vol)
    reaper.UpdateArrange()
    result = {
      track_guid = reaper.GetTrackGUID(dtr),
      track_name = tostring(new_name),
      cloned_start_s = clip0,
      cloned_end_s = clip1,
      dest_s = dest + (clip0 - s0),
      length_s = clip1 - clip0,
    }
  end)
  reaper.Undo_EndBlock(string.format("Orpheus: clone region [%.2f-%.2f] to %.2f on %s",
    s0, s1, dest, tostring(new_name)), -1)
  if not ok_all then err(tostring(loop_err)) end
  return result
end

-- nudge_take_offset — slide the AUDIO inside every audio item on a track by adding
-- delta_s to each active take's D_STARTOFFS, WITHOUT moving the items themselves
-- (motivating case: record-latency sync repair — the take plays delta_s earlier while
-- item positions, and therefore deliberate ADT offsets between clone tracks, stay put).
-- positive delta_s = audio plays EARLIER on the timeline. A resulting negative
-- start-offset is clamped to 0 and COUNTED, never silent; MIDI takes are counted and
-- left alone (audio-only contract, same as clone_track_items). ONE undo block.
HANDLERS.nudge_take_offset = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local delta = tonumber(p.delta_s)
  if delta == nil then err("nudge_take_offset: delta_s is required") end
  local n = reaper.CountTrackMediaItems(tr)
  if n == 0 then err("no items to nudge on track: " .. tostring(p.track)) end

  reaper.Undo_BeginBlock()
  local nudged, skipped_midi, clamped = 0, 0, 0
  local ok_all, loop_err = pcall(function()
    for i = 0, n - 1 do
      local item = reaper.GetTrackMediaItem(tr, i)
      local take = reaper.GetActiveTake(item)
      if take and reaper.TakeIsMIDI(take) then
        skipped_midi = skipped_midi + 1
      elseif take then
        local cur = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
        local new_offs = cur + delta
        if new_offs < 0 then new_offs = 0; clamped = clamped + 1 end
        reaper.SetMediaItemTakeInfo_Value(take, "D_STARTOFFS", new_offs)
        nudged = nudged + 1
      end
    end
    reaper.UpdateArrange()
  end)
  reaper.Undo_EndBlock(string.format("Orpheus: nudge take offset %+.3fs", delta), -1)
  if not ok_all then err(tostring(loop_err)) end
  return {
    track_guid = reaper.GetTrackGUID(tr),
    nudged = nudged,
    skipped_midi = skipped_midi,
    clamped = clamped,
    delta_s = delta,
  }
end

-- add_sampler_zones — build a MULTISAMPLED instrument from user-owned one-shots: one
-- ReaSamplOmatic5000 per zone, each pinned to a note range with semitone pitch-tracking
-- anchored at the zone's root (the sampled note). RS5K live layout (2026-07-31 dump):
-- 3/4 note range start/end (n/127), 5/6 pitch@start/end — normalized over -80..+80 st
-- (semis+80)/160, verify-live: empirically confirmed by pitch-measured renders.
-- Motivating case: slicing real instrument plucks out of the user's own stems and
-- playing MIDI through them (owned-audio sampling, zero external content).
HANDLERS.add_sampler_zones = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local zones = p.zones
  if type(zones) ~= "table" or #zones == 0 then
    err("add_sampler_zones: zones is required and must be non-empty")
  end
  if #zones > 32 then err("too many zones: " .. #zones .. " > 32") end
  reaper.Undo_BeginBlock()
  local added, failed = 0, {}
  local ok_all, loop_err = pcall(function()
    for i, z in ipairs(zones) do
      local file = z.file
      local lo = math.floor(tonumber(z.note_lo) or -1)
      local hi = math.floor(tonumber(z.note_hi) or -1)
      local root = math.floor(tonumber(z.root) or -1)
      if not file or lo < 0 or hi > 127 or hi < lo or root < 0 then
        failed[#failed + 1] = { zone = i, error = "invalid zone spec" }
      else
        local fx = reaper.TrackFX_AddByName(tr, "ReaSamplOmatic5000", false, -1)
        if fx < 0 then
          failed[#failed + 1] = { zone = i, error = "could not add ReaSamplOmatic5000" }
        else
          reaper.TrackFX_SetNamedConfigParm(tr, fx, "FILE0", file)
          reaper.TrackFX_SetNamedConfigParm(tr, fx, "DONE", "")
          reaper.TrackFX_SetParamNormalized(tr, fx, 3, lo / 127.0)
          reaper.TrackFX_SetParamNormalized(tr, fx, 4, hi / 127.0)
          reaper.TrackFX_SetParamNormalized(tr, fx, 5, ((lo - root) + 80) / 160.0)
          reaper.TrackFX_SetParamNormalized(tr, fx, 6, ((hi - root) + 80) / 160.0)
          reaper.TrackFX_SetParamNormalized(tr, fx, 11, 0.0)   -- one-shot (natural decay)
          added = added + 1
        end
      end
    end
  end)
  reaper.Undo_EndBlock(string.format("Orpheus: add %d sampler zones", added), -1)
  if not ok_all then err(tostring(loop_err)) end
  return { track_guid = reaper.GetTrackGUID(tr), zones_added = added, failed = failed }
end

-- delete_track — remove a track from the project (user-mandated 2026-07-30; the
-- original no-delete design stance was overridden by explicit request). Resolves by
-- name/GUID/index like every verb; the result echoes what was deleted so the caller
-- can log it. One undo block — a deletion must be one Ctrl+Z away from recovery.
HANDLERS.delete_track = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local guid = reaper.GetTrackGUID(tr)
  local _, name = reaper.GetTrackName(tr, "")
  reaper.Undo_BeginBlock()
  reaper.DeleteTrack(tr)
  reaper.Undo_EndBlock("Orpheus: delete track " .. tostring(name), -1)
  if reaper.TrackList_AdjustWindows then reaper.TrackList_AdjustWindows(false) end
  return { deleted = true, name = name, track_guid = guid }
end

-- set_fx_preset / get_fx_preset — select a plugin's FACTORY PRESET by index and read
-- back the active preset name + total count. The preset list is the fastest route to
-- musical sounds on instrument VSTs (TAL-NoiseMaker ships ~256 host-visible programs;
-- param-by-param sound design cannot compete for breadth).
HANDLERS.set_fx_preset = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local fx = math.floor(tonumber(p.fx_index) or -1)
  local idx = math.floor(tonumber(p.preset_index) or -1)
  if fx < 0 or idx < 0 then err("set_fx_preset: fx_index and preset_index required (0-based)") end
  local ok = reaper.TrackFX_SetPresetByIndex(tr, fx, idx)
  if not ok then err("could not set preset " .. idx .. " on fx " .. fx) end
  local _, name = reaper.TrackFX_GetPreset(tr, fx, "")
  return { track_guid = reaper.GetTrackGUID(tr), preset_index = idx, preset_name = name or "" }
end

HANDLERS.get_fx_preset = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local fx = math.floor(tonumber(p.fx_index) or -1)
  if fx < 0 then err("get_fx_preset: fx_index required") end
  local idx, count = reaper.TrackFX_GetPresetIndex(tr, fx)
  local _, name = reaper.TrackFX_GetPreset(tr, fx, "")
  return { preset_index = idx, preset_count = count, preset_name = name or "" }
end

-- rename_track — set a track's name (project labeling).
HANDLERS.rename_track = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local name = p.new_name
  if not name or name == "" then err("rename_track: new_name is required") end
  reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", name, true)
  return { track_guid = reaper.GetTrackGUID(tr), name = name }
end

-- add_region — a NAMED, optionally colored REGION spanning [start_s, end_s]: the visual
-- section dividers (chorus/verse/...) point markers cannot express. Seconds, not bars —
-- section boundaries come from audio analysis, which speaks time.
HANDLERS.add_region = function(p)
  local s0 = tonumber(p.start_s)
  local s1 = tonumber(p.end_s)
  if not (s0 and s1) or s1 <= s0 then
    err("add_region: start_s and end_s required, end_s > start_s")
  end
  local color = 0
  if p.color then
    local hexs = tostring(p.color):gsub("^#", "")
    if not hexs:match("^%x%x%x%x%x%x$") then
      err("add_region: color must be '#RRGGBB' (got: " .. tostring(p.color) .. ")")
    end
    color = reaper.ColorToNative(tonumber(hexs:sub(1, 2), 16), tonumber(hexs:sub(3, 4), 16),
      tonumber(hexs:sub(5, 6), 16)) | 0x1000000
  end
  local idx = reaper.AddProjectMarker2(0, true, s0, s1, p.name or "", -1, color)
  return { name = p.name or "", start_s = s0, end_s = s1, index = idx }
end

-- set_track_color — custom track color from "#RRGGBB" (project organization's visual
-- half: group the active stack / reference / retired tiers by color theme).
HANDLERS.set_track_color = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local hexs = tostring(p.color or ""):gsub("^#", "")
  if not hexs:match("^%x%x%x%x%x%x$") then
    err("set_track_color: color must be '#RRGGBB' (got: " .. tostring(p.color) .. ")")
  end
  local r = tonumber(hexs:sub(1, 2), 16)
  local g = tonumber(hexs:sub(3, 4), 16)
  local b = tonumber(hexs:sub(5, 6), 16)
  reaper.SetMediaTrackInfo_Value(tr, "I_CUSTOMCOLOR", reaper.ColorToNative(r, g, b) | 0x1000000)
  if reaper.TrackList_AdjustWindows then reaper.TrackList_AdjustWindows(false) end
  return { track_guid = reaper.GetTrackGUID(tr), color = "#" .. hexs:upper() }
end

-- move_track — move one track to a 1-based position (project organization: group the
-- active stack on top, reference material below, retired scratch at the bottom).
-- ReorderSelectedTracks needs a selection, so the CURRENT selection is snapshotted and
-- restored — the user's selection state must survive a programmatic reorder.
HANDLERS.move_track = function(p)
  local tr, rerr = resolve_track(p.track)
  if not tr then err(rerr) end
  local count = reaper.CountTracks(0)
  local to = math.floor(tonumber(p.to_index) or 0)
  if to < 1 or to > count then err("move_track: to_index out of range 1.." .. count) end
  local sel = {}
  for i = 0, count - 1 do
    local t = reaper.GetTrack(0, i)
    sel[#sel + 1] = { t, reaper.GetMediaTrackInfo_Value(t, "I_SELECTED") }
  end
  reaper.SetOnlyTrackSelected(tr)
  reaper.ReorderSelectedTracks(to - 1, 0)   -- 0-based "move before this index"
  for _, pair in ipairs(sel) do
    reaper.SetMediaTrackInfo_Value(pair[1], "I_SELECTED", pair[2])
  end
  if reaper.TrackList_AdjustWindows then reaper.TrackList_AdjustWindows(false) end
  local new_index = -1
  for i = 0, reaper.CountTracks(0) - 1 do
    if reaper.GetTrack(0, i) == tr then new_index = i + 1; break end
  end
  return { track_guid = reaper.GetTrackGUID(tr), new_index = new_index }
end

-- rebuild_peaks — rebuild the waveform-peak caches for ALL items (REAPER draws items
-- from .reapeaks summary files; replacing a media file's CONTENT in place leaves the
-- cache describing the old audio, so items render blank until re-peaked — the visual
-- sibling of the refresh_media header trap below). Uses the select-all/rebuild/unselect
-- action trio; item selection is not restorable state we track, so it ends deselected.
HANDLERS.rebuild_peaks = function(_)
  reaper.Main_OnCommand(40182, 0)   -- select all items
  reaper.Main_OnCommand(40441, 0)   -- rebuild peaks for selected items
  reaper.Main_OnCommand(40289, 0)   -- unselect all items
  reaper.UpdateArrange()
  return { rebuilt = true }
end

-- refresh_media — force every media source to close and re-open (REAPER's own
-- "Item: Set all media offline" (40100) then "online" (40101) action pair). Needed
-- after a media FILE is replaced on disk while the project has it open: REAPER's
-- PCM sources cache the parsed header (bit depth, data offset) and a changed file
-- silently misreads until re-opened (live incident 2026-07-30: tuned-vocal swap
-- rendered digital silence until refreshed). Touches no selection/render state.
HANDLERS.refresh_media = function(_)
  reaper.Main_OnCommand(40100, 0)   -- all media offline
  reaper.Main_OnCommand(40101, 0)   -- all media online (re-opens every source)
  reaper.UpdateArrange()
  return { refreshed = true }
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

-- --------------------------------------------------------------------------- --
-- Hot reload — the bridge watches its OWN installed file and swaps in updated
-- HANDLERS without the user re-running the action. `orpheus-mcp install-bridge`
-- overwrites the script file; within a couple of seconds the running loop notices
-- the content changed, loads the new file (with ORPHEUS_NO_AUTORUN so the new copy
-- does NOT start a second defer loop), and mutates the LIVE HANDLERS table in
-- place — dispatch() closes over that same table, so new/changed verbs take effect
-- immediately. The loop/tick/heartbeat core itself is NOT hot-swapped (changes to
-- it still need one manual re-run; handler changes — the common case — never do).
-- A broken new file is REPORTED to the console and the old handlers keep serving.
-- --------------------------------------------------------------------------- --

-- Swap the live handler table's contents for another module's handlers, in place.
-- Exposed on M so tests can drive the swap without the file-watch glue.
local function swap_handlers(new_handlers)
  for k in pairs(HANDLERS) do HANDLERS[k] = nil end
  for k, v in pairs(new_handlers) do HANDLERS[k] = v end
end
M.swap_handlers = swap_handlers

local SELF_PATH = nil          -- resolved at autostart via get_action_context
local SELF_CONTENT = nil       -- last-loaded file content (change detector)
local RELOAD_CHECK_EVERY_S = 2.0
local last_reload_check = 0.0

local function read_self()
  if not SELF_PATH then return nil end
  local f = io.open(SELF_PATH, "rb")
  if not f then return nil end
  local content = f:read("*a")
  f:close()
  return content
end

local function check_hot_reload()
  local now = os.clock()
  if now - last_reload_check < RELOAD_CHECK_EVERY_S then return end
  last_reload_check = now
  local content = read_self()
  if not content or content == SELF_CONTENT then return end
  SELF_CONTENT = content   -- consume the change even if the load fails (no retry storm)
  local chunk, cerr = load(content, "@" .. tostring(SELF_PATH))
  if not chunk then
    if reaper.ShowConsoleMsg then
      reaper.ShowConsoleMsg("Orpheus bridge: hot-reload PARSE error (old verbs still live): "
        .. tostring(cerr) .. "\n")
    end
    return
  end
  local saved = _G.ORPHEUS_NO_AUTORUN
  _G.ORPHEUS_NO_AUTORUN = true   -- the new copy must NOT start its own defer loop
  local ok, new_M = pcall(chunk)
  _G.ORPHEUS_NO_AUTORUN = saved
  if ok and type(new_M) == "table" and type(new_M.handlers) == "table" then
    swap_handlers(new_M.handlers)
    if reaper.ShowConsoleMsg then
      reaper.ShowConsoleMsg("Orpheus bridge: hot-reloaded updated handlers.\n")
    end
  elseif reaper.ShowConsoleMsg then
    reaper.ShowConsoleMsg("Orpheus bridge: hot-reload RUN error (old verbs still live): "
      .. tostring(new_M) .. "\n")
  end
end

local function tick()
  touch_heartbeat()
  check_hot_reload()
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
  -- Resolve our own installed path for the hot-reload watcher. get_action_context is
  -- only meaningful when actually run as a REAPER action; absent (or failing), the
  -- watcher stays disarmed and the bridge behaves exactly as before.
  if reaper.get_action_context then
    local ok, _, script_path = pcall(reaper.get_action_context)
    if ok and script_path and script_path ~= "" then
      SELF_PATH = script_path
      SELF_CONTENT = read_self()
    end
  end
  if reaper and reaper.ShowConsoleMsg then
    reaper.ShowConsoleMsg("Orpheus bridge listening in: " .. BRIDGE_DIR ..
      (SELF_PATH and " (hot-reload armed)" or "") .. "\n")
  end
  loop()
end

return M
