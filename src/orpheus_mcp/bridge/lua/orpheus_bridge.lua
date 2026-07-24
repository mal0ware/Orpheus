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
local WAV_32F_RENDER_FORMAT = "ZXZhdxwAAAABAAAAAAAAAAIAAAAAAAAAABAAAAMAAAAgAAAA"

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
