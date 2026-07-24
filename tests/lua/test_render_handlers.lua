-- Lua-side tests for render_track_region in orpheus_bridge.lua (§15.2 — the bridge verb).
-- Run: lua tests/lua/test_render_handlers.lua  (also driven in-process via lupa by
-- scripts/run_lua_tests.py). Self-contained: stubs the `reaper` global with an in-memory
-- project that models tracks + I_SOLO, the master track's I_FXEN, the RENDER_* project
-- keys, the time selection, and a Main_OnCommand(42230) that writes a REAL wav file so the
-- non-silent assertion is exercised. This is the Lua half of the §15.2 contract; the
-- Python half is tests/test_render_region.py against the behavioural fake — both must agree.

local passed, failed = 0, 0
local function ok(cond, msg)
  if cond then passed = passed + 1
  else failed = failed + 1; io.stderr:write("FAIL: " .. tostring(msg) .. "\n") end
end
local function eq(a, b, msg)
  ok(a == b, (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end

-- A directory we can actually write render files into, across CI (linux) and the dev box.
local TMP = os.getenv("TEMP") or os.getenv("TMP") or os.getenv("TMPDIR") or "/tmp"

-- --------------------------------------------------------------------------- --
-- In-memory REAPER stub (render surface)
-- --------------------------------------------------------------------------- --

local master = { i_fxen = 1 }

local proj = {
  play_state = 0,
  tracks = {
    { guid = "{T1}", name = "Drums", i_solo = 0, has_items = true },
    { guid = "{T2}", name = "Bass",  i_solo = 2, has_items = true },  -- pre-soloed (SIP)
    { guid = "{T3}", name = "Empty", i_solo = 0, has_items = false }, -- no items -> silent render
  },
  master = master,
  render = {   -- RENDER_* keys, seeded with sentinels we assert get restored intact
    RENDER_FILE = "/original/dir",
    RENDER_PATTERN = "original_pattern",
    RENDER_FORMAT = "ORIGINAL_FORMAT_BLOB",
    RENDER_SRATE = 44100,
    RENDER_CHANNELS = 1,
    RENDER_SETTINGS = 3,     -- non-zero sentinel (stems) — render must reset then restore to 3
    RENDER_BOUNDSFLAG = 1,   -- entire project — render must set 2 then restore to 1
    RENDER_TAILFLAG = 7,
    RENDER_TAILMS = 500,
    RENDER_ADDTOPROJ = 1,
    RENDER_STARTPOS = 0,
    RENDER_ENDPOS = 0,
  },
  srate = 48000,
  time_sel_s = 3.0, time_sel_e = 9.0,   -- the user's existing time selection (must be restored)
  commands = {},                        -- every Main_OnCommand id, in order
  render_time_solos = nil,              -- I_SOLO of each track AT render time (proves the clear loop + SIP)
  render_time_master_fxen = nil,        -- master I_FXEN AT render time (proves the bypass)
}

local function find_track(t)
  for _, tr in ipairs(proj.tracks) do if tr == t then return tr end end
end

local function u32le(n)
  return string.char(n % 256, math.floor(n / 256) % 256,
    math.floor(n / 65536) % 256, math.floor(n / 16777216) % 256)
end

-- A canonical 16-byte-fmt PCM WAV: RIFF/WAVE + "fmt " (16 bytes) + "data" right at byte 37,
-- i.e. the payload starts at the fixed byte 45 a naive header-skip assumes AND where the
-- data-chunk-aware scan finds it too — this is the case where both approaches agree.
local function write_fake_wav(path, nonsilent)
  local payload = nonsilent and string.rep("\1", 128) or string.rep("\0", 128)
  local fmt_data = string.rep("F", 16)   -- fmt chunk body — contents irrelevant to the scan
  local f = assert(io.open(path, "wb"))
  f:write("RIFF")
  f:write(u32le(4 + (8 + #fmt_data) + (8 + #payload)))
  f:write("WAVE")
  f:write("fmt "); f:write(u32le(#fmt_data)); f:write(fmt_data)
  f:write("data"); f:write(u32le(#payload)); f:write(payload)
  f:close()
end

-- A REAPER-realistic 32-bit-float WAV layout: RIFF/WAVE + "fmt " (18-byte IEEE-float fmt
-- chunk) + a "fact" chunk BEFORE "data" — i.e. "data" is NOT at the fixed byte 45 a naive
-- 44-byte header skip assumes. Proves the data-chunk-aware scan finds "data" regardless.
local function write_fake_wav_ext(path, nonsilent)
  local payload = nonsilent and string.rep("\1", 128) or string.rep("\0", 128)
  local fmt_data = string.rep("F", 18)   -- fmt chunk body — contents irrelevant to the scan
  local fact_data = string.rep("N", 4)   -- fact chunk body — contents irrelevant to the scan
  local f = assert(io.open(path, "wb"))
  f:write("RIFF")
  f:write(u32le(4 + (8 + #fmt_data) + (8 + #fact_data) + (8 + #payload)))
  f:write("WAVE")
  f:write("fmt "); f:write(u32le(#fmt_data)); f:write(fmt_data)
  f:write("fact"); f:write(u32le(#fact_data)); f:write(fact_data)
  f:write("data"); f:write(u32le(#payload)); f:write(payload)
  f:close()
end

reaper = {
  GetExtState = function() return "" end,
  GetAppVersion = function() return "7.0/luatest" end,
  RecursiveCreateDirectory = function() end,
  ShowConsoleMsg = function() end,
  defer = function() end,
  EnumerateFiles = function() return nil end,

  GetPlayState = function() return proj.play_state end,
  CountTracks = function() return #proj.tracks end,
  GetTrack = function(_, i) return proj.tracks[i + 1] end,       -- 0-based in
  GetTrackGUID = function(tr) return tr.guid end,
  GetTrackName = function(tr) return true, tr.name end,
  GetMasterTrack = function() return proj.master end,

  GetMediaTrackInfo_Value = function(tr, key)
    if key == "I_FXEN" then return tr.i_fxen or 1 end            -- master (or any track)
    if key == "I_SOLO" then return tr.i_solo or 0 end
    return 0
  end,
  SetMediaTrackInfo_Value = function(tr, key, val)
    if key == "I_FXEN" then tr.i_fxen = val
    elseif key == "I_SOLO" then
      local t = find_track(tr); if t then t.i_solo = val end
    end
    return true
  end,

  GetSetProjectInfo = function(_, key, val, is_set)
    if key == "PROJECT_SRATE" then return proj.srate end
    if is_set then proj.render[key] = val; return val end
    return proj.render[key] or 0
  end,
  GetSetProjectInfo_String = function(_, key, val, is_set)
    if is_set then proj.render[key] = val; return true, val end
    return true, proj.render[key] or ""
  end,

  GetSet_LoopTimeRange = function(is_set, _is_loop, s, e)
    if is_set then proj.time_sel_s, proj.time_sel_e = s, e; return s, e end
    return proj.time_sel_s, proj.time_sel_e
  end,

  Main_OnCommand = function(id)
    proj.commands[#proj.commands + 1] = id
    if id == 42230 then
      -- Capture the state IN EFFECT at render time (proves clear-all-solos + SIP + bypass ran).
      proj.render_time_solos = {}
      for i, tr in ipairs(proj.tracks) do proj.render_time_solos[i] = tr.i_solo end
      proj.render_time_master_fxen = proj.master.i_fxen
      -- The soloed track drives the mix; a track with items renders audio, an empty one silence.
      local target
      for _, tr in ipairs(proj.tracks) do if tr.i_solo == 2 then target = tr end end
      local path = (proj.render.RENDER_FILE or TMP) .. "/" .. (proj.render.RENDER_PATTERN or "out") .. ".wav"
      if proj.use_ext_header then
        write_fake_wav_ext(path, target ~= nil and target.has_items)
      else
        write_fake_wav(path, target ~= nil and target.has_items)
      end
    end
  end,
}

-- --------------------------------------------------------------------------- --
-- Load the bridge WITHOUT starting the defer loop
-- --------------------------------------------------------------------------- --

_G.ORPHEUS_NO_AUTORUN = true
local here = arg[0]:match("(.*[/\\])") or "./"
local M = assert(loadfile(here .. "../../src/orpheus_mcp/bridge/lua/orpheus_bridge.lua"))()

local function call(fn, params) return M.dispatch(fn, params or {}) end

local function has_command(id)
  for _, c in ipairs(proj.commands) do if c == id then return true end end
  return false
end

-- --------------------------------------------------------------------------- --
-- 1. Happy path: render track 1 (has items) — succeeds, dispatches 42230, restores all.
-- --------------------------------------------------------------------------- --
local bn = "render_ok_" .. tostring(os.time())
local r = call("render_track_region", { track = "1", start_s = 1.0, end_s = 2.5,
  out_dir = TMP, basename = bn })
eq(r.ok, true, "render_track_region ok")
eq(r.result.start_s, 1.0, "result echoes start_s")
eq(r.result.end_s, 2.5, "result echoes end_s")
eq(r.result.sr, 48000, "result sr is the project sample rate")
ok(r.result.path:sub(-4) == ".wav", "result path is a .wav")

-- 1a. 42230 dispatched; the "render project (using the most recent settings)" action —
--     NEVER 40015 (which pops the render dialog / can render whole-project).
ok(has_command(42230), "Main_OnCommand(42230) dispatched")
ok(not has_command(40015), "40015 was NEVER dispatched")

-- 1b. During the render the solos were cleared everywhere and SIP (=2) applied to the
--     target only; the master FX chain was bypassed (I_FXEN=0).
eq(proj.render_time_solos[1], 2, "render-time: target (track 1) is SIP=2")
eq(proj.render_time_solos[2], 0, "render-time: pre-soloed track 2 cleared to 0")
eq(proj.render_time_solos[3], 0, "render-time: track 3 stays 0")
eq(proj.render_time_master_fxen, 0, "render-time: master FX bypassed (I_FXEN=0)")

-- 1c. Full restore after a successful render: solos, master FX, RENDER_* keys, time selection.
eq(proj.tracks[1].i_solo, 0, "restore: track 1 solo back to 0")
eq(proj.tracks[2].i_solo, 2, "restore: track 2's pre-existing SIP solo restored")
eq(proj.tracks[3].i_solo, 0, "restore: track 3 solo back to 0")
eq(proj.master.i_fxen, 1, "restore: master FX re-enabled")
eq(proj.render.RENDER_FILE, "/original/dir", "restore: RENDER_FILE (string key)")
eq(proj.render.RENDER_PATTERN, "original_pattern", "restore: RENDER_PATTERN (string key)")
eq(proj.render.RENDER_FORMAT, "ORIGINAL_FORMAT_BLOB", "restore: RENDER_FORMAT (string key)")
eq(proj.render.RENDER_SETTINGS, 3, "restore: RENDER_SETTINGS (numeric key)")
eq(proj.render.RENDER_BOUNDSFLAG, 1, "restore: RENDER_BOUNDSFLAG (numeric key)")
eq(proj.render.RENDER_CHANNELS, 1, "restore: RENDER_CHANNELS (numeric key)")
eq(proj.render.RENDER_SRATE, 44100, "restore: RENDER_SRATE (numeric key)")
eq(proj.time_sel_s, 3.0, "restore: time selection start")
eq(proj.time_sel_e, 9.0, "restore: time selection end")
os.remove(r.result.path)

-- --------------------------------------------------------------------------- --
-- 2. Error path (silent render): render an EMPTY track -> non-silent assertion fails ->
--    ok=false with a "silent" message, and ALL state is still restored (§15.2).
-- --------------------------------------------------------------------------- --
proj.commands = {}
local bn2 = "render_silent_" .. tostring(os.time())
local e = call("render_track_region", { track = "3", start_s = 4.0, end_s = 6.0,
  out_dir = TMP, basename = bn2 })
eq(e.ok, false, "silent render -> ok=false")
ok(tostring(e.error):find("silent"), "error message mentions 'silent'")

-- error-path restore held: nothing leaked despite the mid-render failure.
eq(proj.tracks[1].i_solo, 0, "error-path restore: track 1 solo")
eq(proj.tracks[2].i_solo, 2, "error-path restore: track 2 SIP solo")
eq(proj.tracks[3].i_solo, 0, "error-path restore: track 3 solo")
eq(proj.master.i_fxen, 1, "error-path restore: master FX re-enabled")
eq(proj.render.RENDER_FILE, "/original/dir", "error-path restore: RENDER_FILE")
eq(proj.render.RENDER_SETTINGS, 3, "error-path restore: RENDER_SETTINGS")
eq(proj.time_sel_s, 3.0, "error-path restore: time selection start")
os.remove(TMP .. "/" .. bn2 .. ".wav")  -- best effort; the file is all-zero silence

-- --------------------------------------------------------------------------- --
-- 3. Transport guard: a render is refused while the transport is playing/recording.
-- --------------------------------------------------------------------------- --
proj.play_state = 1
local pe = call("render_track_region", { track = "1", start_s = 0.0, end_s = 1.0,
  out_dir = TMP, basename = "should_not_write" })
eq(pe.ok, false, "render while playing -> ok=false")
ok(tostring(pe.error):find("playing"), "error mentions transport playing")
proj.play_state = 0

-- --------------------------------------------------------------------------- --
-- 4. Extended-header WAV (fake "fact" chunk before "data", as REAPER's real 32-bit-float
--    render produces) with a SILENT payload -> the data-chunk-aware scan must still find
--    the real "data" marker and trip the silence error, not misread fmt/fact bytes as audio
--    (the exact bug a fixed 44-byte header skip has: it would land inside the fact chunk's
--    non-zero filler and could read a genuinely silent render as non-silent).
-- --------------------------------------------------------------------------- --
proj.commands = {}
proj.use_ext_header = true
local bn4 = "render_ext_silent_" .. tostring(os.time())
local e4 = call("render_track_region", { track = "3", start_s = 0.0, end_s = 1.0,
  out_dir = TMP, basename = bn4 })
eq(e4.ok, false, "extended-header silent render -> ok=false")
ok(tostring(e4.error):find("silent"), "extended-header error message mentions 'silent'")
proj.use_ext_header = false
os.remove(TMP .. "/" .. bn4 .. ".wav")

-- --------------------------------------------------------------------------- --
-- 5. Unresolved track: must fail BEFORE any state mutation (§15.2) — the tuple-checked
--    resolve_track() now runs (and errors) before the solo-clear loop, master-FX bypass,
--    render-target write, and Main_OnCommand dispatch, so nothing below it ever executes.
-- --------------------------------------------------------------------------- --
proj.commands = {}
local pre_solos = {}
for i, tr in ipairs(proj.tracks) do pre_solos[i] = tr.i_solo end
local pre_master_fxen = proj.master.i_fxen
local pre_render_file = proj.render.RENDER_FILE
local pre_time_sel_s, pre_time_sel_e = proj.time_sel_s, proj.time_sel_e

local ue = call("render_track_region", { track = "does-not-exist", start_s = 0.0, end_s = 1.0,
  out_dir = TMP, basename = "should_not_write2" })
eq(ue.ok, false, "unresolved track -> ok=false")
ok(tostring(ue.error):find("no track"), "error mentions unresolved track")
ok(not has_command(42230), "unresolved track: render was never dispatched")
for i, tr in ipairs(proj.tracks) do
  eq(tr.i_solo, pre_solos[i], "unresolved track: solo state untouched on track " .. i)
end
eq(proj.master.i_fxen, pre_master_fxen, "unresolved track: master FX untouched")
eq(proj.render.RENDER_FILE, pre_render_file, "unresolved track: RENDER_FILE untouched")
eq(proj.time_sel_s, pre_time_sel_s, "unresolved track: time selection start untouched")
eq(proj.time_sel_e, pre_time_sel_e, "unresolved track: time selection end untouched")

print(string.format("lua render handlers: %d passed, %d failed", passed, failed))
os.exit(failed == 0 and 0 or 1)
