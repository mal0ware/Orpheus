-- Lua-side tests for the Task 11.1 FX handlers in orpheus_bridge.lua:
-- add_fx_by_name / set_fx_param / get_fx_params.
-- Run: lua tests/lua/test_fx_handlers.lua
--
-- Self-contained (no shell / no real REAPER), same pattern as test_m1_handlers.lua: stubs
-- the `reaper` global with an in-memory project, then drives the handlers through
-- M.dispatch. Unit/range translation (dB/Hz/ms <-> normalized) is PYTHON-side (tools/mix.py
-- + data/fx_param_maps.json) — this suite only proves the bridge's raw normalized-space
-- contract: add_fx_by_name(track, name) -> {track, fx_index, already_present}; set_fx_param
-- writes a normalized 0..1 value; get_fx_params reads {fx_name, params: [{index, name,
-- value}]} back.

local passed, failed = 0, 0
local function ok(cond, msg)
  if cond then passed = passed + 1
  else failed = failed + 1; io.stderr:write("FAIL: " .. tostring(msg) .. "\n") end
end
local function eq(a, b, msg)
  ok(a == b, (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end
local function approx(a, b, msg)
  ok(math.abs(a - b) < 1e-6, (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end

-- --------------------------------------------------------------------------- --
-- In-memory REAPER stub — tracks + a per-track FX chain (name, params, num_params).
-- --------------------------------------------------------------------------- --

local proj = {
  tracks = {},   -- array of { guid, name, fx = { {name=, params={[idx]=val}, num_params=} } }
  guid_seq = 0,
}

local function next_guid()
  proj.guid_seq = proj.guid_seq + 1
  return string.format("{LUAFX-%04d}", proj.guid_seq)
end

reaper = {
  GetExtState = function(a, b) if a == "orpheus" and b == "bridge_dir" then return "/tmp/x" end return "" end,
  GetAppVersion = function() return "7.0/luatest" end,
  RecursiveCreateDirectory = function() end,
  ShowConsoleMsg = function() end,
  defer = function() end,
  EnumerateFiles = function() return nil end,

  CountTracks = function() return #proj.tracks end,
  GetTrack = function(_, i) return proj.tracks[i + 1] end,  -- 0-based in
  GetTrackGUID = function(tr) return tr.guid end,
  GetTrackName = function(tr) return true, tr.name end,
  InsertTrackAtIndex = function(idx)  -- 0-based
    local tr = { guid = next_guid(), name = "", fx = {} }
    table.insert(proj.tracks, idx + 1, tr)
  end,
  GetSetMediaTrackInfo_String = function(tr, key, val, set)
    if key == "P_NAME" and set then tr.name = val end
    return true, tr.name
  end,

  -- FX chain: tr.fx = {} array of { name, params = {[idx]=normalized}, num_params }.
  TrackFX_AddByName = function(tr, name, _rec, mode)
    tr.fx = tr.fx or {}
    if mode == 0 then  -- find-only
      for i, existing in ipairs(tr.fx) do
        if existing.name == name then return i - 1 end
      end
      return -1
    end
    tr.fx[#tr.fx + 1] = { name = name, params = {}, num_params = 4 }
    return #tr.fx - 1
  end,
  TrackFX_SetParamNormalized = function(tr, fx_idx, param_idx, value)
    local fx = tr.fx[fx_idx + 1]
    if not fx then return false end
    fx.params[param_idx] = value
    return true
  end,
  TrackFX_GetParamNormalized = function(tr, fx_idx, param_idx)
    local fx = tr.fx[fx_idx + 1]
    if not fx then return 0.0 end
    return fx.params[param_idx] or 0.0
  end,
  TrackFX_GetNumParams = function(tr, fx_idx)
    local fx = tr.fx[fx_idx + 1]
    return fx and fx.num_params or 0
  end,
  TrackFX_GetParamName = function(_tr, _fx_idx, param_idx, _buf)
    return true, "param" .. tostring(param_idx)
  end,
  TrackFX_GetFXName = function(tr, fx_idx, _buf)
    local fx = tr.fx[fx_idx + 1]
    return true, fx and fx.name or ""
  end,
}

-- --------------------------------------------------------------------------- --
-- Load the bridge WITHOUT starting the defer loop
-- --------------------------------------------------------------------------- --

_G.ORPHEUS_NO_AUTORUN = true
local here = arg[0]:match("(.*[/\\])") or "./"
local M = assert(loadfile(here .. "../../src/orpheus_mcp/bridge/lua/orpheus_bridge.lua"))()

local function call(fn, params)
  return M.dispatch(fn, params or {})
end

local function create_track(name)
  return call("create_track", { name = name }).result.guid
end

-- 1. add_fx_by_name: first call adds, is NOT already_present, and returns a stable fx_index.
local guid = create_track("Bus")
local first = call("add_fx_by_name", { track = guid, name = "ReaEQ" })
eq(first.ok, true, "add_fx_by_name ok")
eq(first.result.already_present, false, "first add not already present")
eq(first.result.fx_index, 0, "first fx lands at index 0")
eq(first.result.track, guid, "add_fx_by_name echoes the track guid")

-- 2. add_fx_by_name is IDEMPOTENT: a second call for the same plugin does not stack a
-- duplicate instance and reports already_present=true at the SAME index.
local second = call("add_fx_by_name", { track = guid, name = "ReaEQ" })
eq(second.ok, true, "second add_fx_by_name ok")
eq(second.result.already_present, true, "second add reports already_present")
eq(second.result.fx_index, 0, "second add resolves to the same fx_index")

local track_obj
for _, tr in ipairs(proj.tracks) do
  if tr.guid == guid then track_obj = tr end
end
eq(#track_obj.fx, 1, "idempotent add did not stack a duplicate FX instance")

-- 3. a DIFFERENT plugin on the same track gets its own fx_index.
local comp = call("add_fx_by_name", { track = guid, name = "ReaComp" })
eq(comp.result.already_present, false, "different plugin is not already present")
eq(comp.result.fx_index, 1, "second distinct plugin lands at index 1")

-- 4. set_fx_param -> get_fx_params round-trips a normalized value at the wire level.
local set_res = call("set_fx_param", { track = guid, fx_index = 0, param_index = 1, value = 0.75 })
eq(set_res.ok, true, "set_fx_param ok")
eq(set_res.result.ok, true, "set_fx_param result.ok")

local params = call("get_fx_params", { track = guid, fx_index = 0 })
eq(params.ok, true, "get_fx_params ok")
eq(params.result.fx_name, "ReaEQ", "get_fx_params reports the fx name")
ok(#params.result.params >= 2, "get_fx_params enumerates params")
approx(params.result.params[2].value, 0.75, "set value round-trips at index 1 (Lua 1-based -> param_index 1)")
eq(params.result.params[2].index, 1, "returned param carries its 0-based index")
ok(params.result.params[2].name ~= nil and params.result.params[2].name ~= "", "returned param carries a name")

-- an UNSET param on the same fx reads back its untouched default (0.0).
approx(params.result.params[1].value, 0.0, "unset param defaults to 0.0")

-- 5. set_fx_param on an out-of-range fx_index -> ok=false, never silently no-ops.
local bad = call("set_fx_param", { track = guid, fx_index = 99, param_index = 0, value = 0.5 })
eq(bad.ok, false, "set_fx_param on a missing fx -> ok=false")

-- 6. unknown track -> ok=false for every FX verb.
eq(call("add_fx_by_name", { track = "{NOPE}", name = "ReaEQ" }).ok, false, "add_fx_by_name unknown track -> ok=false")
eq(call("get_fx_params", { track = "{NOPE}", fx_index = 0 }).ok, false, "get_fx_params unknown track -> ok=false")

print(string.format("lua fx handlers: %d passed, %d failed", passed, failed))
os.exit(failed == 0 and 0 or 1)
