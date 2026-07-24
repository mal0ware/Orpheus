"""A behavioural fake of a REAPER project + the M1 bridge handlers.

This is the *executable spec* for the M1 handlers in ``orpheus_bridge.lua``: it models
enough of REAPER's data (tracks, MIDI items/takes, tempo, time signature) and the exact
beats↔PPQ math the Lua side performs, so the Python tools can be tested over the real wire
protocol without launching REAPER. The Lua handlers and these handlers MUST agree — the
Lua-side tests (tests/lua/test_bridge.lua) guard the other half of that contract.

THE INVARIANT under test: the model speaks beats; PPQ/tempo conversion lives in the bridge.
A note written at beat B must read back at beat B (tests/test_midi_roundtrip.py).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

# REAPER's MIDI takes use 960 ticks per quarter note by default. The conversion the
# bridge performs is beats(QN) -> PPQ and back; we mirror it exactly here.
PPQ_PER_QN = 960


@dataclass
class FakeNote:
    pitch: int
    start_ppq: int
    end_ppq: int
    velocity: int = 96
    channel: int = 0


@dataclass
class FakeTake:
    notes: list[FakeNote] = field(default_factory=list)
    # The take's item start, in project quarter-notes — PPQ is relative to this.
    item_start_qn: float = 0.0
    # The take's item END, in project quarter-notes. Models D_POSITION + D_LENGTH so a
    # regression test can assert the item grows to cover notes written by a later call
    # (e.g. arrange_song writing section 2 at a higher at_bar into the same shared track).
    item_end_qn: float = 0.0


@dataclass
class FakeFxInstance:
    """One FX loaded on a track's chain, for the Task 11.1 add/set/get FX verbs.

    ``params`` holds only EXPLICITLY-SET normalized (0..1) values, keyed by param index —
    a fresh FX reports no params set yet (mirrors a real plugin's compiled-in defaults,
    which this fake does not model). ``num_params`` is a floor on how many params
    ``get_fx_params`` enumerates; it grows automatically to cover any index that has been
    set (see ``get_fx_params`` below), the same way a real FX's param count is fixed at
    load time but this fake doesn't need to know it up front.
    """

    name: str
    params: dict[int, float] = field(default_factory=dict)
    num_params: int = 4


@dataclass
class FakeTrack:
    guid: str
    name: str = ""
    volume_db: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    takes: list[FakeTake] = field(default_factory=list)
    fx: list[str] = field(default_factory=list)
    # The Task 11.1 FX chain (add_fx_by_name / set_fx_param / get_fx_params) — kept
    # separate from `fx` (which add_instrument/list_tracks use) so this task's additions
    # never touch that established behavior.
    fx_chain: list[FakeFxInstance] = field(default_factory=list)
    # I_SOLO as REAPER models it: 0=off, 1=solo, 2=solo-in-place (SIP). Kept distinct from
    # the `solo` bool (which list_tracks reports) because render_track_region round-trips
    # the exact 0/1/2 value and asserts SIP (=2) on the target during a render (§15.2).
    i_solo: int = 0


@dataclass
class FakeReaperProject:
    """The mutable state a real REAPER session would hold for the M1 surface."""

    tempo: float = 120.0
    ts_num: int = 4
    ts_den: int = 4
    tracks: list[FakeTrack] = field(default_factory=list)
    play_state: int = 0
    installed_fx: list[str] = field(default_factory=list)
    markers: list[dict] = field(default_factory=list)
    _guid_seq: int = 0

    # -- render state (§15.2 render_track_region) -------------------------- #
    srate: int = 48000
    master_fxen: int = 1  # master-FX-chain enable (I_FXEN on the master track): 0=bypassed
    render_settings: dict = field(default_factory=dict)  # RENDER_* keys (dir/pattern/format/...)
    render_mutex_held: bool = False  # bridge-side non-reentrant render lock
    time_sel: tuple[float, float] = (0.0, 0.0)  # the current time selection (loop range)
    _render_snapshot: dict | None = None  # last full pre-render snapshot (for best-effort restore)
    # What was in effect at the instant the render wrote its file — lets a test prove the SIP
    # solo (=2) and master-FX bypass (=0) were ACTIVE *during* the render, not merely before.
    render_observed_master_fxen: int | None = None
    render_observed_target_solo: int | None = None

    # -- identity / lookup ------------------------------------------------- #

    def _next_guid(self) -> str:
        self._guid_seq += 1
        return f"{{ORPHEUS-{self._guid_seq:04d}}}"

    def resolve_track(self, ref: str) -> FakeTrack:
        ref = str(ref)
        if ref.startswith("{"):
            for tr in self.tracks:
                if tr.guid == ref:
                    return tr
            raise ValueError(f"no track with GUID {ref}")
        if ref.lstrip("+-").isdigit():
            idx = int(ref) - 1  # 1-based
            if 0 <= idx < len(self.tracks):
                return self.tracks[idx]
            raise ValueError(f"track index out of range: {ref}")
        for tr in self.tracks:
            if tr.name == ref:
                return tr
        raise ValueError(f"no track named '{ref}'")

    # -- musical-time math (the bridge's job, mirrored) -------------------- #

    def qn_per_bar(self) -> float:
        return self.ts_num * (4.0 / self.ts_den)

    def bar_start_qn(self, bar: int) -> float:
        bar = bar if bar and bar >= 1 else 1
        return (bar - 1) * self.qn_per_bar()

    @staticmethod
    def qn_to_ppq(qn: float) -> int:
        return round(qn * PPQ_PER_QN)

    @staticmethod
    def ppq_to_qn(ppq: float) -> float:
        return ppq / PPQ_PER_QN


def _write_wav(path: Path, *, nonsilent: bool, sample_rate: int) -> None:
    """Write a real, tiny mono WAV so render_track_region's non-silent assertion is
    testable end-to-end. A track with items renders a 440 Hz sine (non-zero samples); an
    empty track renders digital silence (all-zero samples) — exactly the §15.2 failure the
    assertion catches. 16-bit PCM here for simplicity; the real render is 32-bit float."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 512
    if nonsilent:
        frames = b"".join(
            struct.pack("<h", int(10000 * math.sin(2 * math.pi * 440 * i / sample_rate)))
            for i in range(n)
        )
    else:
        frames = b"\x00\x00" * n
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)


def _wav_is_silent(path: Path) -> bool:
    with wave.open(str(path), "rb") as w:
        data = w.readframes(w.getnframes())
    return all(b == 0 for b in data)


def make_handlers(project: FakeReaperProject) -> dict:
    """Build the M1 handler dict (fn -> callable(params) -> result) over a project."""

    def get_connection_status(_):
        return {"reaper_version": "7.0/fake", "bridge_dir": "<fake>"}

    def get_project_info(_):
        return {
            "tempo": project.tempo,
            "time_signature": [project.ts_num, project.ts_den],
            "length": 0.0,
            "num_tracks": len(project.tracks),
            "playing": (project.play_state & 1) == 1,
        }

    def list_tracks(_):
        return [
            {
                "index": i + 1,
                "guid": tr.guid,
                "name": tr.name,
                "volume_db": tr.volume_db,
                "pan": tr.pan,
                "mute": tr.mute,
                "solo": tr.solo,
                "num_items": len(tr.takes),
            }
            for i, tr in enumerate(project.tracks)
        ]

    def set_tempo(p):
        project.tempo = float(p["bpm"])
        return {"tempo": project.tempo}

    def set_time_signature(p):
        project.ts_num = int(p["numerator"])
        project.ts_den = int(p["denominator"])
        return {"time_signature": [project.ts_num, project.ts_den]}

    def play_stop_record(p):
        states = {"play": 1, "stop": 0, "record": 5}
        cmd = p["command"]
        if cmd not in states:
            raise ValueError(f"unknown transport command: {cmd}")
        project.play_state = states[cmd]
        return {"command": cmd, "play_state": project.play_state}

    def create_track(p):
        tr = FakeTrack(guid=project._next_guid(), name=p.get("name", ""))
        idx = p.get("index")
        if idx is None:
            project.tracks.append(tr)
            at = len(project.tracks)
        else:
            at = max(1, min(int(idx), len(project.tracks) + 1))
            project.tracks.insert(at - 1, tr)
        return {"guid": tr.guid, "index": at, "name": tr.name}

    def create_midi_item(p):
        tr = project.resolve_track(p["track"])
        start_qn = project.bar_start_qn(p.get("start_bar", 1))
        take = FakeTake(item_start_qn=start_qn)
        tr.takes.append(take)
        return {
            "track": tr.guid,
            "item_index": len(tr.takes) - 1,
            "start_bar": p.get("start_bar", 1),
            "length_bars": p.get("length_bars", 1),
            "start_qn": start_qn,
        }

    def insert_midi_notes(p):
        tr = project.resolve_track(p["track"])
        notes = p.get("notes") or []
        if len(notes) > 512:
            raise ValueError(f"too many notes in one call: {len(notes)} > 512")
        base_qn = project.bar_start_qn(p.get("at_bar", 1))
        max_end = max((n["start_beat"] + n["duration_beats"] for n in notes), default=0.0)
        if not tr.takes:
            tr.takes.append(FakeTake(item_start_qn=base_qn, item_end_qn=base_qn + max_end))
        take = tr.takes[0]
        # Grow the item to cover this call's notes too — mirrors the Lua fix: a later
        # call at a higher at_bar on the SAME shared track must not write notes past the
        # item's right edge.
        take.item_end_qn = max(take.item_end_qn, base_qn + max_end)
        for n in notes:
            start_ppq = project.qn_to_ppq(base_qn + n["start_beat"])
            end_ppq = project.qn_to_ppq(base_qn + n["start_beat"] + n["duration_beats"])
            take.notes.append(
                FakeNote(
                    pitch=n["pitch"],
                    start_ppq=start_ppq,
                    end_ppq=end_ppq,
                    velocity=n.get("velocity", 96),
                    channel=n.get("channel", 0),
                )
            )
        take.notes.sort(key=lambda x: (x.start_ppq, x.pitch))
        return {"track": tr.guid, "inserted": len(notes), "at_bar": p.get("at_bar", 1)}

    def get_track_midi(p):
        tr = project.resolve_track(p["track"])
        if not tr.takes:
            return {"track": tr.guid, "notes": []}
        base_qn = project.bar_start_qn(p.get("at_bar", 1))
        take = tr.takes[0]
        out = []
        for n in take.notes:
            start_qn = project.ppq_to_qn(n.start_ppq)
            end_qn = project.ppq_to_qn(n.end_ppq)
            out.append(
                {
                    "pitch": n.pitch,
                    "start_beat": start_qn - base_qn,
                    "duration_beats": end_qn - start_qn,
                    "velocity": n.velocity,
                    "channel": n.channel,
                }
            )
        return {"track": tr.guid, "notes": out}

    def transpose_notes(p):
        tr = project.resolve_track(p["track"])
        if not tr.takes:
            raise ValueError("track has no MIDI take to transpose")
        semis = int(p.get("semitones", 0))
        moved = 0
        for n in tr.takes[0].notes:
            np = n.pitch + semis
            if 0 <= np <= 127:
                n.pitch = np
                moved += 1
        return {"track": tr.guid, "transposed": moved, "semitones": semis}

    def clear_track_midi(p):
        tr = project.resolve_track(p["track"])
        cleared = 0
        if tr.takes:
            cleared = len(tr.takes[0].notes)
            tr.takes[0].notes.clear()
        return {"track": tr.guid, "cleared": cleared}

    def list_installed_fx(_):
        return {"fx": list(project.installed_fx)}

    def add_instrument(p):
        tr = project.resolve_track(p["track"])
        kind = p["kind"]
        if kind == "drumkit":
            if "ReaSamplOmatic5000" in tr.fx:
                return {"track": tr.guid, "loaded": "drumkit", "already_present": True}
            for _voice, _path in (p.get("samples") or {}).items():
                tr.fx.append("ReaSamplOmatic5000")
            return {"track": tr.guid, "loaded": "drumkit", "already_present": False}
        name = p["name"]
        if name in tr.fx:
            return {"track": tr.guid, "loaded": name, "already_present": True}
        tr.fx.append(name)
        return {"track": tr.guid, "loaded": name, "already_present": False}

    def add_marker(p):
        entry = {"name": p["name"], "bar": int(p["bar"])}
        project.markers.append(entry)
        return {"name": entry["name"], "bar": entry["bar"], "index": len(project.markers)}

    # -- FX verbs (§8 step 3, Task 11.1) -- the executable spec for the Lua handlers ---- #

    def add_fx_by_name(p):
        tr = project.resolve_track(p["track"])
        name = p["name"]
        for i, existing_fx in enumerate(tr.fx_chain):
            if existing_fx.name == name:
                return {"track": tr.guid, "fx_index": i, "already_present": True}
        tr.fx_chain.append(FakeFxInstance(name=name))
        return {"track": tr.guid, "fx_index": len(tr.fx_chain) - 1, "already_present": False}

    def set_fx_param(p):
        tr = project.resolve_track(p["track"])
        fx_index = int(p["fx_index"])
        if not (0 <= fx_index < len(tr.fx_chain)):
            raise ValueError(f"fx index out of range: {fx_index}")
        tr.fx_chain[fx_index].params[int(p["param_index"])] = float(p["value"])
        return {"ok": True}

    def get_fx_params(p):
        tr = project.resolve_track(p["track"])
        fx_index = int(p["fx_index"])
        if not (0 <= fx_index < len(tr.fx_chain)):
            raise ValueError(f"fx index out of range: {fx_index}")
        fx = tr.fx_chain[fx_index]
        count = max(fx.num_params, (max(fx.params) + 1) if fx.params else 0)
        params = [
            {"index": i, "name": f"param{i}", "value": fx.params.get(i, 0.0)}
            for i in range(count)
        ]
        return {"fx_name": fx.name, "params": params}

    # -- render_track_region (§15.2) — the executable spec for the Lua handler ------ #

    def _render_snapshot():
        """Full pre-render snapshot: RENDER_* keys, every track's I_SOLO, master I_FXEN,
        and the time selection — everything a render mutates. Stored on the project so the
        error path can best-effort restore it (mirrors LAST_RENDER_SNAPSHOT in the Lua)."""
        snap = {
            "render_settings": dict(project.render_settings),
            "solos": {tr.guid: tr.i_solo for tr in project.tracks},
            "master_fxen": project.master_fxen,
            "time_sel": project.time_sel,
        }
        project._render_snapshot = snap
        return snap

    def _render_restore(snap):
        if snap is None:
            return
        project.render_settings = dict(snap["render_settings"])
        for tr in project.tracks:
            if tr.guid in snap["solos"]:
                tr.i_solo = snap["solos"][tr.guid]
        project.master_fxen = snap["master_fxen"]
        project.time_sel = snap["time_sel"]
        project._render_snapshot = None

    def render_track_region(p):
        # Step 1 (serialization): a render already in flight refuses a new one — and must
        # NOT touch the in-flight render's mutex on the way out (early return, no acquire).
        if project.render_mutex_held:
            raise ValueError("render busy")
        project.render_mutex_held = True
        try:
            if project.play_state != 0:
                raise ValueError("transport is playing/recording")
            snap = _render_snapshot()
            tr = project.resolve_track(p["track"])
            for t in project.tracks:  # step 3: clear ALL solos first
                t.i_solo = 0
            tr.i_solo = 2  # SIP on the target (§15.2, deliberate — keeps sends)
            project.master_fxen = 0  # step 4: bypass master FX
            project.time_sel = (float(p["start_s"]), float(p["end_s"]))  # step 5
            # step 6: point the render at out_dir/basename.wav, WAV/32f, project srate, master mix
            project.render_settings.update(
                {
                    "RENDER_FILE": p["out_dir"],
                    "RENDER_PATTERN": p["basename"],
                    "RENDER_FORMAT": "wav-32f",
                    "RENDER_SRATE": project.srate,
                    "RENDER_CHANNELS": 2,
                    "RENDER_SETTINGS": 0,
                    "RENDER_BOUNDSFLAG": 2,
                }
            )
            path = Path(p["out_dir"]) / f"{p['basename']}.wav"
            # Record what is in effect at render time so a test can prove SIP + master bypass
            # were ACTIVE during the render (not merely set then immediately reverted).
            project.render_observed_target_solo = tr.i_solo
            project.render_observed_master_fxen = project.master_fxen
            _write_wav(path, nonsilent=bool(tr.takes), sample_rate=project.srate)
            # step 7: the render must exist and not be digital silence.
            if _wav_is_silent(path):
                raise ValueError(f"render is silent (digital silence): {path}")
            _render_restore(snap)  # step 8
            return {
                "path": str(path),
                "start_s": p["start_s"],
                "end_s": p["end_s"],
                "sr": project.srate,
            }
        except Exception:
            _render_restore(project._render_snapshot)  # restore in the error path too (§15.2)
            raise
        finally:
            project.render_mutex_held = False  # cleared in EVERY path

    return {
        "get_connection_status": get_connection_status,
        "get_project_info": get_project_info,
        "list_tracks": list_tracks,
        "set_tempo": set_tempo,
        "set_time_signature": set_time_signature,
        "play_stop_record": play_stop_record,
        "create_track": create_track,
        "create_midi_item": create_midi_item,
        "insert_midi_notes": insert_midi_notes,
        "get_track_midi": get_track_midi,
        "transpose_notes": transpose_notes,
        "clear_track_midi": clear_track_midi,
        "list_installed_fx": list_installed_fx,
        "add_instrument": add_instrument,
        "add_marker": add_marker,
        "add_fx_by_name": add_fx_by_name,
        "set_fx_param": set_fx_param,
        "get_fx_params": get_fx_params,
        "render_track_region": render_track_region,
    }


class FakeReaperBridge:
    """Stand-in for the in-REAPER Lua poll loop. Mirrors the wire protocol exactly.

    A real, threaded implementation of the file protocol the Lua loop speaks: read
    request_N.json → dispatch → write response_N.json atomically → delete request;
    touch heartbeat.lock. It is the executable spec for orpheus_bridge.lua, not a mock.
    """

    def __init__(
        self, bridge_dir: Path, handlers: dict, *, beat: bool = True, answer: bool = True
    ):
        self.dir = Path(bridge_dir)
        self.handlers = handlers
        self._beat = beat
        self._answer = answer
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> FakeReaperBridge:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        if self._beat:
            # Single heartbeat writer is the thread; just wait until it's beaten once.
            deadline = time.monotonic() + 1.0
            while not (self.dir / "heartbeat.lock").exists() and time.monotonic() < deadline:
                time.sleep(0.005)
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._beat:
                # Heartbeat content is never parsed (only its mtime is checked by
                # BridgeClient.is_alive()); skipping one tick is harmless and the next
                # tick retries. Must not let a transient OSError propagate — an
                # uncaught exception here kills the poller thread for the rest of the
                # test (see _atomic_write for why this can happen on Windows).
                with contextlib.suppress(OSError):
                    self._atomic_write(self.dir / "heartbeat.lock", str(time.time()))
            if self._answer:
                for req_file in sorted(self.dir.glob("request_*.json")):
                    self._handle(req_file)
            time.sleep(0.005)

    def _handle(self, req_file: Path) -> None:
        try:
            req = json.loads(req_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            return  # half-written or already gone — atomicity means we just retry next tick
        result = self._dispatch(req.get("fn"), req.get("params") or {})
        self._atomic_write(self.dir / f"response_{req['id']}.json", json.dumps(result))
        req_file.unlink(missing_ok=True)

    def _dispatch(self, fn: str, params: dict) -> dict:
        if fn == "__batch__":
            try:
                results = [
                    self._dispatch(c["fn"], c.get("params") or {})["result"]
                    for c in params["calls"]
                ]
                return {"ok": True, "result": results}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
        handler = self.handlers.get(fn)
        if handler is None:
            return {"ok": False, "error": f"unknown fn: {fn}"}
        try:
            return {"ok": True, "result": handler(params)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # os.replace() below needs delete-share access to `path`. On Windows, a concurrent
    # reader that briefly opens the same path — e.g. BridgeClient.is_alive()'s
    # stat() of heartbeat.lock, called from a *different* thread on every dispatch —
    # can make the OS refuse that rename for a few milliseconds with
    # "PermissionError: [WinError 5] Access is denied". This is a real, if narrow,
    # threading race between this poller thread and any concurrent reader thread:
    # both live in the same process (unlike the real Lua bridge, which is a separate
    # OS process), so their file handles can transiently collide. The read side of
    # this protocol already treats such races as retryable (see _handle and
    # BridgeClient._await_response); the write side needs the same tolerance.
    _REPLACE_RETRIES = 25
    _REPLACE_RETRY_DELAY_S = 0.002

    @classmethod
    def _atomic_write(cls, path: Path, text: str) -> None:
        import tempfile

        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        last_exc: OSError | None = None
        for _ in range(cls._REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                return
            except OSError as exc:
                last_exc = exc
                time.sleep(cls._REPLACE_RETRY_DELAY_S)
        # Retries exhausted (genuinely stuck, not just a transient collision) — clean
        # up our temp file and surface the real error rather than leaving it orphaned.
        with contextlib.suppress(OSError):
            os.remove(tmp)
        assert last_exc is not None
        raise last_exc
