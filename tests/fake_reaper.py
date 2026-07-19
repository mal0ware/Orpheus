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

import json
import os
import threading
import time
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
        if not tr.takes:
            tr.takes.append(FakeTake(item_start_qn=base_qn))
        take = tr.takes[0]
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

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        import tempfile

        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
