"""``project_changes``: what changed in the REAPER project since the last check.

Pure diffing of two ``get_project_snapshot`` results (spec 2026-10-23 REAPER sync §2.3), plus
one stored snapshot per project under ``<cache root>/snapshots/``. Tracks and items are matched
by GUID, markers and regions by their displayed number. Float noise is ignored: under 1 ms for
times, 0.05 dB for volume, 0.005 for pan, 0.01 BPM for tempo.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from orpheus_mcp.analysis.cachefs import _default_cache_root
from orpheus_mcp.models import ProjectChange, ProjectChanges, ProjectIdentity

TIME_EPS = 0.001
DB_EPS = 0.05
PAN_EPS = 0.005
BPM_EPS = 0.01


def _as_list(value: Any) -> list:
    """The bridge encodes an empty Lua table as ``{}``; treat it as an empty list."""
    return list(value) if isinstance(value, list) else []


def _where(bar: Any, beat: Any) -> str:
    if bar is None:
        return ""
    if beat is None or abs(float(beat) - 1.0) < 0.01:
        return f"bar {bar}"
    return f"bar {bar} beat {float(beat):g}"


def _kind_word(item: dict) -> str:
    return {"midi": "MIDI item", "audio": "Audio item"}.get(item.get("kind", ""), "Item")


def _stable_order(old: list[str], new: list[str]) -> set[str]:
    """GUIDs in the longest common subsequence of two orders: the tracks that did NOT move."""
    n, m = len(old), len(new)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if old[i] == new[j] else max(dp[i + 1][j], dp[i][j + 1])
    keep: set[str] = set()
    i = j = 0
    while i < n and j < m:
        if old[i] == new[j]:
            keep.add(old[i])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return keep


def _diff_tracks(old: dict, new: dict, out: list[ProjectChange]) -> None:
    ot = {t["guid"]: t for t in _as_list(old.get("tracks"))}
    nt = {t["guid"]: t for t in _as_list(new.get("tracks"))}
    for g, t in nt.items():
        if g not in ot:
            out.append(ProjectChange(
                kind="track_added", track=t["name"], after=t["index"],
                summary=f"Track '{t['name']}' added at position {t['index']}.",
            ))
    for g, t in ot.items():
        if g not in nt:
            out.append(ProjectChange(kind="track_removed", track=t["name"],
                                     summary=f"Track '{t['name']}' removed."))
    common_old = [t["guid"] for t in _as_list(old.get("tracks")) if t["guid"] in nt]
    common_new = [t["guid"] for t in _as_list(new.get("tracks")) if t["guid"] in ot]
    stayed = _stable_order(common_old, common_new)
    for g in common_new:
        a, b = ot[g], nt[g]
        name = b["name"]
        if a["name"] != b["name"]:
            out.append(ProjectChange(kind="track_renamed", track=name, before=a["name"],
                                     after=b["name"],
                                     summary=f"Track '{a['name']}' renamed to '{b['name']}'."))
        if g not in stayed:
            out.append(ProjectChange(kind="track_moved", track=name, before=a["index"],
                                     after=b["index"],
                                     summary=f"Track '{name}' moved from position {a['index']} "
                                             f"to {b['index']}."))
        if bool(a.get("mute")) != bool(b.get("mute")):
            word = "muted" if b.get("mute") else "unmuted"
            out.append(ProjectChange(kind="track_mute", track=name, before=a.get("mute"),
                                     after=b.get("mute"), summary=f"Track '{name}' {word}."))
        if bool(a.get("solo")) != bool(b.get("solo")):
            word = "soloed" if b.get("solo") else "unsoloed"
            out.append(ProjectChange(kind="track_solo", track=name, before=a.get("solo"),
                                     after=b.get("solo"), summary=f"Track '{name}' {word}."))
        va, vb = float(a.get("volume_db", 0.0)), float(b.get("volume_db", 0.0))
        if abs(va - vb) >= DB_EPS:
            out.append(ProjectChange(kind="track_volume", track=name, before=va, after=vb,
                                     summary=f"Track '{name}' volume {va:.1f} dB -> {vb:.1f} dB."))
        pa, pb = float(a.get("pan", 0.0)), float(b.get("pan", 0.0))
        if abs(pa - pb) >= PAN_EPS:
            out.append(ProjectChange(kind="track_pan", track=name, before=pa, after=pb,
                                     summary=f"Track '{name}' pan {pa:+.2f} -> {pb:+.2f}."))
        if a.get("fx_count") != b.get("fx_count"):
            out.append(ProjectChange(kind="track_fx_count", track=name, before=a.get("fx_count"),
                                     after=b.get("fx_count"),
                                     summary=f"Track '{name}' effects: {a.get('fx_count')} -> "
                                             f"{b.get('fx_count')}."))


def _items_by_guid(snap: dict) -> dict[str, tuple[str, str, dict]]:
    """item GUID -> (track GUID, track name, item)."""
    out: dict[str, tuple[str, str, dict]] = {}
    for t in _as_list(snap.get("tracks")):
        for item in _as_list(t.get("items")):
            if item.get("guid"):
                out[item["guid"]] = (t["guid"], t["name"], item)
    return out


def _diff_items(old: dict, new: dict, out: list[ProjectChange]) -> None:
    oi, ni = _items_by_guid(old), _items_by_guid(new)
    for g, (_tg, track, it) in ni.items():
        if g not in oi:
            where = _where(it.get("bar"), it.get("beat"))
            out.append(ProjectChange(kind="item_added", track=track, bar=it.get("bar"),
                                     beat=it.get("beat"),
                                     summary=f"New {_kind_word(it).lower()} on '{track}'"
                                             + (f" at {where}." if where else ".")))
    for g, (_tg, track, it) in oi.items():
        if g not in ni:
            where = _where(it.get("bar"), it.get("beat"))
            out.append(ProjectChange(kind="item_removed", track=track, bar=it.get("bar"),
                                     beat=it.get("beat"),
                                     summary=f"{_kind_word(it)} removed from '{track}'"
                                             + (f" (was at {where})." if where else ".")))
    for g, (track_guid, track, b) in ni.items():
        if g not in oi:
            continue
        old_guid, old_track, a = oi[g]
        where = _where(b.get("bar"), b.get("beat"))
        at = f" at {where}" if where else ""
        moved_track = old_guid != track_guid  # by GUID: a renamed track is the same track
        moved_time = abs(float(a["position_s"]) - float(b["position_s"])) >= TIME_EPS
        if moved_track or moved_time:
            src = _where(a.get("bar"), a.get("beat")) or f"{a['position_s']:.2f} s"
            dst = where or f"{b['position_s']:.2f} s"
            hop = f" from '{old_track}' to '{track}'" if moved_track else f" on '{track}'"
            out.append(ProjectChange(kind="item_moved", track=track, bar=b.get("bar"),
                                     beat=b.get("beat"), before=src, after=dst,
                                     summary=f"{_kind_word(b)} moved{hop}: {src} -> {dst}."))
        la, lb = float(a["length_s"]), float(b["length_s"])
        if abs(la - lb) >= TIME_EPS:
            out.append(ProjectChange(kind="item_resized", track=track, bar=b.get("bar"),
                                     beat=b.get("beat"), before=la, after=lb,
                                     summary=f"{_kind_word(b)} on '{track}'{at} resized: "
                                             f"{la:.2f} s -> {lb:.2f} s."))
        if a.get("midi_hash") != b.get("midi_hash") and b.get("kind") == "midi":
            out.append(ProjectChange(kind="item_notes_changed", track=track, bar=b.get("bar"),
                                     beat=b.get("beat"),
                                     summary=f"Notes changed in the MIDI item on '{track}'{at}."))
        if a.get("source") != b.get("source") and b.get("kind") == "audio":
            out.append(ProjectChange(kind="item_source_changed", track=track, bar=b.get("bar"),
                                     beat=b.get("beat"), before=a.get("source"),
                                     after=b.get("source"),
                                     summary=f"Audio item on '{track}'{at} now plays "
                                             f"'{os.path.basename(str(b.get('source')))}'."))


def _diff_marks(old: dict, new: dict, key: str, word: str, out: list[ProjectChange]) -> None:
    start = "start_s" if key == "regions" else "time_s"
    bar_k = "start_bar" if key == "regions" else "bar"
    beat_k = "start_beat" if key == "regions" else "beat"
    om = {m["index"]: m for m in _as_list(old.get(key))}
    nm = {m["index"]: m for m in _as_list(new.get(key))}
    for i, m in nm.items():
        where = _where(m.get(bar_k), m.get(beat_k))
        if i not in om:
            out.append(ProjectChange(kind=f"{word}_added", bar=m.get(bar_k), beat=m.get(beat_k),
                                     summary=f"{word.capitalize()} '{m['name']}' added"
                                             + (f" at {where}." if where else ".")))
            continue
        a = om[i]
        if a["name"] != m["name"]:
            out.append(ProjectChange(kind=f"{word}_renamed", bar=m.get(bar_k), beat=m.get(beat_k),
                                     before=a["name"], after=m["name"],
                                     summary=f"{word.capitalize()} '{a['name']}' renamed to "
                                             f"'{m['name']}'."))
        moved = abs(float(a[start]) - float(m[start])) >= TIME_EPS
        if key == "regions":
            moved = moved or abs(float(a["end_s"]) - float(m["end_s"])) >= TIME_EPS
        if moved:
            src = _where(a.get(bar_k), a.get(beat_k)) or f"{a[start]:.2f} s"
            out.append(ProjectChange(kind=f"{word}_moved", bar=m.get(bar_k), beat=m.get(beat_k),
                                     before=src, after=where,
                                     summary=f"{word.capitalize()} '{m['name']}' moved: {src} -> "
                                             f"{where or str(round(m[start], 2)) + ' s'}."))
    for i, m in om.items():
        if i not in nm:
            out.append(ProjectChange(kind=f"{word}_removed", bar=m.get(bar_k), beat=m.get(beat_k),
                                     summary=f"{word.capitalize()} '{m['name']}' removed."))


def _tempo_signature(points: list) -> list[tuple]:
    return [
        (round(float(p["time_s"]) / TIME_EPS), round(float(p["bpm"]) / BPM_EPS),
         tuple(p["meter"]), bool(p.get("linear")))
        for p in points
    ]


def _describe_tempo(p: dict) -> str:
    num, den = p["meter"]
    where = _where(p.get("bar"), p.get("beat")) or "start"
    return f"{float(p['bpm']):g} BPM {num}/{den} at {where}"


def diff_snapshots(old: dict, new: dict) -> list[ProjectChange]:
    """Every change from snapshot ``old`` to snapshot ``new``, tracks first."""
    out: list[ProjectChange] = []
    _diff_tracks(old, new, out)
    _diff_items(old, new, out)
    _diff_marks(old, new, "markers", "marker", out)
    _diff_marks(old, new, "regions", "region", out)
    ot, nt = _as_list(old.get("tempo_map")), _as_list(new.get("tempo_map"))
    if _tempo_signature(ot) != _tempo_signature(nt):
        out.append(ProjectChange(
            kind="tempo_map_changed",
            before=[_describe_tempo(p) for p in ot],
            after=[_describe_tempo(p) for p in nt],
            summary="Tempo/time-signature map changed: " + "; ".join(_describe_tempo(p) for p in nt)
            + ".",
        ))
    la = float(old.get("project", {}).get("length_s", 0.0))
    lb = float(new.get("project", {}).get("length_s", 0.0))
    if abs(la - lb) >= TIME_EPS:
        out.append(ProjectChange(kind="project_length_changed", before=la, after=lb,
                                 summary=f"Project length {la:.1f} s -> {lb:.1f} s."))
    return out


# --------------------------------------------------------------------------- store


def snapshot_dir() -> Path:
    return _default_cache_root() / "snapshots"


def project_key(project: dict) -> str:
    """One snapshot file per project: the saved path, or the name for an unsaved project."""
    ident = project.get("path") or f"unsaved:{project.get('name', '')}"
    return hashlib.sha1(ident.encode("utf-8")).hexdigest()  # noqa: S324 (a file name, not security)


def load_snapshot(key: str) -> dict | None:
    path = snapshot_dir() / f"{key}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_snapshot(key: str, snap: dict) -> None:
    d = snapshot_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{key}.json.part"
    tmp.write_text(json.dumps(snap), encoding="utf-8")
    os.replace(tmp, d / f"{key}.json")


# --------------------------------------------------------------------------- the check


def check_changes(snap: dict) -> ProjectChanges:
    """Compare a fresh snapshot with the stored one for the same project, store the fresh
    one, and report. The first check of a project only records a baseline."""
    proj = snap.get("project") or {}
    identity = ProjectIdentity(
        name=proj.get("name") or "",
        path=proj.get("path") or "",
        length_s=float(proj.get("length_s") or 0.0),
        state_change_count=proj.get("state_change_count"),
    )
    key = project_key(proj)
    prev = load_snapshot(key)
    if prev is None:
        save_snapshot(key, snap)
        return ProjectChanges(
            project=identity, baseline_recorded=True,
            note="First check of this project: baseline recorded. Later checks report what "
                 "changed since the previous one.",
        )
    prev_count = (prev.get("project") or {}).get("state_change_count")
    if prev_count is not None and prev_count == identity.state_change_count:
        return ProjectChanges(
            project=identity, previous_state_change_count=prev_count,
            note="No changes since the last check: REAPER's change counter has not moved.",
        )
    changes = diff_snapshots(prev, snap)
    save_snapshot(key, snap)
    return ProjectChanges(
        project=identity,
        previous_state_change_count=prev_count,
        changes=changes,
        truncated=bool(snap.get("truncated") or prev.get("truncated")),
        note=None if changes else (
            "REAPER's change counter moved, but nothing this check tracks changed (for example "
            "a selection, an envelope or an effect parameter)."
        ),
    )
