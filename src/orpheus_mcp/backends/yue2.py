"""YuE2 through yue2.cpp's ``yue-synth`` CLI, one process per sketch (spec 2026-10-24 sketch §3).

YuE2's weights are CC BY-NC 4.0, so nothing here runs unless ``ORPHEUS_NC_BACKENDS`` names
``yue2``. The binary and the weights live outside the repo, under ``ORPHEUS_YUE2_DIR``
(default ``~/.orpheus/yue2``); the owner builds and downloads them
(docs/pins/2026-10-yue2-backend.md).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from orpheus_mcp.analysis import model_registry

NC_ENV = "ORPHEUS_NC_BACKENDS"
DIR_ENV = "ORPHEUS_YUE2_DIR"
SKETCH_DIR_ENV = "ORPHEUS_SKETCH_DIR"
EXE_NAME = "yue-synth.exe" if sys.platform == "win32" else "yue-synth"
VAE_NAME = "YuE2-Vae-F32.gguf"
REQUEST_NAME = "request.json"
LOG_NAME = "yue-synth.log"
SETUP_DOC = "docs/pins/2026-10-yue2-backend.md"

_TRUNCATED = re.compile(r"^\[AR\] Semantic song \d+: \d+ tokens \(truncated\)", re.M)
_DONE = re.compile(r"^\[Synth\] Done: ([\d.]+) s of audio, seeds (-?\d+) and (-?\d+)", re.M)
_FATAL = re.compile(r"^.*FATAL.*$", re.M)


class Yue2NotInstalled(RuntimeError):
    """The CLI or a weight file is missing; the message says what and how to set it up."""


class Yue2RenderError(RuntimeError):
    """yue-synth failed, timed out, or wrote no audio."""


def nc_enabled(name: str) -> bool:
    """True when ``ORPHEUS_NC_BACKENDS`` names ``name`` (comma, semicolon or space separated,
    any case). One opt-in per backend."""
    raw = os.environ.get(NC_ENV, "")
    return name.lower() in {p.strip().lower() for p in re.split(r"[,;\s]+", raw) if p.strip()}


def yue2_dir() -> Path:
    env = os.environ.get(DIR_ENV)
    return Path(env) if env else Path.home() / ".orpheus" / "yue2"


def sketch_root() -> Path:
    """Where rendered sketches live: outside the analysis cache, so clearing the cache never
    takes REAPER's sketch items offline."""
    env = os.environ.get(SKETCH_DIR_ENV)
    return Path(env) if env else Path.home() / ".orpheus" / "sketches"


@dataclass(frozen=True)
class Yue2Install:
    command: list[str]  # the CLI to run (tests put a fake script here)
    model: Path
    vae: Path
    bin_dir: Path  # put first on the child's PATH, for the ggml DLLs beside the binary


def find_install(root: Path | None = None) -> Yue2Install:
    """The yue-synth binary and the two GGUF files under ``root`` (default ``yue2_dir()``).
    The backbone prefers Q8_0, the quantisation measured on the 3080."""
    root = root or yue2_dir()
    build = root / "yue2.cpp" / "build"
    exe = next((p for p in (build / "Release" / EXE_NAME, build / EXE_NAME) if p.is_file()), None)
    models = root / "models"
    backbones = sorted(models.glob("YuE2-3B-*.gguf")) if models.is_dir() else []
    model = next((p for p in backbones if p.name == "YuE2-3B-Q8_0.gguf"),
                 backbones[0] if backbones else None)
    vae = models / VAE_NAME
    missing = []
    if exe is None:
        missing.append(f"yue2.cpp/build/Release/{EXE_NAME} (build yue2.cpp)")
    if model is None:
        missing.append("models/YuE2-3B-*.gguf (Q8_0 preferred)")
    if not vae.is_file():
        missing.append(f"models/{VAE_NAME}")
    if missing:
        raise Yue2NotInstalled(
            f"YuE2 is not set up in {root}: missing " + "; ".join(missing) + ". Build yue2.cpp "
            f"and download the two GGUF files as in {SETUP_DOC} (the owner downloads the "
            f"CC BY-NC weights; Orpheus never does), or point {DIR_ENV} at the folder.")
    assert exe is not None and model is not None
    return Yue2Install(command=[str(exe)], model=model, vae=vae, bin_dir=exe.parent)


def build_request(*, style: str, lyrics: str, abc: str, cot: str, duration: float,
                  lm_seed: int, seed: int) -> dict:
    """The request JSON for one sketch. Only the owner's own score is ever rendered, so
    ``abc`` must be present and ``cot`` is ``full`` or ``melody``; seeds are explicit so the
    sketch can be named and replayed."""
    if cot not in ("full", "melody"):
        raise ValueError(f"cot must be 'full' or 'melody' for a scored sketch, not {cot!r}")
    if not abc.strip():
        raise ValueError("a sketch needs the lead sheet's score (abc)")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if lm_seed < 0 or seed < 0:
        raise ValueError("seeds must be explicit (0 or more)")
    return {"style": style, "lyrics": lyrics, "abc": abc, "cot": cot,
            "duration": float(duration), "lm_seed": int(lm_seed), "seed": int(seed),
            "output_format": "wav24"}


@dataclass(frozen=True)
class RenderResult:
    wav: Path
    request: Path
    replay: Path | None
    log: Path
    render_s: float
    audio_s: float | None
    truncated: bool
    lm_seed: int
    seed: int


def render(install: Yue2Install, request: dict, out_dir: Path, stem: str, *,
           timeout: float | None = None) -> RenderResult:
    """Run yue-synth once: ``out_dir/request.json`` in, ``out_dir/<stem>.wav`` and its replay
    ``<stem>.json`` out, the whole log kept as ``yue-synth.log``. Holds Orpheus's GPU
    semaphore for the run so its own GPU stages wait."""
    if stem.lower() == Path(REQUEST_NAME).stem:
        raise ValueError("the output stem 'request' would overwrite request.json "
                         "(yue-synth writes <out-stem>.json)")
    out_dir.mkdir(parents=True, exist_ok=True)
    req_path = out_dir / REQUEST_NAME
    req_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    wav, log_path = out_dir / f"{stem}.wav", out_dir / LOG_NAME
    cmd = [*install.command, "--model", str(install.model), "--vae", str(install.vae),
           "--request", str(req_path), "--out", str(wav)]
    env = dict(os.environ)
    env["PATH"] = str(install.bin_dir) + os.pathsep + env.get("PATH", "")
    limit = timeout if timeout is not None else 300.0 + 3.0 * float(request["duration"])

    started = time.monotonic()
    with model_registry.GPU_SEMAPHORE:  # §15.5: no in-process GPU stage runs alongside YuE2
        try:
            proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                                  cwd=out_dir, env=env, timeout=limit, check=False)
        except subprocess.TimeoutExpired as exc:
            log_path.write_text(_text(exc.stderr) + _text(exc.stdout), encoding="utf-8")
            raise Yue2RenderError(f"yue-synth did not finish in {limit:.0f} s and was stopped; "
                                  f"log: {log_path}") from exc
    render_s = time.monotonic() - started
    log = (proc.stderr or "") + (proc.stdout or "")
    log_path.write_text(log, encoding="utf-8")

    if proc.returncode != 0 or not wav.is_file():
        reason = "; ".join(m.strip() for m in _FATAL.findall(log)) or _tail(log)
        raise Yue2RenderError(f"yue-synth failed (exit {proc.returncode}): {reason}; "
                              f"log: {log_path}")
    done = _DONE.search(log)
    replay = wav.with_suffix(".json")
    return RenderResult(
        wav=wav, request=req_path, replay=replay if replay.is_file() else None, log=log_path,
        render_s=render_s, audio_s=float(done.group(1)) if done else None,
        truncated=bool(_TRUNCATED.search(log)),
        lm_seed=int(done.group(2)) if done else int(request["lm_seed"]),
        seed=int(done.group(3)) if done else int(request["seed"]))


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def _tail(log: str, lines: int = 8) -> str:
    return " | ".join(log.strip().splitlines()[-lines:]) or "no output"
