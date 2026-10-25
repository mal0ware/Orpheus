"""The YuE2 backend (spec 2026-10-24 sketch §3): install discovery, the non-commercial opt-in,
the request, and one CLI run per render. The CLI here is tests/fixtures/fake_yue_synth.py,
which prints the real binary's log lines and writes a short WAV plus the replay JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from orpheus_mcp.analysis import model_registry
from orpheus_mcp.backends import yue2

FAKE_CLI = Path(__file__).parent / "fixtures" / "fake_yue_synth.py"


def _install(tmp_path: Path) -> yue2.Yue2Install:
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / "YuE2-3B-Q8_0.gguf").write_bytes(b"x")
    (models / "YuE2-Vae-F32.gguf").write_bytes(b"x")
    return yue2.Yue2Install(command=[sys.executable, "-I", str(FAKE_CLI)],
                            model=models / "YuE2-3B-Q8_0.gguf",
                            vae=models / "YuE2-Vae-F32.gguf", bin_dir=tmp_path / "bin")


def _request(**kw):
    base = {"style": "indie pop, male vocal", "lyrics": "[Verse]\nHold on", "abc": "X:1\n",
            "cot": "full", "duration": 12.0, "lm_seed": 7, "seed": 7}
    base.update(kw)
    return yue2.build_request(**base)


# -- opt-in ---------------------------------------------------------------------------- #

@pytest.mark.parametrize(("value", "on"), [
    (None, False), ("", False), ("yue2", True), ("YuE2", True), ("songbloom, yue2", True),
    ("songbloom;yue2", True), ("yue", False), ("yue2x", False)])
def test_nc_opt_in_names_each_backend(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv(yue2.NC_ENV, raising=False)
    else:
        monkeypatch.setenv(yue2.NC_ENV, value)
    assert yue2.nc_enabled("yue2") is on


# -- install discovery ------------------------------------------------------------------ #

def _lay_out(root: Path, *, exe=True, models=("YuE2-3B-Q8_0.gguf", "YuE2-Vae-F32.gguf")):
    release = root / "yue2.cpp" / "build" / "Release"
    release.mkdir(parents=True)
    if exe:
        (release / yue2.EXE_NAME).write_bytes(b"x")
    (root / "models").mkdir()
    for m in models:
        (root / "models" / m).write_bytes(b"x")
    return release


def test_find_install_uses_the_env_folder(tmp_path, monkeypatch):
    release = _lay_out(tmp_path)
    monkeypatch.setenv(yue2.DIR_ENV, str(tmp_path))
    found = yue2.find_install()
    assert found.command == [str(release / yue2.EXE_NAME)]
    assert found.model.name == "YuE2-3B-Q8_0.gguf"
    assert found.vae.name == "YuE2-Vae-F32.gguf"
    assert found.bin_dir == release


def test_default_install_folder_is_under_the_home_folder(monkeypatch):
    monkeypatch.delenv(yue2.DIR_ENV, raising=False)
    assert yue2.yue2_dir() == Path.home() / ".orpheus" / "yue2"


def test_q8_is_preferred_over_other_quantisations(tmp_path):
    _lay_out(tmp_path, models=("YuE2-3B-Q4_K_M.gguf", "YuE2-3B-Q8_0.gguf", "YuE2-3B-F16.gguf",
                               "YuE2-Vae-F32.gguf"))
    assert yue2.find_install(tmp_path).model.name == "YuE2-3B-Q8_0.gguf"


def test_another_backbone_is_used_when_q8_is_missing(tmp_path):
    _lay_out(tmp_path, models=("YuE2-3B-Q4_K_M.gguf", "YuE2-Vae-F32.gguf"))
    assert yue2.find_install(tmp_path).model.name == "YuE2-3B-Q4_K_M.gguf"


@pytest.mark.parametrize(("exe", "models", "missing"), [
    (False, ("YuE2-3B-Q8_0.gguf", "YuE2-Vae-F32.gguf"), "yue-synth"),
    (True, ("YuE2-Vae-F32.gguf",), "YuE2-3B-*.gguf"),
    (True, ("YuE2-3B-Q8_0.gguf",), "YuE2-Vae-F32.gguf")])
def test_missing_pieces_give_setup_steps(tmp_path, exe, models, missing):
    _lay_out(tmp_path, exe=exe, models=models)
    with pytest.raises(yue2.Yue2NotInstalled) as info:
        yue2.find_install(tmp_path)
    text = str(info.value)
    assert missing in text and yue2.DIR_ENV in text and "2026-10-yue2-backend.md" in text


def test_nothing_installed_at_all(tmp_path):
    with pytest.raises(yue2.Yue2NotInstalled, match="yue-synth"):
        yue2.find_install(tmp_path / "nowhere")


# -- the request ------------------------------------------------------------------------ #

def test_request_carries_exactly_the_agreed_fields():
    req = _request()
    assert req == {"style": "indie pop, male vocal", "lyrics": "[Verse]\nHold on",
                   "abc": "X:1\n", "cot": "full", "duration": 12.0, "lm_seed": 7, "seed": 7,
                   "output_format": "wav24"}


@pytest.mark.parametrize("bad", [{"cot": "off"}, {"duration": 0}, {"lm_seed": -1},
                                 {"abc": ""}])
def test_request_refuses_what_a_sketch_never_sends(bad):
    with pytest.raises(ValueError):
        _request(**bad)


# -- one render through the fake CLI --------------------------------------------------- #

def test_render_writes_request_wav_replay_and_log(tmp_path):
    out = tmp_path / "sketch"
    res = yue2.render(_install(tmp_path), _request(), out, "Verse 1 #7")
    assert res.wav == out / "Verse 1 #7.wav" and res.wav.is_file()
    assert res.request == out / "request.json"
    assert json.loads(res.request.read_text(encoding="utf-8")) == _request()  # not clobbered
    assert res.replay == out / "Verse 1 #7.json"
    assert json.loads(res.replay.read_text(encoding="utf-8"))["semantic_tokens"] == "1,2,3"
    assert "[Synth] Done" in res.log.read_text(encoding="utf-8")
    assert res.truncated is False
    assert res.audio_s == pytest.approx(0.5)
    assert (res.lm_seed, res.seed) == (7, 7)
    assert res.render_s > 0


def test_the_cli_gets_the_model_files_and_its_own_folder_on_path(tmp_path):
    install = _install(tmp_path)
    out = tmp_path / "sketch"
    yue2.render(install, _request(), out, "Chorus #7")
    call = json.loads((out / "fake_cli_call.json").read_text(encoding="utf-8"))
    argv = call["argv"]
    assert argv[argv.index("--model") + 1] == str(install.model)
    assert argv[argv.index("--vae") + 1] == str(install.vae)
    assert argv[argv.index("--request") + 1] == str(out / "request.json")
    assert argv[argv.index("--out") + 1] == str(out / "Chorus #7.wav")
    assert call["path"].split(";" if sys.platform == "win32" else ":")[0] == str(
        install.bin_dir)


def test_an_output_stem_that_would_clobber_the_request_is_refused(tmp_path):
    with pytest.raises(ValueError, match="request"):
        yue2.render(_install(tmp_path), _request(), tmp_path / "s", "request")


def test_truncation_is_read_from_the_log(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_YUE_MODE", "truncate")
    res = yue2.render(_install(tmp_path), _request(), tmp_path / "s", "Verse #7")
    assert res.truncated is True


def test_a_failed_render_reports_the_cli_error_and_keeps_the_log(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_YUE_MODE", "fatal")
    out = tmp_path / "s"
    with pytest.raises(yue2.Yue2RenderError, match="unknown output format") as info:
        yue2.render(_install(tmp_path), _request(), out, "Verse #7")
    assert str(out / "yue-synth.log") in str(info.value)
    assert (out / "yue-synth.log").is_file()


def test_a_non_zero_exit_fails_even_when_audio_was_written(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_YUE_MODE", "fatal_after_wav")
    with pytest.raises(yue2.Yue2RenderError, match="cannot write"):
        yue2.render(_install(tmp_path), _request(), tmp_path / "s", "Verse #7")


def test_a_render_that_hangs_is_stopped(tmp_path, monkeypatch):
    def slow(*_a, **kw):
        raise subprocess.TimeoutExpired(cmd="yue-synth", timeout=kw["timeout"], stderr="[AR] x")
    monkeypatch.setattr(yue2.subprocess, "run", slow)
    with pytest.raises(yue2.Yue2RenderError, match="did not finish"):
        yue2.render(_install(tmp_path), _request(), tmp_path / "s", "Verse #7", timeout=1)


def test_the_gpu_semaphore_is_held_while_the_cli_runs(tmp_path, monkeypatch):
    real_run, seen = subprocess.run, []

    def spy(*a, **kw):
        free = model_registry.GPU_SEMAPHORE.acquire(blocking=False)
        if free:
            model_registry.GPU_SEMAPHORE.release()
        seen.append(free)
        return real_run(*a, **kw)

    monkeypatch.setattr(yue2.subprocess, "run", spy)
    yue2.render(_install(tmp_path), _request(), tmp_path / "s", "Verse #7")
    assert seen == [False]
    assert model_registry.GPU_SEMAPHORE.acquire(blocking=False)  # released afterwards
    model_registry.GPU_SEMAPHORE.release()


def test_sketch_folder_is_outside_the_analysis_cache(monkeypatch):
    monkeypatch.delenv(yue2.SKETCH_DIR_ENV, raising=False)
    assert yue2.sketch_root() == Path.home() / ".orpheus" / "sketches"
    monkeypatch.setenv(yue2.SKETCH_DIR_ENV, "/elsewhere")
    assert yue2.sketch_root() == Path("/elsewhere")
