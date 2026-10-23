"""Lyric ASR device choice: a native cuDNN abort must never take the MCP server down.

On the owner's desktop, ctranslate2 4.4 (pinned below 4.5 by whisperx 3.3.1) links cuDNN 8, while
torch ships cuDNN 9. ``WhisperModel`` loads fine on CUDA and the first transcribe aborts the whole
process ("Could not locate cudnn_ops_infer64_8.dll", exit 127). Production used to pick CUDA
whenever torch saw a GPU. Now ``auto`` proves CUDA in a throwaway child first, and
``ORPHEUS_ASR_DEVICE`` overrides the choice.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

import orpheus_mcp.analysis.lyrics as lyrics


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(lyrics, "_CUDA_PROBE_RESULT", None)
    monkeypatch.delenv("ORPHEUS_ASR_DEVICE", raising=False)


def _cuda_available(monkeypatch, available: bool) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


def _no_probe(monkeypatch) -> None:
    monkeypatch.setattr(lyrics, "_probe_cuda_asr_in_child", lambda: pytest.fail("probed"))


def test_env_cpu_forces_cpu_without_probing(monkeypatch):
    _cuda_available(monkeypatch, True)
    _no_probe(monkeypatch)
    monkeypatch.setenv("ORPHEUS_ASR_DEVICE", "cpu")
    assert lyrics.resolve_asr_device() == "cpu"


def test_env_cuda_trusts_the_user_without_probing(monkeypatch):
    # Also what the probe child itself runs with, so the probe can never recurse.
    _cuda_available(monkeypatch, True)
    _no_probe(monkeypatch)
    monkeypatch.setenv("ORPHEUS_ASR_DEVICE", " CUDA ")
    assert lyrics.resolve_asr_device() == "cuda"


def test_env_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv("ORPHEUS_ASR_DEVICE", "gpu")
    with pytest.raises(ValueError, match="ORPHEUS_ASR_DEVICE"):
        lyrics.resolve_asr_device()


def test_auto_without_cuda_is_cpu_and_never_probes(monkeypatch):
    _cuda_available(monkeypatch, False)
    _no_probe(monkeypatch)
    assert lyrics.resolve_asr_device() == "cpu"


@pytest.mark.parametrize(("probe_ok", "device"), [(True, "cuda"), (False, "cpu")])
def test_auto_with_cuda_follows_the_child_probe(monkeypatch, probe_ok, device):
    _cuda_available(monkeypatch, True)
    monkeypatch.setattr(lyrics, "_probe_cuda_asr_in_child", lambda: probe_ok)
    assert lyrics.resolve_asr_device() == device


def test_auto_probes_once_per_process(monkeypatch):
    _cuda_available(monkeypatch, True)
    calls: list[int] = []
    monkeypatch.setattr(lyrics, "_probe_cuda_asr_in_child", lambda: calls.append(1) or False)
    assert lyrics.resolve_asr_device() == "cpu"
    assert lyrics.resolve_asr_device() == "cpu"
    assert calls == [1]


def _fake_run(returncode: int, stdout: str):
    def run(cmd, **kwargs):
        assert kwargs["env"]["ORPHEUS_ASR_DEVICE"] == "cuda"  # child skips its own probe
        assert kwargs.get("encoding") == "utf-8"  # never the cp1252 locale default
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    return run


@pytest.mark.parametrize(
    ("returncode", "stdout", "usable"),
    [
        (0, "OK cuda\n", True),
        (127, "", False),  # the native cuDNN abort, contained in the child
        (0, "OK cpu\n", False),  # the child fell back, so CUDA is not proven
    ],
)
def test_child_probe_reads_the_child_verdict(monkeypatch, returncode, stdout, usable):
    monkeypatch.setattr(lyrics.subprocess, "run", _fake_run(returncode, stdout))
    assert lyrics._probe_cuda_asr_in_child() is usable


def test_child_probe_timeout_counts_as_unusable(monkeypatch):
    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(lyrics.subprocess, "run", run)
    assert lyrics._probe_cuda_asr_in_child() is False


def test_load_uses_the_resolved_device(monkeypatch, tmp_path):
    monkeypatch.setattr(lyrics, "resolve_asr_device", lambda: "cpu")
    built: dict[str, str] = {}

    class _FakeModel:
        def __init__(self, model_dir, device, compute_type):
            built.update(device=device, compute_type=compute_type)

    fake = types.SimpleNamespace(WhisperModel=_FakeModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    asr = lyrics.FasterWhisperAsr()
    monkeypatch.setattr(asr, "_checkpoint_dir", lambda: str(tmp_path))
    asr._load()
    assert built == {"device": "cpu", "compute_type": "int8"}


@pytest.mark.desktop
@pytest.mark.gpu
def test_real_probe_contains_the_abort_and_returns_a_verdict():
    """On the desktop the child dies natively (cuDNN 8 missing); the parent must survive and
    get False. Elsewhere it may well be True. Either way: a bool, and we are still running."""
    pytest.importorskip("faster_whisper")
    assert lyrics._probe_cuda_asr_in_child() in (True, False)
