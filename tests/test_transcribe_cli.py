"""Task 0.5 tests: T5 lane (isolated basic-pitch subprocess env).

CI-safe tests mock ``subprocess.run`` and assert the exact argv contract plus CSV parsing
(header verified against basic-pitch's upstream ``save_note_events``: ``start_time_s,
end_time_s,pitch_midi,velocity,pitch_bend``). The one real-engine test is
``@pytest.mark.desktop`` — it needs the isolated env actually installed
(``uv run python scripts/setup_transcribe_env.py``) and is deselected in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orpheus_mcp.analysis import transcribe_cli
from orpheus_mcp.analysis.transcribe_cli import NoteEvent, TranscribeEnvMissing, transcribe

_FIXTURE_CSV = (
    "start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\n"
    "0.023,0.512,60,90\n"
    "0.512,1.001,64,80,10,12,15\n"
)


def _fake_cli(tmp_path: Path) -> Path:
    cli = tmp_path / "env" / "basic-pitch.exe"
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text("", encoding="utf-8")
    return cli


# --------------------------------------------------------------------------- env-missing error
def test_transcribe_raises_when_env_missing(tmp_path, monkeypatch):
    missing_cli = tmp_path / "no-such-env" / "basic-pitch.exe"
    monkeypatch.setattr(transcribe_cli, "transcribe_env_cli", lambda: missing_cli)

    with pytest.raises(
        TranscribeEnvMissing, match=r"uv run python scripts/setup_transcribe_env\.py"
    ):
        transcribe(tmp_path / "mix.wav", tmp_path / "out")


# --------------------------------------------------------------------------- subprocess contract
def test_transcribe_invokes_expected_argv_and_parses_csv(tmp_path, monkeypatch):
    cli = _fake_cli(tmp_path)
    monkeypatch.setattr(transcribe_cli, "transcribe_env_cli", lambda: cli)

    wav = tmp_path / "mix.wav"
    wav.write_bytes(b"")
    out_dir = tmp_path / "out"

    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mix_basic_pitch.csv").write_text(_FIXTURE_CSV, encoding="utf-8")
        return transcribe_cli.subprocess.CompletedProcess(argv, 0, stdout="✨ done", stderr="")

    monkeypatch.setattr(transcribe_cli.subprocess, "run", fake_run)

    events = transcribe(wav, out_dir)

    assert captured["argv"] == [str(cli), str(out_dir), str(wav), "--save-note-events"]
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    # Inside the MCP server, stdout IS the JSON-RPC channel: the child's banner must be
    # captured, never inherited, and decoded as UTF-8 rather than the cp1252 locale.
    assert captured["capture_output"] is True
    assert captured["encoding"] == "utf-8"
    assert events == [
        NoteEvent(0.023, 0.512, 60, 90, None),
        NoteEvent(0.512, 1.001, 64, 80, [10, 12, 15]),
    ]


def test_transcribe_failure_raises_with_the_childs_stderr(tmp_path, monkeypatch):
    cli = _fake_cli(tmp_path)
    monkeypatch.setattr(transcribe_cli, "transcribe_env_cli", lambda: cli)

    def fake_run(argv, **kwargs):
        return transcribe_cli.subprocess.CompletedProcess(
            argv, 2, stdout="", stderr="Traceback (most recent call last)\nValueError: bad wav"
        )

    monkeypatch.setattr(transcribe_cli.subprocess, "run", fake_run)
    with pytest.raises(transcribe_cli.subprocess.CalledProcessError) as info:
        transcribe(tmp_path / "mix.wav", tmp_path / "out")
    assert "bad wav" in info.value.stderr


# --------------------------------------------------------------------------- CSV parsing
def test_parse_note_events_csv_skips_malformed_rows():
    text = (
        "start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\n"
        "0.0,0.5,60,90\n"
        "not,a,valid,row\n"
        "1.0,1.5,62,70\n"
    )
    events = transcribe_cli._parse_note_events_csv(text)
    assert len(events) == 2
    assert events[0].pitch_midi == 60
    assert events[1].pitch_midi == 62


def test_parse_note_events_csv_raises_when_all_rows_malformed():
    text = (
        "start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\n"
        "bad,row,one,x\n"
        "bad,row,two,y\n"
    )
    with pytest.raises(ValueError, match="all 2 data row"):
        transcribe_cli._parse_note_events_csv(text)


def test_parse_note_events_csv_empty_is_not_an_error():
    text = "start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\n"
    assert transcribe_cli._parse_note_events_csv(text) == []


# --------------------------------------------------------------------------- real-engine smoke
@pytest.mark.desktop
def test_transcribe_real_env_on_ci_sparse_fixture(tmp_path):
    """Runs the real basic-pitch subprocess against the ci_sparse fixture mix.

    Deselected in CI (`-m "not desktop and not gpu"`); needs the isolated env actually
    installed via `uv run python scripts/setup_transcribe_env.py`.
    """
    mix = Path("tests/fixtures/audio/ci_sparse/mix.flac")
    if not mix.is_file():
        pytest.skip("ci_sparse fixture not present")

    cli = transcribe_cli.transcribe_env_cli()
    if not cli.is_file():
        pytest.skip("isolated basic-pitch env not set up (scripts/setup_transcribe_env.py)")

    events = transcribe(mix, tmp_path / "out")

    assert isinstance(events, list)
    for e in events:
        assert isinstance(e, NoteEvent)
        assert e.end_time_s >= e.start_time_s
        assert 0 <= e.pitch_midi <= 127
