"""Minimal SCNet-XL inference: checkpoint + wav in, FLAC stems out.

This is a dependency-light re-implementation of MSST's generic-mode ``demix`` overlap-add
loop (upstream ``utils/model_utils.py`` @ commit 83d495d) — same chunking, reflect padding,
linear fade windowing, and batched accumulation, but without ml_collections / matplotlib /
tqdm / torch.distributed. It only ever runs SCNet (the generic, non-htdemucs branch), so all
the demucs-specific code paths are dropped.

Public surface:
    load_scnet(checkpoint_path, config_path=None, device=None) -> (model, config)
    separate_array(model, config, mix, device) -> dict[source_name, np.ndarray]
    separate_file(checkpoint_path, wav_in, out_dir, ...) -> dict[source_name, Path]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

_DEFAULT_CONFIG = Path(__file__).with_name("config_scnet_large.yaml")


def _load_config(config_path: str | Path | None) -> Any:
    from omegaconf import OmegaConf

    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
    return OmegaConf.load(path)


def load_scnet(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    device: str | None = None,
) -> tuple[torch.nn.Module, Any]:
    """Instantiate SCNet from ``config`` and load ``checkpoint_path`` weights onto ``device``.

    ``config_path=None`` uses the bundled SCNet-large config (the architecture the pinned
    ``model_scnet_ep_54_sdr_9.8051.ckpt`` was trained with). ZFTurbo checkpoints are a bare
    ``state_dict`` (or occasionally wrapped under a ``"state_dict"`` key) — both are handled.
    """
    import torch

    from orpheus_mcp._vendor.msst.scnet import SCNet

    config = _load_config(config_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SCNet(**dict(config.model))

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config


def _windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    import torch

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def separate_array(
    model: torch.nn.Module,
    config: Any,
    mix: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    """Run SCNet over ``mix`` (shape ``(channels, time)``, float32) via overlap-add chunks.

    Returns ``{source_name: waveform (channels, time)}`` trimmed back to the input length.
    """
    import numpy as np
    import torch
    import torch.nn.functional as functional

    sources = list(config.model.sources)
    chunk_size = int(config.audio.chunk_size)
    num_overlap = int(config.inference.num_overlap)
    batch_size = int(config.inference.get("batch_size", 1))
    use_amp = bool(config.training.get("use_amp", True)) and device.startswith("cuda")

    mix_t = torch.as_tensor(np.asarray(mix, dtype=np.float32))

    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix_t.shape[-1]
    window = _windowing_array(chunk_size, fade_size)

    if length_init > 2 * border and border > 0:
        mix_t = functional.pad(mix_t, (border, border), mode="reflect")

    req_shape = (len(sources),) + mix_t.shape
    result = torch.zeros(req_shape, dtype=torch.float32)
    counter = torch.zeros(req_shape, dtype=torch.float32)

    with torch.amp.autocast("cuda", enabled=use_amp), torch.inference_mode():
        i = 0
        batch_data: list[torch.Tensor] = []
        batch_locations: list[tuple[int, int]] = []
        while i < mix_t.shape[1]:
            part = mix_t[:, i : i + chunk_size].to(device)
            chunk_len = part.shape[-1]
            pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
            part = functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)
            batch_data.append(part)
            batch_locations.append((i, chunk_len))
            i += step

            if len(batch_data) >= batch_size or i >= mix_t.shape[1]:
                arr = torch.stack(batch_data, dim=0)
                out = model(arr)

                win = window.clone()
                if i - step == 0:  # first chunk: no fade-in
                    win[:fade_size] = 1
                if i >= mix_t.shape[1]:  # last chunk: no fade-out
                    win[-fade_size:] = 1

                for j, (start, seg_len) in enumerate(batch_locations):
                    chunk = out[j, ..., :seg_len].float().cpu()
                    result[..., start : start + seg_len] += chunk * win[..., :seg_len]
                    counter[..., start : start + seg_len] += win[..., :seg_len]

                batch_data.clear()
                batch_locations.clear()

    # Samples no chunk's non-zero window ever touched (possible only for clips shorter than
    # 2*border, where the reflect padding above is skipped) would divide 0/0 -> NaN; clamp the
    # counter so those stay a clean 0 instead. Real songs are always padded and fully covered.
    counter = torch.where(counter == 0, torch.ones_like(counter), counter)
    estimated = (result / counter).numpy()
    if length_init != estimated.shape[-1]:  # strip the reflect padding
        estimated = estimated[..., border : border + length_init]
    return {name: estimated[idx] for idx, name in enumerate(sources)}


def separate_file(
    checkpoint_path: str | Path,
    wav_in: str | Path,
    out_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    device: str | None = None,
    output_format: str = "FLAC",
) -> dict[str, Path]:
    """Separate ``wav_in`` with SCNet-XL, writing one ``<source>.flac`` per stem to ``out_dir``.

    Returns ``{source_name: written_path}``. Audio I/O is via soundfile; mono input is
    duplicated to the model's expected channel count.
    """
    import numpy as np
    import soundfile as sf
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, config = load_scnet(checkpoint_path, config_path, device)
    audio_channels = int(config.model.audio_channels)

    data, sr = sf.read(str(wav_in), dtype="float32", always_2d=True)  # (frames, channels)
    mix = data.T  # (channels, frames)
    if mix.shape[0] == 1 and audio_channels > 1:
        mix = np.repeat(mix, audio_channels, axis=0)

    stems = separate_array(model, config, mix, device)

    written: dict[str, Path] = {}
    ext = output_format.lower()
    subtype = "PCM_16" if ext == "flac" else None
    for name, wav in stems.items():
        dest = out_dir / f"{name}.{ext}"
        sf.write(str(dest), wav.T, sr, subtype=subtype)
        written[name] = dest
    return written
