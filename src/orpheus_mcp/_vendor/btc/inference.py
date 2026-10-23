"""Minimal BTC large-voca chord inference: checkpoint + mono audio in, per-frame chord
predictions out.

A dependency-light re-implementation of the inference-only slice of BTC-ISMIR19 @ 2682317
(upstream ``test.py`` + ``utils/mir_eval_modules.py``): the log-magnitude CQT feature
extraction, the 170-class ``idx → voca-chord`` vocabulary, and the chunked self-attention
forward loop. No training / CRF / mir_eval / dataset / CLI code is vendored.

Public surface:
    load_btc(checkpoint_path, device=None) -> (model, mean, std, device)
    audio_to_features(mono_22050) -> np.ndarray  # (frames, 144), log-magnitude CQT
    predict_frames(model, mean, std, features, device) -> (idx[np.int64], prob[np.float32])
    idx_to_voca() -> list[str]  # length 170, index → upstream voca-chord label string

``FEATURE_PER_SECOND`` (== INST_LEN / TIMESTEP) is the model's frame period in seconds — the
same ``feature_per_second`` upstream derives, used by the caller to place segment times.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

# Upstream ``run_config.yaml`` (model block) with the large-voca overrides from ``test.py``
# (``num_chords = 170``). Everything else is the shipped architecture the pinned checkpoint
# ``btc_model_large_voca.pt`` was trained with.
_MODEL_CONFIG: dict[str, Any] = {
    "feature_size": 144,
    "timestep": 108,
    "num_chords": 170,
    "input_dropout": 0.2,
    "layer_dropout": 0.2,
    "attention_dropout": 0.2,
    "relu_dropout": 0.2,
    "num_layers": 8,
    "num_heads": 4,
    "hidden_size": 128,
    "total_key_depth": 128,
    "total_value_depth": 128,
    "filter_size": 128,
    "probs_out": False,
}

# Upstream feature params (``run_config.yaml`` ``mp3`` + ``feature`` blocks).
SONG_HZ = 22050
INST_LEN = 10.0
_N_BINS = 144
_BINS_PER_OCTAVE = 24
_HOP_LENGTH = 2048
TIMESTEP = _MODEL_CONFIG["timestep"]
FEATURE_PER_SECOND = INST_LEN / TIMESTEP

# Upstream ``mir_eval_modules`` root/quality ordering — the 14-quality large vocabulary.
_ROOT_LIST = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_QUALITY_LIST = (
    "min", "maj", "dim", "aug", "min6", "maj6", "min7",
    "minmaj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4",
)  # fmt: skip


def idx_to_voca() -> list[str]:
    """The upstream ``idx2voca_chord`` mapping as a length-170 list. Index 169 == ``"N"``
    (no chord), 168 == ``"X"`` (unknown); 0..167 are ``root:quality`` (bare ``root`` when the
    quality is ``maj``, matching upstream, which drops the ``:maj``)."""
    table = ["X"] * 170
    table[169] = "N"
    table[168] = "X"
    for i in range(168):
        root = _ROOT_LIST[i // 14]
        quality = _QUALITY_LIST[i % 14]
        table[i] = root if i % 14 == 1 else f"{root}:{quality}"
    return table


def load_btc(
    checkpoint_path: str | Path, device: str | None = None
) -> tuple[Any, float, float, str]:
    """Instantiate ``BTC_model`` and load the large-voca checkpoint. Returns
    ``(model.eval(), mean, std, device)``.

    ``weights_only=False`` is required (the 2019 checkpoint pickles ``mean``/``std`` floats
    alongside the ``state_dict``, which torch 2.6's default ``weights_only=True`` refuses) and
    SAFE here: the file is sha256-pinned in ``data/model_registry.json`` and verified by
    ``fetch-models`` before it is ever read, so no untrusted pickle is executed."""
    import torch

    from orpheus_mcp._vendor.btc.btc_model import BTC_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BTC_model(config=_MODEL_CONFIG).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, float(checkpoint["mean"]), float(checkpoint["std"]), device


def audio_to_features(mono: np.ndarray) -> np.ndarray:
    """Log-magnitude CQT features for a mono waveform already at ``SONG_HZ`` (22050 Hz),
    replicating upstream ``audio_file_to_features``: fixed ``INST_LEN``-second CQT chunks
    concatenated along time, ``log(|cqt| + 1e-6)``, returned transposed to ``(frames, 144)``."""
    import librosa
    import numpy as np

    y = np.ascontiguousarray(mono, dtype=np.float32)
    step = int(SONG_HZ * INST_LEN)
    chunks: list[np.ndarray] = []
    cur = 0
    while len(y) > cur + step:
        seg = y[cur : cur + step]
        chunks.append(
            librosa.cqt(
                seg, sr=SONG_HZ, n_bins=_N_BINS,
                bins_per_octave=_BINS_PER_OCTAVE, hop_length=_HOP_LENGTH,
            )
        )
        cur += step
    chunks.append(
        librosa.cqt(
            y[cur:], sr=SONG_HZ, n_bins=_N_BINS,
            bins_per_octave=_BINS_PER_OCTAVE, hop_length=_HOP_LENGTH,
        )
    )
    feature = np.concatenate(chunks, axis=1)
    feature = np.log(np.abs(feature) + 1e-6)
    return feature.T  # (frames, 144)


def predict_frames(
    model: Any, mean: float, std: float, features: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame chord index + softmax confidence for ``features`` ((frames, 144)).

    Replicates upstream ``test.py``: standardize by ``(x - mean) / std``, zero-pad the time
    axis up to a multiple of ``TIMESTEP``, and run each ``TIMESTEP``-frame window through the
    bi-directional self-attention stack + output projection. The padded tail frames are dropped
    from the returned arrays (length == ``features.shape[0]``). ``prob`` is the per-frame max
    softmax probability (the caller averages it into a segment confidence)."""
    import numpy as np
    import torch
    import torch.nn.functional as functional

    n_real = int(features.shape[0])
    feat = (np.asarray(features, dtype=np.float32) - mean) / std
    num_pad = TIMESTEP - (n_real % TIMESTEP)
    if num_pad == TIMESTEP:
        num_pad = 0
    if num_pad:
        feat = np.pad(feat, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feat.shape[0] // TIMESTEP

    idx_all: list[np.ndarray] = []
    prob_all: list[np.ndarray] = []
    with torch.no_grad():
        model.eval()
        tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        for t in range(num_instance):
            window = tensor[:, TIMESTEP * t : TIMESTEP * (t + 1), :]
            encoded, _ = model.self_attn_layers(window)
            logits = model.output_layer.output_projection(encoded)  # (1, TIMESTEP, 170)
            probs = functional.softmax(logits, dim=-1).squeeze(0)  # (TIMESTEP, 170)
            top_prob, top_idx = torch.max(probs, dim=-1)
            idx_all.append(top_idx.cpu().numpy().astype(np.int64))
            prob_all.append(top_prob.cpu().numpy().astype(np.float32))

    if not idx_all:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    idx = np.concatenate(idx_all)[:n_real]
    prob = np.concatenate(prob_all)[:n_real]
    return idx, prob
