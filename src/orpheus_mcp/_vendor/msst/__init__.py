"""Vendored SCNet inference path from ZFTurbo/Music-Source-Separation-Training (MSST).

Upstream: https://github.com/ZFTurbo/Music-Source-Separation-Training
Pinned commit: 83d495dfc81b2ede9bc62f4209619f8bdfd14995
License: MIT (see ``LICENSE`` in this directory — (c) 2024 Roman Solovyev / ZFTurbo).

Scope of the vendoring (Task 0.3, §4 SCNet slot ``4stem-b/scnet-xl``):
- ``scnet.py``      — the SCNet model definition, upstream verbatim.
- ``separation.py`` — the dual-path SeparationNet SCNet depends on, upstream verbatim.
- ``config_scnet_large.yaml`` — upstream ``configs/config_musdb18_scnet_large.yaml`` (the
  SCNet-large / "XL" architecture the pinned ``model_scnet_ep_54_sdr_9.8051.ckpt`` matches),
  verbatim.
- ``inference.py`` — a MINIMAL, dependency-light re-implementation of MSST's generic
  overlap-add ``demix`` loop (``utils/model_utils.py``), written here rather than vendored so
  we do not drag in ml_collections / matplotlib / tqdm / torch.distributed. It runs SCNet-XL
  inference given a checkpoint path + a wav in, writing FLAC stems out.

NO training code, dataset code, or CLI is vendored — inference only. Runtime deps
(``torch``, ``omegaconf``, ``soundfile``, ``numpy``) all live in the ``separation`` extra.
"""

from __future__ import annotations

MSST_UPSTREAM_COMMIT = "83d495dfc81b2ede9bc62f4209619f8bdfd14995"

__all__ = ["MSST_UPSTREAM_COMMIT"]
