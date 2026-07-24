"""Vendored BTC (Bi-directional Transformer for Chord recognition) inference path.

Upstream: https://github.com/jayg996/BTC-ISMIR19  (ISMIR 2019)
Pinned commit: 2682317be668032e6e4b269ded36adaa2ad57df0
License: MIT (see ``LICENSE`` in this directory — (c) 2019 Jonggwon Park).

Scope of the vendoring (Task 9.1, §13 chord MODEL lane, ``chord/btc-large-voca``):
- ``transformer_modules.py`` — upstream ``utils/transformer_modules.py``, VERBATIM except one
  mechanical numpy-2 fix: ``np.arange(...).astype(np.float)`` → ``.astype(float)`` (``np.float``
  was removed in numpy 1.24). The self-attention / multi-head-attention / feed-forward blocks
  the model is built from.
- ``btc_model.py`` — upstream ``btc_model.py``, VERBATIM except: the two ``from utils.*`` imports
  are rewritten to this package, and the training-only ``if __name__ == "__main__"`` smoke block
  (which needed ``utils.hparams``) is dropped. The ``BTC_model`` definition itself is untouched.
- ``inference.py`` — a MINIMAL, dependency-light inference surface written HERE (not vendored):
  the large-voca feature extraction (log-magnitude CQT, upstream ``audio_file_to_features``
  parameters), the 170-class ``idx → voca-chord`` vocabulary (upstream
  ``mir_eval_modules.idx2voca_chord``), checkpoint loading, and the per-frame forward loop
  (upstream ``test.py``). No training code, dataset code, CRF, mir_eval, or CLI is vendored.

The pinned ``btc_model_large_voca.pt`` checkpoint (registry ``chord/btc-large-voca``,
sha256-verified by ``fetch-models``) is a 2019-era pickle carrying ``{"model", "mean", "std"}``;
``inference.load_btc`` loads it with ``weights_only=False`` — safe precisely because the file is
hash-pinned in the registry, so no untrusted pickle is ever executed (torch 2.6 defaults
``weights_only=True``, which cannot unpickle this checkpoint). Runtime deps (``torch``,
``librosa``, ``numpy``) all live in the ``[analysis]`` extra; CPU inference works (``map_location``
+ device fallback), so a box without CUDA still runs the model when the registry routes to it.
"""

from __future__ import annotations

BTC_UPSTREAM_COMMIT = "2682317be668032e6e4b269ded36adaa2ad57df0"

__all__ = ["BTC_UPSTREAM_COMMIT"]
