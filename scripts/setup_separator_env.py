"""Create the isolated ``audio-separator`` uv env (Deviations D9).

Every ``audio-separator`` release transitively caps ``fastmcp<3.4`` (Task 0.1 finding), so it
cannot share the main env with the MCP server. It lives in its own uv venv and is driven only
through its CLI as a subprocess (never ``import audio_separator`` in the main env) — the same
isolation pattern the plan uses for basic-pitch (Task 0.5).

    uv venv ~/.orpheus_separator_env
    uv pip install "audio-separator[gpu]"                 # onnxruntime-gpu + deps + a CPU torch
    uv pip install torch --index-url .../whl/cu124 \       # then OVERRIDE torch with the CUDA build
        --reinstall-package torch

**Why the two-step torch install (Task 4.1 carried-problem fix):** ``audio-separator[gpu]``'s
``[gpu]`` extra only guarantees ``onnxruntime-gpu``; its ``torch`` dependency is unconstrained
(``torch>=2.3``) and resolves to the **CPU** wheel from PyPI (Task 0.3's pin run observed
``torch ... +cpu``, ``cuda.is_available() == False`` → "running in CPU mode"). The RoFormer /
Demucs (MDXC/Demucs-arch) models in audio-separator run on *torch*, not onnxruntime, so a CPU
torch means CPU separation regardless of onnxruntime-gpu. We therefore force the CUDA build from
the same ``cu124`` index the main ``pyproject`` pins (§ ``[[tool.uv.index]] pytorch-cu124``),
AFTER audio-separator is installed, with ``--reinstall-package torch`` so uv replaces the already
-satisfying CPU wheel instead of keeping it (a bare ``uv pip install torch --index-url cu124``
no-ops — the installed CPU torch already satisfies ``torch``). ``--index-url cu124`` restricts the
resolution to CUDA wheels, so the reinstall can only pick a ``+cu124`` build (currently 2.6.0,
matching the main env). ``--cpu`` skips this override (CI / GPU-less boxes).

Usage:  ``python scripts/setup_separator_env.py``  (add ``--cpu`` for the CPU-only wheel).

The env is created ONCE at install time. ``analysis/separation.py``'s ``AudioSeparatorEngine``
(Phase 4) and ``scripts/pin_models.py``'s separation pin run invoke ``audio-separator`` out of
this env via ``separator_env_cli()``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# The main pyproject's CUDA wheel index (``[[tool.uv.index]] name = "pytorch-cu124"``) — the
# separator env's torch is forced from here so it matches the GPU-verified main env build.
_TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu124"


def separator_env_dir() -> Path:
    """Location of the isolated audio-separator env: ``~/.orpheus_separator_env``."""
    return Path.home() / ".orpheus_separator_env"


def separator_env_cli() -> Path:
    """Path to the ``audio-separator`` console script inside the isolated env.

    Windows lays console scripts under ``Scripts/`` with a ``.exe`` suffix; POSIX uses
    ``bin/``. Returns the expected path whether or not it exists yet (callers check).
    """
    root = separator_env_dir()
    if sys.platform == "win32":
        return root / "Scripts" / "audio-separator.exe"
    return root / "bin" / "audio-separator"


def separator_env_python() -> Path:
    """Path to the Python interpreter inside the isolated env (for verification runs)."""
    root = separator_env_dir()
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def setup(*, gpu: bool = True) -> Path:
    """Create the env and install audio-separator into it. Returns the CLI path.

    With ``gpu=True`` the CUDA torch override runs after ``audio-separator[gpu]`` (see module
    docstring) and ``verify_cuda`` confirms ``torch.cuda.is_available()`` — raising if it is
    still CPU, so a silently-degraded env can never pass install unnoticed.
    """
    root = separator_env_dir()
    subprocess.run(["uv", "venv", str(root)], check=True)
    extra = "gpu" if gpu else "cpu"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(root), f"audio-separator[{extra}]"],
        check=True,
    )
    if gpu:
        # Override the CPU torch the [gpu] extra pulled with the CUDA (cu124) build. See the
        # module docstring for why the reinstall + explicit index are both required.
        subprocess.run(
            [
                "uv", "pip", "install", "--python", str(root),
                "torch", "--index-url", _TORCH_CUDA_INDEX_URL,
                "--reinstall-package", "torch",
            ],
            check=True,
        )
        verify_cuda()
    cli = separator_env_cli()
    print(f"audio-separator env ready: {cli}")
    return cli


def verify_cuda() -> bool:
    """Confirm the isolated env has a CUDA torch AND onnxruntime-gpu's CUDA provider.

    Runs a probe inside the env's interpreter (never importing torch/onnxruntime into the main
    env). Returns True on success; raises :class:`RuntimeError` if torch reports no CUDA — the
    exact failure Task 0.3 hit — so the install fails loudly instead of yielding a CPU env.
    """
    probe = (
        "import torch, onnxruntime as ort;"
        "cuda = torch.cuda.is_available();"
        "prov = 'CUDAExecutionProvider' in ort.get_available_providers();"
        "print(f'torch={torch.__version__} cuda={cuda} ort_cuda={prov}');"
        "raise SystemExit(0 if cuda else 1)"
    )
    result = subprocess.run(
        [str(separator_env_python()), "-c", probe], capture_output=True, text=True, check=False
    )
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(
            "The audio-separator env still resolved a CPU torch (torch.cuda.is_available() is "
            f"False):\n{result.stdout}\n{result.stderr}\nRe-run without --cpu on a CUDA box, or "
            "check that the cu124 index is reachable."
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Install the CPU wheel instead of the CUDA (gpu) build (skips the cu124 override).",
    )
    args = parser.parse_args()
    setup(gpu=not args.cpu)


if __name__ == "__main__":
    main()
