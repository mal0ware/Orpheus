# YuE2 backend spike — yue2.cpp on the RTX 3080 (2026-10-23)

**Decision: GO** (owner, 2026-10-23, after listening to both renders). YuE2 runs comfortably on
this PC through yue2.cpp at Q8_0; the lead-sheet + ABC work targets YuE2's dialect, and the sketch
renderer gets built (spec `docs/specs/2026-10-23-orpheus-definition-design.md` §4 steps 5 and 7).

## What was installed (outside every repo)

Everything lives under `C:\Users\mal0s\.orpheus\yue2\`. Nothing YuE2-related is in this repo.

| Item | Source | Version / size | Licence |
|---|---|---|---|
| yue2.cpp source | `git clone --recurse-submodules https://github.com/ServeurpersoCom/yue2.cpp` | master `8af0174` (2026-10-16), ggml fork `cc5df49`; 73 MB on disk with `.git` | MIT |
| `models/YuE2-3B-Q8_0.gguf` | `https://huggingface.co/Serveurperso/YuE2-GGUF/resolve/main/` | 3,810,232,064 B, sha256 `41121ce97786d7795a325bcf123ca196956c03bb252c9e75384cfc1f2fc19e6b` ✓ | CC BY-NC 4.0 |
| `models/YuE2-Vae-F32.gguf` | same | 530,497,344 B, sha256 `93e49dfb1970e89ad64cacb17cf13b5d05f6bb30ef7ed3adae3050bcb728638a` ✓ | CC BY-NC 4.0 |

No sign-in was needed. Both hashes matched the repo's published values.

## Build

Toolchain already on the PC: Visual Studio 2022 Build Tools (MSVC 14.44), CUDA Toolkit 13.2,
CMake 4.3.2, Ninja 1.13. The repo's `buildcuda.cmd` hard-codes VS 2026 and fails here, so the
documented 2022 path was used:

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G "Ninja Multi-Config" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS%
```

About 3 minutes; binaries in `build\Release\` (`yue-synth.exe`, `yue-server.exe`, `yue-plan.exe`,
`yue-transcribe.exe`, …). Runtime DLLs: `cudart64_13`, `cublas64_13`, `cublasLt64_13` from
`CUDA\v13.2\bin\x64` (on PATH). No cuDNN.

## Runs

RTX 3080 10240 MiB, driver 621.01. Desktop idle before the runs: **1981 MiB** used (browser,
Discord, Spotify, wallpaper engine and others open). Orpheus's own GPU stages were idle. VRAM was
sampled every 250 ms with `nvidia-smi --query-gpu=memory.used`.

Command (both runs): `yue-synth.exe --model models\YuE2-3B-Q8_0.gguf --vae models\YuE2-Vae-F32.gguf
--request <run>.json --out <run>.wav`, default `--max-seq` (24576), `steps` 32, seeds 831001,
`output_format` `wav24`.

| Run | Request | Audio out | Wall time | Peak VRAM (total) | Above idle |
|---|---|---|---|---|---|
| 1, model-planned | `cot: "full"`, no `abc`, `duration: 60`, original test lyrics (2 verses + 2 choruses), style "mellow indie pop … 92 BPM" | 60.00 s, 48 kHz stereo, −13.5 dBFS RMS | **28.8 s** (pipeline 27.2 s, 2.2× real time) | **7229 MiB** | ~5.2 GB |
| 2, bring-your-own score | `cot: "full"`, `abc` = YuE's `examples/score.abc` (Apache-2.0; 8 bars, C major, 88 BPM, `% verse` / `% chorus`, Ins `Z4`), `duration: 30` | 22.24 s, 48 kHz stereo, −15.3 dBFS RMS | **8.9 s** (2.6× real time) | **7128 MiB** | ~5.1 GB |

Notes from the logs:

- Run 1 planned a 1103-token score, then the semantic stage stopped at **1500 tokens (truncated)**:
  exactly `duration × 25`. The planned song was longer than the 60 s cap, so the audio ends mid-song.
  yue2.cpp reports truncation only on stderr (`[AR] Semantic song 0: 1500 tokens (truncated)`),
  never in its output files, so Orpheus must read the log (CLI) or `GET /logs` (server).
- Run 2 rendered our score at its own length (8 bars at 88 BPM ≈ 21.8 s plus tail) with the
  lyrics we sent. Bring-your-own-score works through the CLI.
- yue-synth writes a replay JSON next to the output, named `<out-stem>.json`. With `--out run1.wav`
  that **overwrote the request file `run1.json`**. Orpheus must use different stems for requests
  and outputs.
- Output is peak-normalised (`peak_clip` 10): peaks hit 1.0.
- Models load in about 1 s each and unload between stages (default `EVICT_STRICT`), so the CLI
  holds VRAM only while it runs.

## What this means for the design

- **Headroom:** about 3 GB free at peak with a normal desktop. Orpheus's own GPU stages (SCNet,
  PANNs, BTC) must not run at the same time; `GPU_SEMAPHORE` only covers in-process work, so the
  sketch renderer must hold it while the external process runs.
- **CLI per render** fits well: about 1 s of model load per stage, nothing resident between
  renders, and errors and truncation are on stderr. The HTTP server (`yue-server`, port 8087) is
  the alternative if renders need queueing or cancelling. To be decided in the sketch-renderer spec.
- **Scores must be validated before rendering:** yue2.cpp has no ABC check and tokenizes whatever
  it gets. Run 1's model-planned score is a real example of the dialect: `L:1/32`, ties across
  bars, `Z`/`Z2` instrument rests, `% intro` / `% verse` / `% chorus` / `% interlude`, chords
  `Dm7`, `G7`, `Cmaj7`.
- **Sample rate:** 48 kHz output; REAPER resamples on import, but the import should keep the
  file's own rate.

Not measured yet: a full-length (3–4 min) song, `--max-seq 8192`, the server, and SheetSage2.
