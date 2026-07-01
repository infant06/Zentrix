# ZenLLM / Zentrix — Windows Setup, Usage & Troubleshooting

This guide covers building and running ZenLLM (the `zentrix` Python bridge and
the `zenllm` CLI/server) on Windows with an NVIDIA GPU, plus fixes for the
issues encountered during bring-up on a low-end machine
(AMD Ryzen 5 3550H, GTX 1650 4 GB, CUDA 12.4).

---

## 1. Prerequisites

- Visual Studio 2022 Build Tools (MSVC x64 toolchain).
- CUDA Toolkit 12.4 (or matching your driver).
- Rust (stable) + `maturin` (installed inside your Python venv).
- A GGUF model, e.g. `qwen2.5-0.5b-instruct-q4_0.gguf`.

## 2. The single most important build setting: `CUDA_COMPUTE_CAP`

CUDA kernels are dispatched by *runtime* compute capability but the device code
path is baked in at *compile* time. If the kernels are compiled for the wrong
architecture they fail at launch with `CUDA_ERROR_ILLEGAL_ADDRESS`.

**Always pin the compute capability to your GPU before building.**

| GPU family | Example card | `CUDA_COMPUTE_CAP` |
|------------|--------------|--------------------|
| Turing     | GTX 1650, RTX 20xx | `75` |
| Ampere     | RTX 30xx, A100 | `80` / `86` |
| Ada        | RTX 40xx | `89` |

The build prints the arch it compiled for:

```
warning: zen-kernels-quant: compiling CUDA kernels for compute capability sm_75 ...
```

Confirm this matches your GPU.

## 3. Building the Python bridge (`zentrix`)

Use the x64 Native Tools Command Prompt (or the provided `build_cuda_sm75.bat`,
which bundles the MSVC env + `CUDA_COMPUTE_CAP=75` + venv activation):

```bat
set CUDA_COMPUTE_CAP=75
call "<venv>\Scripts\activate.bat"
cd crates\zen-python
maturin develop --release --features "cuda,code-execution"
```

If you switch GPUs/arch, force a clean rebuild of the kernel crates:

```bat
cargo clean -p zen-kernels-quant -p zen-core
```

## 4. Building the CLI / server (`zenllm`, ollama-like)

```bat
set CUDA_COMPUTE_CAP=75
cargo build --release -p zenllm --features "cuda,code-execution"
```

Run an OpenAI/Anthropic-compatible server with a web UI and hardware-aware
auto-mode:

```bat
target\release\zenllm.exe serve auto -m Qwen/Qwen2.5-0.5B-Instruct-GGUF ^
    --format gguf -f qwen2.5-0.5b-instruct-q4_0.gguf
```

Endpoints:
- OpenAI-compatible API: `http://localhost:<port>/v1`
- Anthropic-compatible API: `http://localhost:<port>`
- Swagger docs: `http://localhost:<port>/docs`
- Web UI: `http://localhost:<port>/ui`

Use `zenllm --help` and `zenllm serve --help` for the full option set.

## 5. Using the ZenDNA Light launcher (Python)

```bat
python src\main.py info                 # detected hardware + auto load plan
python src\main.py ask "your prompt"    # one-shot generation
python src\main.py run                  # interactive chat REPL
```

Flags: `--model PATH`, `--mode {auto|gpu-only|cpu-only|partial-offload|flexload}`,
`--dtype {Auto|F16|BF16|F32}`, `--gpu-layers N`, `--max-tokens N`.

### Auto-mode (self-evaluation)

In `auto` mode the launcher reads GGUF metadata (layer count, size), detects
VRAM/RAM, and picks:
- **gpu-only** when the model fits in usable VRAM,
- **partial-offload** (N layers on GPU, rest on CPU) when it doesn't,
- **cpu-only** when it fits in RAM but not VRAM,
- **flexload** (experimental) when it exceeds both.

---

## 6. Troubleshooting

### `CUDA_ERROR_ILLEGAL_ADDRESS` during the prompt/prompt step
Two distinct root causes were found and fixed:

1. **Windows FFI truncation (fixed).** The rotary-embedding kernel bindings
   declared 64-bit CUDA args (`num_tokens`, strides, `stream`) as `c_long`,
   which is **32-bit on Windows (LLP64)**. This truncated the stream handle and
   corrupted stride arguments, causing out-of-bounds access. Fixed by using
   `i64` in `zen-kernels-quant/src/rotary/ffi.rs` and its call sites. If you
   write new FFI to CUDA `int64_t` params, **always use `i64`, never `c_long`.**

2. **Wrong compiled arch.** See §2 — pin `CUDA_COMPUTE_CAP`.

Debugging tip: set `CUDA_LAUNCH_BLOCKING=1` so the error is attributed to the
exact faulting kernel instead of surfacing at a later async sync point.

### `device mismatch in rms-norm, lhs: Cpu, rhs: Cuda` (partial offload) — fixed
When tail layers are offloaded to CPU, the final norm/output head (always on the
primary device) received a hidden state left on CPU. Fixed by moving the hidden
state back to the primary device before the final norm in the quantized models
(`quantized_qwen`, `quantized_qwen3`, `quantized_qwen3_moe`; `quantized_llama`
already did this).

### `ImportError: DLL load failed while importing zentrix`
Since Python 3.8, extension modules do **not** resolve dependent DLLs via `PATH`.
Register the CUDA `bin` directory before importing:

```python
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
import zentrix
```

The launcher does this automatically (see `core/llm_engine.py`). Required DLLs:
`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`, `curand64_10.dll`,
plus the driver's `nvcuda.dll`.

### `maturin: command not found`
Activate the venv first (`call <venv>\Scripts\activate.bat`). In PowerShell use
`$env:CUDA_COMPUTE_CAP="75"` (not `set`), and note `vcvars64.bat` does not persist
its environment into PowerShell — prefer the x64 Native Tools Command Prompt.

### NVCC redefinition error for `__hmax_nan` / `__hmin_nan` (CUDA 12.4)
CUDA 12.4 already provides these half intrinsics; remove/guard the duplicate
definitions in candle-kernels' `compatibility.cuh`. This is a compile-time fix
and is unrelated to the runtime illegal-address issue above.

### Escape hatch for GGUF matmul kernels
Set `ZEN_DISABLE_FAST_QMATMUL=1` to bypass the custom MMVQ/MMQ CUDA kernels and
use candle's built-in quantized matmul (diagnostic / fallback only).

### `Planner rejected load: insufficient VRAM` for a small local GGUF
The CLI planner estimates a model's memory need from the model **name at f16**
(`estimate_model_metadata(model_id, "f16")`), not from the actual quantized
file size. For a local GGUF this over-estimates and can wrongly reject
`--mode gpu-only`. Workarounds:
- Add `--allow-fallback` (auto will pick GPU/partial/CPU that fits), or
- Use `--mode auto --allow-fallback` (recommended), or
- Pin layers explicitly with `--gpu-layers N` / `-n 0:N`.

Known limitation: the planner should read the real GGUF file size for accurate
VRAM fitting (the Python launcher in `zendna light` already does this). Until
that lands, prefer `auto --allow-fallback` on the CLI.
