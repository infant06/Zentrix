# ZenLLM Usage Guide

This guide covers building, running, and configuring ZenLLM (the `zenllm` CLI /
server and the `zentrix` Python bridge).

## 1. Build

Pin the CUDA compute capability for your GPU so kernels are compiled correctly:

| GPU family | Examples | `CUDA_COMPUTE_CAP` |
|------------|----------|--------------------|
| Turing | GTX 16xx, RTX 20xx | `75` |
| Ampere | RTX 30xx, A100 | `80` / `86` |
| Ada | RTX 40xx | `89` |
| Hopper | H100 | `90` |

```bash
# CLI + Python bindings, CUDA
CUDA_COMPUTE_CAP=86 ./scripts/build.sh all           # Linux/macOS
powershell -File scripts\build.ps1 -ComputeCap 75 -Target all   # Windows

# CPU-only
./scripts/build.sh cli --cpu
```

Artifacts:
- CLI/server binary: `target/release/zenllm`
- Python module: installed into your active venv as `zentrix`

## 2. Get a model

```bash
# From HuggingFace (GGUF repo)
zenllm pull Qwen/Qwen2.5-0.5B-Instruct-GGUF

# From an Ollama registry or direct URL
zenllm pull ollama:llama3.2:1b
zenllm pull https://example.com/model.gguf

# List locally registered models
zenllm models
```

## 3. Run inference

### Interactive chat
```bash
zenllm run --mode auto --allow-fallback \
    auto -m Qwen/Qwen2.5-0.5B-Instruct-GGUF --format gguf \
    -f qwen2.5-0.5b-instruct-q4_0.gguf --tok-model-id Qwen/Qwen2.5-0.5B-Instruct
```

### One-shot (`-i`)
```bash
zenllm run --mode gpu-only -i "Write a haiku about Rust." \
    auto -m <repo-or-dir> --format gguf -f <file.gguf> --tok-model-id <tok-repo>
```

Notes:
- Flags that belong to `run` (`--mode`, `--gpu-layers`, `--allow-fallback`,
  `-i`) go **before** the `auto`/`text` subcommand; model-selection flags
  (`-m`, `--format`, `-f`, `--tok-model-id`) go **after** it.
- For local GGUF files, prefer `--mode auto --allow-fallback` (see the planner
  note in the troubleshooting doc).

## 4. Serve an OpenAI-compatible API

```bash
zenllm serve auto -m Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    --format gguf -f qwen2.5-0.5b-instruct-q4_0.gguf --host 0.0.0.0 --port 8080
```

Then use any OpenAI client:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"zen","messages":[{"role":"user","content":"Hello"}]}'
```

Surfaces: OpenAI `/v1`, Anthropic-compatible root, Web UI at `/ui`, Swagger at
`/docs`. Disable the UI with `--no-ui`.

## 5. Execution modes and offload

| Mode | Use when |
|------|----------|
| `auto` | You want ZenLLM to decide (recommended) |
| `gpu-only` | Model fits fully in VRAM |
| `cpu-only` | No GPU, or debugging |
| `partial-offload` | Model is larger than VRAM but you have RAM headroom |
| `flexload` *(experimental)* | Model larger than RAM+VRAM |

Manual layer split (partial offload): `--gpu-layers N` or `-n 0:N` puts `N`
transformer layers on GPU 0 and the rest on CPU.

## 6. Python API (`zentrix`)

```python
import zentrix, json

# Inspect hardware
print(json.loads(zentrix.detect_hardware()))

which = zentrix.Which.GGUF(
    tok_model_id="Qwen/Qwen2.5-0.5B-Instruct",
    quantized_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    quantized_filename="qwen2.5-0.5b-instruct-q4_0.gguf",
    dtype=zentrix.ModelDType.Auto,        # Auto | F16 | BF16 | F32
)
runner = zentrix.Runner(
    which=which,
    mode="auto",                          # auto|gpu-only|cpu-only|partial-offload|flexload
    num_device_layers=None,               # e.g. ["12"] to force 12 layers on GPU
    max_seqs=16,
    no_paged_attn=True,
)
req = zentrix.ChatCompletionRequest(
    messages=[{"role": "user", "content": "Hello!"}], model="zen", max_tokens=128)
resp = runner.send_chat_completion_request(req)
print(resp.choices[0].message.content)
```

On Windows (Python ≥ 3.8), register the CUDA `bin` directory before importing
`zentrix` (dependent DLLs are not resolved via PATH):

```python
import os
os.add_dll_directory(r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin")
import zentrix
```

## 7. Useful environment variables

| Variable | Purpose |
|----------|---------|
| `CUDA_COMPUTE_CAP` | Compute capability to compile CUDA kernels for (build time) |
| `CUDA_LAUNCH_BLOCKING=1` | Serialize kernel launches for precise error attribution |
| `ZEN_DISABLE_FAST_QMATMUL=1` | Bypass custom MMQ/MMVQ kernels (diagnostic fallback) |
| `RUST_LOG=zen_core=debug` | Verbose runtime logging |

## 8. Diagnostics & benchmarking

```bash
zenllm doctor     # environment + hardware health
zenllm runtime    # planner decisions per mode
zenllm bench ...  # generation throughput benchmark
```

For error fixes (illegal memory access, DLL load failures, device mismatch,
planner rejections), see
[WINDOWS_SETUP_AND_TROUBLESHOOTING.md](WINDOWS_SETUP_AND_TROUBLESHOOTING.md).
