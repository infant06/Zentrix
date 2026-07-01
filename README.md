# ZenLLM · Zentrix Runtime

**A Rust-first, hardware-aware LLM runtime.** ZenLLM runs quantized language
models efficiently across GPU, CPU, and hybrid CPU+GPU configurations, and
exposes them through an OpenAI-compatible server, a CLI, and Python bindings.

> Derived from [mistral.rs](https://github.com/EricLBuehler/mistral.rs) and built
> on [candle](https://github.com/huggingface/candle), ZenLLM adds intelligent
> hardware-aware loading, hybrid offload, a persistent memory layer, and a
> unified control surface.

License: MIT · Status: **early / active development**

---

## Highlights

- **Hardware-aware auto mode** — detects CPU, RAM, GPU, and VRAM, then picks the
  best execution path: full GPU, hybrid CPU+GPU offload, or CPU-only.
- **Explicit execution modes** — `gpu-only`, `cpu-only`, `partial-offload`,
  `auto` (and experimental `flexload`).
- **GGUF quantized inference** — q4_0/q4_1/q5/q6/q8 and k-quants via custom CUDA
  MMQ/MMVQ kernels, with a candle fallback.
- **OpenAI + Anthropic compatible server** with a built-in web UI and Swagger docs.
- **CLI** for `run`, `serve`, `pull`, `doctor`, `bench`, `tune`, and `vector`.
- **Python bindings** (`zentrix`) for embedding the runtime in Python apps.
- **Memory layer** — vector search (TurboVec-style) with Lance persistent storage.
- **Multi-backend** — CUDA, Metal (Apple Silicon), and CPU; Intel MKL / Apple
  Accelerate BLAS options.

## Architecture

```
User → CLI / Web UI / OpenAI API
        │
        ▼
   Request Layer → Scheduler → Model Manager → Backend Manager
                                                   │
                                   ┌───────────────┼───────────────┐
                                   ▼               ▼               ▼
                                  CPU            CPU+GPU           GPU
                                             (partial offload)
        Memory Engine: TurboVec (retrieval) + Lance (storage)
```

The workspace is organized as focused crates: `zen-core` (runtime/pipeline),
`zen-kernels-*` (CUDA/Metal kernels), `zen-hardware` (detection),
`zen-orchestrator` (load planner), `zen-api` (server), `zen-cli` (`zenllm`
binary), `zen-python` (`zentrix` bindings), `zen-vector` + `zen-memory` (memory).

## Requirements

- Rust ≥ 1.88
- For CUDA: NVIDIA driver + CUDA Toolkit (11.x/12.x), and on Windows the MSVC
  build tools.
- For the Python bridge: Python ≥ 3.10 and `maturin`.

## Build

Pin the CUDA compute capability for your GPU (Turing `75`, Ampere `80`/`86`,
Ada `89`, Hopper `90`) so kernels compile for the right architecture:

```bash
# Linux / macOS
CUDA_COMPUTE_CAP=86 ./scripts/build.sh all

# Windows (from an x64 Native Tools Command Prompt for VS)
powershell -File scripts\build.ps1 -ComputeCap 75 -Target all
```

CPU-only build: add `--cpu` (shell) or `-Cpu` (PowerShell).

See **[docs/WINDOWS_SETUP_AND_TROUBLESHOOTING.md](docs/WINDOWS_SETUP_AND_TROUBLESHOOTING.md)**
for a detailed Windows walkthrough and fixes for common errors.

## Run

### CLI — one-shot and interactive

```bash
# Pull a model (GGUF from HuggingFace)
zenllm pull Qwen/Qwen2.5-0.5B-Instruct-GGUF

# One-shot
zenllm run --mode auto --allow-fallback -i "Explain Rust ownership in one line." \
    auto -m Qwen/Qwen2.5-0.5B-Instruct-GGUF --format gguf \
    -f qwen2.5-0.5b-instruct-q4_0.gguf --tok-model-id Qwen/Qwen2.5-0.5B-Instruct

# Interactive chat: drop the -i flag
```

### Server — OpenAI/Anthropic compatible + web UI

```bash
zenllm serve auto -m Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    --format gguf -f qwen2.5-0.5b-instruct-q4_0.gguf
```

- OpenAI API: `http://localhost:<port>/v1`
- Anthropic API: `http://localhost:<port>`
- Web UI: `http://localhost:<port>/ui`
- Swagger docs: `http://localhost:<port>/docs`

### Diagnostics

```bash
zenllm doctor        # hardware, CUDA, and environment health check
zenllm runtime       # show planner decisions for available modes
```

### Python (`zentrix`)

```python
import zentrix
which = zentrix.Which.GGUF(
    tok_model_id="Qwen/Qwen2.5-0.5B-Instruct",
    quantized_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    quantized_filename="qwen2.5-0.5b-instruct-q4_0.gguf",
)
runner = zentrix.Runner(which=which, mode="auto")   # or gpu-only / cpu-only / partial-offload
req = zentrix.ChatCompletionRequest(
    messages=[{"role": "user", "content": "Hello!"}], model="zen", max_tokens=64)
print(runner.send_chat_completion_request(req).choices[0].message.content)
```

## Execution modes

| Mode | Behavior |
|------|----------|
| `auto` | Detects hardware + model size and selects the best fit |
| `gpu-only` | Entire model on GPU (fastest; requires enough VRAM) |
| `cpu-only` | Runs entirely on CPU (portable fallback) |
| `partial-offload` | Splits layers across GPU and CPU for low-VRAM devices |
| `flexload` *(experimental)* | Layer streaming for models larger than RAM+VRAM |

## Project status

Verified working: GPU-only, CPU-only, auto, and hybrid partial-offload inference
(GGUF), the CLI, `doctor`, and the OpenAI-compatible server path.

Work in progress / not yet fully wired end-to-end:
- **FlexLoad** disk-streaming execution (the planner selects it, but execution is
  not complete).
- The CLI planner estimates VRAM from the model name rather than the actual GGUF
  file size — prefer `--mode auto --allow-fallback` for local GGUF files.
- Memory (vector/Lance) and multimodal/audio paths are present but lightly tested.

See the research paper for design rationale and the engineering notes for the
Windows/Turing bring-up.

## Documentation

- [Research paper](docs/ZenLLM_Research_Paper.md) — design, hardware-aware
  loading, and engineering findings.
- [Windows setup & troubleshooting](docs/WINDOWS_SETUP_AND_TROUBLESHOOTING.md)
- [Usage guide](docs/USAGE.md)

## Contributing

Issues and PRs are welcome. Please run `cargo fmt`, `cargo clippy`, and the test
suite before submitting. For kernel changes, note the target compute capability
you built and tested against.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party attributions
(mistral.rs, candle, llama.cpp, vLLM, AirLLM, Lance).
