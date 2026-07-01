# ZenLLM: A Hardware-Aware, Rust-First Runtime for Quantized LLM Inference on Commodity Hardware

**ZenLLM Contributors**
*Technical Report — 2026*

## Abstract

Large language model (LLM) inference is increasingly bottlenecked not by model
quality but by the mismatch between model memory footprints and the memory
available on the hardware people actually own. Existing runtimes optimize for
different points in the design space: `llama.cpp` for portability, vLLM for
datacenter throughput, Ollama for developer ergonomics, and `mistral.rs` for a
clean Rust-first architecture. ZenLLM unifies these strengths behind a single
runtime that *reasons about the host hardware* before it loads a model, choosing
between full-GPU, CPU-only, and hybrid CPU+GPU execution automatically. We
describe ZenLLM's architecture, its hardware-aware load planner, and its
quantized CUDA kernel layer. We also report a detailed engineering case study of
bringing GPU inference up on a low-end Turing GPU under Windows, where we
identify and fix three concrete defects — a Windows LLP64 FFI truncation bug in
the rotary-embedding kernel bindings, an architecture-pinning build issue, and a
device-placement bug in hybrid offload — that together made the difference
between a hard crash and correct, interactive inference at ~18–22 tokens/s on a
4 GB GPU. We conclude with an honest account of what is complete, what is
experimental, and what remains.

## 1. Introduction

The practical question for most users is not "which model is best?" but "which
model can I run on the machine in front of me, and how fast?". A 4 GB laptop GPU,
an 8-core CPU, and 16 GB of RAM describe a very large fraction of real
developer hardware, yet most inference stacks either assume a datacenter GPU or
fall back to CPU-only execution, leaving hybrid configurations — where some
layers run on the GPU and the rest on the CPU — underused and often buggy.

ZenLLM is a Rust-first runtime, derived from `mistral.rs` and built on the
`candle` tensor library, whose central thesis is that **the runtime should adapt
to the hardware, not the other way around**. It detects the CPU, RAM, GPU, and
VRAM available, estimates the model's memory footprint, and selects an execution
strategy: full GPU, hybrid partial offload, CPU-only, or (planned) FlexLoad
layer streaming for models larger than combined RAM+VRAM.

This report makes three contributions:

1. A description of ZenLLM's crate-structured architecture and its
   hardware-aware load planner (§3–§4).
2. A description of the quantized GGUF kernel layer and its dispatch/fallback
   design (§5), and the hybrid-offload device mapping (§6).
3. A reproducible engineering case study (§8) of low-end GPU bring-up, including
   three root-cause fixes that are broadly relevant to anyone shipping CUDA FFI
   from Rust on Windows.

## 2. Background and Related Work

**llama.cpp** popularized GGUF quantization and highly portable CPU/GPU kernels,
including the MMQ (matrix-matrix quantized) and MMVQ (matrix-vector quantized)
CUDA kernels that ZenLLM adapts. **vLLM** introduced PagedAttention and
continuous batching for high datacenter throughput. **Ollama** demonstrated the
value of a friendly CLI, a model registry, and a local server. **mistral.rs**
provided a clean, async, Rust-first engine with a strong scheduler and candle
backend. **AirLLM** showed that layer streaming can run very large models on
tiny GPUs by loading and evicting layers on demand.

ZenLLM's design borrows deliberately: GGUF kernels from llama.cpp, PagedAttention
concepts from vLLM, the CLI/registry/server surface from Ollama, the runtime
skeleton from mistral.rs, and the FlexLoad concept from AirLLM. Its novel
emphasis is the *planner* that ties hardware detection to execution strategy, and
the memory engine (TurboVec + Lance) for persistent, cross-session retrieval.

## 3. System Architecture

ZenLLM is organized as a Rust workspace of focused crates:

- `zen-core` — the runtime: pipelines, quantized model implementations,
  scheduler, KV cache, device mapping, attention.
- `zen-kernels-quant`, `zen-kernels-paged`, `zen-kernels-flash` — CUDA/Metal
  kernels for quantized matmul, paged attention, and flash attention.
- `zen-hardware` — CPU/RAM/GPU/VRAM detection across NVIDIA, AMD, Intel, Apple.
- `zen-orchestrator` — the load planner that maps hardware + model metadata to an
  execution plan.
- `zen-api` — the OpenAI/Anthropic-compatible HTTP server and web UI.
- `zen-cli` — the `zenllm` binary (`run`, `serve`, `pull`, `doctor`, `bench`, …).
- `zen-python` — the `zentrix` PyO3 bindings.
- `zen-vector`, `zen-memory` — vector retrieval and Lance-backed storage.

The request flow is: *client → CLI/UI/API → request layer → scheduler → model
manager → backend manager → inference engine → CPU/GPU*. The runtime is
deliberately not an agent; it provides infrastructure (APIs, scheduling,
hardware abstraction, memory) that higher-level systems build on.

## 4. Hardware-Aware Execution

### 4.1 Detection

`zen-hardware` enumerates CPU cores and vendor, total/available RAM, and each
accelerator's kind (NVIDIA/AMD/Intel/AppleMetal), name, total/available VRAM, and
compute capability (via `nvidia-smi` for NVIDIA).

### 4.2 The load planner

Given detected hardware and model metadata, the planner (`zen-orchestrator`)
produces a decision among `GpuOnly`, `PartialOffload`, `CpuOnly`, and `FlexLoad`.
The core logic, for a requested `Auto` mode, is:

1. If the model's estimated footprint fits in usable VRAM → **GpuOnly**.
2. Else if part of it fits and RAM can hold the remainder → **PartialOffload**,
   computing how many transformer layers fit in VRAM.
3. Else if it fits in RAM → **CpuOnly**.
4. Else → **FlexLoad** (stream layers from disk).

Explicit modes (`GpuOnly`, `CpuOnly`, …) are honored, with optional fallback if
the request cannot be satisfied.

### 4.3 Two planners, one principle

ZenLLM currently has two planner surfaces: the Rust `zen-orchestrator` planner
used by the server/CLI, and a launcher-level planner (in the reference Python app)
that reads the *actual* GGUF file — its true layer count and on-disk size — to
compute an exact plan. The GGUF-aware approach is more accurate for local
quantized files; unifying the two is future work (§10).

## 5. Quantized Kernel Layer

GGUF weights are executed through `GgufMatMul`, which dispatches by batch size:

- **Batch 1–8 (decode):** MMVQ (matrix-vector quantized) kernels.
- **Batch > 8 (prefill):** MMQ (matrix-matrix quantized) kernels.
- **Fallback:** candle's built-in quantized matmul (F32), selectable at runtime
  via `ZEN_DISABLE_FAST_QMATMUL=1`.

The MMQ/MMVQ kernels are adapted from llama.cpp and select tile sizes and code
paths based on compute capability (e.g., Turing tensor-core MMA vs. DP4A). A key
subtlety, discussed in §8, is that the *host* dispatches by runtime compute
capability while the *device* code path is fixed at compile time — so the kernels
must be compiled for the exact target architecture.

## 6. Hybrid Offload and FlexLoad

In partial offload, a `DeviceMapper` assigns each transformer layer to a device;
the hidden state is moved to a layer's device before that layer executes. Weights
that are not per-layer — token embeddings, the final RMS norm, and the output
head — live on the primary device. A correct implementation must therefore move
the hidden state back to the primary device before the final norm; failing to do
so produces a device mismatch (§8.3).

FlexLoad generalizes this to streaming: layers are loaded, executed, and evicted
so that peak memory is bounded by a working set rather than the whole model. The
planner already selects FlexLoad when appropriate; end-to-end execution through
the Python bridge is not yet complete and is marked experimental.

## 7. Memory Engine

To support long sessions and retrieval-augmented use without depending solely on
the model's context window, ZenLLM integrates a two-tier memory layer: a
SIMD-accelerated nearest-neighbor search layer (TurboVec-style, exposed via
`zen-vector`) over persistent, columnar Lance storage (`zen-memory`). This
enables chat, project, and document memory to survive restarts. These components
are present and wired into the CLI (`zenllm vector`, `--enable-search`) but are
lightly tested relative to the core inference path.

## 8. Engineering Case Study: Low-End GPU Bring-Up

We brought GPU inference up on a representative low-end configuration:

- CPU: AMD Ryzen 5 3550H (8 threads), 16 GB RAM.
- GPU: NVIDIA GeForce GTX 1650, 4 GB VRAM, compute capability 7.5 (Turing).
- CUDA Toolkit 12.4, Windows, MSVC toolchain.
- Model: Qwen2.5-0.5B-Instruct, GGUF q4_0.

Initially, any GPU prompt crashed with
`CUDA_ERROR_ILLEGAL_ADDRESS`. We isolated three distinct defects.

### 8.1 Architecture pinning

CUDA kernels are dispatched by runtime compute capability but the device code
path is baked in at compile time. The quantized-kernel build defaulted to
`sm_80` when the compute capability was not explicitly set; the resulting binary
could not launch correctly on the `sm_75` Turing GPU. **Fix:** pin
`CUDA_COMPUTE_CAP=75` (surfaced as a build-time diagnostic) so all kernel crates
compile for Turing. This is necessary but, as it turned out, not sufficient.

### 8.2 A Windows LLP64 FFI truncation bug (root cause)

With `CUDA_LAUNCH_BLOCKING=1` we obtained a precise backtrace pinning the fault
to the rotary-embedding (RoPE) kernel rather than the matmul (the matmul was
merely where the asynchronous error surfaced). The Rust bindings for the CUDA
rotary kernel declared 64-bit `int64_t` parameters (`num_tokens`, `query_stride`,
`key_stride`, and the CUDA stream handle) as `c_long`.

On Linux (LP64) `c_long` is 64-bit and the code worked; on **Windows (LLP64)
`c_long` is 32-bit**. The stream handle was truncated and the stride arguments
were corrupted, so the kernel computed out-of-bounds byte offsets — an illegal
memory access. The bug was invisible on CPU (no FFI) and independent of dtype and
architecture, which is why it masqueraded as a kernel/arch problem. **Fix:** use
`i64` (64-bit on all platforms) in the FFI declaration and call sites. General
lesson: when binding to CUDA `int64_t`, always use Rust `i64`, never `c_long`.

### 8.3 Device mismatch in hybrid offload

Once GPU-only inference worked, explicit partial offload
(`num_device_layers`) crashed with
`device mismatch in rms-norm, lhs: Cpu, rhs: Cuda`. When tail layers were
offloaded to the CPU, the hidden state was left on the CPU, but the final RMS
norm and output head reside on the primary (GPU) device. **Fix:** move the hidden
state back to the primary device before the final norm. We applied this to the
affected quantized models (Qwen2, Qwen3, Qwen3-MoE); the Llama implementation
already did so.

### 8.4 Windows DLL discovery

A separate, non-fatal issue: since Python 3.8, extension modules do not resolve
dependent DLLs via `PATH`. The bindings must register the CUDA `bin` directory
with `os.add_dll_directory` before import; otherwise `import zentrix` fails with
a DLL load error despite the DLLs being present.

## 9. Evaluation

We report a preliminary, single-configuration measurement on the hardware in §8,
using the Qwen2.5-0.5B q4_0 model and a short one-shot prompt through the
`zenllm` CLI:

| Metric | Value |
|--------|-------|
| Time to first token | 1.57 s |
| Prompt throughput | 22.5 tokens/s (35-token prompt) |
| Decode throughput | 17.9 tokens/s (198 tokens generated) |

CPU-only and auto/hybrid modes also produce correct output on the same machine.
These numbers are illustrative of functional correctness and interactive latency
on a 4 GB GPU rather than a throughput benchmark; a systematic evaluation across
models, quantization levels, batch sizes, and GPUs is future work. We deliberately
avoid reporting comparative benchmarks we have not run.

## 10. Limitations and Future Work

- **FlexLoad execution** is not complete end-to-end; only the planning decision
  exists.
- **Planner accuracy for local GGUF:** the CLI planner estimates footprint from
  the model name at f16 rather than the actual quantized file size, and can
  over-reject `gpu-only`. Unifying it with the GGUF-aware launcher planner is
  planned. Workaround: `--mode auto --allow-fallback`.
- **Coverage:** the device-placement fix has been applied to the Qwen/Llama
  family; other architectures should be audited for the same pattern.
- **Memory and multimodal** paths are present but under-tested.
- **Performance work** (continuous batching, KV-cache reuse, prefix caching,
  CUDA graphs, dynamic quantization) is planned per the project roadmap.

## 11. Conclusion

ZenLLM demonstrates that a hardware-aware runtime can make commodity, low-VRAM
hardware a first-class target for quantized LLM inference. The core value is not
a single kernel but the discipline of *deciding how to load a model based on the
machine it will run on*, combined with correct hybrid execution. The engineering
case study shows how a small number of platform-specific defects — an FFI integer
width, a compile-time architecture, and a device-placement omission — stand
between "crashes on GPU" and "interactive inference on a 4 GB laptop GPU". We
release ZenLLM under the MIT license to invite scrutiny and contribution.

## References

1. G. Gerganov et al. *llama.cpp*. https://github.com/ggerganov/llama.cpp
2. W. Kwon et al. *Efficient Memory Management for Large Language Model Serving
   with PagedAttention* (vLLM). SOSP 2023.
3. *Ollama*. https://github.com/ollama/ollama
4. E. Buehler. *mistral.rs*. https://github.com/EricLBuehler/mistral.rs
5. *candle*. https://github.com/huggingface/candle
6. *AirLLM*. https://github.com/lyogavin/airllm
7. *Lance*. https://github.com/lancedb/lance
8. NVIDIA. *CUDA C++ Programming Guide* (compute capabilities; LLP64 vs LP64 ABI).
