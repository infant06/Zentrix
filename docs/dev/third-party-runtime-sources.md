# Third-Party Runtime Sources Migration Report

## AirLLM
**Source Path:** `C:\AiResearch\airllm`

### Overview
AirLLM is a Python-based library designed for running large language models on low-VRAM machines through layer-wise inference and offloading.
It currently contains the following key components:
- `air_llm/`: Core Python library for layer inference and memory management.
- `training/` / `eval/`: Scripts for fine-tuning and evaluation.
- `rlhf/`: Reinforcement learning from human feedback components.

### Migration Strategy
The `zen-memory` crate already incorporates the core philosophical architecture of AirLLM (layer streaming, ephemeral block initialization, and chunk-based forward passes).
We do not need to port Python code directly. Instead, we should:
1. Identify any advanced caching, speculative layer fetching, or compression mechanisms in AirLLM that our `FlexLoadPlan` and `LayerStreamer` might be missing.
2. Adapt any custom mathematical layer-compression algorithms from AirLLM if applicable.
3. AirLLM is heavily integrated with PyTorch and HuggingFace Accelerate. Our implementation relies on Safetensors and Candle directly in Rust, providing a much lower-overhead native execution path.

---

## TurboVec
**Source Path:** `C:\AiResearch\turbovec`

### Overview
TurboVec is a Rust-based vectorized and hardware-accelerated computation engine with a Python binding.
It currently contains:
- `turbovec/`: Core Rust library.
- `turbovec-python/`: Python bindings (likely PyO3).
- `benchmarks/`: Performance tests.

### Migration Strategy
TurboVec should be adapted and rebranded as the `zen-vector` crate within the Zenllm workspace.
The process should involve:
1. **Source Inspection**: Review `turbovec/src/` to identify the core SIMD, BLAS, or custom matrix multiplication routines.
2. **Rebranding**: Change the package name to `zen-vector` and update all namespaces.
3. **Integration**: Link `zen-vector` to `zen-core` or `zen-hardware` as a backend accelerator for specific Candle operators or custom quantized kernels.
4. We must ensure no heavy build scripts or large binary dependencies (unless explicitly required and optimized) bloat the Zenllm compile time. We will avoid blindly copying workspace configurations.
