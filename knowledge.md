# ZenLLM Knowledge Base

ZenLLM is a next-generation, Rust-first AI runtime designed to unify the strengths of modern inference engines while solving their limitations in memory efficiency, hardware utilization, multi-model orchestration, and long-context management.

Originally derived from the proven foundations of mistral.rs, ZenLLM extends beyond a traditional inference engine by introducing intelligent hardware-aware execution, flexible model loading strategies, persistent memory capabilities, and future-ready distributed architectures.

Unlike existing runtimes that primarily focus on model execution, ZenLLM aims to become a complete runtime platform capable of supporting text, vision, audio, embeddings, multimodal models, and future modalities through a unified architecture.

ZenLLM automatically adapts itself to available hardware. It detects CPU capabilities, GPU memory, system RAM, storage bandwidth, and backend availability to determine the most efficient execution path. Models can run entirely on GPU, partially offloaded between CPU and GPU, completely on CPU, or through FlexLoad, a layer-streaming approach inspired by AirLLM that enables extremely large models to operate on limited hardware.

To support long-running sessions and future agent ecosystems, ZenLLM incorporates a memory architecture built around TurboVec and Lance. This allows conversations, project knowledge, and external documents to be stored, retrieved, and reused efficiently without depending solely on the model's context window.

ZenLLM remains runtime-focused. It does not attempt to perform planning or agent reasoning itself. Instead, it provides the infrastructure required by higher-level systems such as Agent Zen. Through OpenAI-compatible APIs, scheduler mechanisms, hardware abstraction layers, and memory services, ZenLLM acts as the execution engine upon which future AI workspaces and autonomous systems can be built.

The long-term goal is to create a runtime that combines the throughput of vLLM, the portability of Ollama, the efficiency of llama.cpp, the elegance of mistral.rs, and the memory capabilities required for next-generation AI applications.

## Why Mistral.rs Was Chosen

We started from mistral.rs because:

**Strengths:**
- Rust-first architecture
- High performance
- Candle backend
- GGUF support
- Excellent scheduler design
- Clean codebase
- Async execution
- Good portability

**Weaknesses:**
- Limited memory system
- No persistent cross-chat memory
- No FlexLoad
- Limited model lifecycle management
- Single-model oriented
- No runtime UI ecosystem
- No hardware-aware intelligent loading

Therefore, mistral.rs became the foundation rather than the final product.

## Why TurboVec Was Integrated

TurboVec was chosen because vector search eventually becomes a bottleneck.

Traditional vector search:
`Embedding` → `Vector DB` → `Retrieval`
can become slow when memories grow.

**TurboVec brings:**
- SIMD acceleration
- AVX2, AVX512, NEON
- Fast nearest-neighbor search
- Lower latency retrieval
- High throughput

**Benefits to ZenLLM:**
- Faster memory retrieval
- Better long conversations
- Agent memory support
- Project memory support
- RAG acceleration

TurboVec becomes the search engine of the memory layer.

## Why Lance Was Chosen

TurboVec alone is not enough. Vectors need persistent storage.

**Lance provides:**
- Columnar storage
- Efficient indexing
- Metadata support
- Versioning
- Large dataset handling
- Incremental updates

**Benefits:**
Chat memory, project memory, document memory, code memory, and embeddings can survive across sessions. Without Lance, memory disappears after restart. 

Lance becomes the storage engine while TurboVec becomes the retrieval accelerator.

## Why AirLLM Inspired FlexLoad

Large models remain inaccessible to most users. Normally, running a 70B model on a 4GB GPU is impossible. AirLLM introduced layer streaming.

**Benefits:**
- Reduced VRAM requirements
- CPU-assisted execution
- Disk-assisted loading
- Dynamic layer swapping

ZenLLM extends this into CPU mode, GPU mode, Hybrid mode, and **FlexLoad mode**, allowing large models to run on commodity hardware.

## Architecture

`User` → `CLI / UI / API` → `Request Layer` → `Scheduler` → `Model Manager` → `Backend Manager` → `Inference Engine` → `CPU / GPU / FlexLoad`

### Model Lifecycle

Unlike Ollama (Loaded vs Unloaded), ZenLLM introduces:
- Not Installed
- Installed but Unloaded
- Warm Loaded
- Active Running

**Benefits:** Faster switching, reduced cold starts, multi-model readiness, and better responsiveness.

### Hardware Modes

- **GPU Only:** Maximum speed. Entire model fits VRAM.
- **Hybrid:** CPU + GPU. Partial layer offload.
- **CPU Only:** Fallback mode. Runs everywhere.
- **FlexLoad:** Layer streaming. Very low VRAM requirements.
- **Auto Mode:** Default mode. ZenLLM calculates CPU, GPU, RAM, VRAM, Storage, Model size, and Context length, and selects the optimal execution path automatically.

## Why UI Exists

The UI is not Agent Zen. It is a runtime control center. Inspired by mistral.rs. 

Users can control:
- Temperature, Top-p, Top-k
- Threads, Batch size, GPU layers, Context length
- CPU/GPU mode
- Loaded models, Memory status, Server status

CLI: `zenllm ui`

## Why OpenAI-Compatible APIs

To make ZenLLM usable by LangChain, OpenWebUI, Continue, OpenHands, Agent Zen, and Custom applications without changing client code.

## Performance Improvement Plan

- **Phase 1:** Faster model loading (Warm pools, Keep frequently used models partially initialized)
- **Phase 2:** FlexLoad (AirLLM-inspired streaming, Run very large models on small hardware)
- **Phase 3:** Memory Engine (TurboVec + Lance, Persistent memory, Cross-session retrieval, Project knowledge, Long conversations)
- **Phase 4:** Smarter memory management (Dynamic eviction: LRU, Priority, Warm pool preservation. Optimize: VRAM, RAM, Disk cache)
- **Phase 5:** Continuous batching (Borrow ideas from vLLM. Higher throughput, Better GPU utilization, More concurrent users)
- **Phase 6:** KV cache reuse (Reuse previous computations. Faster responses, Lower latency, Better long-context performance)
- **Phase 7:** Prefix caching (Common prefixes: System prompts, Templates, Shared contexts are reused)
- **Phase 8:** Flash Attention (Faster attention kernels, Lower memory usage)
- **Phase 9:** CUDA Graphs (Reduce launch overhead, Increase throughput)
- **Phase 10:** Dynamic quantization (Adaptive precision: FP16, Q8, Q6, Q4 depending on available hardware)
- **Phase 11:** Distributed inference (Future support: Multi-GPU, Multi-node, Remote workers, Tensor parallelism, Pipeline parallelism)

## Final ZenLLM Architecture

```text
ZenLLM
│
├── CLI
├── Control Center UI
├── OpenAI Server
│
├── Scheduler
├── Request Queue
│
├── Model Registry
├── Model Manager
├── Warm Pool
│
├── Memory Engine
│     ├── TurboVec
│     └── Lance
│
├── Context Engine
├── Embedding Services
│
├── FlexLoad Engine
│
├── Backend Manager
│     ├── CPU Backend
│     ├── CUDA Backend
│     ├── ROCm Backend
│     ├── Metal Backend
│     ├── Vulkan Backend
│     └── oneAPI Backend
│
├── Diagnostics
├── Benchmarking
│
└── Future
      ├── Continuous batching
      ├── KV cache reuse
      ├── Flash attention
      ├── CUDA graphs
      ├── Distributed inference
      └── Multi-node execution
```

The key principle is: **ZenLLM is a runtime, not an agent.**
