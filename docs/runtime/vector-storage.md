# ZenVector Storage

Zentrix includes an integrated high-performance vector search engine named **ZenVector**.
This subsystem is specifically built for:
- Retrieving embeddings for RAG (Retrieval-Augmented Generation) pipelines.
- Looking up semantic cache hits.
- Managing large vector datasets.

> [!WARNING]
> **Important Note:** ZenVector does **not** accelerate normal LLM token generation or inference. It is an independent retrieval and storage module.

## Architecture

ZenVector provides two layers of operation:

1. **In-Memory Engine (`MemoryVectorStorage`)**
   A highly optimized, quantized vector index capable of sub-millisecond search latency using SIMD intrinsics (AVX-512, NEON). It supports exact `L2`, `Dot Product`, and `Cosine` similarity metrics and seamlessly compresses vectors to 2-4 bits per coordinate with near-optimal distortion.

2. **Persistent Storage (`LanceVectorStorage`)**
   For datasets that are too large to fit in memory, ZenVector integrates an optional persistent storage backend. This layer is gated behind the `lance-storage` feature flag and provides disk-backed vector records with future support for metadata filtering.

## CLI Usage

Zentrix provides a built-in CLI to interact with ZenVector:

```bash
# Import vectors into memory (useful for testing)
zenllm vector import -f data.json

# Import vectors into persistent storage
zenllm vector import -f data.json --persistent --out ./my_vectors

# Search for similar vectors
zenllm vector search --query "0.1,0.2,-0.5,0.8" -k 5

# Show storage statistics
zenllm vector stats
```

## Cargo Features

When compiling Zentrix or `zen-vector`, you can customize the build using feature flags:

- `memory-storage`: (Default) Enables the in-memory `ZenVectorIndex` and `MemoryVectorStorage`.
- `lance-storage`: Enables `LanceVectorStorage` for persistent, disk-backed operations. If this feature fails to build on your platform (e.g., Windows), you can safely omit it.
