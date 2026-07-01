# Runtime Capability Truth Matrix

This document provides a factual, execution-ready audit of Zentrix's current runtime modes.

## 1. cpu-only
- **CLI flag exists:** Yes (`--mode cpu-only`)
- **Planner decision exists:** Yes
- **Real execution exists:** Yes 
- **Benchmarked:** No
- **Uses mock/stub:** No
- **Current limitations:** Very slow generation speed, bottlenecked by CPU matrix multiplication.
- **Exact files implementing it:** `crates/zen-orchestrator/src/planner/mod.rs`
- **Exact command to test:** `cargo run --bin zenllm -- run <model> --mode cpu-only`

## 2. gpu-only
- **CLI flag exists:** Yes (`--mode gpu-only`)
- **Planner decision exists:** Yes
- **Real execution exists:** Yes 
- **Benchmarked:** No
- **Uses mock/stub:** No
- **Current limitations:** Will panic/OOM if the entire model does not fit in available VRAM.
- **Exact files implementing it:** `crates/zen-orchestrator/src/planner/mod.rs`
- **Exact command to test:** `cargo run --bin zenllm -- run <model> --mode gpu-only`

## 3. auto
- **CLI flag exists:** Yes (`--mode auto` or by default)
- **Planner decision exists:** Yes
- **Real execution exists:** Yes (delegates to the selected mode)
- **Benchmarked:** No
- **Uses mock/stub:** No
- **Current limitations:** May aggressively select partial-offload/flexload if VRAM reporting is slightly inaccurate.
- **Exact files implementing it:** `crates/zen-orchestrator/src/planner/mod.rs`
- **Exact command to test:** `cargo run --bin zenllm -- run <model> --mode auto`

## 4. partial-offload
- **CLI flag exists:** Yes (`--mode partial-offload`)
- **Planner decision exists:** Yes
- **Real execution exists:** Yes (Phase 4 complete)
- **Benchmarked:** No
- **Uses mock/stub:** No
- **Current limitations:** Assumes all layers are perfectly uniform in size when making math divisions for VRAM capacity.
- **Exact files implementing it:** `crates/zen-core/src/pipeline.rs`, `crates/zen-orchestrator/src/planner/mod.rs`
- **Exact command to test:** `cargo run --bin zenllm -- run <model> --mode partial-offload`

## 5. flexload
- **CLI flag exists:** Yes (`--mode flexload`)
- **Planner decision exists:** Yes
- **Real execution exists:** Yes (Phase 5 complete: caching, prefetching, and eviction)
- **Benchmarked:** Yes
- **Uses mock/stub:** No
- **Current limitations:** It solves memory but does not solve speed. It is strictly an I/O-bound "Memory Mode", streaming layer weights on demand.
- **Exact files implementing it:** `crates/zen-memory/src/flexload/layer_streamer.rs`, `crates/zen-memory/src/flexload/layer_cache.rs`
- **Exact command to test:** `cargo run --bin zenllm -- run <model> --mode flexload`

## 6. multi-gpu
- **CLI flag exists:** No
- **Planner decision exists:** No (only picks best single accelerator)
- **Real execution exists:** No
- **Benchmarked:** No
- **Uses mock/stub:** Yes (Not implemented)
- **Current limitations:** Does not exist yet. Only single-GPU inference is currently supported.
- **Exact files implementing it:** N/A
- **Exact command to test:** N/A
