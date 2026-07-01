# Mock and Stub Audit

This document tracks placeholders, dummies, and stubs discovered in the codebase that require attention before production release.

## Identified Stubs & Placeholders

| File Path | Symbol/Function | Purpose | Safe to keep? | Must replace before release? |
| :--- | :--- | :--- | :--- | :--- |
| `zen-core/src/models/*.rs` | `MtpModel::mtp_forward` | `unimplemented!()` in all base models for Multi-Token Prediction (MTP). | Yes, but crashes if MTP is explicitly invoked. | No, unless MTP is a launch feature. |
| `zen-core/src/embedding_models/layers.rs` | Multiple functions | `unimplemented!()` inside embedding layer implementations. | No, embedding models may crash during certain forward passes. | Yes, if embedding support is promised. |
| `zen-core/src/lora/qloralinear.rs` | `merge()` | `unimplemented!()` for merging quantized LoRA weights into base weights. | Yes, if users only run unmerged inference. | No, unless `merge` is exposed via CLI. |
| `zen-cli/src/main.rs` | `print_runtime_diagnostics` | Prints a dummy warning when `PartialOffload` is requested, since the backend just falls through to `Normal` execution. | No. | Yes, actual `PartialOffload` execution must be built. |
| `zen-core/src/device_map/mappers.rs` | `DummyDeviceMapper` | Creates a fake single-device map to fulfill NCCL trait requirements without actual multi-gpu splitting. | Yes, serves as a fallback. | No. |
| `zen-core/src/engine/agentic_loop.rs` | `dummy` request message | Swaps out the `RequestMessage` momentarily with a dummy payload during agent context resets. | Yes, standard Rust borrow-checker workaround. | No. |

## Feature Gate Stubs
- `zen-core/src/attention/backends/flash.rs`: Panics with `unimplemented!("Compile with --features flash-attn...")` if the feature is missing but the planner attempts to use it. This is safe and standard Rust behavior.

## Conclusion
The most critical stub is the **PartialOffload execution logic**, which currently exists purely as a CLI argument and Planner selection, but lacks any actual hybrid VRAM/RAM forward pass logic.
