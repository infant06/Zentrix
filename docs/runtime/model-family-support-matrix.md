# Model Family Support Matrix

## Text Generation Models
| Family | Normal Runtime | FlexLoad Compile | FlexLoad Load | FlexLoad Generate | Benchmarked | Known Bugs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama** | Yes | Yes | Yes | Untested | No | None identified yet |
| **Qwen2** | Yes | Yes | Yes | Yes | Yes (Qwen2.5-0.5B) | None |
| **Qwen2.5** | Yes | Yes | Yes | Yes | Yes | None |
| **Mistral** | Yes | Yes | Yes | Untested | No | None identified yet |
| **Mixtral** | Yes | No | No | No | No | Not implemented in FlexLoad |
| **DeepSeekV2** | Yes | Yes | Untested | Untested | No | |
| **DeepSeekV3** | Yes | Yes | Untested | Untested | No | |
| **Gemma** | Yes | Yes | Untested | Untested | No | |
| **Phi-2 / Phi-3**| Yes | Yes | Untested | Untested | No | |
| **Yi** | Unknown | No | No | No | No | |
| **SmolLM** | Yes | No | No | No | No | |

## PartialOffload Support
*Currently, `PartialOffload` is completely unimplemented for execution in all models. It exists purely as a planner flag.*

## Multimodal Models
| Family | Normal Runtime | FlexLoad Support | Notes |
| :--- | :--- | :--- | :--- |
| **Qwen2-VL** | Yes | Broken | Generating attention masks using local layer caches crashes during `broadcast_add` due to `[1, 14, 8, 8]` vs `[8, 16]` dimension mismatch. (See `flexload-cache-mismatch.md`). |
| **Audio Models** | No | No | |
| **Embedding Models**| Yes | No | Embedding uses `Normal` runtime. |

> [!WARNING]
> Do not assume a model works in `FlexLoad` just because it compiles. Only `Qwen2` has been end-to-end benchmarked in `FlexLoad` execution.
