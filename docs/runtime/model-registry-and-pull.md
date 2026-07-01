# Model Registry and Pull Architecture

## Goal
Establish a unified model registry allowing models from multiple sources (Hugging Face, Ollama, URL, local files) to be queried, managed, and executed under Zentrix without forcing the user to run external backends like Ollama for inference. 

## Command Surface Implementation
The CLI will expose the following commands:
1. `zenllm pull hf <repo>` (Hugging Face Hub)
2. `zenllm pull ollama <model>` (Ollama local blob discovery/import)
3. `zenllm pull url <url>` (Direct HTTP/HTTPS file pull)
4. `zenllm models add <path>` (Local model registration)

## Ollama Import Philosophy
- Do **not** require running the Ollama API server as an inference backend.
- Locate the local Ollama blob storage (usually `~/.ollama/models/blobs`).
- Import/pull the model metadata and map the layer weights to `zen-core`.
- Execute the model natively within the `ZenLLM` runtime to take advantage of `PartialOffload` and `FlexLoad`.

## Registry Structure
The registry will maintain a configuration/index file containing metadata for each available model. 

### Stored Metadata Fields
- `model_id` (Unique identifier for the model)
- `source_type`: `hf` | `ollama` | `url` | `local`
- `source_ref` (Original tag, URL, or HF Hub ref)
- `local_path` (Absolute path to the GGUF/Safetensors on disk)
- `format`: `gguf` | `safetensors` | `unknown`
- `architecture` (e.g., Llama, Qwen2, DeepSeek)
- `parameter_count` (e.g., 7B, 70B)
- `quantization` (e.g., Q4_K_M, fp16)
- `context_length` (e.g., 8192, 128000)
- `recommended_mode` (Auto, GpuOnly, PartialOffload, FlexLoad)
- `last_benchmark` (Timestamp or TPS metric for cached references)
- `tags` (List of tags like `instruct`, `vision`)

## Phased Implementation Priority
1. Implement the local registry schema (`zenllm models list`, `zenllm models add`).
2. Wire up Hugging Face downloading (`zenllm pull hf`).
3. Wire up local Ollama discovery (`zenllm pull ollama`).
