# Command Surface Audit

## Target Command Surface

| Target Command | Current State / Command | Status | Notes |
| :--- | :--- | :--- | :--- |
| `zenllm run <model>` | `zenllm run` | Production | Implemented fully. |
| `zenllm server` | `zenllm serve` | Production | Needs rename from `serve` to `server`. |
| `zenllm doctor` | `zenllm doctor` | Experimental | Exists, does diagnostic checks. |
| `zenllm status` | N/A | Missing | |
| `zenllm bench` | `zenllm bench` | Experimental | Exists and functional. |
| `zenllm models list` | N/A | Missing | Will replace `cache list`. |
| `zenllm models add <path>` | N/A | Missing | |
| `zenllm models remove <id>` | N/A | Missing | Will replace `cache delete`. |
| `zenllm models inspect <id>` | N/A | Missing | |
| `zenllm pull <model>` | N/A | Missing | |
| `zenllm pull hf <repo>` | N/A | Missing | |
| `zenllm pull ollama <model>` | N/A | Missing | |
| `zenllm pull url <url>` | N/A | Missing | |
| `zenllm cache list` | `zenllm cache list` | Production | Already exists. To be deprecated/moved to `models list`. |
| `zenllm cache clean` | N/A | Missing | Needs implementation. |
| `zenllm config show` | N/A | Missing | |
| `zenllm config set` | N/A | Missing | |
| `zenllm runtime modes` | N/A | Missing | |
| `zenllm runtime plan <model>`| N/A | Missing | |
| `zenllm hardware` | N/A | Missing | Could be extracted from `doctor`. |
| `zenllm version` | `zenllm --version` | Missing (Command) | Exists as flag, but explicit command is missing. |

## Other Commands (To be evaluated)
| Command | Status | Notes |
| :--- | :--- | :--- |
| `zenllm quantize` | Experimental | Exists. |
| `zenllm tune` | Experimental | Exists. Provides hardware-aware tuning recommendations. |
| `zenllm login` | Production | Exists. Authenticates HF token. |
| `zenllm from-config` | Production | Exists. Runs from a TOML definition. |
| `zenllm completions` | Production | Generates shell completions. |
