use clap::Subcommand;

#[derive(Subcommand, Clone)]
pub enum ModelsCommand {
    /// Add a local model to the registry
    Add {
        /// Local path to the model directory or GGUF/Safetensors file
        path: String,
    },
    /// List all registered models
    List,
    /// Inspect a registered model's metadata
    Inspect {
        /// Model ID
        id: String,
    },
    /// Remove a registered model
    Remove {
        /// Model ID
        id: String,
    },
}

#[derive(Subcommand, Clone)]
pub enum PullCommand {
    /// Pull a model from the Hugging Face Hub
    Hf {
        /// Repository ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        repo: String,
    },
    /// Pull a model directly via an HTTP URL
    Url {
        /// Direct URL to the model file
        url: String,
    },
    /// Pull a model from local Ollama blobs
    Ollama {
        /// Ollama model name (e.g., "qwen2.5:0.5b")
        model: String,
    },
}

#[derive(clap::Subcommand, Clone)]
pub enum RuntimeCommand {
    /// List all supported runtime modes
    Modes,
    /// Generate an execution plan for a registered model
    Plan {
        /// Registered Model ID
        model: String,
    },
}
