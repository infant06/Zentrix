use anyhow::Result;
use crate::args::VectorCommand;
use zen_core::search::vector::VectorSearchProvider;
use zen_vector::provider::ZenVectorSearchProvider;
use zen_vector::storage::json::JsonVectorStorage;

#[cfg(feature = "lance-storage")]
use zen_vector::storage::lance::LanceVectorStorage;

/// Return the default path for the persistent JSON vector store.
fn default_store_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".zenllm")
        .join("vectors")
        .join("store.json")
}

// Helper to build the provider based on CLI config.
async fn load_vector_config() -> crate::config::VectorOptionsToml {
    if let Ok(content) = std::fs::read_to_string("zenllm.toml") {
        if let Ok(config) = toml::from_str::<crate::config::CliConfig>(&content) {
            match config {
                crate::config::CliConfig::Serve(s) => return s.vector,
                crate::config::CliConfig::Run(r) => return r.vector,
            }
        }
    }
    crate::config::VectorOptionsToml::default()
}

pub async fn build_provider(_dim: usize) -> Result<Box<dyn VectorSearchProvider>> {
    let config = load_vector_config().await;
    let storage_type = config.resolve_storage_type()?;

    if storage_type == "lance" {
        #[cfg(feature = "lance-storage")]
        {
            let path = config.path.clone().unwrap_or_else(|| std::path::PathBuf::from(".zenllm/vector_store"));
            match LanceVectorStorage::new(&path, _dim) {
                Ok(storage) => {
                    tracing::info!("Initialized Lance vector storage at {:?}", path);
                    return Ok(Box::new(ZenVectorSearchProvider::new(storage)));
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize Lance storage ({e}). Downgrading to JSON storage.");
                }
            }
        }
        #[cfg(not(feature = "lance-storage"))]
        {
            tracing::warn!("Lance storage is not enabled. Downgrading to JSON storage.");
        }
    }

    // Default: persistent JSON storage
    let path = config.path.unwrap_or_else(default_store_path);
    let storage = JsonVectorStorage::open(&path).map_err(|e| anyhow::anyhow!(e))?;
    tracing::info!("Initialized JSON vector storage at {:?}", path);
    Ok(Box::new(ZenVectorSearchProvider::new(storage)))
}

fn parse_embedding(s: &str) -> anyhow::Result<Vec<f32>> {
    s.split(',')
        .map(|v| v.trim().parse::<f32>().map_err(|e| anyhow::anyhow!("Invalid float '{}': {}", v, e)))
        .collect()
}

pub async fn run_vector_command(cmd: VectorCommand) -> Result<()> {
    match cmd {
        VectorCommand::Stats => {
            // Use dim=0 — JsonVectorStorage ignores it, memory would use it for init only
            let provider = build_provider(0).await?;
            println!("{}", provider.stats().await?);
        }
        VectorCommand::Add { id, embedding, metadata, metadata_json, text } => {
            let emb = parse_embedding(&embedding)?;
            let mut provider = build_provider(emb.len()).await?;
            
            let mut meta_obj = serde_json::Map::new();
            
            if let Some(t) = text {
                meta_obj.insert("text".to_string(), serde_json::Value::String(t));
            }
            if let Some(m) = metadata {
                meta_obj.insert("metadata".to_string(), serde_json::Value::String(m));
            }
            if let Some(mj) = metadata_json {
                match serde_json::from_str::<serde_json::Value>(&mj) {
                    Ok(val) => {
                        if let serde_json::Value::Object(obj) = val {
                            for (k, v) in obj {
                                meta_obj.insert(k, v);
                            }
                        } else {
                            meta_obj.insert("data".to_string(), val);
                        }
                    }
                    Err(e) => anyhow::bail!("Invalid JSON in --metadata-json: {}", e),
                }
            }
            
            let meta_val = if meta_obj.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::Value::Object(meta_obj)
            };
            
            provider.add(&[id.clone()], &[emb], &[meta_val]).await?;
            println!("Successfully added vector {} to storage.", id);
        }
        VectorCommand::Search { embedding, top_k } => {
            let emb = parse_embedding(&embedding)?;
            let provider = build_provider(emb.len()).await?;
            let results = provider.search(&emb, top_k, false).await?;
            if results.is_empty() {
                println!("No results found.");
            } else {
                println!("Found {} results:", results.len());
                for (i, hit) in results.iter().enumerate() {
                    println!("{}. id={}, score={:.4}, metadata={}", i + 1, hit.id, hit.score, hit.metadata);
                }
            }
        }
    }
    Ok(())
}
