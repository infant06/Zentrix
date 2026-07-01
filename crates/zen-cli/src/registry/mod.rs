use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceType {
    Hf,
    Ollama,
    Url,
    Local,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: String,
    pub source_type: SourceType,
    pub source_ref: String,
    pub local_path: PathBuf,
    pub format: String,
    pub architecture: Option<String>,
    pub parameter_count: Option<String>,
    pub quantization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gguf_tensor_type: Option<String>,
    pub context_length: Option<usize>,
    pub recommended_mode: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistryConfig {
    pub models: HashMap<String, ModelMetadata>,
}

pub struct Registry {
    path: PathBuf,
    pub config: RegistryConfig,
}

impl Registry {
    pub fn load() -> Result<Self> {
        let path = dirs::home_dir()
            .context("Could not find home directory")?
            .join(".zenllm")
            .join("models.json");

        let config = if path.exists() {
            let content = fs::read_to_string(&path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            RegistryConfig::default()
        };

        Ok(Self { path, config })
    }

    pub fn save(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.config)?;
        fs::write(&self.path, content)?;
        Ok(())
    }

    pub fn add(&mut self, metadata: ModelMetadata) {
        self.config.models.insert(metadata.model_id.clone(), metadata);
    }

    pub fn remove(&mut self, id: &str) -> bool {
        self.config.models.remove(id).is_some()
    }

    pub fn get(&self, id: &str) -> Option<&ModelMetadata> {
        self.config.models.get(id)
    }

    pub fn list(&self) -> Vec<&ModelMetadata> {
        let mut models: Vec<_> = self.config.models.values().collect();
        models.sort_by_key(|m| &m.model_id);
        models
    }
}

pub fn detect_metadata_from_path(path: &Path) -> Result<ModelMetadata> {
    let mut metadata = ModelMetadata {
        model_id: path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
        source_type: SourceType::Local,
        source_ref: "local".to_string(),
        local_path: path.to_path_buf(),
        format: "unknown".to_string(),
        architecture: None,
        parameter_count: None,
        quantization: None,
        gguf_tensor_type: None,
        context_length: None,
        recommended_mode: Some("auto".to_string()),
        tags: vec![],
    };

    // Try to parse quantization from filename
    let filename_lower = metadata.model_id.to_lowercase();
    let common_quants = ["q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0"];
    for q in common_quants {
        if filename_lower.contains(q) {
            metadata.quantization = Some(q.to_string());
            break;
        }
    }

    if let Some(ext) = path.extension() {
        if ext == "gguf" {
            metadata.format = "gguf".to_string();
            if let Ok(mut file) = std::fs::File::open(path) {
                if let Ok(content) = candle_core::quantized::gguf_file::Content::read(&mut file) {
                    // Try to extract metadata
                    if let Some(arch) = content.metadata.get("general.architecture") {
                        if let candle_core::quantized::gguf_file::Value::String(s) = arch {
                            metadata.architecture = Some(s.clone());
                            
                            // Context length
                            let ctx_key = format!("{}.context_length", s);
                            if let Some(ctx) = content.metadata.get(&ctx_key) {
                                match ctx {
                                    candle_core::quantized::gguf_file::Value::U32(v) => metadata.context_length = Some(*v as usize),
                                    candle_core::quantized::gguf_file::Value::U64(v) => metadata.context_length = Some(*v as usize),
                                    _ => {}
                                }
                            }
                        }
                    }
                    if let Some(quant) = content.metadata.get("general.file_type") {
                        if let candle_core::quantized::gguf_file::Value::U32(v) = quant {
                            metadata.gguf_tensor_type = Some(format!("type_{}", v));
                            if metadata.quantization.is_none() {
                                let mapped = match v {
                                    2 => "q4_0",
                                    3 => "q4_1",
                                    8 => "q8_0",
                                    9 => "q8_1",
                                    10 => "q2_k",
                                    11 => "q3_k",
                                    12 => "q4_k",
                                    13 => "q5_k",
                                    14 => "q6_k",
                                    15 => "q8_k",
                                    _ => "unknown",
                                };
                                if mapped != "unknown" {
                                    metadata.quantization = Some(mapped.to_string());
                                } else {
                                    metadata.quantization = Some(format!("type_{}", v));
                                }
                            }
                        }
                    }
                }
            }
        } else if ext == "safetensors" {
            metadata.format = "safetensors".to_string();
            // TODO: read safetensors headers if necessary, but safetensors generally store metadata in a separate config.json
        } else if path.is_dir() {
            metadata.format = "safetensors".to_string(); // Assuming dir format
        }
    }

    Ok(metadata)
}
