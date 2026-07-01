use anyhow::{Context, Result};
use std::io::Write;
use crate::args::PullCommand;
use crate::registry::{detect_metadata_from_path, Registry, SourceType};

pub async fn run_pull_command(cmd: PullCommand) -> Result<()> {
    let mut registry = Registry::load()?;

    match cmd {
        PullCommand::Hf { repo } => {
            println!("Pulling from Hugging Face: {}", repo);
            let dest_dir = std::env::current_dir().unwrap().join("models");
            std::fs::create_dir_all(&dest_dir)?;
            let api = hf_hub::api::sync::Api::new()?;
            let repo_api = api.model(repo.clone());
            // This is blocking hf_hub call. We can use tokio::task::spawn_blocking.
            let info = tokio::task::spawn_blocking(move || repo_api.info()).await??;
            
            println!("Found {} files in {}", info.siblings.len(), repo);
            let mut target_gguf = None;
            for sibling in &info.siblings {
                let fname = &sibling.rfilename;
                if fname.ends_with(".gguf") || fname.ends_with(".safetensors") {
                    if target_gguf.is_none() || fname.contains("Q4_K") {
                        target_gguf = Some(fname.clone());
                    }
                }
            }
            
            let mut downloaded = vec![];
            for sibling in &info.siblings {
                let fname = &sibling.rfilename;
                let is_target_weights = Some(fname) == target_gguf.as_ref();
                let is_config = fname.ends_with("config.json") || fname.ends_with("tokenizer.json") || fname.ends_with("tokenizer_config.json");
                
                if is_target_weights || is_config {
                    println!("Downloading {}...", fname);
                    let fname_clone = fname.clone();
                    let repo_api_clone = api.model(repo.clone());
                    let path = tokio::task::spawn_blocking(move || repo_api_clone.get(&fname_clone)).await??;
                    
                    let dest_path = dest_dir.join(fname);
                    if let Some(parent) = dest_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::copy(&path, &dest_path)?;
                    
                    println!(" ✅ -> {}", dest_path.display());
                    downloaded.push(dest_path);
                }
            }
            if downloaded.is_empty() {
                println!("No model weight files found in repo {}.", repo);
                return Ok(());
            }
            if let Some(p) = downloaded.iter().find(|p| p.extension().map(|e| e == "gguf" || e == "safetensors").unwrap_or(false)) {
                if let Ok(metadata) = detect_metadata_from_path(p) {
                    registry.add(metadata.clone());
                    registry.save()?;
                    println!("✅ Registered '{}' in local registry", metadata.model_id);
                }
            }
        }
        PullCommand::Url { url } => {
            println!("Downloading from URL: {}", url);
            let fname = url.split('/').last().unwrap_or("model.gguf");
            let dest_dir = std::env::current_dir().unwrap().join("models");
            std::fs::create_dir_all(&dest_dir)?;
            let dest_path = dest_dir.join(fname);
            let partial_path = dest_path.with_extension("partial");
            
            println!("Downloading {} -> {}", url, dest_path.display());
            
            let client = reqwest::Client::new();
            let mut response = client.get(&url).send().await.context("Failed to download from URL")?;
            
            if !response.status().is_success() {
                anyhow::bail!("Download failed: HTTP status {}", response.status());
            }
            
            // Check for HTML response
            if let Some(ct) = response.headers().get(reqwest::header::CONTENT_TYPE) {
                if let Ok(ct_str) = ct.to_str() {
                    if ct_str.contains("text/html") {
                        anyhow::bail!("Download failed: HTML response received (not a GGUF model)");
                    }
                }
            }
            
            let mut file = std::fs::File::create(&partial_path)?;
            let mut total_bytes = 0;
            
            while let Some(chunk) = response.chunk().await? {
                file.write_all(&chunk)?;
                total_bytes += chunk.len();
            }
            
            // Rename partial to final
            std::fs::rename(&partial_path, &dest_path)?;
            
            println!("✅ Downloaded {} bytes to {}", total_bytes, dest_path.display());
            if let Ok(metadata) = detect_metadata_from_path(&dest_path) {
                registry.add(metadata.clone());
                registry.save()?;
                println!("✅ Registered '{}' in local registry", metadata.model_id);
            } else {
                anyhow::bail!("Download failed: Not a valid GGUF model");
            }
        }
        PullCommand::Ollama { model } => {
            println!("Importing from Ollama: {}", model);
            let dest_dir = std::env::current_dir().unwrap().join("models");
            std::fs::create_dir_all(&dest_dir)?;
            let ollama_dir = dirs::home_dir().unwrap().join(".ollama").join("models");
            if !ollama_dir.exists() {
                anyhow::bail!("Ollama models directory not found at {}", ollama_dir.display());
            }
            let model_parts: Vec<&str> = model.splitn(2, ':').collect();
            let model_name = model_parts[0];
            let model_tag = model_parts.get(1).copied().unwrap_or("latest");
            let manifest_path = ollama_dir.join("manifests").join("registry.ollama.ai").join("library").join(model_name).join(model_tag);
            if !manifest_path.exists() {
                anyhow::bail!("Ollama model manifest not found at {}. Is the model pulled with `ollama pull {}`?", manifest_path.display(), model);
            }
            let manifest_str = std::fs::read_to_string(&manifest_path)?;
            let manifest: serde_json::Value = serde_json::from_str(&manifest_str)?;
            let blobs_dir = ollama_dir.join("blobs");
            if let Some(layers) = manifest["layers"].as_array() {
                let model_layer = layers.iter()
                    .filter(|l| l["mediaType"].as_str().map(|t| t.contains("model")).unwrap_or(false))
                    .max_by_key(|l| l["size"].as_u64().unwrap_or(0));
                if let Some(layer) = model_layer {
                    let digest = layer["digest"].as_str().unwrap_or("").replace(':', "-");
                    let blob_path = blobs_dir.join(&digest);
                    if blob_path.exists() {
                        let dest_path = dest_dir.join(format!("{}.gguf", model.replace(':', "-")));
                        std::fs::copy(&blob_path, &dest_path)?;
                        let metadata = crate::registry::ModelMetadata {
                            model_id: format!("ollama/{}", model),
                            source_type: SourceType::Ollama,
                            source_ref: model.clone(),
                            local_path: dest_path.clone(),
                            format: "gguf".to_string(),
                            architecture: None,
                            parameter_count: None,
                            quantization: None,
                            gguf_tensor_type: None,
                            context_length: None,
                            recommended_mode: Some("auto".to_string()),
                            tags: vec!["ollama".to_string()],
                        };
                        registry.add(metadata);
                        registry.save()?;
                        println!("✅ Imported '{}' from Ollama: {}", model, blob_path.display());
                    } else {
                        anyhow::bail!("Ollama blob not found at {}", blob_path.display());
                    }
                }
            }
        }
    }
    
    Ok(())
}
