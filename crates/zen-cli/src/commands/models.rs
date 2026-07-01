use anyhow::Result;
use comfy_table::{presets::UTF8_FULL, Cell, ContentArrangement, Table};
use std::path::PathBuf;

use crate::args::ModelsCommand;
use crate::registry::{detect_metadata_from_path, Registry};

pub fn run_models_command(cmd: ModelsCommand) -> Result<()> {
    let mut registry = Registry::load()?;

    match cmd {
        ModelsCommand::Add { path } => {
            let p = PathBuf::from(&path);
            if !p.exists() {
                anyhow::bail!("Path {} does not exist", path);
            }
            let metadata = detect_metadata_from_path(&p)?;
            println!("Discovered model: {}", metadata.model_id);
            if let Some(ref arch) = metadata.architecture {
                println!("  Architecture: {}", arch);
            }
            if let Some(ctx) = metadata.context_length {
                println!("  Context Length: {}", ctx);
            }
            registry.add(metadata.clone());
            registry.save()?;
            println!("✅ Successfully registered model '{}'", metadata.model_id);
        }
        ModelsCommand::List => {
            let models = registry.list();
            if models.is_empty() {
                println!("No models registered. Use `zenllm models add <path>` to register one.");
                return Ok(());
            }

            let mut table = Table::new();
            table
                .load_preset(UTF8_FULL)
                .set_content_arrangement(ContentArrangement::Dynamic)
                .set_header(vec![
                    Cell::new("Model ID"),
                    Cell::new("Architecture"),
                    Cell::new("Format"),
                    Cell::new("Mode"),
                ]);

            for m in models {
                table.add_row(vec![
                    Cell::new(&m.model_id),
                    Cell::new(m.architecture.as_deref().unwrap_or("unknown")),
                    Cell::new(&m.format),
                    Cell::new(m.recommended_mode.as_deref().unwrap_or("auto")),
                ]);
            }
            println!("{table}");
        }
        ModelsCommand::Inspect { id } => {
            if let Some(m) = registry.get(&id) {
                let json = serde_json::to_string_pretty(m)?;
                println!("{}", json);
            } else {
                println!("Model '{}' not found in registry.", id);
            }
        }
        ModelsCommand::Remove { id } => {
            if registry.remove(&id) {
                registry.save()?;
                println!("✅ Removed model '{}'", id);
            } else {
                println!("Model '{}' not found in registry.", id);
            }
        }
    }

    Ok(())
}
