use anyhow::Result;
use crate::args::RuntimeCommand;

pub fn run_status_command() -> Result<()> {
    println!("ZentrixLLM Runtime Status");
    println!("=========================");
    
    // Memory and HW
    let hw = zen_hardware::detect_hardware();
    
    println!("System Memory:");
    println!("  RAM: {:.2} GB / {:.2} GB", hw.memory.ram_available as f64 / 1e9, hw.memory.ram_total as f64 / 1e9);
    
    println!("\nHardware Detection:");
    println!("  CPU: {}", hw.cpu.vendor);
    
    let acc = hw.accelerators.iter()
        .filter(|a| a.kind != zen_hardware::AcceleratorType::CpuOnly)
        .max_by_key(|a| a.vram_available);
    if let Some(a) = acc {
        println!("  GPU: {} ({:.2} GB VRAM)", a.name, a.vram_total as f64 / 1e9);
    } else {
        println!("  GPU: none");
    }

    println!("\nFeatures:");
    println!("  CUDA Available: {}", cfg!(feature = "cuda"));
    println!("  Metal Available: {}", cfg!(feature = "metal"));
    
    let models_count = crate::registry::Registry::load().map(|r| r.config.models.len()).unwrap_or(0);
    println!("\nRegistry:");
    println!("  Models Registered: {}", models_count);
    
    println!("\nCapabilities:");
    println!("  FlexLoad: Enabled");
    println!("  PartialOffload: Enabled");
    println!("  Multi-GPU: Disabled");
    
    Ok(())
}

pub fn run_runtime_command(cmd: RuntimeCommand) -> Result<()> {
    match cmd {
        RuntimeCommand::Modes => {
            println!("Supported Runtime Modes:");
            println!("  - normal          (Full GPU/CPU resident)");
            println!("  - cpu-only        (Forced CPU execution)");
            println!("  - gpu-only        (Forced GPU execution)");
            println!("  - partial-offload (Mixed execution)");
            println!("  - flexload        (Paged VRAM execution)");
        }
        RuntimeCommand::Plan { model } => {
            let registry = crate::registry::Registry::load()?;
            let metadata = if let Some(m) = registry.get(&model) {
                Some(m.clone())
            } else {
                let p = std::path::PathBuf::from(&model);
                if p.exists() {
                    crate::registry::detect_metadata_from_path(&p).ok()
                } else {
                    None
                }
            };

            if let Some(m) = metadata {
                let hw = zen_hardware::detect_hardware();
                let arch = m.architecture.clone().unwrap_or_else(|| "unknown".to_string());
                
                let quant = m.quantization.clone().unwrap_or_else(|| {
                    if model.to_lowercase().contains("q4_0") {
                        "q4_0".to_string()
                    } else if model.to_lowercase().contains("q8") {
                        "q8".to_string()
                    } else if model.to_lowercase().contains("q2_k") {
                        "q2_k".to_string()
                    } else {
                        "f16".to_string()
                    }
                });

                let mut est_meta = zen_modelhub::metadata::estimate_model_metadata(&model, &quant);
                if let Some(ctx) = m.context_length {
                    est_meta.context_window = ctx;
                }

                let input = zen_orchestrator::planner::PlannerInput {
                    hardware: hw.clone(),
                    metadata: est_meta.clone(),
                    requested_mode: zen_core::RuntimeMode::Auto,
                    fallback_allowed: true,
                    cuda_enabled: cfg!(feature = "cuda"),
                    metal_enabled: cfg!(feature = "metal"),
                };

                let decision = zen_orchestrator::planner::create_load_plan(&input);

                println!("Execution Plan:");
                println!("  Model: {}", m.model_id);
                println!("  Architecture: {}", arch);
                println!("  Format: {}", m.format);
                println!("  Quantization: {}", quant);
                if let Some(tt) = &m.gguf_tensor_type {
                    println!("  gguf_tensor_type: {}", tt);
                }
                
                let params = m.parameter_count.clone().unwrap_or_else(|| {
                    let b = est_meta.num_parameters as f64 / 1_000_000_000.0;
                    format!("{:.1}B", b)
                });
                println!("  Parameters: {}", params);
                println!("  Context Length: {}", est_meta.context_window);

                println!("\nHardware:");
                println!("  CPU: {}", hw.cpu.vendor);
                println!("  RAM Available: {:.1} GB", hw.memory.ram_available as f64 / 1e9);

                let acc = hw.accelerators.iter()
                    .filter(|a| a.kind != zen_hardware::AcceleratorType::CpuOnly)
                    .max_by_key(|a| a.vram_available);
                if let Some(a) = acc {
                    println!("  Accelerator: {} ({:.1} GB VRAM)", a.name, a.vram_available as f64 / 1e9);
                } else {
                    println!("  Accelerator: none");
                }

                println!("\nPlanner:");
                match decision {
                    zen_orchestrator::planner::PlannerDecision::Accepted(plan) => {
                        let req_mode = match plan.requested_mode {
                            zen_core::RuntimeMode::Auto => "auto",
                            zen_core::RuntimeMode::GpuOnly => "gpu-only",
                            zen_core::RuntimeMode::CpuOnly => "cpu-only",
                            zen_core::RuntimeMode::PartialOffload => "partial-offload",
                            zen_core::RuntimeMode::FlexLoad => "flexload",
                            zen_core::RuntimeMode::MultiGpu => "multi-gpu",
                        };
                        let sel_mode = match plan.selected_mode {
                            zen_core::RuntimeMode::Auto => "auto",
                            zen_core::RuntimeMode::GpuOnly => "gpu-only",
                            zen_core::RuntimeMode::CpuOnly => "cpu-only",
                            zen_core::RuntimeMode::PartialOffload => "partial-offload",
                            zen_core::RuntimeMode::FlexLoad => "flexload",
                            zen_core::RuntimeMode::MultiGpu => "multi-gpu",
                        };
                        println!("  Requested Mode: {}", req_mode);
                        println!("  Selected Mode: {}", sel_mode);
                        
                        match plan.gpu_layers {
                            Some(n) => println!("  GPU Layers: {}", n),
                            None => println!("  GPU Layers: none"),
                        }
                        
                        match plan.cpu_layers {
                            Some(n) => println!("  CPU Layers: {}", n),
                            None => println!("  CPU Layers: all"),
                        }
                        
                        println!("  Fallback Used: {}", plan.fallback_used);
                        println!("  Reason: {}", plan.reason);
                    }
                    zen_orchestrator::planner::PlannerDecision::Rejected { reason, .. } => {
                        println!("  Rejected: {}", reason);
                    }
                }
            } else {
                println!("Model '{}' not found in registry or is not a valid path.", model);
            }
        }
    }
    Ok(())
}
