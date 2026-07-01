use zen_core::RuntimeMode;
use zen_hardware::{AcceleratorType, HardwareInfo};
use zen_modelhub::metadata::ModelMetadata;

#[derive(Debug, Clone)]
pub struct PlannerInput {
    pub hardware: HardwareInfo,
    pub metadata: ModelMetadata,
    pub requested_mode: RuntimeMode,
    pub fallback_allowed: bool,
    pub cuda_enabled: bool,
    pub metal_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct TensorParallelPlan {
    pub world_size: usize,
    pub backend: String,
}

#[derive(Debug, Clone)]
pub struct LoadPlan {
    pub requested_mode: RuntimeMode,
    pub selected_mode: RuntimeMode,
    pub gpu_layers: Option<usize>,
    pub cpu_layers: Option<usize>,
    pub accelerator_name: Option<String>,
    pub fallback_used: bool,
    pub reason: String,
    pub tensor_parallel: Option<TensorParallelPlan>,
}

#[derive(Debug, Clone)]
pub enum PlannerDecision {
    Accepted(LoadPlan),
    Rejected {
        reason: String,
        suggestion: Option<RuntimeMode>,
    },
}

pub fn create_load_plan(input: &PlannerInput) -> PlannerDecision {
    let hw = &input.hardware;
    let meta = &input.metadata;

    // Single best accelerator rule
    let best_accelerator = hw.accelerators.iter()
        .filter(|a| a.kind != AcceleratorType::CpuOnly)
        .max_by_key(|a| a.vram_available);

    let mut best_vram = best_accelerator.map(|a| a.vram_available).unwrap_or(0);
    let accelerator_name = best_accelerator.map(|a| a.name.clone());
    let ram = hw.memory.ram_available;

    let mut unusable_reason = None;
    if let Some(acc) = best_accelerator {
        if acc.kind == AcceleratorType::Nvidia && !input.cuda_enabled {
            best_vram = 0;
            unusable_reason = Some("NVIDIA GPU detected, but current binary was built without CUDA feature. Falling back to CPU-only.");
        } else if acc.kind == AcceleratorType::AppleMetal && !input.metal_enabled {
            best_vram = 0;
            unusable_reason = Some("Apple Silicon detected, but current binary was built without Metal feature. Falling back to CPU-only.");
        }
    }

    let min_partial_offload_threshold = 500 * 1024 * 1024; // 500MB

    let decision = match input.requested_mode {
        RuntimeMode::Auto => {
            if best_vram >= meta.estimated_vram_usage {
                PlannerDecision::Accepted(LoadPlan {
                    requested_mode: RuntimeMode::Auto,
                    selected_mode: RuntimeMode::GpuOnly,
                    gpu_layers: Some(meta.layer_count),
                    cpu_layers: Some(0),
                    accelerator_name,
                    fallback_used: false,
                    reason: "Sufficient VRAM available for full GPU offload.".to_string(),
                    tensor_parallel: None,
                })
            } else if best_vram > min_partial_offload_threshold {
                let bytes_per_layer = meta.estimated_vram_usage / meta.layer_count.max(1);
                let fit_layers = best_vram / bytes_per_layer.max(1);
                
                PlannerDecision::Accepted(LoadPlan {
                    requested_mode: RuntimeMode::Auto,
                    selected_mode: RuntimeMode::PartialOffload,
                    gpu_layers: Some(fit_layers),
                    cpu_layers: Some(meta.layer_count.saturating_sub(fit_layers)),
                    accelerator_name,
                    fallback_used: false,
                    reason: format!("Partial offload: {} layers fit in VRAM.", fit_layers),
                    tensor_parallel: None,
                })
            } else if ram >= meta.estimated_ram_usage {
                PlannerDecision::Accepted(LoadPlan {
                    requested_mode: RuntimeMode::Auto,
                    selected_mode: RuntimeMode::CpuOnly,
                    gpu_layers: Some(0),
                    cpu_layers: Some(meta.layer_count),
                    accelerator_name: None,
                    fallback_used: false,
                    reason: unusable_reason.unwrap_or("Insufficient VRAM. Sufficient RAM available for CPU-only.").to_string(),
                    tensor_parallel: None,
                })
            } else {
                PlannerDecision::Accepted(LoadPlan {
                    requested_mode: RuntimeMode::Auto,
                    selected_mode: RuntimeMode::FlexLoad,
                    gpu_layers: None,
                    cpu_layers: None,
                    accelerator_name: None,
                    fallback_used: false,
                    reason: unusable_reason.unwrap_or("Insufficient RAM and VRAM. Falling back to low-memory FlexLoad streaming.").to_string(),
                    tensor_parallel: None,
                })
            }
        }
        RuntimeMode::GpuOnly => {
            if best_vram >= meta.estimated_vram_usage {
                PlannerDecision::Accepted(LoadPlan {
                    requested_mode: RuntimeMode::GpuOnly,
                    selected_mode: RuntimeMode::GpuOnly,
                    gpu_layers: Some(meta.layer_count),
                    cpu_layers: Some(0),
                    accelerator_name,
                    fallback_used: false,
                    reason: "User requested GpuOnly. Sufficient VRAM available.".to_string(),
                    tensor_parallel: None,
                })
            } else if input.fallback_allowed {
                let mut fallback_decision = create_load_plan(&PlannerInput {
                    hardware: input.hardware.clone(),
                    metadata: input.metadata.clone(),
                    requested_mode: RuntimeMode::Auto,
                    fallback_allowed: false,
                    cuda_enabled: input.cuda_enabled,
                    metal_enabled: input.metal_enabled,
                });
                
                if let PlannerDecision::Accepted(ref mut plan) = fallback_decision {
                    plan.requested_mode = RuntimeMode::GpuOnly;
                    plan.fallback_used = true;
                    plan.reason = format!("Fallback triggered: {}", plan.reason);
                }
                fallback_decision
            } else {
                PlannerDecision::Rejected {
                    reason: "User requested GpuOnly, but insufficient VRAM is available. Fallback is disabled.".to_string(),
                    suggestion: Some(RuntimeMode::Auto),
                }
            }
        }
        RuntimeMode::CpuOnly => {
            if ram >= meta.estimated_ram_usage {
                PlannerDecision::Accepted(LoadPlan {
                    requested_mode: RuntimeMode::CpuOnly,
                    selected_mode: RuntimeMode::CpuOnly,
                    gpu_layers: Some(0),
                    cpu_layers: Some(meta.layer_count),
                    accelerator_name: None,
                    fallback_used: false,
                    reason: "User requested CpuOnly. Sufficient RAM available.".to_string(),
                    tensor_parallel: None,
                })
            } else if input.fallback_allowed {
                PlannerDecision::Accepted(LoadPlan {
                    requested_mode: RuntimeMode::CpuOnly,
                    selected_mode: RuntimeMode::FlexLoad,
                    gpu_layers: None,
                    cpu_layers: None,
                    accelerator_name: None,
                    fallback_used: true,
                    reason: "Fallback triggered: Insufficient RAM for CPU-only, using FlexLoad.".to_string(),
                    tensor_parallel: None,
                })
            } else {
                PlannerDecision::Rejected {
                    reason: "User requested CpuOnly, but insufficient RAM is available. Fallback is disabled.".to_string(),
                    suggestion: Some(RuntimeMode::FlexLoad),
                }
            }
        }
        RuntimeMode::PartialOffload => {
            if best_vram > min_partial_offload_threshold {
                let bytes_per_layer = meta.estimated_vram_usage / meta.layer_count.max(1);
                let fit_layers = best_vram / bytes_per_layer.max(1);
                
                PlannerDecision::Accepted(LoadPlan {
                    requested_mode: RuntimeMode::PartialOffload,
                    selected_mode: RuntimeMode::PartialOffload,
                    gpu_layers: Some(fit_layers),
                    cpu_layers: Some(meta.layer_count.saturating_sub(fit_layers)),
                    accelerator_name,
                    fallback_used: false,
                    reason: format!("User requested PartialOffload. {} layers fit in VRAM.", fit_layers),
                    tensor_parallel: None,
                })
            } else if input.fallback_allowed {
                if ram >= meta.estimated_ram_usage {
                    PlannerDecision::Accepted(LoadPlan {
                        requested_mode: RuntimeMode::PartialOffload,
                        selected_mode: RuntimeMode::CpuOnly,
                        gpu_layers: Some(0),
                        cpu_layers: Some(meta.layer_count),
                        accelerator_name: None,
                        fallback_used: true,
                        reason: "Fallback triggered: Insufficient VRAM for partial offload. Using CPU-only.".to_string(),
                        tensor_parallel: None,
                    })
                } else {
                    PlannerDecision::Accepted(LoadPlan {
                        requested_mode: RuntimeMode::PartialOffload,
                        selected_mode: RuntimeMode::FlexLoad,
                        gpu_layers: None,
                        cpu_layers: None,
                        accelerator_name: None,
                        fallback_used: true,
                        reason: "Fallback triggered: Insufficient RAM/VRAM. Using FlexLoad.".to_string(),
                        tensor_parallel: None,
                    })
                }
            } else {
                PlannerDecision::Rejected {
                    reason: "User requested PartialOffload, but insufficient VRAM is available. Fallback is disabled.".to_string(),
                    suggestion: Some(RuntimeMode::CpuOnly),
                }
            }
        }
        RuntimeMode::FlexLoad => {
            if meta.format == "gguf" {
                PlannerDecision::Accepted(LoadPlan {
                    requested_mode: RuntimeMode::FlexLoad,
                    selected_mode: if best_vram > min_partial_offload_threshold { RuntimeMode::PartialOffload } else { RuntimeMode::CpuOnly },
                    gpu_layers: None,
                    cpu_layers: None,
                    accelerator_name: None,
                    fallback_used: true,
                    reason: "FlexLoad streaming currently supports Safetensor models only. Falling back to non-streaming mode.".to_string(),
                    tensor_parallel: None,
                })
            } else {
                PlannerDecision::Accepted(LoadPlan {
                    requested_mode: RuntimeMode::FlexLoad,
                    selected_mode: RuntimeMode::FlexLoad,
                    gpu_layers: None,
                    cpu_layers: None,
                    accelerator_name: None,
                    fallback_used: false,
                    reason: "User requested FlexLoad ultra-low-memory streaming.".to_string(),
                    tensor_parallel: None,
                })
            }
        }
        RuntimeMode::MultiGpu => {
            PlannerDecision::Rejected {
                reason: "MultiGpu mode is currently unimplemented in the execution layer. Coming soon.".to_string(),
                suggestion: Some(RuntimeMode::GpuOnly),
            }
        }
    };

    if let PlannerDecision::Accepted(ref plan) = decision {
        zen_telemetry::Telemetry::record_planner_decision(
            &format!("{:?}", plan.selected_mode),
            plan.gpu_layers.unwrap_or(0),
            plan.cpu_layers.unwrap_or(0),
        );
    } else if let PlannerDecision::Rejected { .. } = decision {
        zen_telemetry::Telemetry::record_planner_decision("Rejected", 0, 0);
    }

    decision
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_hardware::{CpuInfo, MemoryInfo};

    fn dummy_hw(vram: usize, ram: usize) -> HardwareInfo {
        let mut accelerators = vec![];
        if vram > 0 {
            accelerators.push(zen_hardware::AcceleratorInfo {
                kind: zen_hardware::AcceleratorType::Nvidia,
                name: "TestGPU".to_string(),
                vram_total: vram,
                vram_available: vram,
                compute_capability: None,
            });
        }
        HardwareInfo {
            cpu: CpuInfo { cores: 8, vendor: "Test".to_string() },
            memory: MemoryInfo { ram_total: ram, ram_available: ram },
            accelerators,
        }
    }

    fn dummy_meta(vram: usize, ram: usize) -> ModelMetadata {
        ModelMetadata {
            num_parameters: 7_000_000_000,
            model_dtype: "bf16".to_string(),
            estimated_vram_usage: vram,
            estimated_ram_usage: ram,
            layer_count: 32,
            context_window: 8192,
        }
    }

    #[test]
    fn test_auto_selects_gpu_when_vram_enough() {
        let hw = dummy_hw(10_000_000_000, 16_000_000_000);
        let meta = dummy_meta(8_000_000_000, 4_000_000_000);
        let input = PlannerInput {
            hardware: hw,
            metadata: meta,
            requested_mode: RuntimeMode::Auto,
            fallback_allowed: false,
        };
        
        match create_load_plan(&input) {
            PlannerDecision::Accepted(plan) => {
                assert!(matches!(plan.selected_mode, RuntimeMode::GpuOnly));
                assert_eq!(plan.gpu_layers, Some(32));
            }
            _ => panic!("Expected Accepted"),
        }
    }

    #[test]
    fn test_auto_selects_partial_offload_when_some_vram() {
        let hw = dummy_hw(4_000_000_000, 16_000_000_000);
        let meta = dummy_meta(8_000_000_000, 4_000_000_000);
        let input = PlannerInput {
            hardware: hw,
            metadata: meta,
            requested_mode: RuntimeMode::Auto,
            fallback_allowed: false,
        };
        
        match create_load_plan(&input) {
            PlannerDecision::Accepted(plan) => {
                assert!(matches!(plan.selected_mode, RuntimeMode::PartialOffload));
                // 4GB out of 8GB for 32 layers = 16 layers
                assert_eq!(plan.gpu_layers, Some(16));
                assert_eq!(plan.cpu_layers, Some(16));
            }
            _ => panic!("Expected Accepted"),
        }
    }

    #[test]
    fn test_auto_selects_cpu_when_no_gpu_but_ram_enough() {
        let hw = dummy_hw(0, 16_000_000_000);
        let meta = dummy_meta(8_000_000_000, 4_000_000_000);
        let input = PlannerInput {
            hardware: hw,
            metadata: meta,
            requested_mode: RuntimeMode::Auto,
            fallback_allowed: false,
        };
        
        match create_load_plan(&input) {
            PlannerDecision::Accepted(plan) => {
                assert!(matches!(plan.selected_mode, RuntimeMode::CpuOnly));
                assert_eq!(plan.cpu_layers, Some(32));
            }
            _ => panic!("Expected Accepted"),
        }
    }

    #[test]
    fn test_auto_selects_flexload_when_ram_too_low() {
        let hw = dummy_hw(0, 2_000_000_000);
        let meta = dummy_meta(8_000_000_000, 4_000_000_000);
        let input = PlannerInput {
            hardware: hw,
            metadata: meta,
            requested_mode: RuntimeMode::Auto,
            fallback_allowed: false,
        };
        
        match create_load_plan(&input) {
            PlannerDecision::Accepted(plan) => {
                assert!(matches!(plan.selected_mode, RuntimeMode::FlexLoad));
            }
            _ => panic!("Expected Accepted"),
        }
    }

    #[test]
    fn test_gpu_only_rejects_when_no_vram_and_no_fallback() {
        let hw = dummy_hw(2_000_000_000, 16_000_000_000);
        let meta = dummy_meta(8_000_000_000, 4_000_000_000);
        let input = PlannerInput {
            hardware: hw,
            metadata: meta,
            requested_mode: RuntimeMode::GpuOnly,
            fallback_allowed: false,
        };
        
        match create_load_plan(&input) {
            PlannerDecision::Rejected { suggestion, .. } => {
                assert!(matches!(suggestion, Some(RuntimeMode::Auto)));
            }
            _ => panic!("Expected Rejected"),
        }
    }

    #[test]
    fn test_gpu_only_falls_back_when_allowed() {
        let hw = dummy_hw(4_000_000_000, 16_000_000_000);
        let meta = dummy_meta(8_000_000_000, 4_000_000_000);
        let input = PlannerInput {
            hardware: hw,
            metadata: meta,
            requested_mode: RuntimeMode::GpuOnly,
            fallback_allowed: true,
        };
        
        match create_load_plan(&input) {
            PlannerDecision::Accepted(plan) => {
                assert!(matches!(plan.requested_mode, RuntimeMode::GpuOnly));
                assert!(matches!(plan.selected_mode, RuntimeMode::PartialOffload));
                assert!(plan.fallback_used);
                assert_eq!(plan.gpu_layers, Some(16));
            }
            _ => panic!("Expected Accepted"),
        }
    }
}
