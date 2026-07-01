//! Interactive mode command implementation

mod interactive;

pub(crate) use interactive::interactive_mode;
use interactive::OneshotInput;

use anyhow::Result;
use tracing::info;

use zen_core::initialize_logging;
use zen_api::zentrix_for_server_builder::ZentrixForServerBuilder;

#[cfg(feature = "code-execution")]
use super::serve::build_code_exec_config;
use super::serve::{
    apply_agent_mode, apply_quant_resolution, convert_to_model_selected, extract_device_settings,
    extract_isq_setting, extract_paged_attn_settings, extract_sandbox_settings, load_mcp_config,
    log_agent_runtime, validate_agent_options,
};
use crate::args::{AgentCliOptions, GlobalOptions, ModelType, RuntimeOptions, SandboxOptions};

/// Run the model in interactive or one-shot mode
#[allow(clippy::too_many_arguments)]
pub async fn run_interactive(
    mut model_type: ModelType,
    mut runtime: RuntimeOptions,
    agent_options: AgentCliOptions,
    sandbox: SandboxOptions,
    global: GlobalOptions,
    thinking: Option<bool>,
    input: Option<String>,
    images: Vec<String>,
    videos: Vec<String>,
    audios: Vec<String>,
) -> Result<()> {
    initialize_logging();

    agent_options.apply_to(&mut runtime);
    apply_agent_mode(&mut runtime);
    validate_agent_options(&runtime)?;
    log_agent_runtime(&runtime, None);

    // Convert our clean args to ModelSelected
    let matformer = runtime.matformer_selection();
    apply_quant_resolution(&mut model_type, &global.token_source, &matformer).await?;
    let model_selected = convert_to_model_selected(&model_type, &matformer)?;

    // Extract settings
    let (
        paged_attn,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_block_size,
        paged_cache_type,
    ) = extract_paged_attn_settings(&model_type);
    let device_opts = extract_device_settings(&model_type);
    let mut cpu = device_opts.cpu;
    let mut device_layers = device_opts.device_layers.clone();

    let hw = zen_hardware::HardwareDetector::detect();
    let model_id_str = super::serve::model_id_of(&model_type);
    let mut metadata = zen_modelhub::metadata::estimate_model_metadata(model_id_str, "f16");
    
    metadata.format = match &model_selected {
        zen_core::ModelSelected::GGUF { .. } | zen_core::ModelSelected::LoraGGUF { .. } | zen_core::ModelSelected::XLoraGGUF { .. } => "gguf".to_string(),
        zen_core::ModelSelected::GGML { .. } | zen_core::ModelSelected::LoraGGML { .. } | zen_core::ModelSelected::XLoraGGML { .. } => "ggml".to_string(),
        _ => "safetensors".to_string(),
    };

    let planner_input = zen_orchestrator::planner::PlannerInput {
        hardware: hw,
        metadata,
        requested_mode: runtime.mode.clone().into(),
        fallback_allowed: !matches!(runtime.mode, crate::args::RuntimeModeArg::GpuOnly),
        cuda_enabled: cfg!(feature = "cuda"),
        metal_enabled: cfg!(feature = "metal"),
    };

    let decision = zen_orchestrator::planner::create_load_plan(&planner_input);
    match decision {
        zen_orchestrator::planner::PlannerDecision::Accepted(plan) => {
            tracing::info!("Planner decision: {}", plan.reason);
            
            runtime.mode = match plan.selected_mode {
                zen_core::RuntimeMode::Auto => crate::args::RuntimeModeArg::Auto,
                zen_core::RuntimeMode::GpuOnly => crate::args::RuntimeModeArg::GpuOnly,
                zen_core::RuntimeMode::CpuOnly => crate::args::RuntimeModeArg::CpuOnly,
                zen_core::RuntimeMode::PartialOffload => crate::args::RuntimeModeArg::PartialOffload,
                zen_core::RuntimeMode::FlexLoad => crate::args::RuntimeModeArg::FlexLoad,
                zen_core::RuntimeMode::MultiGpu => crate::args::RuntimeModeArg::MultiGpu,
            };
            
            if matches!(plan.selected_mode, zen_core::RuntimeMode::GpuOnly) {
                cpu = false;
                device_layers = None;
            } else if matches!(plan.selected_mode, zen_core::RuntimeMode::CpuOnly) {
                cpu = true;
                device_layers = None;
            } else if matches!(plan.selected_mode, zen_core::RuntimeMode::PartialOffload) {
                cpu = false;
                if runtime.gpu_layers != "auto" {
                    if device_layers.is_none() {
                        if let Ok(num) = runtime.gpu_layers.parse::<usize>() {
                            device_layers = Some(vec![format!("0:{}", num)]);
                        }
                    }
                } else if device_layers.is_none() {
                    if let Some(gl) = plan.gpu_layers {
                        device_layers = Some(vec![format!("0:{}", gl)]);
                        tracing::info!("Planner auto-configured {} GPU layers for PartialOffload", gl);
                    }
                }
            } else if matches!(plan.selected_mode, zen_core::RuntimeMode::FlexLoad) {
                cpu = true;
                device_layers = None;
            }
        }
        zen_orchestrator::planner::PlannerDecision::Rejected { reason, suggestion } => {
            anyhow::bail!("Planner rejected load: {}. Suggestion: {:?}", reason, suggestion);
        }
    }

    super::serve::print_runtime_diagnostics(device_opts, &runtime);
    let isq = extract_isq_setting(&model_type);

    // Build the zentrix instance
    let mut builder = ZentrixForServerBuilder::new()
        .with_model(model_selected)
        .with_max_seqs(runtime.max_seqs)
        .with_no_kv_cache(runtime.no_kv_cache)
        .with_token_source(global.token_source)
        .with_interactive_mode(true)
        .with_prefix_cache_n(if input.is_some() { 0 } else { runtime.prefix_cache_n })
        .set_paged_attn(paged_attn)
        .with_cpu(cpu)
        .with_enable_search(runtime.enable_search)
        .with_seed_optional(global.seed)
        .with_log_optional(global.log.as_ref().map(|p| p.to_string_lossy().to_string()))
        .with_chat_template_optional(
            runtime
                .chat_template
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
        )
        .with_jinja_explicit_optional(
            runtime
                .jinja_explicit
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
        )
        .with_num_device_layers_optional(device_layers)
        .with_in_situ_quant_optional(isq)
        .with_paged_attn_gpu_mem_optional(paged_attn_gpu_mem)
        .with_paged_attn_gpu_mem_usage_optional(paged_attn_gpu_mem_usage)
        .with_paged_ctxt_len_optional(paged_ctxt_len)
        .with_paged_attn_block_size_optional(paged_attn_block_size)
        .with_mtp_config_optional(runtime.mtp_config())
        .with_paged_attn_cache_type(paged_cache_type)
        .with_mode(runtime.mode.into());

    if let Some(model) = runtime.search_embedding_model {
        builder = builder.with_search_embedding_model(model.into());
    }

    let mcp_client_config = load_mcp_config(runtime.mcp_config.as_deref())?;
    builder = builder.with_mcp_config_optional(mcp_client_config);

    if runtime.enable_search {
        let provider = crate::commands::vector::build_provider(0).await?;
        builder = builder.with_vector_search_provider(std::sync::Arc::from(provider));
    }

    let sandbox_policy = extract_sandbox_settings(sandbox);

    #[cfg(feature = "code-execution")]
    {
        let config = build_code_exec_config(&runtime, sandbox_policy);
        builder = builder.with_code_exec_config_optional(config);
    }
    #[cfg(not(feature = "code-execution"))]
    let _ = sandbox_policy;

    let zentrix = builder.build().await?;

    if let Some(text) = input {
        info!("Model loaded, running one-shot mode...");
        #[cfg(feature = "code-execution")]
        let do_code_exec = runtime.enable_code_execution;
        #[cfg(not(feature = "code-execution"))]
        let do_code_exec = false;

        interactive::oneshot_mode(
            zentrix.clone(),
            runtime.enable_search,
            do_code_exec,
            runtime.code_exec_permission.into(),
            thinking,
            OneshotInput {
                text,
                images,
                videos,
                audios,
            },
        )
        .await;
    } else {
        #[cfg(feature = "code-execution")]
        let do_code_exec = runtime.enable_code_execution;
        #[cfg(not(feature = "code-execution"))]
        let do_code_exec = false;

        info!("Model loaded, starting interactive mode...");
        interactive::interactive_mode(
            zentrix.clone(),
            runtime.enable_search,
            do_code_exec,
            runtime.code_exec_permission.into(),
            thinking,
        )
        .await;
    }

    Ok(())
}
