use clap::Parser;
use either::Either;
use indexmap::IndexMap;
use serde::Serialize;
use std::time::Instant;
use tokio::sync::mpsc::channel;
use tracing::info;
use zen_core::{
    get_model_dtype, get_auto_device_map_params, LoaderBuilder, ModelSelected,
    RuntimeMode, TokenSource, DeviceMapSetting, ZenBuilder, Request, NormalRequest, RequestMessage, SamplingParams,
    Constraint, Response,
};
use candle_core::Device;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    #[arg(long, default_value_t = 16)]
    max_tokens: usize,

    #[arg(long, default_value_t = 1)]
    runs: usize,

    #[arg(long, default_value_t = false)]
    skip_flexload: bool,

    #[arg(long)]
    output_json: Option<String>,

    #[arg(long, default_value_t = String::from("Qwen/Qwen2.5-0.5B-Instruct"))]
    model_id: String,

    #[arg(long, default_value_t = false)]
    use_flexload: bool,
}

#[derive(Serialize)]
struct BenchmarkResult {
    mode: String,
    model: String,
    load_time_ms: u128,
    ttft_ms: f64,
    tokens_per_sec: f64,
    peak_ram_mb: usize,
    peak_vram_mb: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    info!("Starting zen-bench-runtime");
    if args.dry_run {
        info!("DRY RUN MODE: Skipping actual execution.");
        return Ok(());
    }

    let mode = if args.use_flexload && !args.skip_flexload {
        RuntimeMode::FlexLoad
    } else {
        RuntimeMode::Auto
    };

    let model_selected = ModelSelected::Plain {
        model_id: args.model_id.clone(),
        tokenizer_json: None,
        arch: None,
        dtype: zen_core::ModelDType::Auto,
        topology: None,
        organization: None,
        write_uqff: None,
        from_uqff: None,
        imatrix: None,
        calibration_file: None,
        max_seq_len: args.max_tokens + 128,
        max_batch_size: 1,
        hf_cache_path: None,
        matformer_config_path: None,
        matformer_slice_name: None,
    };

    let start_load = Instant::now();
    let dtype = get_model_dtype(&model_selected)?;
    let loader = LoaderBuilder::new(model_selected.clone()).with_mode(mode).build()?;
    
    // Default device (CPU) for basic benchmark to avoid CUDA dep issues on generic environments
    let device = Device::Cpu; 

    // We skip cache_config (PagedAttention) since we just want a basic smoke test
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &dtype,
        &device,
        false,
        DeviceMapSetting::Auto(get_auto_device_map_params(&model_selected)?),
        None,
        None,
    )?;

    let zentrix = ZenBuilder::new(pipeline, zen_core::SchedulerConfig::DefaultScheduler {
        method: zen_core::DefaultSchedulerMethod::Fixed(std::num::NonZeroUsize::new(1).unwrap()),
    }, false, None)
    .build()
    .await;

    let load_time_ms = start_load.elapsed().as_millis();
    info!("Model loaded in {} ms", load_time_ms);

    let mut all_ttft = vec![];
    let mut all_tps = vec![];

    for r in 0..args.runs {
        let (tx, mut rx) = channel(10_000);
        let req = Request::Normal(Box::new(NormalRequest {
            id: zentrix.next_request_id(),
            messages: RequestMessage::Chat {
                messages: vec![{
                    let mut m = IndexMap::new();
                    m.insert("role".to_string(), Either::Left("user".to_string()));
                    m.insert("content".to_string(), Either::Left("Explain quantum mechanics in simple terms.".to_string()));
                    m
                }],
                enable_thinking: None,
                reasoning_effort: None,
            },
            sampling_params: SamplingParams {
                max_len: Some(args.max_tokens),
                ..SamplingParams::deterministic()
            },
            response: tx,
            return_logprobs: false,
            is_streaming: true,
            constraint: Constraint::None,
            suffix: None,
            tools: None,
            tool_choice: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
            enable_code_execution: false,
            code_execution_permission: None,
            code_execution_approval_notifier: None,
            agent_permission: None,
            agent_approval_handler: None,
            agent_approval_notifier: None,
            max_tool_rounds: None,
            tool_dispatch_url: None,
            model_id: None,
            truncate_sequence: false,
            session_id: None,
            files: None,
        }));

        let sender = zentrix.get_sender(None).unwrap();
        sender.send(req).await?;

        let start_gen = Instant::now();
        let mut first_token_time = None;
        let mut token_count = 0;

        while let Some(resp) = rx.recv().await {
            match resp {
                Response::Chunk(_) | Response::CompletionChunk(_) => {
                    if first_token_time.is_none() {
                        first_token_time = Some(start_gen.elapsed().as_millis());
                    }
                    token_count += 1;
                }
                Response::Done(_) | Response::CompletionDone(_) => {
                    break;
                }
                Response::ValidationError(e) | Response::InternalError(e) => {
                    eprintln!("Run {}: Error: {:?}", r, e);
                    break;
                }
                _ => {}
            }
        }

        let total_gen_time = start_gen.elapsed().as_secs_f64();
        let tps = token_count as f64 / total_gen_time;
        
        all_ttft.push(first_token_time.unwrap_or(0));
        all_tps.push(tps);
        info!("Run {}: TTFT = {} ms, TPS = {}", r, first_token_time.unwrap_or(0), tps);
    }

    let avg_ttft = all_ttft.iter().sum::<u128>() as f64 / args.runs as f64;
    let avg_tps = all_tps.iter().sum::<f64>() / args.runs as f64;

    let result = BenchmarkResult {
        mode: if args.use_flexload && !args.skip_flexload { "flexload".into() } else { "normal".into() },
        model: args.model_id.clone(),
        load_time_ms,
        ttft_ms: avg_ttft,
        tokens_per_sec: avg_tps,
        peak_ram_mb: 0, // Mocked for smoke test
        peak_vram_mb: 0,
    };

    if let Some(out) = args.output_json {
        let json = serde_json::to_string_pretty(&result)?;
        std::fs::write(out, json)?;
    }

    Ok(())
}
