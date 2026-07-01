use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub num_parameters: usize,
    pub model_dtype: String,
    pub estimated_vram_usage: usize,
    pub estimated_ram_usage: usize,
    pub layer_count: usize,
    pub context_window: usize,
    pub format: String,
}

pub fn estimate_model_metadata(model_id: &str, dtype: &str) -> ModelMetadata {
    let lowercase_id = model_id.to_lowercase();
    
    // Heuristic parameter sizing
    let mut num_parameters = 7_000_000_000; // Default 7B
    if lowercase_id.contains("8b") {
        num_parameters = 8_000_000_000;
    } else if lowercase_id.contains("0.5b") {
        num_parameters = 500_000_000;
    } else if lowercase_id.contains("1.5b") || lowercase_id.contains("1b") {
        num_parameters = 1_500_000_000;
    } else if lowercase_id.contains("3b") {
        num_parameters = 3_000_000_000;
    } else if lowercase_id.contains("14b") {
        num_parameters = 14_000_000_000;
    } else if lowercase_id.contains("32b") {
        num_parameters = 32_000_000_000;
    } else if lowercase_id.contains("70b") || lowercase_id.contains("72b") {
        num_parameters = 70_000_000_000;
    }

    // Heuristic bytes per parameter
    let bytes_per_param = match dtype.to_lowercase().as_str() {
        "f32" | "float32" => 4.0,
        "f16" | "float16" | "bf16" | "bfloat16" => 2.0,
        "int8" | "q8" => 1.0,
        "int4" | "q4" => 0.5,
        _ => 2.0, // Assume half precision by default
    };

    let base_memory_usage = (num_parameters as f64 * bytes_per_param) as usize;
    
    // Add overhead heuristics (KV cache buffers, context windows, etc.)
    // 20% overhead for safety
    let estimated_vram_usage = (base_memory_usage as f64 * 1.2) as usize;
    
    // If we load completely into VRAM, RAM usage is minimal, but we assume
    // some staging buffer in RAM.
    let estimated_ram_usage = (base_memory_usage as f64 * 0.3) as usize;

    let layer_count = match num_parameters {
        n if n < 2_000_000_000 => 24,
        n if n < 10_000_000_000 => 32,
        n if n < 20_000_000_000 => 40,
        n if n < 40_000_000_000 => 64,
        _ => 80,
    };

    ModelMetadata {
        num_parameters,
        model_dtype: dtype.to_string(),
        estimated_vram_usage,
        estimated_ram_usage,
        layer_count,
        context_window: 8192, // Default standard context window
        format: "safetensors".to_string(), // Default format
    }
}
