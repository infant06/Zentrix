use metrics::{counter, gauge, histogram};
use std::time::Duration;

pub struct Telemetry;

impl Telemetry {
    pub fn record_layer_fetch_latency(layer_id: usize, duration: Duration) {
        histogram!("zenllm_layer_fetch_latency_seconds", "layer_id" => layer_id.to_string()).record(duration.as_secs_f64());
    }

    pub fn record_layer_staging_latency(layer_id: usize, duration: Duration) {
        histogram!("zenllm_layer_staging_latency_seconds", "layer_id" => layer_id.to_string()).record(duration.as_secs_f64());
    }

    pub fn record_layer_eviction_latency(layer_id: usize, duration: Duration) {
        histogram!("zenllm_layer_eviction_latency_seconds", "layer_id" => layer_id.to_string()).record(duration.as_secs_f64());
    }

    pub fn record_planner_decision(decision_type: &str, num_layers_gpu: usize, num_layers_cpu: usize) {
        counter!("zenllm_planner_decisions_total", "decision_type" => decision_type.to_string()).increment(1);
        gauge!("zenllm_planner_gpu_layers").set(num_layers_gpu as f64);
        gauge!("zenllm_planner_cpu_layers").set(num_layers_cpu as f64);
    }

    pub fn record_cache_hit(layer_id: usize, location: &str) {
        counter!("zenllm_cache_hits_total", "layer_id" => layer_id.to_string(), "location" => location.to_string()).increment(1);
    }

    pub fn record_cache_miss(layer_id: usize) {
        counter!("zenllm_cache_misses_total", "layer_id" => layer_id.to_string()).increment(1);
    }

    pub fn record_ttft(duration: Duration) {
        histogram!("zenllm_ttft_seconds").record(duration.as_secs_f64());
    }

    pub fn record_tps(tps: f64) {
        gauge!("zenllm_tps").set(tps);
    }
}
