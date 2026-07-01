use super::device_stage::StageDevice;

#[derive(Debug, Clone)]
pub struct FlexLoadConfig {
    pub prefetch_ahead_count: usize,
    pub max_ram_bytes: u64,
    pub max_vram_bytes: u64,
    pub prefer_pinned_memory: bool,
    pub target_device: StageDevice,
    pub eviction_policy: super::layer_cache::CachePolicy,
}

#[derive(Debug, Clone)]
pub struct FlexLoadPlan {
    pub ordered_layer_ids: Vec<usize>,
    pub layer_sizes: Vec<u64>,
    pub target_device: StageDevice,
    pub prefetch_schedule: Vec<usize>,
}
