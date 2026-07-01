pub mod offload;
pub mod flexload;
pub mod layer_streaming;
pub mod memory_map;
pub mod cpu_cache;
pub mod gpu_cache;

#[derive(Debug, Clone, Default)]
pub enum MemoryMode {
    #[default]
    Auto,
    Offload,
    FlexLoad,
}

#[derive(Debug, Clone)]
pub struct OffloadPlan {
    pub gpu_layers: usize,
    pub cpu_layers: usize,
}

pub use flexload::{FlexLoadConfig, FlexLoadPlan, LayerStreamer, StageDevice};

#[derive(Debug, Clone)]
pub enum LayerPlacement {
    Gpu,
    Cpu,
    Disk,
}

#[derive(Debug, Clone, Default)]
pub struct MemoryBudget {
    pub vram_bytes_total: usize,
    pub vram_bytes_available: usize,
    pub ram_bytes_total: usize,
    pub ram_bytes_available: usize,
}
