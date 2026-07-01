#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageDevice {
    Disk,
    CpuRam,
    PinnedCpu,
    Gpu { index: usize },
    UnifiedMemory,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerState {
    NotLoaded,
    Queued,
    Loading,
    InRam,
    MovingToDevice,
    OnDevice,
    Evicting,
    Evicted,
    Failed(String),
}
