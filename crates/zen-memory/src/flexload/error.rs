#[derive(Debug)]
pub enum FlexLoadError {
    LayerNotFound(usize),
    DeviceOutOfMemory(String),
    IoError(String),
    InvalidStateTransition(String),
}

impl std::fmt::Display for FlexLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LayerNotFound(id) => write!(f, "Layer {} not found", id),
            Self::DeviceOutOfMemory(msg) => write!(f, "Device OOM: {}", msg),
            Self::IoError(msg) => write!(f, "IO Error: {}", msg),
            Self::InvalidStateTransition(msg) => write!(f, "Invalid State Transition: {}", msg),
        }
    }
}

impl std::error::Error for FlexLoadError {}
