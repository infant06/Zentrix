use std::sync::OnceLock;

const ZENTRIX_CUDA_GRAPHS: &str = "ZENTRIX_CUDA_GRAPHS";
const ZEN_CUDA_GRAPHS: &str = "ZEN_CUDA_GRAPHS";
const ZENTRIX_FLASHINFER_DECODE: &str = "ZENTRIX_FLASHINFER_DECODE";
const ZEN_FLASHINFER_DECODE: &str = "ZEN_FLASHINFER_DECODE";

static CUDA_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
static FLASHINFER_DECODE_ENABLED: OnceLock<bool> = OnceLock::new();

fn env_flag(new_name: &str, old_name: &str, default: bool) -> bool {
    crate::utils::env::env_or_legacy(new_name, old_name)
        .map(|value| {
            if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on") {
                true
            } else if matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off") {
                false
            } else {
                default
            }
        })
        .unwrap_or(default)
}

pub(crate) fn cuda_graphs_enabled() -> bool {
    *CUDA_GRAPHS_ENABLED.get_or_init(|| {
        env_flag(ZENTRIX_CUDA_GRAPHS, ZEN_CUDA_GRAPHS, true)
    })
}

pub(crate) fn flashinfer_decode_enabled() -> bool {
    *FLASHINFER_DECODE_ENABLED.get_or_init(|| {
        env_flag(ZENTRIX_FLASHINFER_DECODE, ZEN_FLASHINFER_DECODE, true)
    })
}
