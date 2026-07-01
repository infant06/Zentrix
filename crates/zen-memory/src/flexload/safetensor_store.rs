use std::path::PathBuf;
use std::collections::{HashMap, HashSet};
use safetensors::SafeTensors;

use super::error::FlexLoadError;
use super::layer_streamer::LayerStore;
#[cfg(feature = "candle")]
use super::layer_weights::CandleLayerWeights;

pub struct LayerPrefixPattern {
    pub prefix: String, // e.g. "model.layers."
}

pub struct SafetensorsLayerStore {
    pub paths: Vec<PathBuf>,
    pub layer_prefix_pattern: LayerPrefixPattern,
}

#[cfg(feature = "candle")]
impl LayerStore for SafetensorsLayerStore {
    type LayerWeights = CandleLayerWeights;

    fn layer_ids(&self) -> Vec<usize> {
        let mut layer_ids = HashSet::new();
        for path in &self.paths {
            let file = match std::fs::File::open(path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let buffer = match unsafe { memmap2::MmapOptions::new().map(&file) } {
                Ok(b) => b,
                Err(_) => continue,
            };
            let st = match SafeTensors::deserialize(&buffer) {
                Ok(st) => st,
                Err(_) => continue,
            };

            for name in st.names() {
                if name.starts_with(&self.layer_prefix_pattern.prefix) {
                    let suffix = &name[self.layer_prefix_pattern.prefix.len()..];
                    if let Some(id_str) = suffix.split('.').next() {
                        if let Ok(id) = id_str.parse::<usize>() {
                            layer_ids.insert(id);
                        }
                    }
                }
            }
        }
        let mut ids: Vec<usize> = layer_ids.into_iter().collect();
        ids.sort_unstable();
        ids
    }

    fn load_layer(&self, layer_id: usize) -> Result<Self::LayerWeights, FlexLoadError> {
        let layer_prefix = format!("{}{}.", self.layer_prefix_pattern.prefix, layer_id);
        let mut tensors = HashMap::new();

        for path in &self.paths {
            let file = match std::fs::File::open(path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let buffer = match unsafe { memmap2::MmapOptions::new().map(&file) } {
                Ok(b) => b,
                Err(_) => continue,
            };
            let st = match SafeTensors::deserialize(&buffer) {
                Ok(st) => st,
                Err(_) => continue,
            };

            for name in st.names() {
                if name.starts_with(&layer_prefix) {
                    let tensor_view = st.tensor(name).map_err(|e| FlexLoadError::IoError(format!("Safetensors error: {:?}", e)))?;
                    let data = tensor_view.data();
                    let dtype = match tensor_view.dtype() {
                        safetensors::Dtype::F16 => candle_core::DType::F16,
                        safetensors::Dtype::F32 => candle_core::DType::F32,
                        safetensors::Dtype::BF16 => candle_core::DType::BF16,
                        _ => return Err(FlexLoadError::IoError(format!("Unsupported dtype for tensor: {}", name))),
                    };
                    let shape = tensor_view.shape();
                    let candle_tensor = candle_core::Tensor::from_raw_buffer(data, dtype, shape, &candle_core::Device::Cpu)
                        .map_err(|e| FlexLoadError::IoError(e.to_string()))?;

                    let relative_name = &name[layer_prefix.len()..];
                    tensors.insert(relative_name.to_string(), candle_tensor);
                }
            }
        }

        if tensors.is_empty() {
            return Err(FlexLoadError::LayerNotFound(layer_id));
        }

        Ok(CandleLayerWeights {
            layer_id,
            tensors,
        })
    }

    fn layer_size_bytes(&self, layer_id: usize) -> Result<u64, FlexLoadError> {
        let layer_prefix = format!("{}{}.", self.layer_prefix_pattern.prefix, layer_id);
        let mut total_bytes = 0;

        for path in &self.paths {
            let file = match std::fs::File::open(path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let buffer = match unsafe { memmap2::MmapOptions::new().map(&file) } {
                Ok(b) => b,
                Err(_) => continue,
            };
            let st = match SafeTensors::deserialize(&buffer) {
                Ok(st) => st,
                Err(_) => continue,
            };

            for name in st.names() {
                if name.starts_with(&layer_prefix) {
                    let tensor_view = st.tensor(name).map_err(|e| FlexLoadError::IoError(format!("Safetensors error: {:?}", e)))?;
                    total_bytes += tensor_view.data().len() as u64;
                }
            }
        }

        if total_bytes == 0 {
            return Err(FlexLoadError::LayerNotFound(layer_id));
        }

        Ok(total_bytes)
    }
}

#[cfg(test)]
#[cfg(feature = "candle")]
mod tests {
    use super::*;
    use candle_core::{Tensor, Device};
    use tempfile::NamedTempFile;

    #[test]
    fn test_safetensors_layer_store_parsing() -> Result<(), Box<dyn std::error::Error>> {
        let temp_file = NamedTempFile::new()?;
        
        let tensor = Tensor::zeros((10, 10), candle_core::DType::F32, &Device::Cpu)?;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("model.layers.0.weight".to_string(), tensor);
        
        // Save the dummy candle tensor to a safetensors file
        candle_core::safetensors::save(&tensors, temp_file.path())?;

        let store = SafetensorsLayerStore {
            paths: vec![temp_file.path().to_path_buf()],
            layer_prefix_pattern: LayerPrefixPattern { prefix: "model.layers.".to_string() },
        };

        let ids = store.layer_ids();
        assert_eq!(ids, vec![0]);

        let layer = store.load_layer(0)?;
        assert_eq!(layer.layer_id, 0);
        assert!(layer.tensors.contains_key("weight"));
        
        let size = store.layer_size_bytes(0)?;
        assert_eq!(size, 10 * 10 * 4);

        Ok(())
    }
}
