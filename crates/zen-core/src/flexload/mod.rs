use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use zen_memory::flexload::CandleLayerWeights;

pub mod runners;

/// A generic interface that an architecture must implement to support FlexLoad streaming.
pub trait FlexLoadableModel {
    /// Loads foundational modules like embeddings, RMS norms, and lm_head 
    /// that sit in VRAM permanently.
    fn load_foundations(&mut self, vb: VarBuilder) -> Result<()>;

    /// Executes a single transformer layer dynamically loaded from RAM/Disk.
    /// The tensors arrive inside `CandleLayerWeights` and are immediately wrapped
    /// into a local module block, computed, and dropped.
    fn forward_streamed_layer(
        &self,
        layer_idx: usize,
        x: &Tensor,
        layer_weights: CandleLayerWeights,
        // Optional context for KV caches/Flash Attention etc.
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
        attention_mask: &crate::device_map::DeviceMappedMask,
    ) -> Result<Tensor>;

    /// Returns the sliding window size if the model uses sliding window attention.
    fn get_sliding_window(&self) -> Option<usize> {
        None
    }
}

pub struct DictionaryBackend {
    pub tensors: std::collections::HashMap<String, Tensor>,
}

impl candle_nn::var_builder::SimpleBackend for DictionaryBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: candle_core::DType,
        dev: &candle_core::Device,
    ) -> Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, dev)?;
        if tensor.shape() != &s {
            candle_core::bail!("shape mismatch for {name}: expected {:?}, got {:?}", s, tensor.shape())
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: candle_core::DType, dev: &candle_core::Device) -> Result<Tensor> {
        let tensor = self.tensors.get(name).ok_or_else(|| {
            let keys: Vec<&String> = self.tensors.keys().collect();
            println!("DictionaryBackend failed to find: {name}. Available keys: {:?}", keys);
            candle_core::Error::CannotFindTensor {
                path: name.to_string(),
            }
        })?;
        tensor.to_device(dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}
