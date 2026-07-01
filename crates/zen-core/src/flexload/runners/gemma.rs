use std::sync::Arc;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use zen_memory::flexload::CandleLayerWeights;
use zen_kernels_quant::safetensors::ShardedSafeTensors;

use crate::models::gemma::{Config, Model as Gemma};
use crate::flexload::FlexLoadableModel;

impl_flexload_runner!(FlexGemmaRunner, Gemma, Config, embed_tokens, norm, lm_head);

impl FlexLoadableModel for FlexGemmaRunner {
    fn load_foundations(&mut self, _vb: VarBuilder) -> Result<()> {
        Ok(())
    }

    fn forward_streamed_layer(
        &self,
        layer_idx: usize,
        x: &Tensor,
        layer_weights: CandleLayerWeights,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
        attention_mask: &crate::device_map::DeviceMappedMask,
    ) -> Result<Tensor> {
        let dtype = x.dtype();
        let device = x.device().clone();
        
        let backend = crate::flexload::DictionaryBackend {
            tensors: layer_weights.tensors,
        };
        
        let sharded_vb = ShardedSafeTensors::wrap(Box::new(backend), dtype, device.clone());
        let comm = self.base_model.mapper.get_comm_for(layer_idx)?;
        
        let head_dim = self.cfg.head_dim;
        let is_gptx = false;
        
        let rope = Arc::new(crate::layers::RotaryEmbedding::new(
            self.cfg.rope_theta as f32,
            head_dim,
            self.cfg.max_position_embeddings,
            &device,
            is_gptx,
            dtype,
        )?);
        
        let block = crate::models::gemma::DecoderLayer::new(
            rope,
            &self.cfg,
            sharded_vb,
            &*self.base_model.mapper,
            layer_idx,
            false,
            None,
            &comm,
        )?;
        
        // Fetch KV cache for this layer
        let cache = &mut self.base_model.cache.normal().0;
        let mask_for_layer = attention_mask.get(x.device());
        
        let out = block.forward(x, &mask_for_layer, &mut cache[layer_idx], ctx, layer_idx)?;
        
        Ok(out)
    }
}
