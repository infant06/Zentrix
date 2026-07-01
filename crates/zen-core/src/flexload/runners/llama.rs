use std::sync::Arc;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use zen_memory::flexload::CandleLayerWeights;
use zen_kernels_quant::safetensors::ShardedSafeTensors;

use crate::models::llama::{Config, Llama, Block};
use crate::flexload::FlexLoadableModel;

impl_flexload_runner!(FlexLlamaRunner, Llama, Config, wte, ln_f, lm_head);

impl FlexLoadableModel for FlexLlamaRunner {
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
        
        let sharded_vb = ShardedSafeTensors::wrap(Box::new(backend), dtype, device);
        let comm = self.base_model.mapper.get_comm_for(layer_idx)?;
        
        let rope = Arc::new(crate::layers::Llama3RotaryEmbedding::new_llama3(
            dtype,
            &self.cfg,
            x.device(),
            false, // is_gptx
        )?);
        
        // PagedAttention logic can be passed in if needed, None for eager fallback
        let paged_attn = None;
        
        // Dynamically instantiate the Block!
        let block = Block::load(
            sharded_vb,
            &self.cfg,
            &*self.base_model.mapper,
            layer_idx,
            false, // loading_isq
            rope,
            paged_attn,
            &comm,
        )?;
        
        // Fetch KV cache for this layer
        let cache = &mut crate::pipeline::NormalModel::cache(&self.base_model).normal().0;
        let mask_for_layer = attention_mask.get(x.device());
        
        // Execute block
        let out = block.forward(x, &mask_for_layer, &mut cache[layer_idx], ctx, layer_idx)?;
        
        // Block is dropped, releasing VRAM!
        Ok(out)
    }
}


