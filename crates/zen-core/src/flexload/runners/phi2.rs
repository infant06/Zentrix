use std::sync::Arc;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use zen_memory::flexload::CandleLayerWeights;
use zen_kernels_quant::safetensors::ShardedSafeTensors;
use crate::models::phi2::{DecoderLayer, Model, Config};

use crate::flexload::FlexLoadableModel;

impl_flexload_runner!(
    FlexPhi2Runner,
    Model,
    Config,
    embed_tokens,
    final_layernorm,
    lm_head
);

impl FlexLoadableModel for FlexPhi2Runner {
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
        
        let rope = Arc::new(crate::layers::RotaryEmbedding::new_partial(
            self.cfg.rope_theta as f32,
            (self.cfg.partial_rotary_factor * self.cfg.head_dim() as f64) as usize,
            self.cfg.max_position_embeddings,
            &device,
            false,
            dtype,
        )?);

        let paged_attn = None;

        let block = DecoderLayer::new(
            &self.cfg,
            sharded_vb,
            &*self.base_model.mapper,
            layer_idx,
            false,
            rope,
            paged_attn,
            &comm,
        )?;

        let cache = &mut crate::pipeline::NormalModel::cache(&self.base_model).normal().0;
        let mask_for_layer = attention_mask.get(x.device());

        let out = block.forward(x, &mask_for_layer, &mut cache[layer_idx], ctx, layer_idx)?;
        Ok(out)
    }
}
