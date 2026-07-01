use std::sync::Arc;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use zen_memory::flexload::CandleLayerWeights;
use zen_kernels_quant::safetensors::ShardedSafeTensors;
use crate::models::deepseek3::{DecoderLayer, DeepSeekV3, DeepSeekV3Config};
use crate::flexload::FlexLoadableModel;

impl_flexload_runner!(
    FlexDeepSeekV3Runner,
    DeepSeekV3,
    DeepSeekV3Config,
    embed_tokens,
    norm,
    lm_head
);

impl FlexLoadableModel for FlexDeepSeekV3Runner {
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
        
        let rope_cfg = crate::layers::DeepSeekV2RopeConfig {
            rope_scaling: self.cfg.rope_scaling.clone(),
            max_position_embeddings: self.cfg.max_position_embeddings,
            rope_theta: self.cfg.rope_theta,
            qk_rope_head_dim: self.cfg.qk_rope_head_dim,
        };
        let rope = Arc::new(crate::layers::DeepSeekV2RotaryEmbedding::new(
            &rope_cfg,
            dtype,
            &device,
        )?);

        let paged_attn = None;

        let block = DecoderLayer::new(
            rope,
            &self.cfg,
            sharded_vb,
            &*self.base_model.mapper,
            layer_idx,
            false,
            paged_attn,
            &comm,
            device.clone(),
        )?;

        let cache = &mut crate::pipeline::NormalModel::cache(&self.base_model).normal().0;
        let mask_for_layer = attention_mask.get(x.device());

        let out = block.forward(x, &mask_for_layer, &mut cache[layer_idx], ctx, layer_idx)?;
        Ok(out)
    }
}
