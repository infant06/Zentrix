use std::sync::Arc;
use candle_core::Result;
use candle_nn::VarBuilder;
use zen_memory::flexload::CandleLayerWeights;
use zen_kernels_quant::safetensors::ShardedSafeTensors;
use crate::models::qwen2::{DecoderLayer, Model, Config};
use crate::flexload::FlexLoadableModel;

impl_flexload_runner!(
    FlexQwen2Runner,
    Model,
    Config,
    embed_tokens,
    norm,
    lm_head
);

impl FlexLoadableModel for FlexQwen2Runner {
    fn load_foundations(&mut self, _vb: VarBuilder) -> Result<()> {
        Ok(())
    }

    fn forward_streamed_layer(
        &self,
        layer_idx: usize,
        x: &candle_core::Tensor,
        layer_weights: CandleLayerWeights,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
        attention_mask: &crate::device_map::DeviceMappedMask,
    ) -> candle_core::Result<candle_core::Tensor> {
        let dtype = x.dtype();
        let device = x.device().clone();

        let backend = crate::flexload::DictionaryBackend {
            tensors: layer_weights.tensors,
        };
        
        let sharded_vb = ShardedSafeTensors::wrap(Box::new(backend), dtype, device.clone());
        
        let comm = self.base_model.mapper.get_comm_for(layer_idx)?;
        let rope = Arc::new(crate::layers::RotaryEmbedding::new(
            self.cfg.rope_theta as f32,
            self.cfg.hidden_size / self.cfg.num_attention_heads,
            self.cfg.max_position_embeddings,
            &device,
            false,
            dtype,
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
        )?;

        let cache = &mut crate::pipeline::NormalModel::cache(&self.base_model).normal().0;
        let mask_for_layer = attention_mask.get(x.device());

        let out = block.forward(x, &mask_for_layer, &mut cache[layer_idx], ctx, layer_idx)?;
        Ok(out)
    }
}
