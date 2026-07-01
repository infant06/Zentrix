#[macro_export]
macro_rules! impl_flexload_runner {
    (
        $runner_name:ident,
        $base_model_name:ident,
        $config_type:ty,
        $embed_field:ident,
        $norm_field:ident,
        $lm_head_field:ident
    ) => {
        pub struct $runner_name {
            pub base_model: $base_model_name,
            pub cfg: $config_type,
            pub device: candle_core::Device,
            pub layer_streamer: std::sync::Arc<std::sync::Mutex<zen_memory::flexload::LayerStreamer<zen_memory::flexload::SafetensorsLayerStore, zen_memory::flexload::CandleDeviceStager>>>,
        }

        impl $runner_name {
            pub fn new(
                base_model: $base_model_name,
                cfg: $config_type,
                paths: &Box<dyn crate::pipeline::ModelPaths>,
                num_hidden_layers: usize,
            ) -> Self {
                let store = zen_memory::flexload::SafetensorsLayerStore {
                    paths: paths.get_weight_filenames().to_vec(),
                    layer_prefix_pattern: zen_memory::flexload::LayerPrefixPattern { prefix: "model.layers.".to_string() },
                };
                let device_stager = zen_memory::flexload::CandleDeviceStager;
                
                let target_stage_device = match crate::pipeline::NormalModel::device(&base_model) {
                    candle_core::Device::Cpu => zen_memory::flexload::StageDevice::CpuRam,
                    #[cfg(feature = "cuda")]
                    d @ candle_core::Device::Cuda(_) => zen_memory::flexload::StageDevice::Gpu { 
                        index: match d.location() {
                            candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id,
                            _ => 0,
                        }
                    },
                    _ => zen_memory::flexload::StageDevice::CpuRam,
                };

                let plan = zen_memory::flexload::FlexLoadPlan {
                    target_device: target_stage_device.clone(),
                    ordered_layer_ids: (0..num_hidden_layers).collect(),
                    layer_sizes: vec![1024 * 1024 * 1024; num_hidden_layers],
                    prefetch_schedule: (0..num_hidden_layers).collect(),
                };
                let config = zen_memory::flexload::FlexLoadConfig {
                    prefetch_ahead_count: 1,
                    max_ram_bytes: 32 * 1024 * 1024 * 1024,
                    max_vram_bytes: 8 * 1024 * 1024 * 1024,
                    prefer_pinned_memory: false,
                    target_device: target_stage_device,
                    eviction_policy: zen_memory::flexload::layer_cache::CachePolicy::StaticPinning,
                };
                let streamer = zen_memory::flexload::LayerStreamer::new(
                    std::sync::Arc::new(store),
                    std::sync::Arc::new(device_stager),
                    config,
                    plan,
                );

                Self {
                    device: crate::pipeline::NormalModel::device(&base_model).clone(),
                    base_model,
                    cfg,
                    layer_streamer: std::sync::Arc::new(std::sync::Mutex::new(streamer)),
                }
            }
        }

        impl crate::pipeline::IsqModel for $runner_name {
            fn residual_tensors(&self) -> Vec<(String, candle_core::Tensor)> {
                <$base_model_name as crate::pipeline::IsqModel>::residual_tensors(&self.base_model)
            }
        }

        impl crate::amoe::AnyMoeBaseModelMixin for $runner_name {}
        impl crate::speculative::SpeculativeTargetMixin for $runner_name {}

        impl crate::pipeline::NormalModel for $runner_name {
            fn forward(
                &self,
                input_ids: &candle_core::Tensor,
                ctx: &mut crate::pipeline::ModelForwardContext<'_>,
            ) -> candle_core::Result<candle_core::Tensor> {
                let mut x = candle_core::Module::forward(&self.base_model.$embed_field, input_ids)?;
                
                // Pre-calculate attention mask once before layer loop, like base model does
                let mut cache_guard = crate::pipeline::NormalModel::cache(&self.base_model).normal();
                let cache = &mut cache_guard.0;
                let mask_cache = ctx.mask_cache(cache);
                let attention_mask = crate::layers::CausalMasker.make_causal_mask(
                    input_ids,
                    &mask_cache,
                    x.dtype(),
                    &crate::layers_masker::CausalMaskConfig {
                        sliding_window: self.get_sliding_window(),
                        ..Default::default()
                    },
                ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let attention_mask = if ctx.is_first_prompt_chunk() { attention_mask } else { crate::attention::AttentionMask::None };
                let attention_mask = crate::device_map::DeviceMappedMask::new(attention_mask, &*self.base_model.mapper)?;
                drop(cache_guard);

                // Get the number of hidden layers generically 
                let num_hidden_layers = crate::pipeline::NormalModel::config(&self.base_model).num_layers;

                for layer_idx in 0..num_hidden_layers {
                    let mut streamer = self.layer_streamer.lock().unwrap();
                    streamer.prepare_layer(layer_idx).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    let layer_weights = streamer.get_layer(layer_idx).ok_or_else(|| {
                        candle_core::Error::Msg(format!("Layer {} not found after prepare", layer_idx))
                    })?.clone();
                    drop(streamer);

                    x = self.forward_streamed_layer(layer_idx, &x, layer_weights, ctx, &attention_mask)?;

                    let mut streamer = self.layer_streamer.lock().unwrap();
                    streamer.release_layer(layer_idx).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                }
                
                x = x.to_device(&self.device)?;
                x = candle_core::Module::forward(&self.base_model.$norm_field, &x)?;
                x = ctx.logits(&x)?;
                self.base_model.$lm_head_field.forward(&x)
            }

            fn xlora_forward(
                &self,
                input_ids: &candle_core::Tensor,
                input_ids_full: &candle_core::Tensor,
                seqlen_offsets: &[usize],
                seqlen_offsets_full: &[usize],
                no_kv_cache: bool,
                non_granular_state: &Option<crate::xlora_models::NonGranularState>,
                context_lens: Vec<(usize, usize)>,
                position_ids: Vec<usize>,
                flash_params: &crate::pipeline::text_models_inputs_processor::FlashParams,
                flash_params_full: &crate::pipeline::text_models_inputs_processor::FlashParams,
            ) -> candle_core::Result<candle_core::Tensor> {
                self.base_model.xlora_forward(
                    input_ids, input_ids_full, seqlen_offsets, seqlen_offsets_full,
                    no_kv_cache, non_granular_state, context_lens, position_ids,
                    flash_params, flash_params_full
                )
            }

            fn is_xlora(&self) -> bool {
                self.base_model.is_xlora()
            }

            fn device(&self) -> &candle_core::Device {
                &self.device
            }

            fn cache(&self) -> &crate::pipeline::EitherCache {
                crate::pipeline::NormalModel::cache(&self.base_model)
            }

            fn max_seq_len(&self) -> usize {
                self.base_model.max_seq_len()
            }

            fn config(&self) -> &crate::paged_attention::ModelConfigMetadata {
                crate::pipeline::NormalModel::config(&self.base_model)
            }
        }
    };
}
