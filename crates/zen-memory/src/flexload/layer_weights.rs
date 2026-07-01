use std::collections::HashMap;

#[cfg(feature = "candle")]
use candle_core::Tensor;

#[cfg(feature = "candle")]
#[derive(Clone)]
pub struct CandleLayerWeights {
    pub layer_id: usize,
    pub tensors: HashMap<String, Tensor>,
}
