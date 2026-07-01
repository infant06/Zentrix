use std::collections::HashMap;

use super::error::FlexLoadError;
use super::layer_streamer::DeviceStager;
use super::device_stage::StageDevice;

#[cfg(feature = "candle")]
use super::layer_weights::CandleLayerWeights;
#[cfg(feature = "candle")]
use candle_core::Device;

#[cfg(feature = "candle")]
pub struct CandleDeviceStager;

#[cfg(feature = "candle")]
impl DeviceStager for CandleDeviceStager {
    type HostLayer = CandleLayerWeights;
    type DeviceLayer = CandleLayerWeights;

    fn move_to_device(
        &self,
        layer: Self::HostLayer,
        device: StageDevice,
    ) -> Result<Self::DeviceLayer, FlexLoadError> {
        let target_device = match device {
            StageDevice::CpuRam | StageDevice::PinnedCpu | StageDevice::UnifiedMemory | StageDevice::Disk | StageDevice::Unknown => Device::Cpu,
            #[cfg(feature = "cuda")]
            StageDevice::Gpu { index } => Device::new_cuda(index).map_err(|e| FlexLoadError::IoError(e.to_string()))?,
            #[cfg(not(feature = "cuda"))]
            StageDevice::Gpu { .. } => return Err(FlexLoadError::InvalidStateTransition("CUDA feature not enabled".to_string())),
        };

        let mut moved_tensors = HashMap::new();
        for (name, tensor) in layer.tensors {
            let moved_tensor = tensor.to_device(&target_device).map_err(|e| FlexLoadError::IoError(e.to_string()))?;
            moved_tensors.insert(name, moved_tensor);
        }

        Ok(CandleLayerWeights {
            layer_id: layer.layer_id,
            tensors: moved_tensors,
        })
    }

    fn evict(&self, _layer_id: usize) -> Result<(), FlexLoadError> {
        // In Rust, releasing the layer from the HashMap drops the Tensors.
        // If they were on CUDA, Candle will automatically free the CudaSlice.
        Ok(())
    }
}
