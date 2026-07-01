use anyhow::{anyhow, Result};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::Path;

pub struct OnnxEngine {
    session: Session,
}

impl OnnxEngine {
    pub fn init_env() -> Result<()> {
        ort::init()
            .with_name("zen-onnx")
            .commit();
        Ok(())
    }

    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| anyhow!(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!(e.to_string()))?;

        Ok(Self { session })
    }

    pub fn run_forward(&mut self, input_name: &str, input_data: ndarray::ArrayD<f32>) -> Result<ndarray::ArrayD<f32>> {
        let tensor = Tensor::from_array(input_data).map_err(|e| anyhow!(e.to_string()))?;
        let inputs = ort::inputs![input_name => tensor];
        let outputs = self.session.run(inputs)
            .map_err(|e| anyhow!(e.to_string()))?;
        
        let output_name = outputs.keys().next().ok_or_else(|| anyhow!("No outputs"))?;
        let (shape, data) = outputs[output_name].try_extract_tensor::<f32>()
            .map_err(|e| anyhow!(e.to_string()))?;
            
        let ndarray_shape = shape.iter().map(|&x| x as usize).collect::<Vec<_>>();
        let arr = ndarray::ArrayD::from_shape_vec(ndarray_shape, data.to_vec())
            .map_err(|e| anyhow!(e.to_string()))?;
            
        Ok(arr)
    }
}
