use std::future::Future;

pub mod json;
pub mod memory;
#[cfg(feature = "lance-storage")]
pub mod lance;

pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<String>,
}

pub trait VectorStorage: Send + Sync {
    fn add(&mut self, records: &[VectorRecord]) -> impl Future<Output = Result<(), String>> + Send;
    fn search(&self, query: &[f32], k: usize) -> impl Future<Output = Result<Vec<(String, f32, Option<String>)>, String>> + Send;
    fn remove(&mut self, id: &str) -> impl Future<Output = Result<(), String>> + Send;
    fn stats(&self) -> impl Future<Output = Result<String, String>> + Send;
    fn build_index(&mut self) -> impl Future<Output = Result<(), String>> + Send {
        async { Ok(()) }
    }
}
