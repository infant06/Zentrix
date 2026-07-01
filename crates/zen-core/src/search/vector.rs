use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchHit {
    pub id: String,
    pub score: f32,
    pub metadata: serde_json::Value,
    /// Optional raw vector (usually omitted to save memory/bandwidth)
    pub vector: Option<Vec<f32>>,
}

/// Core trait representing an abstracted vector search index suitable for RAG, cache, or embeddings.
#[async_trait]
pub trait VectorSearchProvider: Send + Sync {
    /// Search for `k` nearest neighbors.
    /// `include_vector` determines whether the raw vector payload should be returned.
    async fn search(&self, query: &[f32], k: usize, include_vector: bool) -> Result<Vec<VectorSearchHit>>;
    
    /// Add multiple vectors into the index.
    async fn add(&mut self, ids: &[String], embeddings: &[Vec<f32>], metadata: &[serde_json::Value]) -> Result<()>;
    
    /// Retrieve statistics about the underlying index.
    async fn stats(&self) -> Result<String>;
}

/// Core trait for creating embeddings dynamically within the runtime.
#[async_trait]
pub trait EmbeddingIndexProvider: Send + Sync {
    /// Encode a text string into a flat sequence of floats.
    async fn create_embedding(&self, text: &str) -> Result<Vec<f32>>;
}
