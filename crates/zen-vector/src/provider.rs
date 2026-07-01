use anyhow::Result;
use async_trait::async_trait;
use zen_core::search::vector::{VectorSearchHit, VectorSearchProvider};
use crate::storage::VectorStorage;

pub struct ZenVectorSearchProvider<S: VectorStorage + Send + Sync> {
    storage: S,
}

impl<S: VectorStorage + Send + Sync> ZenVectorSearchProvider<S> {
    pub fn new(storage: S) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl<S: VectorStorage + Send + Sync> VectorSearchProvider for ZenVectorSearchProvider<S> {
    async fn search(&self, query: &[f32], k: usize, _include_vector: bool) -> Result<Vec<VectorSearchHit>> {
        let hits = self.storage.search(query, k).await.map_err(|e| anyhow::anyhow!(e))?;
        Ok(hits.into_iter().map(|(id, score, meta)| VectorSearchHit {
            id,
            score,
            metadata: meta.as_deref()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or(serde_json::Value::Null),
            vector: None,
        }).collect())
    }

    async fn add(&mut self, ids: &[String], embeddings: &[Vec<f32>], metadata: &[serde_json::Value]) -> Result<()> {
        let mut records = Vec::with_capacity(embeddings.len());
        for (i, emb) in embeddings.iter().enumerate() {
            records.push(crate::storage::VectorRecord {
                id: ids[i].clone(),
                vector: emb.clone(),
                metadata: Some(metadata[i].to_string()),
            });
        }
        self.storage.add(&records).await.map_err(|e| anyhow::anyhow!(e))?;
        Ok(())
    }

    async fn stats(&self) -> Result<String> {
        self.storage.stats().await.map_err(|e| anyhow::anyhow!(e))
    }
}
