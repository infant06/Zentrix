use crate::metrics::SimilarityMetric;
use crate::ZenVectorEngine;
use crate::ConstructError;
use std::sync::RwLock;

pub struct ZenVectorIndex {
    engine: RwLock<ZenVectorEngine>,
    metric: SimilarityMetric,
    dim: usize,
}

impl ZenVectorIndex {
    pub fn new(dim: usize, bit_width: usize, metric: SimilarityMetric) -> Result<Self, ConstructError> {
        let engine = ZenVectorEngine::new(dim, bit_width)?;
        Ok(Self {
            engine: RwLock::new(engine),
            metric,
            dim,
        })
    }

    pub fn add_vector(&self, vector: &[f32]) {
        self.add_batch(vector);
    }

    pub fn add_batch(&self, vectors: &[f32]) {
        let mut engine = self.engine.write().unwrap();
        // Depending on metric, we could normalize (for Cosine) or transform here.
        // For now, we pass to the underlying MIPS engine directly.
        match self.metric {
            SimilarityMetric::Cosine => {
                // Normalize vectors for cosine
                let mut norm_vecs = vectors.to_vec();
                for chunk in norm_vecs.chunks_mut(self.dim) {
                    let norm: f32 = chunk.iter().map(|v| v * v).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for v in chunk.iter_mut() {
                            *v /= norm;
                        }
                    }
                }
                engine.add(&norm_vecs);
            }
            _ => {
                engine.add(vectors);
            }
        }
    }

    pub fn search_top_k(&self, queries: &[f32], k: usize) -> crate::SearchResults {
        let engine = self.engine.read().unwrap();
        
        match self.metric {
            SimilarityMetric::Cosine => {
                let mut norm_queries = queries.to_vec();
                for chunk in norm_queries.chunks_mut(self.dim) {
                    let norm: f32 = chunk.iter().map(|v| v * v).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for v in chunk.iter_mut() {
                            *v /= norm;
                        }
                    }
                }
                engine.search(&norm_queries, k)
            }
            _ => {
                engine.search(queries, k)
            }
        }
    }

    pub fn remove(&self, id: usize) -> usize {
        let mut engine = self.engine.write().unwrap();
        engine.swap_remove(id)
    }

    pub fn clear(&self) {
        let mut engine = self.engine.write().unwrap();
        let bit_width = engine.bit_width();
        let dim = engine.dim();
        *engine = ZenVectorEngine::new(dim, bit_width).unwrap();
    }
    
    pub fn cosine() {} // placeholder
    pub fn dot() {} // placeholder
    pub fn l2() {} // placeholder
    pub fn quantized8() {} // placeholder
}
