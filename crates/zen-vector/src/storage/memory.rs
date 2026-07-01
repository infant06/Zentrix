use std::collections::HashMap;
use super::{VectorRecord, VectorStorage};
use crate::index::ZenVectorIndex;
use crate::metrics::SimilarityMetric;

pub struct MemoryVectorStorage {
    index: ZenVectorIndex,
    id_to_index: HashMap<String, usize>,
    index_to_id: HashMap<usize, String>,
    next_index: usize,
    dim: usize,
    padded_dim: usize,
}

impl MemoryVectorStorage {
    pub fn new(dim: usize, metric: SimilarityMetric) -> Result<Self, String> {
        let padded_dim = (dim + 7) & !7;
        let index = ZenVectorIndex::new(padded_dim, 4, metric).map_err(|e| format!("{:?}", e))?;
        Ok(Self {
            index,
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            next_index: 0,
            dim,
            padded_dim,
        })
    }
}

impl VectorStorage for MemoryVectorStorage {
    async fn add(&mut self, records: &[VectorRecord]) -> Result<(), String> {
        for record in records {
            if record.vector.len() != self.dim {
                return Err(format!("Vector dimension mismatch: expected {}, got {}", self.dim, record.vector.len()));
            }
            if self.id_to_index.contains_key(&record.id) {
                continue; // simple skip for now, could implement upsert
            }
            
            let mut padded_vec = vec![0.0; self.padded_dim];
            padded_vec[..self.dim].copy_from_slice(&record.vector);
            self.index.add_vector(&padded_vec);
            
            let current_idx = self.next_index;
            self.id_to_index.insert(record.id.clone(), current_idx);
            self.index_to_id.insert(current_idx, record.id.clone());
            self.next_index += 1;
        }
        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32, Option<String>)>, String> {
        if query.len() != self.dim {
            return Err(format!("Query dimension mismatch: expected {}, got {}", self.dim, query.len()));
        }
        
        let mut padded_query = vec![0.0; self.padded_dim];
        padded_query[..self.dim].copy_from_slice(query);
        let results = self.index.search_top_k(&padded_query, k);
        let mut final_results = Vec::new();
        
        for (idx, score) in results.indices.iter().zip(results.scores.iter()) {
            if let Some(id) = self.index_to_id.get(&(*idx as usize)) {
                final_results.push((id.clone(), *score, None));
            }
        }
        
        Ok(final_results)
    }

    async fn remove(&mut self, id: &str) -> Result<(), String> {
        if let Some(idx) = self.id_to_index.remove(id) {
            let swapped_idx = self.index.remove(idx);
            self.index_to_id.remove(&idx);
            
            if swapped_idx != idx {
                // The last element was swapped into the hole.
                // We need to update the mapping for the swapped element.
                if let Some(swapped_id) = self.index_to_id.remove(&self.next_index.saturating_sub(1)) {
                    self.id_to_index.insert(swapped_id.clone(), idx);
                    self.index_to_id.insert(idx, swapped_id);
                }
            }
            self.next_index = self.next_index.saturating_sub(1);
            Ok(())
        } else {
            Err("ID not found".to_string())
        }
    }

    async fn stats(&self) -> Result<String, String> {
        Ok(format!("MemoryVectorStorage: {} vectors, dim: {}", self.next_index, self.dim))
    }
}
