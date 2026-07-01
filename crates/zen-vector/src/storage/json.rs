use std::path::PathBuf;
use std::collections::HashMap;
use super::{VectorRecord, VectorStorage};

/// Simple persistent vector storage backed by a JSON file.
/// Each add/remove immediately serializes to disk so data survives across
/// separate CLI invocations. Search uses brute-force cosine similarity.
pub struct JsonVectorStorage {
    path: PathBuf,
    records: HashMap<String, Vec<f32>>,
    metadata: HashMap<String, Option<String>>,
}

impl JsonVectorStorage {
    /// Open or create a JSON store at the given path.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, String> {
        let path: PathBuf = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }

        let (records, metadata) = if path.exists() {
            let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
            let root: serde_json::Value = serde_json::from_str(&content).unwrap_or_default();
            let recs = root["vectors"].as_object().map(|obj| {
                obj.iter().filter_map(|(k, v)| {
                    let vec: Vec<f32> = v["embedding"].as_array()?
                        .iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect();
                    Some((k.clone(), vec))
                }).collect::<HashMap<_, _>>()
            }).unwrap_or_default();

            let metas = root["vectors"].as_object().map(|obj| {
                obj.iter().map(|(k, v)| {
                    let meta = v["metadata"].as_str().map(|s| s.to_string());
                    (k.clone(), meta)
                }).collect::<HashMap<_, _>>()
            }).unwrap_or_default();

            (recs, metas)
        } else {
            (HashMap::new(), HashMap::new())
        };

        Ok(Self { path, records, metadata })
    }

    fn save(&self) -> Result<(), String> {
        let mut vectors_obj = serde_json::Map::new();
        for (id, emb) in &self.records {
            let meta_val = match self.metadata.get(id).and_then(|m| m.as_deref()) {
                Some(m) => serde_json::Value::String(m.to_string()),
                None => serde_json::Value::Null,
            };
            vectors_obj.insert(id.clone(), serde_json::json!({
                "embedding": emb,
                "metadata": meta_val,
            }));
        }
        let root = serde_json::json!({ "vectors": vectors_obj });
        let content = serde_json::to_string_pretty(&root).map_err(|e| e.to_string())?;
        std::fs::write(&self.path, content).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn cosine_score(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
    }
}

impl VectorStorage for JsonVectorStorage {
    async fn add(&mut self, records: &[VectorRecord]) -> Result<(), String> {
        for rec in records {
            self.records.insert(rec.id.clone(), rec.vector.clone());
            self.metadata.insert(rec.id.clone(), rec.metadata.clone());
        }
        self.save()
    }

    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32, Option<String>)>, String> {
        let mut scored: Vec<(String, f32, Option<String>)> = self.records.iter()
            .map(|(id, emb)| {
                let meta = self.metadata.get(id).and_then(|m| m.clone());
                (id.clone(), Self::cosine_score(query, emb), meta)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    async fn remove(&mut self, id: &str) -> Result<(), String> {
        self.records.remove(id);
        self.metadata.remove(id);
        self.save()
    }

    async fn stats(&self) -> Result<String, String> {
        let n = self.records.len();
        let dim = self.records.values().next().map(|v| v.len()).unwrap_or(0);
        Ok(format!(
            "JsonVectorStorage: {} vectors (dim={}) — stored at {:?}",
            n, dim, self.path
        ))
    }
}
