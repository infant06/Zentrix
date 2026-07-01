use super::{VectorRecord, VectorStorage};
use std::path::PathBuf;
use std::sync::Arc;
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteParams, WriteMode};

/// Persistent vector storage backed by Lance.
pub struct LanceVectorStorage {
    path: PathBuf,
    dim: usize,
}

impl LanceVectorStorage {
    pub fn new(path: impl Into<PathBuf>, dim: usize) -> Result<Self, String> {
        Ok(Self {
            path: path.into(),
            dim,
        })
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dim as i32,
                ),
                false,
            ),
            Field::new("metadata", DataType::Utf8, true),
        ]))
    }
    
    fn path_str(&self) -> &str {
        self.path.to_str().unwrap_or(".zenllm/vector_store")
    }
}

impl VectorStorage for LanceVectorStorage {
    async fn add(&mut self, records: &[VectorRecord]) -> Result<(), String> {
        if records.is_empty() {
            return Ok(());
        }

        let mut ids = Vec::with_capacity(records.len());
        let mut vectors = Vec::with_capacity(records.len() * self.dim);
        let mut metadatas = Vec::with_capacity(records.len());

        for rec in records {
            if rec.vector.len() != self.dim {
                return Err(format!("Vector dimension mismatch: expected {}, got {}", self.dim, rec.vector.len()));
            }
            ids.push(rec.id.clone());
            vectors.extend_from_slice(&rec.vector);
            metadatas.push(rec.metadata.clone());
        }

        let id_array = Arc::new(StringArray::from(ids)) as Arc<dyn arrow_array::Array>;
        
        let float_array = Float32Array::from(vectors);
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array = Arc::new(FixedSizeListArray::new(field, self.dim as i32, Arc::new(float_array), None)) as Arc<dyn arrow_array::Array>;
        
        let metadata_array = Arc::new(StringArray::from(
            metadatas.into_iter().map(|m| m.unwrap_or_default()).collect::<Vec<_>>()
        )) as Arc<dyn arrow_array::Array>;

        let schema = self.schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![id_array, vector_array, metadata_array],
        ).map_err(|e| e.to_string())?;

        let mut write_params = WriteParams::default();
        write_params.mode = WriteMode::Append;

        let reader = arrow_array::RecordBatchIterator::new(vec![Ok(batch)], schema);

        Dataset::write(
            reader,
            self.path_str(),
            Some(write_params)
        ).await.map_err(|e| e.to_string())?;

        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32, Option<String>)>, String> {
        let dataset = match Dataset::open(self.path_str()).await {
            Ok(d) => d,
            Err(_) => return Ok(vec![]),
        };
        
        if dataset.count_rows(None).await.unwrap_or(0) == 0 {
            return Ok(vec![]);
        }
        
        let query_array = Float32Array::from(query.to_vec());
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let query_fsl = FixedSizeListArray::new(field, self.dim as i32, Arc::new(query_array), None);
        
        let mut scanner = dataset.scan();
        scanner.nearest("vector", &query_fsl, k).map_err(|e| e.to_string())?;
        scanner.project(&["id", "_distance", "metadata"]).map_err(|e| e.to_string())?;
        
        let stream = scanner.try_into_stream().await.map_err(|e| e.to_string())?;
        let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(|e| e.to_string())?;
        
        let mut results = Vec::new();
        for batch in batches {
            let id_col = batch.column_by_name("id").and_then(|c| c.as_any().downcast_ref::<StringArray>()).ok_or("missing id")?;
            let dist_col = batch.column_by_name("_distance").and_then(|c| c.as_any().downcast_ref::<Float32Array>()).ok_or("missing dist")?;
            let meta_col = batch.column_by_name("metadata").and_then(|c| c.as_any().downcast_ref::<StringArray>());
            for i in 0..batch.num_rows() {
                let id = id_col.value(i).to_string();
                let dist = dist_col.value(i);
                let score = 1.0 / (1.0 + dist);
                let meta = meta_col.map(|c| {
                    if c.is_null(i) || c.value(i).is_empty() {
                        None
                    } else {
                        Some(c.value(i).to_string())
                    }
                }).flatten();
                results.push((id, score, meta));
            }
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }

    async fn remove(&mut self, id: &str) -> Result<(), String> {
        let mut dataset = match Dataset::open(self.path_str()).await {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        let predicate = format!("id = '{}'", id.replace('\'', "''"));
        dataset.delete(&predicate).await.map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn stats(&self) -> Result<String, String> {
        match Dataset::open(self.path_str()).await {
            Ok(d) => {
                let rows = d.count_rows(None).await.unwrap_or(0);
                Ok(format!(
                    "LanceVectorStorage at {:?} — {} vectors (dim={})",
                    self.path, rows, self.dim
                ))
            }
            Err(_) => Ok(format!(
                "LanceVectorStorage at {:?} (empty, not yet created)",
                self.path
            )),
        }
    }

    async fn build_index(&mut self) -> Result<(), String> {
        let mut dataset = match Dataset::open(self.path_str()).await {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        
        if dataset.count_rows(None).await.unwrap_or(0) < 256 {
            // IVF requires minimum number of rows
            return Ok(());
        }

        use lance::index::vector::VectorIndexParams;
        use lance_index::traits::DatasetIndexExt;
        use lance_index::IndexType;
        use lance_linalg::distance::MetricType;
        let params = VectorIndexParams::ivf_pq(32, 8, 2, MetricType::L2, 50);
        
        let _ = dataset.create_index(&["vector"], IndexType::Vector, None, &params, true).await.map_err(|e| e.to_string())?;
        Ok(())
    }
}
