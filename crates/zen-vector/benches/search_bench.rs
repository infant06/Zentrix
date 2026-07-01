use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use zen_vector::TurboQuantIndex;

fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn naive_dot(queries: &[f32], vectors: &[f32], dim: usize, nq: usize, n_vectors: usize, k: usize) -> Vec<f32> {
    let mut all_top_scores = Vec::with_capacity(nq * k);

    for q_idx in 0..nq {
        let q_base = q_idx * dim;
        let query = &queries[q_base..q_base + dim];
        
        let mut scores = Vec::with_capacity(n_vectors);
        for v_idx in 0..n_vectors {
            let v_base = v_idx * dim;
            let vec = &vectors[v_base..v_base + dim];
            let dot: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            scores.push(dot);
        }

        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        all_top_scores.extend(scores.into_iter().take(k));
    }
    
    all_top_scores
}

fn naive_cosine(queries: &[f32], vectors: &[f32], dim: usize, nq: usize, n_vectors: usize, k: usize) -> Vec<f32> {
    let mut all_top_scores = Vec::with_capacity(nq * k);

    for q_idx in 0..nq {
        let q_base = q_idx * dim;
        let query = &queries[q_base..q_base + dim];
        
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut scores = Vec::with_capacity(n_vectors);
        for v_idx in 0..n_vectors {
            let v_base = v_idx * dim;
            let vec = &vectors[v_base..v_base + dim];
            let dot: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            let v_norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cosine = if q_norm * v_norm > 0.0 { dot / (q_norm * v_norm) } else { 0.0 };
            scores.push(cosine);
        }

        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        all_top_scores.extend(scores.into_iter().take(k));
    }
    
    all_top_scores
}

fn std_chunked_baseline(queries: &[f32], vectors: &[f32], dim: usize, nq: usize, n_vectors: usize, k: usize) -> Vec<f32> {
    let mut all_top_scores = Vec::with_capacity(nq * k);

    for q_idx in 0..nq {
        let q_base = q_idx * dim;
        let query = &queries[q_base..q_base + dim];
        
        let mut scores = Vec::with_capacity(n_vectors);
        for vec in vectors.chunks_exact(dim) {
            let dot: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            scores.push(dot);
        }

        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        all_top_scores.extend(scores.into_iter().take(k));
    }
    
    all_top_scores
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    
    let dim = 1536;
    let n_vectors = 10_000;
    let n_queries = 10;
    let k = 10;

    let vectors = generate_random_vectors(n_vectors, dim, 42);
    let queries = generate_random_vectors(n_queries, dim, 43);

    // Prepare TurboQuantIndex
    let mut index = TurboQuantIndex::new(dim, 4).unwrap();
    index.add(&vectors);
    index.prepare(); // pay one-time initialization cost before benchmark

    group.bench_function(BenchmarkId::new("TurboQuantIndex", format!("{dim}d")), |b| {
        b.iter(|| index.search(&queries, k));
    });

    group.bench_function(BenchmarkId::new("Naive Dot", format!("{dim}d")), |b| {
        b.iter(|| naive_dot(&queries, &vectors, dim, n_queries, n_vectors, k));
    });

    group.bench_function(BenchmarkId::new("Naive Cosine", format!("{dim}d")), |b| {
        b.iter(|| naive_cosine(&queries, &vectors, dim, n_queries, n_vectors, k));
    });

    group.bench_function(BenchmarkId::new("Std Chunked Baseline", format!("{dim}d")), |b| {
        b.iter(|| std_chunked_baseline(&queries, &vectors, dim, n_queries, n_vectors, k));
    });

    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
