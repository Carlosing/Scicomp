use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rayon::prelude::*;
use gpu_project::Matrix;
use rand::Rng;

fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Multiplication");

    let sizes = &[2, 4, 10, 100, 200, 400, 600];

    for &rows in sizes {
        for &cols in sizes {
            let mut rng = rand::thread_rng();
            let matrix1_data: Vec<f64> = (0..rows * cols).map(|_| rng.gen_range(0.0..200.0)).collect();
            let matrix2_data: Vec<f64> = (0..cols * rows).map(|_| rng.gen_range(0.0..200.0)).collect();

            let matrix1 = Matrix::new(rows, cols, matrix1_data);
            let matrix2 = Matrix::new(cols, rows, matrix2_data);

            // Benchmark secuencial
            group.bench_with_input(BenchmarkId::new("Sequential", format!("{}x{}", rows, cols)), &(rows, cols), |b, _| {
                b.iter(|| matrix1.multiply(&matrix2))
            });

            // Benchmark paralelo
            group.bench_with_input(BenchmarkId::new("Parallel", format!("{}x{}", rows, cols)), &(rows, cols), |b, _| {
                b.iter(|| matrix1.multiply_parallel(&matrix2))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_matrix_multiplication);
criterion_main!(benches);
