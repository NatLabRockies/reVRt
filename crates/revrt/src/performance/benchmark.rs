//! Benchmarking reVRt
//!
//! Monitor reVRt's performance to guide development and avoid regressions.
//!
//! This benchmarking suite is meant to be run in CI/CD pipelines, so
//! it should not be too long.
//!
//! Cases to consider:
//! - All ones: So we guarantee always the same solution
//! - Small distance but large cost array: It should be impacted by
//!   the cost chunk size only.
//! - Random cost: Would that create too much noise for statistics?
//! - Too many layers: How well we parallelize between layers.
//! - Too many paths in the same area: How well we parallelize
//!   between paths (re-using cost cache).
//! - Single chunk with reasonable size: How well we parallelize
//!   calculating the cost.

use core::time::Duration;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use revrt::ArrayIndex;
use revrt::bench_minimalist;

use ndarray::Array3;
use rand::RngExt;

enum FeaturesType {
    AllOnes,
    Random,
}

/// Create temporary features to support benchmarking
fn features(ni: u64, nj: u64, ci: u64, cj: u64, ftype: FeaturesType) -> std::path::PathBuf {
    let tmp_path = tempfile::TempDir::new().unwrap();

    let store: zarrs::storage::ReadableWritableListableStorage = std::sync::Arc::new(
        zarrs::filesystem::FilesystemStore::new(tmp_path.path())
            .expect("could not open filesystem store"),
    );

    zarrs::group::GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    // Create an array
    // Remember to remove /cost
    for array_path in ["/A", "/B", "/C", "/cost"].iter() {
        let array = zarrs::array::ArrayBuilder::new(
            vec![1, ni, nj], // array shape
            vec![1, ci, cj], // regular chunk shape
            zarrs::array::DataType::Float32,
            zarrs::array::FillValue::from(zarrs::array::ZARR_NAN_F32),
        )
        // .bytes_to_bytes_codecs(vec![]) // uncompressed
        .dimension_names(["band", "y", "x"].into())
        // .storage_transformers(vec![].into())
        .build(store.clone(), array_path)
        .unwrap();

        // Write array metadata to store
        array.store_metadata().unwrap();

        let mut a = vec![];
        match ftype {
            FeaturesType::AllOnes => {
                a.resize((ni * nj).try_into().unwrap(), 1.0);
            }
            FeaturesType::Random => {
                let mut rng = rand::rng();
                for _x in 0..(ni * nj) {
                    a.push(rng.random_range(0.0..=1.0));
                }
            }
        }
        let data: Array3<f32> =
            ndarray::Array::from_shape_vec((1, ni.try_into().unwrap(), nj.try_into().unwrap()), a)
                .unwrap();

        array
            .store_chunks_ndarray(
                &zarrs::array_subset::ArraySubset::new_with_ranges(&[
                    0..1,
                    0..(ni / ci),
                    0..(nj / cj),
                ]),
                data,
            )
            .unwrap();
    }

    tmp_path.keep()
}

/// Standard benchmark with input features all equal to one
fn standard_ones(c: &mut Criterion) {
    let features_path = features(100, 100, 4, 4, FeaturesType::AllOnes);

    c.bench_function("constant_cost", |b| {
        b.iter(|| {
            bench_minimalist(
                black_box(features_path.clone()),
                black_box(vec![ArrayIndex::new_ij(20, 50)]),
                black_box(vec![ArrayIndex::new_ij(5, 50)]),
            )
        })
    });
}

/// Standard benchmark with random input features
fn standard_random(c: &mut Criterion) {
    let features_path = features(100, 100, 4, 4, FeaturesType::Random);

    c.bench_function("random_cost", |b| {
        b.iter(|| {
            bench_minimalist(
                black_box(features_path.clone()),
                black_box(vec![ArrayIndex::new_ij(20, 50)]),
                black_box(vec![ArrayIndex::new_ij(5, 50)]),
            )
        })
    });
}

/// Multiple paths in the same area, to test efficiency of reusing cost.
fn multiple_near_routes(c: &mut Criterion) {
    let features_path = features(100, 100, 4, 4, FeaturesType::AllOnes);

    c.bench_function("multiple_near_routes", |b| {
        b.iter(|| {
            bench_minimalist(
                black_box(features_path.clone()),
                black_box(
                    (19..=22)
                        .flat_map(|row| (48..=51).map(move |col| ArrayIndex::new_ij(row, col)))
                        .collect::<Vec<_>>(),
                ),
                black_box(vec![ArrayIndex::new_ij(10, 50)]),
            )
        })
    });
}

/// Multiple spread routes, to test efficiency of accessing multiple chunks.
fn multiple_spread_routes(c: &mut Criterion) {
    let features_path = features(100, 100, 5, 5, FeaturesType::AllOnes);

    c.bench_function("multiple_spread_routes", |b| {
        b.iter(|| {
            bench_minimalist(
                black_box(features_path.clone()),
                black_box(
                    (40..=60)
                        .step_by(5)
                        .flat_map(|row| {
                            (40..=60)
                                .step_by(5)
                                .map(move |col| ArrayIndex::new_ij(row, col))
                        })
                        .collect::<Vec<_>>(),
                ),
                black_box(vec![ArrayIndex::new_ij(50, 50)]),
            )
        })
    });
}

/// Benchmark with features all equal to one in a single chunk
fn single_chunk(c: &mut Criterion) {
    let features_path = features(100, 100, 1, 1, FeaturesType::AllOnes);

    c.bench_function("single_chunk", |b| {
        b.iter(|| {
            bench_minimalist(
                black_box(features_path.clone()),
                black_box(vec![ArrayIndex::new_ij(20, 50)]),
                black_box(vec![ArrayIndex::new_ij(5, 50)]),
            )
        })
    });
}

/// Benchmark multiple distances with features all equal to one
fn range_distance(c: &mut Criterion) {
    // Away from the border to progressively increas the search radius.
    static X0: u64 = 30;
    let features_path = features(100, 100, 1, 1, FeaturesType::AllOnes);

    let mut group = c.benchmark_group("distance");
    // Create an alternative benchmark definition to run locally only
    for distance in [0, 1, 2, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(distance),
            distance,
            |b, &distance| {
                b.iter(|| {
                    bench_minimalist(
                        black_box(features_path.clone()),
                        black_box(vec![ArrayIndex::new_ij(X0 + distance, 50)]),
                        black_box(vec![ArrayIndex::new_ij(X0, 50)]),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(25));
    targets = standard_ones, standard_random, multiple_near_routes, multiple_spread_routes, single_chunk, range_distance
);
criterion_main!(benches);
