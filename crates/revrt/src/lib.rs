//! # Path optimization with weighted costs
//!
//!

mod benchmark;
mod cost;
mod dataset;
mod error;
#[allow(non_camel_case_types)]
mod ffi;
mod network;
mod routing;
mod solution;

use std::sync::mpsc;

pub use benchmark::bench_minimalist;
use cost::CostFunction;
use error::Result;
use routing::{ParRouting, RouteDefinition, Routing};
use solution::{RevrtRoutingSolutions, Solution};

#[allow(missing_docs)]
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArrayIndex {
    i: u64,
    j: u64,
}

impl ArrayIndex {
    #[allow(missing_docs)]
    pub fn new(i: u64, j: u64) -> Self {
        Self { i, j }
    }
}

impl From<ArrayIndex> for (u64, u64) {
    fn from(ArrayIndex { i, j }: ArrayIndex) -> (u64, u64) {
        (i, j)
    }
}

#[allow(missing_docs)]
pub fn resolve<P: AsRef<std::path::Path>>(
    store_path: P,
    cost_function: &str,
    mem_limit_bytes: u64,
    algorithm: &str,
    start: &[ArrayIndex],
    end: Vec<ArrayIndex>,
) -> Result<RevrtRoutingSolutions> {
    let cost_function = CostFunction::from_json(cost_function)?;
    tracing::trace!("Cost function: {:?}", cost_function);
    let mut simulation: Routing =
        Routing::new(store_path, cost_function, mem_limit_bytes, algorithm)?;
    let result = simulation.compute(start, end).collect();
    Ok(result)
}

#[allow(missing_docs)]
pub(crate) fn resolve_generator<P, I>(
    store_path: P,
    cost_function: &str,
    route_definitions: I,
    tx: mpsc::Sender<(u32, RevrtRoutingSolutions)>,
    mem_limit_bytes: u64,
    algorithm: &str,
) -> Result<()>
where
    P: AsRef<std::path::Path>,
    I: rayon::prelude::IntoParallelIterator<Item = RouteDefinition> + Send + 'static,
    I::Iter: Send,
{
    let cost_function = crate::cost::CostFunction::from_json(cost_function)?;
    tracing::trace!("Cost function: {:?}", cost_function);
    let simulation = ParRouting::new(store_path, cost_function, mem_limit_bytes, algorithm)?;
    simulation.lazy_scout(route_definitions, tx);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[test]
    fn tuple_from_index() {
        let index_tuple: (u64, u64) = From::from(ArrayIndex { i: 2, j: 3 });
        assert_eq!(index_tuple.0, 2);
        assert_eq!(index_tuple.1, 3);
    }

    #[test]
    fn index_into_tuple() {
        let index_tuple: (u64, u64) = ArrayIndex { i: 2, j: 3 }.into();
        assert_eq!(index_tuple.0, 2);
        assert_eq!(index_tuple.1, 3);
    }

    #[test]
    fn vec_contains_index() {
        let vec_of_indices = [ArrayIndex { i: 2, j: 3 }, ArrayIndex { i: 5, j: 6 }];
        assert!(vec_of_indices.contains(&ArrayIndex { i: 5, j: 6 }));
        assert!(!vec_of_indices.contains(&ArrayIndex { i: 8, j: 9 }));
    }

    #[test_case("dijkstra"; "dijkstra")]
    #[test_case("long-range"; "long-range")]
    #[allow(clippy::approx_constant)]
    // Due to truncation solution to handle f32 costs.
    fn minimalist(algorithm: &str) {
        let store_path = dataset::samples::multi_variable_zarr();
        let cost_function = cost::sample::cost_function();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![ArrayIndex { i: 2, j: 3 }];
        let end = vec![ArrayIndex { i: 6, j: 6 }];
        let solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);
        assert!(solutions[0].route().len() > 1);
        assert!(solutions[0].total_cost() > &0.);
    }

    // Due to truncation solution to handle f32 costs.
    #[allow(clippy::approx_constant)]
    #[test_case((1, 1), (1, 1), 1, 0., "dijkstra"; "no movement dijkstra")]
    #[test_case((1, 1), (1, 2), 2, 1., "dijkstra"; "step one cell to the side dijkstra")]
    #[test_case((1, 1), (2, 1), 2, 1., "dijkstra"; "step one cell down dijkstra")]
    #[test_case((1, 1), (2, 2), 2, 1.4142, "dijkstra"; "step one cell diagonally dijkstra")]
    #[test_case((1, 1), (2, 3), 3, 2.4142, "dijkstra"; "step diagonally and across dijkstra")]
    #[test_case((1, 1), (1, 1), 1, 0., "long-range"; "no movement long-range")]
    #[test_case((1, 1), (1, 2), 2, 1., "long-range"; "step one cell to the side long-range")]
    #[test_case((1, 1), (2, 1), 2, 1., "long-range"; "step one cell down long-range")]
    #[test_case((1, 1), (2, 2), 2, 1.4142, "long-range"; "step one cell diagonally long-range")]
    #[test_case((1, 1), (2, 3), 3, 2.4142, "long-range"; "step diagonally and across long-range")]
    fn basic_routing_point_to_point(
        (si, sj): (u64, u64),
        (ei, ej): (u64, u64),
        expected_num_steps: usize,
        expected_cost: f32,
        algorithm: &str,
    ) {
        let store_path = dataset::samples::constant_value_cost_zarr(1.0);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![ArrayIndex { i: si, j: sj }];
        let end = vec![ArrayIndex { i: ei, j: ej }];
        let solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].route().len(), expected_num_steps);
        assert_eq!(solutions[0].total_cost(), &expected_cost);
    }

    #[test_case((1, 1), vec![(1, 4), (3, 1), (4, 4)], (3, 1), 3, 2., "dijkstra"; "different cost endpoints dijkstra")]
    #[test_case((1, 1), vec![(1, 4), (3, 1), (4, 4)], (3, 1), 3, 2., "long-range"; "different cost endpoints long-range")]
    fn basic_routing_one_point_to_many(
        (si, sj): (u64, u64),
        endpoints: Vec<(u64, u64)>,
        expected_endpoint: (u64, u64),
        expected_num_steps: usize,
        expected_cost: f32,
        algorithm: &str,
    ) {
        let store_path = dataset::samples::constant_value_cost_zarr(1.0);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![ArrayIndex { i: si, j: sj }];
        let end = endpoints
            .clone()
            .into_iter()
            .map(|(i, j)| ArrayIndex { i, j })
            .collect();
        let solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].route().len(), expected_num_steps);
        assert_eq!(solutions[0].total_cost(), &expected_cost);
        assert_eq!(solutions[0].route()[0], start[0]);

        let &ArrayIndex { i: ei, j: ej } = solutions[0].route().last().unwrap();
        assert_eq!((ei, ej), expected_endpoint);
    }

    #[test_case((1, 1), vec![(1, 3), (3, 1)], 1., "dijkstra"; "horizontal and vertical dijkstra")]
    #[test_case((3, 3), vec![(3, 5), (1, 1), (3, 1)], 1., "dijkstra"; "horizontal dijkstra")]
    #[test_case((3, 3), vec![(5, 3), (5, 5), (1, 3)], 1., "dijkstra"; "vertical dijkstra")]
    #[test_case((1, 1), vec![(1, 3), (3, 1)], 1., "long-range"; "horizontal and vertical long-range")]
    #[test_case((3, 3), vec![(3, 5), (1, 1), (3, 1)], 1., "long-range"; "horizontal long-range")]
    #[test_case((3, 3), vec![(5, 3), (5, 5), (1, 3)], 1., "long-range"; "vertical long-range")]
    fn routing_one_point_to_many_same_cost_and_length(
        (si, sj): (u64, u64),
        endpoints: Vec<(u64, u64)>,
        cost_array_fill: f32,
        algorithm: &str,
    ) {
        let store_path = dataset::samples::constant_value_cost_zarr(cost_array_fill);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![ArrayIndex { i: si, j: sj }];
        let end = endpoints
            .clone()
            .into_iter()
            .map(|(i, j)| ArrayIndex { i, j })
            .collect();
        let mut solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);

        let s = solutions.swap_remove(0);
        assert_eq!(s.route().len(), 3);
        assert_eq!(s.total_cost(), &(2. * cost_array_fill));
        assert_eq!(s.route()[0], start[0]);

        let &ArrayIndex { i: ei, j: ej } = s.route().last().unwrap();
        assert!(endpoints.contains(&(ei, ej)));
    }

    #[test_case("dijkstra"; "dijkstra")]
    #[test_case("long-range"; "long-range")]
    #[allow(clippy::approx_constant)]
    // Due to truncation solution to handle f32 costs.
    fn routing_many_to_many(algorithm: &str) {
        let store_path = dataset::samples::constant_value_cost_zarr(1.);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![
            ArrayIndex { i: 1, j: 1 },
            ArrayIndex { i: 3, j: 3 },
            ArrayIndex { i: 5, j: 5 },
        ];
        let end = vec![
            ArrayIndex { i: 1, j: 2 },
            ArrayIndex { i: 4, j: 4 },
            ArrayIndex { i: 7, j: 7 },
        ];
        let solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 3);

        let expected_solution = vec![
            (ArrayIndex { i: 1, j: 2 }, 1.0),
            (ArrayIndex { i: 4, j: 4 }, 1.4142),
            (ArrayIndex { i: 4, j: 4 }, 1.4142),
        ];
        for (s, eep) in solutions.into_iter().zip(expected_solution) {
            assert_eq!(s.route().len(), 2);
            assert_eq!(*s.route().last().unwrap(), eep.0);
            assert_eq!(s.total_cost(), &eep.1);
        }
    }

    #[test_case("dijkstra"; "dijkstra")]
    #[test_case("long-range"; "long-range")]
    fn routing_many_to_one(algorithm: &str) {
        let store_path = dataset::samples::constant_value_cost_zarr(1.);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![ArrayIndex { i: 1, j: 1 }, ArrayIndex { i: 5, j: 5 }];
        let end = vec![ArrayIndex { i: 3, j: 3 }];
        let solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 2);

        for s in solutions {
            assert_eq!(s.route().len(), 3);
            assert_eq!(s.total_cost(), &2.8284);
            assert_eq!(*s.route().last().unwrap(), ArrayIndex { i: 3, j: 3 });
        }
    }

    #[test_case("dijkstra"; "dijkstra")]
    #[test_case("long-range"; "long-range")]
    fn test_routing_along_boundary(algorithm: &str) {
        use ndarray::Array3;

        let (ni, nj) = (4, 4);
        let (ci, cj) = (2, 2);

        let store_path = tempfile::TempDir::new().unwrap();

        let store: zarrs::storage::ReadableWritableListableStorage = std::sync::Arc::new(
            zarrs::filesystem::FilesystemStore::new(store_path.path())
                .expect("could not open filesystem store"),
        );

        zarrs::group::GroupBuilder::new()
            .build(store.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();

        let array = zarrs::array::ArrayBuilder::new(
            vec![1, ni, nj], // array shape
            vec![1, ci, cj], // regular chunk shape
            zarrs::array::DataType::Float32,
            zarrs::array::FillValue::from(zarrs::array::ZARR_NAN_F32),
        )
        .dimension_names(["band", "y", "x"].into())
        .build(store.clone(), "/cost")
        .unwrap();

        // Write array metadata to store
        array.store_metadata().unwrap();

        #[rustfmt::skip]
        let a = vec![1., 50.,  1., 1.,
                     1., 50., 50., 1.,
                     1., 50., 50., 1.,
                     1.,  1.,  1., 1.];

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

        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();

        let start = vec![ArrayIndex { i: 0, j: 0 }];
        let end = vec![ArrayIndex { i: 0, j: 2 }];
        let mut solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        assert_eq!(solutions.len(), 1);

        let s = solutions.swap_remove(0);
        // 4 straight moves + 3 diagonal moves
        assert_eq!(s.total_cost(), &8.2426);

        let expected_track = vec![
            ArrayIndex { i: 0, j: 0 },
            ArrayIndex { i: 1, j: 0 },
            ArrayIndex { i: 2, j: 0 },
            ArrayIndex { i: 3, j: 1 },
            ArrayIndex { i: 3, j: 2 },
            ArrayIndex { i: 2, j: 3 },
            ArrayIndex { i: 1, j: 3 },
            ArrayIndex { i: 0, j: 2 },
        ];
        assert_eq!(s.route(), &expected_track);
    }
}
