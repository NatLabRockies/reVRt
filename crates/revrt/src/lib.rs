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
    option: u32,
}

impl ArrayIndex {
    #[allow(missing_docs)]
    pub fn new(i: u64, j: u64) -> Self {
        Self { i, j, option: 0 }
    }

    pub fn new_ij(i: u64, j: u64) -> Self {
        Self { i, j, option: 0 }
    }
}

impl From<ArrayIndex> for (u64, u64) {
    fn from(ArrayIndex { i, j, .. }: ArrayIndex) -> (u64, u64) {
        (i, j)
    }
}

impl From<ArrayIndex> for (u64, u64, u32) {
    fn from(ArrayIndex { i, j, option }: ArrayIndex) -> (u64, u64, u32) {
        (i, j, option)
    }
}

#[allow(missing_docs)]
pub fn resolve_with_routing_options<P: AsRef<std::path::Path>>(
    store_path: P,
    cost_function: &str,
    algorithm: &str,
    start: &[ArrayIndex],
    end: Vec<ArrayIndex>,
    swap_fp: Option<std::path::PathBuf>,
    mem_limit_bytes: u64,
) -> Result<(RevrtRoutingSolutions, Vec<String>)> {
    let cost_function = CostFunction::from_json(cost_function)?;
    let routing_options = cost_function.routing_options.clone();
    tracing::trace!("Cost function: {:?}", cost_function);
    let mut simulation = match swap_fp {
        Some(swap_fp) => Routing::new_wth_swap(
            store_path,
            cost_function,
            swap_fp,
            mem_limit_bytes,
            algorithm,
        )?,
        None => Routing::new(store_path, cost_function, mem_limit_bytes, algorithm)?,
    };
    let result = simulation.compute(start, end).collect();
    Ok((result, routing_options))
}

pub(crate) fn resolve_generator_with_routing_options<P, I>(
    store_path: P,
    cost_function: &str,
    route_definitions: I,
    tx: mpsc::Sender<(u32, RevrtRoutingSolutions)>,
    swap_fp: Option<std::path::PathBuf>,
    mem_limit_bytes: u64,
    algorithm: &str,
) -> Result<Vec<String>>
where
    P: AsRef<std::path::Path>,
    I: rayon::prelude::IntoParallelIterator<Item = RouteDefinition> + Send + 'static,
    I::Iter: Send,
{
    let cost_function = crate::cost::CostFunction::from_json(cost_function)?;
    let routing_options = cost_function.routing_options.clone();
    tracing::trace!("Cost function: {:?}", cost_function);
    let simulation = match swap_fp {
        Some(swap_fp) => ParRouting::new_with_swap(
            store_path,
            cost_function,
            mem_limit_bytes,
            swap_fp,
            algorithm,
        )?,
        None => ParRouting::new(store_path, cost_function, mem_limit_bytes, algorithm)?,
    };
    simulation.lazy_scout(route_definitions, tx);
    Ok(routing_options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[test]
    fn tuple_from_index() {
        let index_tuple: (u64, u64) = From::from(ArrayIndex::new_ij(2, 3));
        assert_eq!(index_tuple.0, 2);
        assert_eq!(index_tuple.1, 3);
    }

    #[test]
    fn index_into_tuple() {
        let index_tuple: (u64, u64) = ArrayIndex::new_ij(2, 3).into();
        assert_eq!(index_tuple.0, 2);
        assert_eq!(index_tuple.1, 3);
    }

    #[test]
    fn vec_contains_index() {
        let vec_of_indices = [ArrayIndex::new_ij(2, 3), ArrayIndex::new_ij(5, 6)];
        assert!(vec_of_indices.contains(&ArrayIndex::new_ij(5, 6)));
        assert!(!vec_of_indices.contains(&ArrayIndex::new_ij(8, 9)));
    }

    #[test]
    fn index_with_option_into_tuple() {
        let index_tuple: (u64, u64, u32) = ArrayIndex {
            i: 2,
            j: 3,
            option: 4,
        }
        .into();
        assert_eq!(index_tuple, (2, 3, 4));
    }

    #[test]
    fn index_with_option_still_converts_to_2d_tuple() {
        let index_tuple: (u64, u64) = ArrayIndex {
            i: 5,
            j: 6,
            option: 2,
        }
        .into();

        assert_eq!(index_tuple, (5, 6));
    }

    #[test_case("astar"; "astar")]
    #[test_case("dijkstra"; "dijkstra")]
    #[test_case("long-range-astar"; "long-range-astar")]
    #[test_case("long-range-dijkstra"; "long-range")]
    #[test_case("bidirectional-long-range-dijkstra"; "bidirectional-long-range")]
    #[allow(clippy::approx_constant)]
    // Due to truncation solution to handle f32 costs.
    fn minimalist(algorithm: &str) {
        let store_path =
            dataset::samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "B", "C", "cost"]);
        let cost_function = cost::sample::cost_function();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![ArrayIndex::new_ij(2, 3)];
        let end = vec![ArrayIndex::new_ij(6, 6)];
        let solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);
        assert!(solutions[0].route().len() > 1);
        assert!(solutions[0].total_cost() > &0.);
    }

    // Due to truncation solution to handle f32 costs.
    #[allow(clippy::approx_constant)]
    #[test_case((1, 1), (1, 1), 1, 0., "astar"; "no movement astar")]
    #[test_case((1, 1), (1, 2), 2, 1., "astar"; "step one cell to the side astar")]
    #[test_case((1, 1), (2, 1), 2, 1., "astar"; "step one cell down astar")]
    #[test_case((1, 1), (2, 2), 2, 1.4142, "astar"; "step one cell diagonally astar")]
    #[test_case((1, 1), (2, 3), 3, 2.4142, "astar"; "step diagonally and across astar")]
    #[test_case((1, 1), (1, 1), 1, 0., "dijkstra"; "no movement dijkstra")]
    #[test_case((1, 1), (1, 2), 2, 1., "dijkstra"; "step one cell to the side dijkstra")]
    #[test_case((1, 1), (2, 1), 2, 1., "dijkstra"; "step one cell down dijkstra")]
    #[test_case((1, 1), (2, 2), 2, 1.4142, "dijkstra"; "step one cell diagonally dijkstra")]
    #[test_case((1, 1), (2, 3), 3, 2.4142, "dijkstra"; "step diagonally and across dijkstra")]
    #[test_case((1, 1), (1, 1), 1, 0., "long-range-astar"; "no movement long-range astar")]
    #[test_case((1, 1), (1, 2), 2, 1., "long-range-astar"; "step one cell to the side long-range astar")]
    #[test_case((1, 1), (2, 1), 2, 1., "long-range-astar"; "step one cell down long-range astar")]
    #[test_case((1, 1), (2, 2), 2, 1.4142, "long-range-astar"; "step one cell diagonally long-range astar")]
    #[test_case((1, 1), (2, 3), 3, 2.4142, "long-range-astar"; "step diagonally and across long-range astar")]
    #[test_case((1, 1), (1, 1), 1, 0., "long-range-dijkstra"; "no movement long-range")]
    #[test_case((1, 1), (1, 2), 2, 1., "long-range-dijkstra"; "step one cell to the side long-range")]
    #[test_case((1, 1), (2, 1), 2, 1., "long-range-dijkstra"; "step one cell down long-range")]
    #[test_case((1, 1), (2, 2), 2, 1.4142, "long-range-dijkstra"; "step one cell diagonally long-range")]
    #[test_case((1, 1), (2, 3), 3, 2.4142, "long-range-dijkstra"; "step diagonally and across long-range")]
    #[test_case((1, 1), (1, 1), 1, 0., "bidirectional-long-range-dijkstra"; "no movement bidirectional long-range")]
    #[test_case((1, 1), (1, 2), 2, 1., "bidirectional-long-range-dijkstra"; "step one cell to the side bidirectional long-range")]
    #[test_case((1, 1), (2, 1), 2, 1., "bidirectional-long-range-dijkstra"; "step one cell down bidirectional long-range")]
    #[test_case((1, 1), (2, 2), 2, 1.4142, "bidirectional-long-range-dijkstra"; "step one cell diagonally bidirectional long-range")]
    #[test_case((1, 1), (2, 3), 3, 2.4142, "bidirectional-long-range-dijkstra"; "step diagonally and across bidirectional long-range")]
    fn basic_routing_point_to_point(
        (si, sj): (u64, u64),
        (ei, ej): (u64, u64),
        expected_num_steps: usize,
        expected_cost: f32,
        algorithm: &str,
    ) {
        let store_path = dataset::samples::uniform_cost_zarr(1, 8, 8, 1, 4, 4, 1.0);
        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"cost"}]}}}"#,
        )
        .unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![ArrayIndex::new_ij(si, sj)];
        let end = vec![ArrayIndex::new_ij(ei, ej)];
        let solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].route().len(), expected_num_steps);
        assert_eq!(solutions[0].total_cost(), &expected_cost);
    }

    #[test_case((1, 1), vec![(1, 4), (3, 1), (4, 4)], (3, 1), 3, 2., "astar"; "different cost endpoints astar")]
    #[test_case((1, 1), vec![(1, 4), (3, 1), (4, 4)], (3, 1), 3, 2., "dijkstra"; "different cost endpoints dijkstra")]
    #[test_case((1, 1), vec![(1, 4), (3, 1), (4, 4)], (3, 1), 3, 2., "long-range-astar"; "different cost endpoints long-range astar")]
    #[test_case((1, 1), vec![(1, 4), (3, 1), (4, 4)], (3, 1), 3, 2., "long-range-dijkstra"; "different cost endpoints long-range")]
    #[test_case((1, 1), vec![(1, 4), (3, 1), (4, 4)], (3, 1), 3, 2., "bidirectional-long-range-dijkstra"; "different cost endpoints bidirectional long-range")]
    fn basic_routing_one_point_to_many(
        (si, sj): (u64, u64),
        endpoints: Vec<(u64, u64)>,
        expected_endpoint: (u64, u64),
        expected_num_steps: usize,
        expected_cost: f32,
        algorithm: &str,
    ) {
        let store_path = dataset::samples::uniform_cost_zarr(1, 8, 8, 1, 4, 4, 1.0);
        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"cost"}]}}}"#,
        )
        .unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![ArrayIndex::new_ij(si, sj)];
        let end = endpoints
            .clone()
            .into_iter()
            .map(|(i, j)| ArrayIndex::new_ij(i, j))
            .collect();
        let solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].route().len(), expected_num_steps);
        assert_eq!(solutions[0].total_cost(), &expected_cost);
        assert_eq!(solutions[0].route()[0], start[0]);

        let &ArrayIndex { i: ei, j: ej, .. } = solutions[0].route().last().unwrap();
        assert_eq!((ei, ej), expected_endpoint);
    }

    #[test]
    fn explicit_barrier_layers_block_routing() {
        let store = dataset::samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(dataset::samples::LayerConfig::constant("cost", 1.0))
            .layer(dataset::samples::LayerConfig::new(
                "barrier",
                dataset::samples::FillStrategy::Values(vec![
                    1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                ]),
            ))
            .build()
            .unwrap();
        let store_path = store.path();

        let cost_function = CostFunction::from_json(
            r#"{
                "routing_options": {
                    "default": {
                        "cost_layers": [{"layer_name": "cost"}],
                        "barrier_layers": [{
                            "layer_name": "barrier",
                            "barrier_operator": "eq",
                            "barrier_threshold": 1.0
                        }]
                    }
                },
                "ignore_invalid_costs": false
            }"#,
        )
        .unwrap();
        let mut simulation = Routing::new(store_path, cost_function, 1_000, "dijkstra").unwrap();
        let start = vec![ArrayIndex::new_ij(1, 1)];
        let end = vec![ArrayIndex::new_ij(0, 0)];

        let solutions = simulation.compute(&start, end).collect::<Vec<_>>();

        assert!(solutions.is_empty());
    }

    #[test_case((1, 1), vec![(1, 3), (3, 1)], 1., "astar"; "horizontal and vertical astar")]
    #[test_case((3, 3), vec![(3, 5), (1, 1), (3, 1)], 1., "astar"; "horizontal astar")]
    #[test_case((3, 3), vec![(5, 3), (5, 5), (1, 3)], 1., "astar"; "vertical astar")]
    #[test_case((1, 1), vec![(1, 3), (3, 1)], 1., "dijkstra"; "horizontal and vertical dijkstra")]
    #[test_case((3, 3), vec![(3, 5), (1, 1), (3, 1)], 1., "dijkstra"; "horizontal dijkstra")]
    #[test_case((3, 3), vec![(5, 3), (5, 5), (1, 3)], 1., "dijkstra"; "vertical dijkstra")]
    #[test_case((1, 1), vec![(1, 3), (3, 1)], 1., "long-range-astar"; "horizontal and vertical long-range astar")]
    #[test_case((3, 3), vec![(3, 5), (1, 1), (3, 1)], 1., "long-range-astar"; "horizontal long-range astar")]
    #[test_case((3, 3), vec![(5, 3), (5, 5), (1, 3)], 1., "long-range-astar"; "vertical long-range astar")]
    #[test_case((1, 1), vec![(1, 3), (3, 1)], 1., "long-range-dijkstra"; "horizontal and vertical long-range")]
    #[test_case((3, 3), vec![(3, 5), (1, 1), (3, 1)], 1., "long-range-dijkstra"; "horizontal long-range")]
    #[test_case((3, 3), vec![(5, 3), (5, 5), (1, 3)], 1., "long-range-dijkstra"; "vertical long-range")]
    #[test_case((1, 1), vec![(1, 3), (3, 1)], 1., "bidirectional-long-range-dijkstra"; "horizontal and vertical bidirectional long-range")]
    #[test_case((3, 3), vec![(3, 5), (1, 1), (3, 1)], 1., "bidirectional-long-range-dijkstra"; "horizontal bidirectional long-range")]
    #[test_case((3, 3), vec![(5, 3), (5, 5), (1, 3)], 1., "bidirectional-long-range-dijkstra"; "vertical bidirectional long-range")]
    fn routing_one_point_to_many_same_cost_and_length(
        (si, sj): (u64, u64),
        endpoints: Vec<(u64, u64)>,
        cost_array_fill: f32,
        algorithm: &str,
    ) {
        let store_path = dataset::samples::uniform_cost_zarr(1, 8, 8, 1, 4, 4, cost_array_fill);
        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"cost"}]}}}"#,
        )
        .unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![ArrayIndex::new_ij(si, sj)];
        let end = endpoints
            .clone()
            .into_iter()
            .map(|(i, j)| ArrayIndex::new_ij(i, j))
            .collect();
        let mut solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);

        let s = solutions.swap_remove(0);
        assert_eq!(s.route().len(), 3);
        assert_eq!(s.total_cost(), &(2. * cost_array_fill));
        assert_eq!(s.route()[0], start[0]);

        let &ArrayIndex { i: ei, j: ej, .. } = s.route().last().unwrap();
        assert!(endpoints.contains(&(ei, ej)));
    }

    #[test_case("astar"; "astar")]
    #[test_case("dijkstra"; "dijkstra")]
    #[test_case("long-range-astar"; "long-range-astar")]
    #[test_case("long-range-dijkstra"; "long-range")]
    #[test_case("bidirectional-long-range-dijkstra"; "bidirectional-long-range")]
    #[allow(clippy::approx_constant)]
    // Due to truncation solution to handle f32 costs.
    fn routing_many_to_many(algorithm: &str) {
        let store_path = dataset::samples::uniform_cost_zarr(1, 8, 8, 1, 4, 4, 1.0);
        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"cost"}]}}}"#,
        )
        .unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![
            ArrayIndex::new_ij(1, 1),
            ArrayIndex::new_ij(3, 3),
            ArrayIndex::new_ij(5, 5),
        ];
        let end = vec![
            ArrayIndex::new_ij(1, 2),
            ArrayIndex::new_ij(4, 4),
            ArrayIndex::new_ij(7, 7),
        ];
        let solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 3);

        let expected_solution = vec![
            (ArrayIndex::new_ij(1, 2), 1.0),
            (ArrayIndex::new_ij(4, 4), 1.4142),
            (ArrayIndex::new_ij(4, 4), 1.4142),
        ];
        for (s, eep) in solutions.into_iter().zip(expected_solution) {
            assert_eq!(s.route().len(), 2);
            assert_eq!(*s.route().last().unwrap(), eep.0);
            assert_eq!(s.total_cost(), &eep.1);
        }
    }

    #[test_case("astar"; "astar")]
    #[test_case("dijkstra"; "dijkstra")]
    #[test_case("long-range-astar"; "long-range-astar")]
    #[test_case("long-range-dijkstra"; "long-range")]
    #[test_case("bidirectional-long-range-dijkstra"; "bidirectional-long-range")]
    fn routing_many_to_one(algorithm: &str) {
        let store_path = dataset::samples::uniform_cost_zarr(1, 8, 8, 1, 4, 4, 1.0);
        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"cost"}]}}}"#,
        )
        .unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();
        let start = vec![ArrayIndex::new_ij(1, 1), ArrayIndex::new_ij(5, 5)];
        let end = vec![ArrayIndex::new_ij(3, 3)];
        let solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        dbg!(&solutions);
        assert_eq!(solutions.len(), 2);

        for s in solutions {
            assert_eq!(s.route().len(), 3);
            assert_eq!(s.total_cost(), &2.8284);
            assert_eq!(*s.route().last().unwrap(), ArrayIndex::new_ij(3, 3));
        }
    }

    #[test_case("astar"; "astar")]
    #[test_case("dijkstra"; "dijkstra")]
    #[test_case("long-range-astar"; "long-range-astar")]
    #[test_case("long-range-dijkstra"; "long-range")]
    #[test_case("bidirectional-long-range-dijkstra"; "bidirectional-long-range")]
    fn test_routing_along_boundary(algorithm: &str) {
        use dataset::samples;

        #[rustfmt::skip]
        let a = vec![1., 50.,  1., 1.,
                     1., 50., 50., 1.,
                     1., 50., 50., 1.,
                     1.,  1.,  1., 1.];
        let store_path = samples::ZarrTestBuilder::new()
            .dimensions(1, 4, 4)
            .chunks(1, 2, 2)
            .layer(samples::LayerConfig::new(
                "cost",
                samples::FillStrategy::Values(a),
            ))
            .build()
            .expect("failed to build zarr test dataset");

        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"cost"}]}}}"#,
        )
        .unwrap();
        let mut simulation = Routing::new(&store_path, cost_function, 1_000, algorithm).unwrap();

        let start = vec![ArrayIndex::new_ij(0, 0)];
        let end = vec![ArrayIndex::new_ij(0, 2)];
        let mut solutions = simulation.compute(&start, end).collect::<Vec<_>>();
        assert_eq!(solutions.len(), 1);

        let s = solutions.swap_remove(0);
        // 4 straight moves + 3 diagonal moves
        assert_eq!(s.total_cost(), &8.2426);

        let expected_track = vec![
            ArrayIndex::new_ij(0, 0),
            ArrayIndex::new_ij(1, 0),
            ArrayIndex::new_ij(2, 0),
            ArrayIndex::new_ij(3, 1),
            ArrayIndex::new_ij(3, 2),
            ArrayIndex::new_ij(2, 3),
            ArrayIndex::new_ij(1, 3),
            ArrayIndex::new_ij(0, 2),
        ];
        assert_eq!(s.route(), &expected_track);
    }
}
