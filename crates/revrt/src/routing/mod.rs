//! Routing module

mod algorithm;
mod astar;
mod features;
mod long_range;
mod scenario;

use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, mpsc};

use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use tracing::{debug, info};

use crate::{ArrayIndex, RevrtRoutingSolutions, Solution, error::Result};
use algorithm::Algorithm;
use algorithm::AlgorithmType;
use scenario::Scenario;

// Percent of the total memory allocation given to the Zarr cache.
// Rest of memory is allocated to the routing algorithm's rayon workers
const CACHE_BUDGET_PERCENT: u64 = 25;

pub(super) struct Routing {
    scenario: Scenario,
    algorithm: Algorithm,
}

impl Routing {
    pub(super) fn compute(
        &mut self,
        start: &[ArrayIndex],
        end: Vec<ArrayIndex>,
    ) -> impl Iterator<Item = Solution<ArrayIndex, f32>> {
        debug!("Starting compute with {} start points", start.len());
        compute_route_attempt_result(&self.scenario, &self.algorithm, start, &end).into_iter()
    }

    pub(super) fn new<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        mem_limit_bytes: u64,
        algorithm: &str,
    ) -> Result<Self> {
        let algorithm = AlgorithmType::from_str(algorithm)?;
        let cache_size = cache_budget_bytes(mem_limit_bytes);
        let rayon_worker_total_budget_bytes = mem_limit_bytes - cache_size;
        let scenario = Scenario::new(store_path, cost_function, cache_size)?;
        let algorithm = Algorithm::from_selection(
            algorithm,
            per_rayon_worker_memory_budget(rayon_worker_total_budget_bytes),
        );

        Ok(Self {
            scenario,
            algorithm,
        })
    }

    pub(super) fn new_wth_swap<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        swap_fp: PathBuf,
        mem_limit_bytes: u64,
        algorithm: &str,
    ) -> Result<Self> {
        let algorithm = AlgorithmType::from_str(algorithm)?;
        let cache_size = cache_budget_bytes(mem_limit_bytes);
        let rayon_worker_total_budget_bytes = mem_limit_bytes - cache_size;
        let scenario = Scenario::new_with_swap(store_path, cost_function, cache_size, swap_fp)?;
        let algorithm = Algorithm::from_selection(
            algorithm,
            per_rayon_worker_memory_budget(rayon_worker_total_budget_bytes),
        );

        Ok(Self {
            scenario,
            algorithm,
        })
    }
}

pub(super) struct RouteDefinition {
    pub(super) route_id: u32,
    pub(super) start_inds: Vec<ArrayIndex>,
    pub(super) end_inds: HashSet<ArrayIndex>,
}

pub(super) struct ParRouting {
    scenario: Arc<Scenario>,
    algorithm: Arc<Algorithm>,
}

impl ParRouting {
    pub(super) fn new<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        mem_limit_bytes: u64,
        algorithm: &str,
    ) -> Result<Self> {
        let algorithm = AlgorithmType::from_str(algorithm)?;
        let cache_size = cache_budget_bytes(mem_limit_bytes);
        let rayon_worker_total_budget_bytes = mem_limit_bytes - cache_size;
        let scenario = Scenario::new(store_path, cost_function, cache_size)?;
        Ok(Self {
            scenario: Arc::new(scenario),
            algorithm: Arc::new(Algorithm::from_selection(
                algorithm,
                per_rayon_worker_memory_budget(rayon_worker_total_budget_bytes),
            )),
        })
    }

    pub(super) fn new_with_swap<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        mem_limit_bytes: u64,
        swap_fp: PathBuf,
        algorithm: &str,
    ) -> Result<Self> {
        let algorithm = AlgorithmType::from_str(algorithm)?;
        let cache_size = cache_budget_bytes(mem_limit_bytes);
        let rayon_worker_total_budget_bytes = mem_limit_bytes - cache_size;
        let scenario = Scenario::new_with_swap(store_path, cost_function, cache_size, swap_fp)?;
        Ok(Self {
            scenario: Arc::new(scenario),
            algorithm: Arc::new(Algorithm::from_selection(
                algorithm,
                per_rayon_worker_memory_budget(rayon_worker_total_budget_bytes),
            )),
        })
    }

    pub(super) fn lazy_scout<I>(
        &self,
        route_definitions: I,
        tx: mpsc::Sender<(u32, RevrtRoutingSolutions)>,
    ) where
        I: IntoParallelIterator<Item = RouteDefinition> + Send + 'static,
        I::Iter: Send,
    {
        let scenario = Arc::clone(&self.scenario);
        let algorithm = Arc::clone(&self.algorithm);
        rayon::spawn(move || {
            let _ = route_definitions.into_par_iter().try_for_each_with(
                tx,
                |sender,
                 RouteDefinition {
                     route_id,
                     start_inds,
                     end_inds,
                 }| {
                    debug!("Computing routes between {start_inds:?} and {end_inds:?}");
                    let result = compute_route_attempt_result(
                        &scenario,
                        &algorithm,
                        &start_inds,
                        &end_inds.iter().cloned().collect::<Vec<_>>(),
                    );
                    let num_routes = result.len();
                    debug!("Finished computing {num_routes} route(s) to {end_inds:?}");
                    sender.send((route_id, result))
                },
            );
        });
    }
}

fn compute_route_attempt_result(
    scenario: &Scenario,
    algorithm: &Algorithm,
    start: &[ArrayIndex],
    end: &[ArrayIndex],
) -> RevrtRoutingSolutions {
    start
        .into_par_iter()
        .filter_map(|start_point| compute_solution_for_start(scenario, algorithm, start_point, end))
        .collect()
}

fn compute_solution_for_start(
    scenario: &Scenario,
    algorithm: &Algorithm,
    start_point: &ArrayIndex,
    end: &[ArrayIndex],
) -> Option<Solution<ArrayIndex, f32>> {
    let grid_shape = scenario.grid_shape();

    // if end_inds.last() == Some(&ArrayIndex { i: 2, j: 6 }) {
    //     use std::thread;
    //     use std::time::Duration;
    //     // let mut rng = rand::rng();
    //     // let delay_secs = rng.random_range(3..=7);
    //     let delay_secs = if start_inds.first() == Some(&ArrayIndex { i: 1, j: 1 }) {
    //         6
    //         // return sender.send(Err(InvalidRouteStart(
    //         //     "start index ArrayIndex { i: 1, j: 1 } is invalid".into(),
    //         // )));
    //     } else {
    //         10
    //     };
    //     // println!("Sleeping {delay_secs}s before yielding");
    //     // io::stdout().flush().expect("Failed to flush stdout");
    //     thread::sleep(Duration::from_secs(delay_secs));
    // }

    for dropped_soft_groups in 0..=scenario.soft_barrier_group_count() {
        if dropped_soft_groups > 0 {
            info!(
                "Retrying route from {:?} with {} soft-barrier group(s) dropped",
                start_point, dropped_soft_groups
            );
        }

        let start_state = scenario
            .allowed_states_at(start_point, dropped_soft_groups)
            .find(|state| *state == *start_point);

        let start_point = match start_state {
            Some(state) => state,
            None => continue,
        };

        let solution = algorithm.compute(
            &start_point,
            end,
            |p| scenario.successors_for_attempt(p, dropped_soft_groups),
            |p| end.contains(p), // |p| end.iter().any(|goal| goal.i == p.i && goal.j == p.j),
            grid_shape,
        );

        if let Some(solution) = solution {
            let dropped_barrier_layers = scenario.dropped_barrier_layers(dropped_soft_groups);
            return Some(solution.record_dropped_barriers(dropped_barrier_layers));
        }
    }

    None
}

// fn compute_solution_for_start(
//     scenario: &Scenario,
//     algorithm: &Algorithm,
//     start_point: &ArrayIndex,
//     end: &[ArrayIndex],
// ) -> Option<Solution<ArrayIndex, f32>> {
//     let grid_shape = scenario.grid_shape();
//     let goal_requires_explicit_option = end.iter().any(|goal| goal.option != 0);

//     for dropped_soft_groups in 0..=scenario.soft_barrier_group_count() {
//         if dropped_soft_groups > 0 {
//             info!(
//                 "Retrying route from {:?} with {} soft-barrier group(s) dropped",
//                 start_point, dropped_soft_groups
//             );
//         }

//         let start_requires_explicit_option =
//             start_point.option != 0 || goal_requires_explicit_option;

//         let start_states = if start_requires_explicit_option {
//             scenario
//                 .allowed_states_at(start_point, dropped_soft_groups)
//                 .into_iter()
//                 .filter(|state| *state == *start_point)
//                 .collect()
//         } else {
//             scenario.allowed_states_at(start_point, dropped_soft_groups)
//         };

//         let solution = start_states
//             .iter()
//             .filter_map(|start_state| {
//                 algorithm.compute(
//                     start_state,
//                     end,
//                     |p| scenario.successors_for_attempt(p, dropped_soft_groups),
//                     |p| {
//                         if goal_requires_explicit_option {
//                             end.iter().any(|goal| goal == p)
//                         } else {
//                             end.iter().any(|goal| goal.i == p.i && goal.j == p.j)
//                         }
//                     },
//                     grid_shape,
//                 )
//             })
//             .min_by(|left, right| left.total_cost.total_cmp(&right.total_cost));

//         if let Some(solution) = solution {
//             let dropped_barrier_layers = scenario.dropped_barrier_layers(dropped_soft_groups);
//             return Some(solution.record_dropped_barriers(dropped_barrier_layers));
//         }
//     }

//     None
// }

const PRECISION_SCALAR: f32 = 1e4;
fn cost_as_u64(cost: f32) -> u64 {
    let cost = cost * PRECISION_SCALAR;
    cost as u64
}

fn unscaled_cost(cost: u64) -> f32 {
    (cost as f32) / PRECISION_SCALAR
}

fn cache_budget_bytes(mem_limit_bytes: u64) -> u64 {
    mem_limit_bytes * CACHE_BUDGET_PERCENT / 100
}

fn per_rayon_worker_memory_budget(total_budget_bytes: u64) -> u64 {
    // Routing uses Rayon global-pool APIs, so this reflects the worker count
    // that will execute the searches, even at initialization
    let worker_count = rayon::current_num_threads().max(1) as u64;
    let per_worker_budget = total_budget_bytes / worker_count;

    debug!(
        "Splitting {} bytes across {} Rayon workers ({} bytes per worker)",
        total_budget_bytes, worker_count, per_worker_budget
    );

    per_worker_budget
}

#[cfg(test)]
mod tests {
    use super::{Algorithm, AlgorithmType, Scenario, compute_route_attempt_result};
    use crate::ArrayIndex;

    fn overhead_switch_cost(_band: u64, _row: u64, _col: u64) -> f32 {
        1.0
    }

    fn underground_switch_cost(_band: u64, _row: u64, _col: u64) -> f32 {
        4.0
    }

    fn overhead_destination_barrier(_band: u64, row: u64, col: u64) -> f32 {
        if row == 0 && col == 1 {
            1.0
        } else {
            0.0
        }
    }

    #[test]
    fn compute_route_attempt_result_tracks_dropped_barriers_per_start() {
        let store = crate::dataset::samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 5)
            .chunks(1, 3, 5)
            .layer(crate::dataset::samples::LayerConfig::ones("cost"))
            .layer(crate::dataset::samples::LayerConfig::new(
                "hard_barrier",
                crate::dataset::samples::FillStrategy::Values(vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]),
            ))
            .layer(crate::dataset::samples::LayerConfig::new(
                "soft_barrier",
                crate::dataset::samples::FillStrategy::Values(vec![
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]),
            ))
            .build()
            .unwrap();
        let cost_function = crate::cost::CostFunction::from_json(
            r#"{
                "routing_options": {
                    "default": {
                        "cost_layers": [{"layer_name": "cost"}],
                        "barrier_layers": [
                            {
                                "layer_name": "hard_barrier",
                                "barrier_operator": "eq",
                                "barrier_threshold": 1.0
                            },
                            {
                                "layer_name": "soft_barrier",
                                "barrier_operator": "eq",
                                "barrier_threshold": 1.0,
                                "barrier_importance": 1
                            }
                        ]
                    }
                },
                "invalid_costs_block_routing": false
            }"#,
        )
        .unwrap();
        let scenario = Scenario::new(store.path(), cost_function, 1_000).unwrap();
        let algorithm = Algorithm::from_selection(AlgorithmType::Dijkstra, 8 * 1024 * 1024);
        let start = [ArrayIndex::new_ij(0, 0), ArrayIndex::new_ij(2, 0)];
        let end = [ArrayIndex::new_ij(0, 4), ArrayIndex::new_ij(2, 4)];

        let result = compute_route_attempt_result(&scenario, &algorithm, &start, &end);

        assert_eq!(result.len(), 2);

        let top_solution = result
            .iter()
            .find(|solution| solution.route().first() == Some(&ArrayIndex::new_ij(0, 0)))
            .unwrap();
        assert_eq!(top_solution.route().last(), Some(&ArrayIndex::new_ij(0, 4)));
        assert_eq!(
            top_solution.dropped_barrier_layers(),
            &vec!["soft_barrier".to_string()]
        );

        let bottom_solution = result
            .iter()
            .find(|solution| solution.route().first() == Some(&ArrayIndex::new_ij(2, 0)))
            .unwrap();
        assert_eq!(
            bottom_solution.route().last(),
            Some(&ArrayIndex::new_ij(2, 4))
        );
        assert!(bottom_solution.dropped_barrier_layers().is_empty());
    }

    #[test]
    fn compute_route_attempt_result_can_switch_options_while_moving() {
        let store = crate::dataset::samples::ZarrTestBuilder::new()
            .dimensions(1, 1, 2)
            .chunks(1, 1, 2)
            .layer(crate::dataset::samples::LayerConfig::custom(
                "overhead_cost",
                overhead_switch_cost,
            ))
            .layer(crate::dataset::samples::LayerConfig::custom(
                "underground_cost",
                underground_switch_cost,
            ))
            .layer(crate::dataset::samples::LayerConfig::custom(
                "overhead_hard_barrier",
                overhead_destination_barrier,
            ))
            .build()
            .unwrap();
        let cost_function = crate::cost::CostFunction::from_json(
            r#"{
                "routing_options": {
                    "overhead": {
                        "cost_layers": [{"layer_name": "overhead_cost"}],
                        "barrier_layers": [
                            {
                                "layer_name": "overhead_hard_barrier",
                                "barrier_operator": "eq",
                                "barrier_threshold": 1.0
                            }
                        ]
                    },
                    "underground": {
                        "cost_layers": [{"layer_name": "underground_cost"}]
                    }
                },
                "invalid_costs_block_routing": false
            }"#,
        )
        .unwrap();
        let scenario = Scenario::new(store.path(), cost_function, 1_000).unwrap();
        let algorithm = Algorithm::from_selection(AlgorithmType::Dijkstra, 8 * 1024 * 1024);
        let start = [ArrayIndex {
            i: 0,
            j: 0,
            option: 0,
        }];
        let end = [ArrayIndex {
            i: 0,
            j: 1,
            option: 1,
        }];

        let result = compute_route_attempt_result(&scenario, &algorithm, &start, &end);

        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].route(),
            &vec![
                ArrayIndex {
                    i: 0,
                    j: 0,
                    option: 0
                },
                ArrayIndex {
                    i: 0,
                    j: 1,
                    option: 1
                },
            ]
        );
        assert_eq!(*result[0].total_cost(), 2.5);
    }

    #[test]
    fn compute_route_attempt_result_applies_transition_costs() {
        let store = crate::dataset::samples::ZarrTestBuilder::new()
            .dimensions(1, 1, 2)
            .chunks(1, 1, 2)
            .layer(crate::dataset::samples::LayerConfig::custom(
                "overhead_cost",
                overhead_switch_cost,
            ))
            .layer(crate::dataset::samples::LayerConfig::custom(
                "underground_cost",
                underground_switch_cost,
            ))
            .layer(crate::dataset::samples::LayerConfig::custom(
                "overhead_hard_barrier",
                overhead_destination_barrier,
            ))
            .build()
            .unwrap();
        let cost_function = crate::cost::CostFunction::from_json(
            r#"{
                "routing_options": {
                    "overhead": {
                        "cost_layers": [{"layer_name": "overhead_cost"}],
                        "barrier_layers": [
                            {
                                "layer_name": "overhead_hard_barrier",
                                "barrier_operator": "eq",
                                "barrier_threshold": 1.0
                            }
                        ]
                    },
                    "underground": {
                        "cost_layers": [{"layer_name": "underground_cost"}]
                    }
                },
                "transition_costs": {
                    "pairwise": [
                        {
                            "from": "overhead",
                            "to": "underground",
                            "cost": 3.0
                        }
                    ]
                },
                "invalid_costs_block_routing": false
            }"#,
        )
        .unwrap();
        let scenario = Scenario::new(store.path(), cost_function, 1_000).unwrap();
        let algorithm = Algorithm::from_selection(AlgorithmType::Dijkstra, 8 * 1024 * 1024);
        let start = [ArrayIndex {
            i: 0,
            j: 0,
            option: 0,
        }];
        let end = [ArrayIndex {
            i: 0,
            j: 1,
            option: 1,
        }];

        let result = compute_route_attempt_result(&scenario, &algorithm, &start, &end);

        assert_eq!(result.len(), 1);
        assert_eq!(*result[0].total_cost(), 5.5);
    }

    #[test]
    fn compute_route_attempt_result_requires_matching_end_option() {
        let store = crate::dataset::samples::ZarrTestBuilder::new()
            .dimensions(1, 1, 2)
            .chunks(1, 1, 2)
            .layer(crate::dataset::samples::LayerConfig::custom(
                "overhead_cost",
                overhead_switch_cost,
            ))
            .layer(crate::dataset::samples::LayerConfig::custom(
                "underground_cost",
                underground_switch_cost,
            ))
            .layer(crate::dataset::samples::LayerConfig::custom(
                "overhead_hard_barrier",
                overhead_destination_barrier,
            ))
            .build()
            .unwrap();
        let cost_function = crate::cost::CostFunction::from_json(
            r#"{
                "routing_options": {
                    "overhead": {
                        "cost_layers": [{"layer_name": "overhead_cost"}],
                        "barrier_layers": [
                            {
                                "layer_name": "overhead_hard_barrier",
                                "barrier_operator": "eq",
                                "barrier_threshold": 1.0
                            }
                        ]
                    },
                    "underground": {
                        "cost_layers": [{"layer_name": "underground_cost"}]
                    }
                },
                "invalid_costs_block_routing": false
            }"#,
        )
        .unwrap();
        let scenario = Scenario::new(store.path(), cost_function, 1_000).unwrap();
        let algorithm = Algorithm::from_selection(AlgorithmType::Dijkstra, 8 * 1024 * 1024);
        let start = [ArrayIndex {
            i: 0,
            j: 0,
            option: 1,
        }];
        let end = [ArrayIndex {
            i: 0,
            j: 1,
            option: 1,
        }];

        let result = compute_route_attempt_result(&scenario, &algorithm, &start, &end);

        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].route(),
            &vec![
                ArrayIndex {
                    i: 0,
                    j: 0,
                    option: 1
                },
                ArrayIndex {
                    i: 0,
                    j: 1,
                    option: 1
                },
            ]
        );
        assert_eq!(*result[0].total_cost(), 4.0);
    }

    // fn any_option_start_cost(band: u64, _row: u64, _col: u64) -> f32 {
    //     if band == 0 { 1.0 } else { 2.0 }
    // }

    // fn first_option_blocked_everywhere(band: u64, _row: u64, _col: u64) -> f32 {
    //     if band == 0 { 1.0 } else { 0.0 }
    // }

    // #[test]
    // fn compute_route_attempt_result_can_start_on_any_allowed_option() {
    //     let store = crate::dataset::samples::ZarrTestBuilder::new()
    //         .dimensions(2, 1, 2)
    //         .chunks(2, 1, 2)
    //         .layer(crate::dataset::samples::LayerConfig::custom(
    //             "cost",
    //             any_option_start_cost,
    //         ))
    //         .layer(crate::dataset::samples::LayerConfig::custom(
    //             "hard_barrier",
    //             first_option_blocked_everywhere,
    //         ))
    //         .build()
    //         .unwrap();
    //     let cost_function = crate::cost::CostFunction::from_json(
    //         r#"{
    //             "routing_options": {
    //                 "overhead": {
    //                     "cost_layers": [{"layer_name": "cost"}],
    //                     "barrier_layers": [
    //                         {
    //                             "layer_name": "hard_barrier",
    //                             "barrier_operator": "eq",
    //                             "barrier_threshold": 1.0
    //                         }
    //                     ]
    //                 },
    //                 "underground": {
    //                     "cost_layers": [{"layer_name": "cost"}],
    //                     "barrier_layers": [
    //                         {
    //                             "layer_name": "hard_barrier",
    //                             "barrier_operator": "eq",
    //                             "barrier_threshold": 1.0
    //                         }
    //                     ]
    //                 }
    //             },
    //             "invalid_costs_block_routing": false
    //         }"#,
    //     )
    //     .unwrap();
    //     let scenario = Scenario::new(store.path(), cost_function, 1_000).unwrap();
    //     let algorithm = Algorithm::from_selection(AlgorithmType::Dijkstra, 8 * 1024 * 1024);
    //     let start = [ArrayIndex::new_ij(0, 0)];
    //     let end = [ArrayIndex::new_ij(0, 1)];

    //     let result = compute_route_attempt_result(&scenario, &algorithm, &start, &end);

    //     assert_eq!(result.len(), 1);
    //     assert_eq!(
    //         result[0].route(),
    //         &vec![
    //             ArrayIndex {
    //                 i: 0,
    //                 j: 0,
    //                 option: 1
    //             },
    //             ArrayIndex {
    //                 i: 0,
    //                 j: 1,
    //                 option: 1
    //             },
    //         ]
    //     );
    //     assert_eq!(*result[0].total_cost(), 2.0);
    // }

    // #[test]
    // fn compute_route_attempt_result_can_end_on_any_allowed_option() {
    //     let store = crate::dataset::samples::ZarrTestBuilder::new()
    //         .dimensions(2, 1, 2)
    //         .chunks(2, 1, 2)
    //         .layer(crate::dataset::samples::LayerConfig::custom(
    //             "cost",
    //             any_option_start_cost,
    //         ))
    //         .layer(crate::dataset::samples::LayerConfig::custom(
    //             "hard_barrier",
    //             first_option_blocked_everywhere,
    //         ))
    //         .build()
    //         .unwrap();
    //     let cost_function = crate::cost::CostFunction::from_json(
    //         r#"{
    //             "routing_options": {
    //                 "overhead": {
    //                     "cost_layers": [{"layer_name": "cost"}],
    //                     "barrier_layers": [
    //                         {
    //                             "layer_name": "hard_barrier",
    //                             "barrier_operator": "eq",
    //                             "barrier_threshold": 1.0
    //                         }
    //                     ]
    //                 },
    //                 "underground": {
    //                     "cost_layers": [{"layer_name": "cost"}],
    //                     "barrier_layers": [
    //                         {
    //                             "layer_name": "hard_barrier",
    //                             "barrier_operator": "eq",
    //                             "barrier_threshold": 1.0
    //                         }
    //                     ]
    //                 }
    //             },
    //             "invalid_costs_block_routing": false
    //         }"#,
    //     )
    //     .unwrap();
    //     let scenario = Scenario::new(store.path(), cost_function, 1_000).unwrap();
    //     let algorithm = Algorithm::from_selection(AlgorithmType::Dijkstra, 8 * 1024 * 1024);
    //     let start = [ArrayIndex {
    //         i: 0,
    //         j: 0,
    //         option: 1,
    //     }];
    //     let end = [ArrayIndex::new_ij(0, 1)];

    //     let result = compute_route_attempt_result(&scenario, &algorithm, &start, &end);

    //     assert_eq!(result.len(), 1);
    //     assert_eq!(
    //         result[0].route(),
    //         &vec![
    //             ArrayIndex {
    //                 i: 0,
    //                 j: 0,
    //                 option: 1
    //             },
    //             ArrayIndex {
    //                 i: 0,
    //                 j: 1,
    //                 option: 1
    //             },
    //         ]
    //     );
    //     assert_eq!(*result[0].total_cost(), 2.0);
    // }
}
