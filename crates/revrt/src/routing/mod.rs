//! Routing module

mod algorithm;
mod bounded;
mod features;
mod scenario;

use std::collections::HashSet;
use std::sync::{Arc, mpsc};

use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use tracing::debug;

use crate::{ArrayIndex, RevrtRoutingSolutions, Solution, error::Result};
use algorithm::Algorithm;
use features::Features;
use scenario::Scenario;

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
        let grid_shape = self.scenario.grid_shape();

        let solution: Vec<Solution<ArrayIndex, f32>> = start
            .into_par_iter()
            .filter_map(|s| {
                self.algorithm.compute(
                    s,
                    |p| self.scenario.successors(p),
                    None::<fn(&ArrayIndex) -> u64>,
                    |p| end.contains(p),
                    grid_shape,
                )
            })
            .collect();

        solution.into_iter()
    }

    pub(super) fn new<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        cache_size: u64,
    ) -> Result<Self> {
        let scenario = Scenario::new(store_path, cost_function, cache_size)?;

        // let algorithm = Algorithm::new();
        let algorithm =
            Algorithm::new_bounded(per_rayon_worker_memory_budget(4 * 1024 * 1024 * 1024));

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
        cache_size: u64,
    ) -> Result<Self> {
        let scenario = Scenario::new(store_path, cost_function, cache_size)?;
        Ok(Self {
            scenario: Arc::new(scenario),
            algorithm: Arc::new(Algorithm::new_bounded(per_rayon_worker_memory_budget(
                75 * 1024 * 1024 * 1024,
            ))),
            // algorithm: Arc::new(Algorithm::new()),
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
                    let routes: RevrtRoutingSolutions = start_inds
                        .into_par_iter()
                        .filter_map(|s| {
                            algorithm.compute(
                                &s,
                                |p| scenario.successors(p),
                                None::<fn(&ArrayIndex) -> u64>,
                                |p| end_inds.contains(p),
                                grid_shape,
                            )
                            // pathfinding::prelude::dijkstra(
                            //     &s,
                            //     |p| scenario.successors(p),
                            //     |p| end_inds.contains(p),
                            // )
                        })
                        // .map(|(route, total_cost)| Solution::new(route, unscaled_cost(total_cost)))
                        .collect();
                    let num_routes = routes.len();
                    debug!("Finished computing {num_routes} to {end_inds:?}");
                    sender.send((route_id, routes))
                },
            );
        });
    }
}

const PRECISION_SCALAR: f32 = 1e4;
fn cost_as_u64(cost: f32) -> u64 {
    let cost = cost * PRECISION_SCALAR;
    cost as u64
}

fn unscaled_cost(cost: u64) -> f32 {
    (cost as f32) / PRECISION_SCALAR
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
