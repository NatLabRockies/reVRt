//! Routing module

mod algorithm;
mod features;
mod long_range;
mod scenario;

use std::collections::HashSet;
use std::str::FromStr;
use std::sync::{Arc, mpsc};

use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use tracing::debug;

use crate::{ArrayIndex, RevrtRoutingSolutions, Solution, error::Result};
use algorithm::Algorithm;
use algorithm::AlgorithmType;
use features::Features;
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
