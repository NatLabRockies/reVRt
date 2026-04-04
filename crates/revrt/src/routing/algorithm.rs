//! Algorithms to find optimal path
//!
//! A collection of different strategies to find optimal paths.
//! Common algorithms are based on the external crate `pathfinding`.

/*
 * pathfinding::dijkstra(start, successor, success)
 * pathfinding::astar(start, successor, heuristic, success)
 * pathfinding::dfs(start, successor, success)
 */

use num_traits::Zero;

use tracing::{debug, warn};

use super::bounded;
use crate::{ArrayIndex, Solution};

const MIN_MEMORY_BUDGET_MB: u64 = 2;

#[derive(Clone, Debug)]
/// Types of algorithms to determine optimal paths
pub(super) enum AlgorithmType {
    // Astar,
    // Dijkstra,
    LongRangeDijkstra,
}

#[derive(Debug)]
pub(super) struct Algorithm {
    algorithm_type: AlgorithmType,
    per_worker_memory_budget_bytes: Option<u64>,
}

#[allow(dead_code)]
/// Manhattan distance
///
/// For a given start point, calculates the shortest manhattan distance to a
/// collection of possible end points, i.e. assume that there are multiple
/// possible ends.
fn manhattan_distance(start: &ArrayIndex, end: &[ArrayIndex]) -> u64 {
    end.iter()
        .map(|end| {
            let di = start.i.abs_diff(end.i);
            let dj = start.j.abs_diff(end.j);
            di + dj
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

impl Algorithm {
    // pub(super) fn new() -> Self {
    //     Self {
    //         algorithm_type: AlgorithmType::Dijkstra,
    //         memory_budget_bytes: None,
    //         budget_coordinator: None,
    //     }
    // }

    pub(super) fn new_bounded(per_worker_memory_budget_bytes: u64) -> Self {
        if per_worker_memory_budget_bytes < MIN_MEMORY_BUDGET_MB * 1024 * 1024 {
            warn!(
                "Long-range Dijkstra per-worker memory budget smaller than the {}MB minimum! Setting to {}MB...",
                MIN_MEMORY_BUDGET_MB, MIN_MEMORY_BUDGET_MB
            );
            Self {
                algorithm_type: AlgorithmType::LongRangeDijkstra,
                per_worker_memory_budget_bytes: Some(MIN_MEMORY_BUDGET_MB * 1024 * 1024),
            }
        } else {
            debug!(
                "Long-range Dijkstra per-worker memory budget set to {}MB",
                per_worker_memory_budget_bytes / (1024 * 1024 * 1024)
            );
            Self {
                algorithm_type: AlgorithmType::LongRangeDijkstra,
                per_worker_memory_budget_bytes: Some(per_worker_memory_budget_bytes),
            }
        }
    }

    #[allow(unused_variables)]
    pub(super) fn compute<C, FN, IN, FH, FS>(
        &self,
        start: &ArrayIndex,
        successors: FN,
        heuristic: Option<FH>,
        success: FS,
        grid_shape: (u64, u64),
    ) -> Option<Solution<ArrayIndex, f32>>
    //) -> Option<Solution<I, C>>
    where
        // I: Eq + Hash + Clone,
        C: Zero + Ord + Copy + From<u64>,
        // I: From<(u64, u64)>,
        // (u64, u64): From<I>,
        // C: Zero + Ord + Copy + From<u64>,
        FN: FnMut(&ArrayIndex) -> IN,
        IN: IntoIterator<Item = (ArrayIndex, C)>,
        FH: FnMut(&ArrayIndex) -> C,
        FS: FnMut(&ArrayIndex) -> bool,
        // Temporary solution while we can't compare f32
        u64: From<C>,
    {
        let ans = match self.algorithm_type {
            // AlgorithmType::Dijkstra => pathfinding::prelude::dijkstra(start, successors, success),
            AlgorithmType::LongRangeDijkstra => {
                let per_worker_memory_budget_bytes = self
                    .per_worker_memory_budget_bytes
                    .expect("Memory budget not set for long-range Dijkstra");
                bounded::long_range_dijkstra(
                    start,
                    successors,
                    success,
                    per_worker_memory_budget_bytes,
                    grid_shape,
                )
            }
        };

        ans.map(|(route, total_cost)| {
            Solution::new(route, super::unscaled_cost(u64::from(total_cost)))
        })
    }
}
