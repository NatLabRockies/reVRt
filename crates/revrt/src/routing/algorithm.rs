//! Algorithms to find optimal path
//!
//! A collection of different strategies to find optimal paths.
//! Common algorithms are based on the external crate `pathfinding`.

/*
 * pathfinding::dijkstra(start, successor, success)
 * pathfinding::astar(start, successor, heuristic, success)
 * pathfinding::dfs(start, successor, success)
 */

use std::str::FromStr;

use num_traits::Zero;

use pathfinding::prelude::dijkstra;
use tracing::{debug, warn};

use super::long_range;
use crate::{ArrayIndex, Solution, error::Error, error::Result};

const MIN_MEMORY_BUDGET_MB: u64 = 2;

/// Types of algorithms to determine optimal paths
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AlgorithmType {
    // Astar,
    Dijkstra,
    LongRangeDijkstra,
}

impl FromStr for AlgorithmType {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "dijkstra" => Ok(Self::Dijkstra),
            "long_range_dijkstra" => Ok(Self::LongRangeDijkstra),
            _ => Err(Error::InvalidAlgorithm {
                value: value.to_string(),
                allowed: &["dijkstra", "long_range_dijkstra"],
            }),
        }
    }
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
    pub(super) fn new() -> Self {
        Self {
            algorithm_type: AlgorithmType::Dijkstra,
            per_worker_memory_budget_bytes: None,
        }
    }

    pub(super) fn new_long_range(per_worker_memory_budget_bytes: u64) -> Self {
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
                per_worker_memory_budget_bytes / (1024 * 1024)
            );
            Self {
                algorithm_type: AlgorithmType::LongRangeDijkstra,
                per_worker_memory_budget_bytes: Some(per_worker_memory_budget_bytes),
            }
        }
    }

    pub(super) fn from_selection(
        algorithm: AlgorithmType,
        per_worker_memory_budget_bytes: u64,
    ) -> Self {
        match algorithm {
            AlgorithmType::Dijkstra => Self::new(),
            AlgorithmType::LongRangeDijkstra => {
                Self::new_long_range(per_worker_memory_budget_bytes)
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
            AlgorithmType::Dijkstra => dijkstra(start, successors, success),
            AlgorithmType::LongRangeDijkstra => {
                let per_worker_memory_budget_bytes = self
                    .per_worker_memory_budget_bytes
                    .expect("Memory budget not set for long-range Dijkstra");
                long_range::long_range_dijkstra(
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

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::AlgorithmType;
    use crate::error::Error;

    #[test]
    fn parses_supported_algorithm_names() {
        assert_eq!(
            AlgorithmType::from_str("dijkstra").unwrap(),
            AlgorithmType::Dijkstra
        );
        assert_eq!(
            AlgorithmType::from_str("long_range_dijkstra").unwrap(),
            AlgorithmType::LongRangeDijkstra
        );
    }

    #[test]
    fn normalizes_case_whitespace_and_hyphens() {
        assert_eq!(
            AlgorithmType::from_str("  Dijkstra  ").unwrap(),
            AlgorithmType::Dijkstra
        );
        assert_eq!(
            AlgorithmType::from_str(" LONG-RANGE-DIJKSTRA ").unwrap(),
            AlgorithmType::LongRangeDijkstra
        );
    }

    #[test]
    fn reports_invalid_algorithm_with_original_value() {
        let invalid_value = " long range ";
        let error = AlgorithmType::from_str(invalid_value).unwrap_err();

        assert!(matches!(
            error,
            Error::InvalidAlgorithm {
                value,
                ..
            } if value == invalid_value
        ));
    }
}
