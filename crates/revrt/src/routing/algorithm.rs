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

use pathfinding::prelude::{astar, dijkstra};
use tracing::{debug, warn};

use super::astar::{astar_successors, octile_heuristic};
use super::long_range;
use crate::{ArrayIndex, Solution, error::Error, error::Result};

const MIN_MEMORY_BUDGET_MB: u64 = 2;

/// Types of algorithms to determine optimal paths
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AlgorithmType {
    Astar,
    Dijkstra,
    LongRangeAstar,
    LongRangeDijkstra,
    BidirectionalLongRangeDijkstra,
}

impl FromStr for AlgorithmType {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "astar" | "a_star" => Ok(Self::Astar),
            "dijkstra" => Ok(Self::Dijkstra),
            "long_range_astar" | "long_range_a_star" => Ok(Self::LongRangeAstar),
            "long_range_dijkstra" => Ok(Self::LongRangeDijkstra),
            "bidirectional_long_range_dijkstra" => Ok(Self::BidirectionalLongRangeDijkstra),
            _ => Err(Error::InvalidAlgorithm {
                value: value.to_string(),
                allowed: &[
                    "astar",
                    "dijkstra",
                    "long_range_astar",
                    "long_range_dijkstra",
                    "bidirectional_long_range_dijkstra",
                ],
            }),
        }
    }
}

#[derive(Debug)]
pub(super) struct Algorithm {
    algorithm_type: AlgorithmType,
    per_worker_memory_budget_bytes: Option<u64>,
}

impl Algorithm {
    pub(super) fn new_astar() -> Self {
        Self {
            algorithm_type: AlgorithmType::Astar,
            per_worker_memory_budget_bytes: None,
        }
    }

    pub(super) fn new() -> Self {
        Self {
            algorithm_type: AlgorithmType::Dijkstra,
            per_worker_memory_budget_bytes: None,
        }
    }

    pub(super) fn new_long_range_astar(per_worker_memory_budget_bytes: u64) -> Self {
        if per_worker_memory_budget_bytes < MIN_MEMORY_BUDGET_MB * 1024 * 1024 {
            warn!(
                "Long-range A* per-worker memory budget smaller than the {}MB minimum! Setting to {}MB...",
                MIN_MEMORY_BUDGET_MB, MIN_MEMORY_BUDGET_MB
            );
            Self {
                algorithm_type: AlgorithmType::LongRangeAstar,
                per_worker_memory_budget_bytes: Some(MIN_MEMORY_BUDGET_MB * 1024 * 1024),
            }
        } else {
            debug!(
                "Long-range A* per-worker memory budget set to {}MB",
                per_worker_memory_budget_bytes / (1024 * 1024)
            );
            Self {
                algorithm_type: AlgorithmType::LongRangeAstar,
                per_worker_memory_budget_bytes: Some(per_worker_memory_budget_bytes),
            }
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

    pub(super) fn new_bidirectional_long_range(per_worker_memory_budget_bytes: u64) -> Self {
        if per_worker_memory_budget_bytes < MIN_MEMORY_BUDGET_MB * 1024 * 1024 {
            warn!(
                "Bidirectional long-range Dijkstra per-worker memory budget smaller than the {}MB minimum! Setting to {}MB...",
                MIN_MEMORY_BUDGET_MB, MIN_MEMORY_BUDGET_MB
            );
            Self {
                algorithm_type: AlgorithmType::BidirectionalLongRangeDijkstra,
                per_worker_memory_budget_bytes: Some(MIN_MEMORY_BUDGET_MB * 1024 * 1024),
            }
        } else {
            debug!(
                "Bidirectional long-range Dijkstra per-worker memory budget set to {}MB",
                per_worker_memory_budget_bytes / (1024 * 1024)
            );
            Self {
                algorithm_type: AlgorithmType::BidirectionalLongRangeDijkstra,
                per_worker_memory_budget_bytes: Some(per_worker_memory_budget_bytes),
            }
        }
    }

    pub(super) fn from_selection(
        algorithm: AlgorithmType,
        per_worker_memory_budget_bytes: u64,
    ) -> Self {
        match algorithm {
            AlgorithmType::Astar => Self::new_astar(),
            AlgorithmType::Dijkstra => Self::new(),
            AlgorithmType::LongRangeAstar => {
                Self::new_long_range_astar(per_worker_memory_budget_bytes)
            }
            AlgorithmType::LongRangeDijkstra => {
                Self::new_long_range(per_worker_memory_budget_bytes)
            }
            AlgorithmType::BidirectionalLongRangeDijkstra => {
                Self::new_bidirectional_long_range(per_worker_memory_budget_bytes)
            }
        }
    }

    pub(super) fn compute<C, FN, IN, FS>(
        &self,
        start: &ArrayIndex,
        goals: &[ArrayIndex],
        successors: FN,
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
        FS: FnMut(&ArrayIndex) -> bool,
        // Temporary solution while we can't compare f32
        u64: From<C>,
    {
        let ans = match self.algorithm_type {
            AlgorithmType::Astar => {
                let min_cost = std::cell::Cell::new(None);
                let mut successors = successors;

                astar(
                    start,
                    |index| astar_successors(index, &mut successors, &min_cost),
                    |index| octile_heuristic(index, goals, &min_cost),
                    success,
                )
            }
            AlgorithmType::Dijkstra => dijkstra(start, successors, success),
            AlgorithmType::LongRangeAstar => {
                let per_worker_memory_budget_bytes = self
                    .per_worker_memory_budget_bytes
                    .expect("Memory budget not set for long-range A*");
                long_range::long_range_astar(
                    start,
                    goals,
                    successors,
                    success,
                    per_worker_memory_budget_bytes,
                    grid_shape,
                )
            }
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
            AlgorithmType::BidirectionalLongRangeDijkstra => {
                let per_worker_memory_budget_bytes = self
                    .per_worker_memory_budget_bytes
                    .expect("Memory budget not set for bidirectional long-range Dijkstra");
                long_range::bidirectional_long_range_dijkstra(
                    start,
                    goals,
                    successors,
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
            AlgorithmType::from_str("astar").unwrap(),
            AlgorithmType::Astar
        );
        assert_eq!(
            AlgorithmType::from_str("dijkstra").unwrap(),
            AlgorithmType::Dijkstra
        );
        assert_eq!(
            AlgorithmType::from_str("long_range_astar").unwrap(),
            AlgorithmType::LongRangeAstar
        );
        assert_eq!(
            AlgorithmType::from_str("long_range_dijkstra").unwrap(),
            AlgorithmType::LongRangeDijkstra
        );
        assert_eq!(
            AlgorithmType::from_str("bidirectional_long_range_dijkstra").unwrap(),
            AlgorithmType::BidirectionalLongRangeDijkstra
        );
    }

    #[test]
    fn normalizes_case_whitespace_and_hyphens() {
        assert_eq!(
            AlgorithmType::from_str(" A-STAR ").unwrap(),
            AlgorithmType::Astar
        );
        assert_eq!(
            AlgorithmType::from_str("  Dijkstra  ").unwrap(),
            AlgorithmType::Dijkstra
        );
        assert_eq!(
            AlgorithmType::from_str(" LONG-RANGE-ASTAR ").unwrap(),
            AlgorithmType::LongRangeAstar
        );
        assert_eq!(
            AlgorithmType::from_str(" LONG-RANGE-DIJKSTRA ").unwrap(),
            AlgorithmType::LongRangeDijkstra
        );
        assert_eq!(
            AlgorithmType::from_str(" bidirectional-long-range-dijkstra ").unwrap(),
            AlgorithmType::BidirectionalLongRangeDijkstra
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
