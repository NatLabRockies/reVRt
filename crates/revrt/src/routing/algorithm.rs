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

#[derive(Clone, Copy, Debug)]
pub(super) enum Algorithm {
    Astar,
    Dijkstra,
    LongRangeAstar { per_worker_memory_budget_bytes: u64 },
    LongRangeDijkstra { per_worker_memory_budget_bytes: u64 },
    BidirectionalLongRangeDijkstra { per_worker_memory_budget_bytes: u64 },
}

impl Algorithm {
    fn normalize_per_worker_memory_budget(
        algorithm_name: &str,
        per_worker_memory_budget_bytes: u64,
    ) -> u64 {
        let min_memory_budget_bytes = MIN_MEMORY_BUDGET_MB * 1024 * 1024;

        if per_worker_memory_budget_bytes < min_memory_budget_bytes {
            warn!(
                "{} per-worker memory budget smaller than the {}MB minimum! Setting to {}MB...",
                algorithm_name, MIN_MEMORY_BUDGET_MB, MIN_MEMORY_BUDGET_MB
            );
            min_memory_budget_bytes
        } else {
            debug!(
                "{} per-worker memory budget set to {}MB",
                algorithm_name,
                per_worker_memory_budget_bytes / (1024 * 1024)
            );
            per_worker_memory_budget_bytes
        }
    }

    pub(super) fn from_selection(
        algorithm: AlgorithmType,
        per_worker_memory_budget_bytes: u64,
    ) -> Self {
        match algorithm {
            AlgorithmType::Astar => Self::Astar,
            AlgorithmType::Dijkstra => Self::Dijkstra,
            AlgorithmType::LongRangeAstar => Self::LongRangeAstar {
                per_worker_memory_budget_bytes: Self::normalize_per_worker_memory_budget(
                    "Long-range A*",
                    per_worker_memory_budget_bytes,
                ),
            },
            AlgorithmType::LongRangeDijkstra => Self::LongRangeDijkstra {
                per_worker_memory_budget_bytes: Self::normalize_per_worker_memory_budget(
                    "Long-range Dijkstra",
                    per_worker_memory_budget_bytes,
                ),
            },
            AlgorithmType::BidirectionalLongRangeDijkstra => Self::BidirectionalLongRangeDijkstra {
                per_worker_memory_budget_bytes: Self::normalize_per_worker_memory_budget(
                    "Bidirectional long-range Dijkstra",
                    per_worker_memory_budget_bytes,
                ),
            },
        }
    }

    pub(super) fn compute<C, FN, IN, FS>(
        &self,
        start: &ArrayIndex,
        goals: &[ArrayIndex],
        successors: FN,
        success: FS,
        grid_shape: (u64, u64, u32),
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
        crate::profiling::scope("routing::Algorithm::compute");
        let ans = match *self {
            Self::Astar => {
                let min_cost = std::cell::Cell::new(None);
                let mut successors = successors;

                astar(
                    start,
                    |index| astar_successors(index, &mut successors, &min_cost),
                    |index| octile_heuristic(index, goals, &min_cost),
                    success,
                )
            }
            Self::Dijkstra => dijkstra(start, successors, success),
            Self::LongRangeAstar {
                per_worker_memory_budget_bytes,
            } => long_range::long_range_astar(
                start,
                goals,
                successors,
                success,
                per_worker_memory_budget_bytes,
                grid_shape,
            ),
            Self::LongRangeDijkstra {
                per_worker_memory_budget_bytes,
            } => long_range::long_range_dijkstra(
                start,
                successors,
                success,
                per_worker_memory_budget_bytes,
                grid_shape,
            ),
            Self::BidirectionalLongRangeDijkstra {
                per_worker_memory_budget_bytes,
            } => long_range::bidirectional_long_range_dijkstra(
                start,
                goals,
                successors,
                per_worker_memory_budget_bytes,
                grid_shape,
            ),
        };

        ans.map(|(route, total_cost)| {
            Solution::new(route, super::unscaled_cost(u64::from(total_cost)))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::{Algorithm, AlgorithmType, MIN_MEMORY_BUDGET_MB};
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

    #[test]
    fn long_range_algorithms_store_normalized_budget() {
        let algorithm = Algorithm::from_selection(AlgorithmType::LongRangeAstar, 1);

        assert!(matches!(
            algorithm,
            Algorithm::LongRangeAstar {
                per_worker_memory_budget_bytes,
            } if per_worker_memory_budget_bytes == MIN_MEMORY_BUDGET_MB * 1024 * 1024
        ));
    }
}
