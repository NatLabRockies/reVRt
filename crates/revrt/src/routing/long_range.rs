//! Long-range (memory-bounded) routing algorithms
//!
//! This module provides a Dijkstra implementation that keeps active
//! frontier state in memory and spills finalized nodes to a swap file.

use std::cell::Cell;

use num_traits::Zero;
use tracing::debug;

use super::astar::{astar_successors, octile_heuristic};
use crate::ArrayIndex;
use crate::network::long_range::{BidirectionalSearchState, FrontierOnlySearchState};

pub(super) fn long_range_astar<C, FN, IN, FS>(
    start: &ArrayIndex,
    goals: &[ArrayIndex],
    mut successors: FN,
    mut success: FS,
    memory_budget_bytes: u64,
    grid_shape: (u64, u64),
) -> Option<(Vec<ArrayIndex>, C)>
where
    C: Zero + Ord + Copy,
    FN: FnMut(&ArrayIndex) -> IN,
    IN: IntoIterator<Item = (ArrayIndex, C)>,
    FS: FnMut(&ArrayIndex) -> bool,
    u64: From<C>,
    C: From<u64>,
{
    debug!(
        "Starting long-range A* with memory budget of {} bytes",
        memory_budget_bytes
    );

    if goals.is_empty() {
        return None;
    }

    if goals.iter().any(|goal| goal == start) {
        return Some((vec![start.clone()], C::zero()));
    }

    let min_cost = Cell::new(None);
    let mut state = FrontierOnlySearchState::new(start, memory_budget_bytes, grid_shape)?;

    while let Some(node) = state.pop_next_node() {
        if success(&node.array_index) {
            debug!(
                "Goal node found terminating at index {:?}",
                &node.array_index
            );
            return state
                .finalize_route(node)
                .map(|(route, cost)| (route, C::from(cost)));
        }

        let neighbors = astar_successors(&node.array_index, &mut successors, &min_cost);
        state.add_successors_tracking_with_estimator(
            &node,
            neighbors,
            |neighbor, cost| {
                cost.saturating_add(u64::from(octile_heuristic::<C>(neighbor, goals, &min_cost)))
            },
            |_, _| {},
        )?;
    }

    None
}

pub(super) fn long_range_dijkstra<C, FN, IN, FS>(
    start: &ArrayIndex,
    mut successors: FN,
    mut success: FS,
    memory_budget_bytes: u64,
    grid_shape: (u64, u64),
) -> Option<(Vec<ArrayIndex>, C)>
where
    C: Zero + Ord + Copy,
    FN: FnMut(&ArrayIndex) -> IN,
    IN: IntoIterator<Item = (ArrayIndex, C)>,
    FS: FnMut(&ArrayIndex) -> bool,
    u64: From<C>,
    C: From<u64>,
{
    debug!(
        "Starting long-range Dijkstra with memory budget of {} bytes",
        memory_budget_bytes
    );
    let mut state = FrontierOnlySearchState::new(start, memory_budget_bytes, grid_shape)?;

    while let Some(node) = state.pop_next_node() {
        if success(&node.array_index) {
            debug!(
                "Goal node found terminating at index {:?}",
                &node.array_index
            );
            return state
                .finalize_route(node)
                .map(|(route, cost)| (route, C::from(cost)));
        }

        state.add_successors(&node, successors(&node.array_index))?;
    }

    None
}

pub(super) fn bidirectional_long_range_dijkstra<C, FN, IN>(
    start: &ArrayIndex,
    goals: &[ArrayIndex],
    successors: FN,
    memory_budget_bytes: u64,
    grid_shape: (u64, u64),
) -> Option<(Vec<ArrayIndex>, C)>
where
    C: Zero + Ord + Copy,
    FN: FnMut(&ArrayIndex) -> IN,
    IN: IntoIterator<Item = (ArrayIndex, C)>,
    u64: From<C>,
    C: From<u64>,
{
    debug!(
        "Starting bidirectional long-range Dijkstra with memory budget of {} bytes",
        memory_budget_bytes
    );

    if goals.is_empty() {
        return None;
    }

    if goals.iter().any(|goal| goal == start) {
        return Some((vec![start.clone()], C::zero()));
    }

    let mut state = BidirectionalSearchState::new(start, goals, memory_budget_bytes, grid_shape)?;

    state
        .run(successors)
        .map(|(route, cost)| (route, C::from(cost)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_astar_finds_shortest_path() {
        let start = ArrayIndex::new_ij(0, 0);
        let goal = ArrayIndex::new_ij(2, 2);

        let ans = long_range_astar(
            &start,
            std::slice::from_ref(&goal),
            |p: &ArrayIndex| {
                let mut out = Vec::new();
                if p.i < 2 {
                    out.push((ArrayIndex::new_ij(p.i + 1, p.j), 1_u64));
                }
                if p.j < 2 {
                    out.push((ArrayIndex::new_ij(p.i, p.j + 1), 1_u64));
                }
                out
            },
            |p| *p == goal,
            2 * 1024 * 1024,
            (3, 3, 1),
        )
        .unwrap();

        assert_eq!(ans.1, 4_u64);
        assert_eq!(ans.0.first(), Some(&start));
        assert_eq!(ans.0.last(), Some(&goal));
    }

    #[test]
    fn bounded_finds_shortest_path() {
        let start = ArrayIndex::new_ij(0, 0);
        let goal = ArrayIndex::new_ij(2, 2);

        let ans = long_range_dijkstra(
            &start,
            |p: &ArrayIndex| {
                let mut out = Vec::new();
                if p.i < 2 {
                    out.push((ArrayIndex::new_ij(p.i + 1, p.j), 1_u64));
                }
                if p.j < 2 {
                    out.push((ArrayIndex::new_ij(p.i, p.j + 1), 1_u64));
                }
                out
            },
            |p| *p == goal,
            2 * 1024 * 1024 * 1024,
            (3, 3, 1),
        )
        .unwrap();

        assert_eq!(ans.1, 4_u64);
        assert_eq!(ans.0.first(), Some(&start));
        assert_eq!(ans.0.last(), Some(&goal));
    }

    #[test]
    fn bounded_astar_rejects_missing_goals() {
        let start = ArrayIndex::new_ij(0, 0);

        let ans = long_range_astar(
            &start,
            &[],
            |_p: &ArrayIndex| Vec::<(ArrayIndex, u64)>::new(),
            |_p| false,
            2 * 1024 * 1024,
            (1, 1, 1),
        );

        assert!(ans.is_none());
    }

    #[test]
    fn bounded_astar_returns_zero_cost_when_start_is_goal() {
        let start = ArrayIndex::new_ij(0, 0);
        let goals = vec![start.clone(), ArrayIndex::new_ij(1, 1)];

        let ans = long_range_astar(
            &start,
            &goals,
            |_p: &ArrayIndex| Vec::<(ArrayIndex, u64)>::new(),
            |_p| false,
            2 * 1024 * 1024,
            (2, 2, 1),
        )
        .unwrap();

        assert_eq!(ans.0, vec![start]);
        assert_eq!(ans.1, 0_u64);
    }

    #[test]
    fn bounded_rejects_too_small_budget() {
        let start = ArrayIndex::new_ij(0, 0);

        let ans = long_range_dijkstra(
            &start,
            |_p: &ArrayIndex| Vec::<(ArrayIndex, u64)>::new(),
            |_p| false,
            1024,
            (1, 1, 1),
        );

        assert!(ans.is_none());
    }

    #[test]
    fn bidirectional_bounded_finds_shortest_path_to_any_goal() {
        let start = ArrayIndex::new_ij(0, 0);
        let goals = vec![ArrayIndex::new_ij(2, 2), ArrayIndex::new_ij(0, 2)];

        let ans = bidirectional_long_range_dijkstra(
            &start,
            &goals,
            |p: &ArrayIndex| {
                let mut out = Vec::new();
                if p.i < 2 {
                    out.push((ArrayIndex::new_ij(p.i + 1, p.j), 1_u64));
                }
                if p.j < 2 {
                    out.push((ArrayIndex::new_ij(p.i, p.j + 1), 1_u64));
                }
                out
            },
            2 * 1024 * 1024,
            (3, 3, 1),
        )
        .unwrap();

        assert_eq!(ans.1, 2_u64);
        assert_eq!(ans.0.first(), Some(&start));
        assert_eq!(ans.0.last(), Some(&ArrayIndex::new_ij(0, 2)));
    }

    #[test]
    fn bidirectional_bounded_rejects_missing_goals() {
        let start = ArrayIndex::new_ij(0, 0);

        let ans = bidirectional_long_range_dijkstra(
            &start,
            &[],
            |_p: &ArrayIndex| Vec::<(ArrayIndex, u64)>::new(),
            2 * 1024 * 1024,
            (1, 1, 1),
        );

        assert!(ans.is_none());
    }
}
