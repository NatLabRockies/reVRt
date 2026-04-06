//! Long-range (memory-bounded) routing algorithms
//!
//! This module provides a Dijkstra implementation that keeps active
//! frontier state in memory and spills finalized nodes to a swap file.

use num_traits::Zero;
use tracing::debug;

use crate::ArrayIndex;
use crate::network::long_range::FrontierOnlySearchState;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_finds_shortest_path() {
        let start = ArrayIndex::new(0, 0);
        let goal = ArrayIndex::new(2, 2);

        let ans = long_range_dijkstra(
            &start,
            |p: &ArrayIndex| {
                let mut out = Vec::new();
                if p.i < 2 {
                    out.push((ArrayIndex::new(p.i + 1, p.j), 1_u64));
                }
                if p.j < 2 {
                    out.push((ArrayIndex::new(p.i, p.j + 1), 1_u64));
                }
                out
            },
            |p| *p == goal,
            2 * 1024 * 1024 * 1024,
            (3, 3),
        )
        .unwrap();

        assert_eq!(ans.1, 4_u64);
        assert_eq!(ans.0.first(), Some(&start));
        assert_eq!(ans.0.last(), Some(&goal));
    }

    #[test]
    fn bounded_rejects_too_small_budget() {
        let start = ArrayIndex::new(0, 0);

        let ans = long_range_dijkstra(
            &start,
            |_p: &ArrayIndex| Vec::<(ArrayIndex, u64)>::new(),
            |_p| false,
            1024,
            (1, 1),
        );

        assert!(ans.is_none());
    }
}
