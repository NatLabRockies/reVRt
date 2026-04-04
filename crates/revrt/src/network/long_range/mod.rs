pub(super) mod swap;
pub(super) mod utilities;

use std::collections::{BinaryHeap, HashMap};

use tracing::debug;

use super::cost::NodeCost;
use crate::ArrayIndex;
use swap::SwapStore;
use utilities::{FinalizedBits, GridIndexer};

const SPILL_BUFFER_ENTRY_BYTES: u64 = 24;
const MAX_PQ_TO_FRONTIER_NODE_RATIO: usize = 4;

#[derive(Clone, Debug)]
pub(crate) struct FinalizedNode {
    slot: usize,
    pub(crate) array_index: ArrayIndex,
    cost: u64,
}

impl FinalizedNode {
    fn route(&self, state: &mut FrontierOnlySearchState) -> Option<Vec<ArrayIndex>> {
        state.reconstruct_path_to(self.slot)
    }
}

#[derive(Debug)]
/// Tracks the mutable state for a search that spills finalized nodes to disk.
///
/// Frontier nodes stay in memory so the search can continue expanding the
/// cheapest known path, while finalized node costs and parent links are
/// written to the swap store for later path reconstruction.
pub(crate) struct FrontierOnlySearchState {
    grid: GridIndexer,
    // Frontier nodes ordered by estimated total cost.
    pq: BinaryHeap<NodeCost<usize, u64>>,
    // Cheapest known cost for each frontier node still tracked in memory.
    // Once a node is finalized, its cost is removed from this map and can
    // be recovered from the swap store during path reconstruction.
    best_node_costs: HashMap<usize, u64>,
    // Parent slot for each frontier node's current best path.
    parents: HashMap<usize, usize>,
    // Compact membership set for nodes whose cheapest path is finalized.
    finalized_bits: FinalizedBits,
    // Spill area for finalized node records needed to rebuild a route.
    swap: SwapStore,
    // Number of finalized nodes removed from the frontier.
    num_nodes_checked: usize,
}

impl FrontierOnlySearchState {
    pub(crate) fn new(
        start: &ArrayIndex,
        memory_budget_bytes: u64,
        grid_shape: (u64, u64),
    ) -> Option<Self> {
        let grid = GridIndexer::new(grid_shape.0, grid_shape.1)?;
        let start_slot = grid.slot_of(start)?;

        let mut state = Self {
            grid,
            pq: BinaryHeap::new(),
            best_node_costs: HashMap::new(),
            parents: HashMap::new(),
            finalized_bits: grid.new_finalized_bits()?,
            swap: SwapStore::new(spill_buffer_capacity(memory_budget_bytes)).ok()?,
            num_nodes_checked: 0,
        };

        state.best_node_costs.insert(start_slot, 0);
        state.pq.push(NodeCost {
            index: start_slot,
            cost: 0,
            estimated_cost: 0,
        });
        Some(state)
    }

    pub(crate) fn pop_next_node(&mut self) -> Option<FinalizedNode> {
        while let Some(NodeCost { index, cost, .. }) = self.pq.pop() {
            if self.finalized_bits.contains(index) {
                continue;
            }

            let Some(current_best) = self.best_node_costs.get(&index).copied() else {
                continue;
            };

            if cost != current_best {
                continue;
            }

            let array_index = self.grid.index_of(index);
            let parent_slot = self.parents.remove(&index);
            self.best_node_costs.remove(&index);

            self.finalized_bits.set(index);
            self.swap.write_slot(index, (cost, parent_slot)).ok()?;
            self.num_nodes_checked += 1;

            return Some(FinalizedNode {
                slot: index,
                array_index,
                cost,
            });
        }

        None
    }

    pub(crate) fn finalize_route(&mut self, node: FinalizedNode) -> Option<(Vec<ArrayIndex>, u64)> {
        let cost = node.cost;
        let route = node.route(self)?;
        Some((route, cost))
    }

    pub(crate) fn add_successors<C, IN>(
        &mut self,
        node: &FinalizedNode,
        successors: IN,
    ) -> Option<()>
    where
        IN: IntoIterator<Item = (ArrayIndex, C)>,
        C: Copy,
        u64: From<C>,
    {
        for (neighbor, edge_cost) in successors {
            self.add_neighbor(node.slot, node.cost, &neighbor, edge_cost);
        }
        self.enforce_memory_budget()
    }

    fn add_neighbor<C>(
        &mut self,
        from_slot: usize,
        from_cost: u64,
        neighbor: &ArrayIndex,
        edge_cost: C,
    ) where
        C: Copy,
        u64: From<C>,
    {
        let Some(neighbor_slot) = self.grid.slot_of(neighbor) else {
            return;
        };

        if self.finalized_bits.contains(neighbor_slot) {
            return;
        }

        let next_cost = from_cost.saturating_add(u64::from(edge_cost));
        let should_update = self
            .best_node_costs
            .get(&neighbor_slot)
            .map(|current_best_cost| next_cost < *current_best_cost)
            .unwrap_or(true);

        if should_update {
            self.best_node_costs.insert(neighbor_slot, next_cost);
            self.parents.insert(neighbor_slot, from_slot);
            self.pq.push(NodeCost {
                index: neighbor_slot,
                cost: next_cost,
                estimated_cost: next_cost,
            });
        }
    }

    pub(crate) fn enforce_memory_budget(&mut self) -> Option<()> {
        if self.pq.len() > self.best_node_costs.len() * MAX_PQ_TO_FRONTIER_NODE_RATIO {
            debug!(
                "Compacting priority queue after checking {} nodes",
                self.num_nodes_checked
            );
            self.pq = compact_pq_set(&self.best_node_costs);
            self.swap.flush().ok()?;
        }

        Some(())
    }

    fn reconstruct_path_to(&mut self, goal_slot: usize) -> Option<Vec<ArrayIndex>> {
        let mut path = Vec::new();
        let mut current_slot = Some(goal_slot);

        while let Some(slot) = current_slot {
            path.push(self.grid.index_of(slot));
            let (_, parent) = self.swap.read_slot(slot).ok()?;
            current_slot = parent;
        }

        path.reverse();
        Some(path)
    }
}

fn compact_pq_set(best_node_costs: &HashMap<usize, u64>) -> BinaryHeap<NodeCost<usize, u64>> {
    best_node_costs
        .iter()
        .map(|(index, cost)| NodeCost {
            index: *index,
            cost: *cost,
            estimated_cost: *cost,
        })
        .collect()
}

fn spill_buffer_capacity(memory_budget_bytes: u64) -> usize {
    let max_entries = memory_budget_bytes.saturating_sub(1) / SPILL_BUFFER_ENTRY_BYTES;
    let capped_entries = max_entries.min(usize::MAX as u64);

    if capped_entries == 0 {
        0
    } else {
        (1_u64 << capped_entries.ilog2()) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_to_goal(
        state: &mut FrontierOnlySearchState,
        goal: ArrayIndex,
        mut successors: impl FnMut(&ArrayIndex) -> Vec<(ArrayIndex, u64)>,
    ) -> Option<(Vec<ArrayIndex>, u64)> {
        while let Some(node) = state.pop_next_node() {
            if node.array_index == goal {
                return state.finalize_route(node);
            }

            state.add_successors(&node, successors(&node.array_index))?;
        }

        None
    }

    #[test]
    fn spill_buffer_capacity_is_power_of_two_under_budget() {
        assert_eq!(spill_buffer_capacity(24), 0);
        assert_eq!(spill_buffer_capacity(25), 1);
        assert_eq!(spill_buffer_capacity(48), 1);
        assert_eq!(spill_buffer_capacity(49), 2);
        assert_eq!(spill_buffer_capacity(1024), 32);
    }

    #[test]
    fn state_spills_when_pressure_exceeds_budget() {
        let start = ArrayIndex::new(10, 10);
        let goal = ArrayIndex::new(15, 15);
        let mut state = FrontierOnlySearchState::new(&start, 2_000, (31, 31)).unwrap();

        let ans = run_to_goal(&mut state, goal.clone(), |p| {
            let mut out = Vec::new();
            for di in -1_i64..=1 {
                for dj in -1_i64..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let ni = p.i as i64 + di;
                    let nj = p.j as i64 + dj;
                    if ni >= 0 && nj >= 0 && ni <= 30 && nj <= 30 {
                        out.push((ArrayIndex::new(ni as u64, nj as u64), 1_u64));
                    }
                }
            }
            out
        })
        .unwrap();

        assert_eq!(ans.0.first(), Some(&start));
        assert_eq!(ans.0.last(), Some(&goal));
    }

    #[test]
    fn state_returns_none_when_frontier_never_reaches_goal() {
        let start = ArrayIndex::new(0, 0);
        let mut state = FrontierOnlySearchState::new(&start, 2_000, (21, 21)).unwrap();

        let ans = run_to_goal(&mut state, ArrayIndex::new(99, 99), |p| {
            let mut out = Vec::new();
            if p.i < 20 {
                out.push((ArrayIndex::new(p.i + 1, p.j), 1_u64));
            }
            if p.j < 20 {
                out.push((ArrayIndex::new(p.i, p.j + 1), 1_u64));
            }
            out
        });

        assert!(ans.is_none());
    }
}
