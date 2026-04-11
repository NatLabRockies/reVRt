pub(super) mod swap;
pub(super) mod utilities;

use std::collections::{BinaryHeap, HashMap, HashSet};

use tracing::debug;

use super::cost::NodeCost;
use crate::ArrayIndex;
use swap::SwapStore;
use utilities::{FinalizedBits, GridIndexer};

const MAX_PQ_TO_FRONTIER_NODE_RATIO: usize = 2;

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
    roots: HashSet<usize>,
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
        Self::new_many(std::slice::from_ref(start), memory_budget_bytes, grid_shape)
    }

    pub(crate) fn new_many(
        starts: &[ArrayIndex],
        memory_budget_bytes: u64,
        grid_shape: (u64, u64),
    ) -> Option<Self> {
        let grid = GridIndexer::new(grid_shape.0, grid_shape.1)?;

        let mut state = Self {
            grid,
            roots: HashSet::new(),
            pq: BinaryHeap::new(),
            best_node_costs: HashMap::new(),
            parents: HashMap::new(),
            finalized_bits: grid.new_finalized_bits()?,
            swap: SwapStore::new(spill_buffer_capacity(memory_budget_bytes)).ok()?,
            num_nodes_checked: 0,
        };

        for start in starts {
            let Some(start_slot) = state.grid.slot_of(start) else {
                continue;
            };

            if !state.roots.insert(start_slot) {
                continue;
            }

            state.best_node_costs.insert(start_slot, 0);
            state.pq.push(NodeCost {
                index: start_slot,
                cost: 0,
                estimated_cost: 0,
            });
        }

        if state.roots.is_empty() {
            None
        } else {
            Some(state)
        }
    }

    pub(crate) fn pop_next_node(&mut self) -> Option<FinalizedNode> {
        while let Some(NodeCost { index, cost, .. }) = self.pq.pop() {
            if self.finalized_bits.contains(index) {
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

    pub(crate) fn has_frontier(&mut self) -> bool {
        self.peek_next_cost().is_some()
    }

    pub(crate) fn peek_next_cost(&mut self) -> Option<u64> {
        while let Some(candidate) = self.pq.peek() {
            if self.finalized_bits.contains(candidate.index) {
                self.pq.pop();
                continue;
            }

            return Some(candidate.cost);
        }

        None
    }

    pub(crate) fn known_cost(&mut self, slot: usize) -> Option<u64> {
        if let Some(cost) = self.best_node_costs.get(&slot).copied() {
            return Some(cost);
        }

        if self.finalized_bits.contains(slot) {
            return self.swap.read_slot(slot).ok().map(|(cost, _)| cost);
        }

        None
    }

    pub(crate) fn reconstruct_known_path_to(
        &mut self,
        goal_slot: usize,
    ) -> Option<Vec<ArrayIndex>> {
        let mut path = Vec::new();
        let mut current_slot = Some(goal_slot);

        while let Some(slot) = current_slot {
            path.push(self.grid.index_of(slot));
            if self.roots.contains(&slot) {
                break;
            }

            if let Some(parent) = self.parents.get(&slot).copied() {
                current_slot = Some(parent);
                continue;
            }

            if self.finalized_bits.contains(slot) {
                let (_, parent) = self.swap.read_slot(slot).ok()?;
                current_slot = parent;
                continue;
            }

            return None;
        }

        path.reverse();
        Some(path)
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
        self.add_successors_tracking(node, successors, |_, _| {})
    }

    pub(crate) fn add_successors_tracking<C, IN, F>(
        &mut self,
        node: &FinalizedNode,
        successors: IN,
        mut on_update: F,
    ) -> Option<()>
    where
        IN: IntoIterator<Item = (ArrayIndex, C)>,
        C: Copy,
        F: FnMut(usize, u64),
        u64: From<C>,
    {
        for (neighbor, edge_cost) in successors {
            if let Some((slot, cost)) =
                self.add_neighbor(node.slot, node.cost, &neighbor, edge_cost)
            {
                on_update(slot, cost);
            }
        }
        self.enforce_memory_budget()
    }

    fn add_neighbor<C>(
        &mut self,
        from_slot: usize,
        from_cost: u64,
        neighbor: &ArrayIndex,
        edge_cost: C,
    ) -> Option<(usize, u64)>
    where
        C: Copy,
        u64: From<C>,
    {
        let neighbor_slot = self.grid.slot_of(neighbor)?;

        if self.finalized_bits.contains(neighbor_slot) {
            return None;
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
            return Some((neighbor_slot, next_cost));
        }

        None
    }

    pub(crate) fn enforce_memory_budget(&mut self) -> Option<()> {
        if self.pq.len() > self.best_node_costs.len() * MAX_PQ_TO_FRONTIER_NODE_RATIO {
            debug!(
                "Compacting priority queue after checking {} nodes: PQ len {}, best_node_costs len {}",
                self.num_nodes_checked,
                self.pq.len(),
                self.best_node_costs.len()
            );
            self.pq = compact_pq_set(&self.best_node_costs);
            debug!("PQ len end: {}", self.pq.len());
            self.swap.flush().ok()?;
        }

        Some(())
    }

    fn reconstruct_path_to(&mut self, goal_slot: usize) -> Option<Vec<ArrayIndex>> {
        self.reconstruct_known_path_to(goal_slot)
    }
}

#[derive(Debug)]
pub(crate) struct BidirectionalSearchState {
    forward: FrontierOnlySearchState,
    backward: FrontierOnlySearchState,
    best_path: Option<(usize, u64)>,
}

impl BidirectionalSearchState {
    pub(crate) fn new(
        start: &ArrayIndex,
        goals: &[ArrayIndex],
        memory_budget_bytes: u64,
        grid_shape: (u64, u64),
    ) -> Option<Self> {
        let per_direction_budget = (memory_budget_bytes / 2).max(1);

        Some(Self {
            forward: FrontierOnlySearchState::new(start, per_direction_budget, grid_shape)?,
            backward: FrontierOnlySearchState::new_many(goals, per_direction_budget, grid_shape)?,
            best_path: None,
        })
    }

    pub(crate) fn run<C, FN, IN>(&mut self, mut successors: FN) -> Option<(Vec<ArrayIndex>, u64)>
    where
        FN: FnMut(&ArrayIndex) -> IN,
        IN: IntoIterator<Item = (ArrayIndex, C)>,
        C: Copy,
        u64: From<C>,
    {
        while self.forward.has_frontier() && self.backward.has_frontier() {
            if self.should_terminate() {
                break;
            }

            let expand_forward = match (
                self.forward.peek_next_cost(),
                self.backward.peek_next_cost(),
            ) {
                (Some(forward_cost), Some(backward_cost)) => forward_cost <= backward_cost,
                _ => break,
            };

            self.expand_direction(expand_forward, &mut successors)?;
        }

        let (meeting_slot, cost) = self.best_path?;
        let route = self.reconstruct_route(meeting_slot)?;
        Some((route, cost))
    }

    fn expand_direction<C, FN, IN>(
        &mut self,
        expand_forward: bool,
        successors: &mut FN,
    ) -> Option<()>
    where
        FN: FnMut(&ArrayIndex) -> IN,
        IN: IntoIterator<Item = (ArrayIndex, C)>,
        C: Copy,
        u64: From<C>,
    {
        let mut candidates = Vec::new();

        {
            let (current, other) = if expand_forward {
                (&mut self.forward, &mut self.backward)
            } else {
                (&mut self.backward, &mut self.forward)
            };

            let node = match current.pop_next_node() {
                Some(node) => node,
                None => return Some(()),
            };

            if let Some(other_cost) = other.known_cost(node.slot) {
                candidates.push((node.slot, node.cost.saturating_add(other_cost)));
            }

            current.add_successors_tracking(
                &node,
                successors(&node.array_index),
                |slot, cost| {
                    if let Some(other_cost) = other.known_cost(slot) {
                        candidates.push((slot, cost.saturating_add(other_cost)));
                    }
                },
            )?;
        }

        for (slot, total_cost) in candidates {
            self.record_candidate(slot, total_cost);
        }

        Some(())
    }

    fn record_candidate(&mut self, slot: usize, total_cost: u64) {
        match self.best_path {
            Some((_, best_cost)) if best_cost <= total_cost => {}
            _ => self.best_path = Some((slot, total_cost)),
        }
    }

    fn should_terminate(&mut self) -> bool {
        let Some((_, best_cost)) = self.best_path else {
            return false;
        };

        let Some(forward_cost) = self.forward.peek_next_cost() else {
            return true;
        };
        let Some(backward_cost) = self.backward.peek_next_cost() else {
            return true;
        };

        forward_cost.saturating_add(backward_cost) >= best_cost
    }

    fn reconstruct_route(&mut self, meeting_slot: usize) -> Option<Vec<ArrayIndex>> {
        let mut forward = self.forward.reconstruct_known_path_to(meeting_slot)?;
        let mut backward = self.backward.reconstruct_known_path_to(meeting_slot)?;
        backward.reverse();
        forward.extend(backward.into_iter().skip(1));
        Some(forward)
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
    let conservative_record_bytes = SwapStore::SPILL_RECORD_BYTES + 8; // Add 8 bytes for HashMap overhead per entry
    let max_entries = memory_budget_bytes.saturating_sub(1) / conservative_record_bytes;
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

    #[test]
    fn multi_source_state_skips_invalid_roots() {
        let starts = vec![
            ArrayIndex::new(0, 0),
            ArrayIndex::new(0, 0),
            ArrayIndex::new(999, 999),
        ];
        let mut state = FrontierOnlySearchState::new_many(&starts, 2_000, (3, 3)).unwrap();

        let node = state.pop_next_node().unwrap();

        assert_eq!(node.array_index, ArrayIndex::new(0, 0));
        assert!(state.pop_next_node().is_none());
    }

    #[test]
    fn reconstruct_known_path_works_for_frontier_nodes() {
        let start = ArrayIndex::new(0, 0);
        let mut state = FrontierOnlySearchState::new(&start, 2_000, (3, 3)).unwrap();
        let node = state.pop_next_node().unwrap();
        state
            .add_successors(&node, vec![(ArrayIndex::new(0, 1), 1_u64)])
            .unwrap();

        let slot = state.grid.slot_of(&ArrayIndex::new(0, 1)).unwrap();
        let path = state.reconstruct_known_path_to(slot).unwrap();

        assert_eq!(path, vec![ArrayIndex::new(0, 0), ArrayIndex::new(0, 1)]);
    }

    #[test]
    fn bidirectional_search_merges_forward_and_backward_paths() {
        let start = ArrayIndex::new(0, 0);
        let goals = vec![ArrayIndex::new(2, 2)];
        let mut state = BidirectionalSearchState::new(&start, &goals, 2_000, (3, 3)).unwrap();

        let (route, cost) = state
            .run(|p| {
                let mut out = Vec::new();
                if p.i > 0 {
                    out.push((ArrayIndex::new(p.i - 1, p.j), 1_u64));
                }
                if p.i < 2 {
                    out.push((ArrayIndex::new(p.i + 1, p.j), 1_u64));
                }
                if p.j > 0 {
                    out.push((ArrayIndex::new(p.i, p.j - 1), 1_u64));
                }
                if p.j < 2 {
                    out.push((ArrayIndex::new(p.i, p.j + 1), 1_u64));
                }
                out
            })
            .unwrap();

        assert_eq!(cost, 4);
        assert_eq!(route.first(), Some(&start));
        assert_eq!(route.last(), Some(&ArrayIndex::new(2, 2)));
    }
}
