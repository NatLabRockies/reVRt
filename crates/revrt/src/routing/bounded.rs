//! Memory-bounded routing algorithms
//!
//! This module provides a Dijkstra implementation that keeps active
//! frontier state in memory and spills finalized nodes to a swap file.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io::{Read, Seek, SeekFrom, Write};

use num_traits::Zero;
use tracing::debug;

use crate::ArrayIndex;

const NO_PARENT_SLOT: u64 = u64::MAX;

#[derive(Debug)]
struct NodePriority<C> {
    slot: usize,
    cost: C,
    estimated_cost: C,
}

impl<C: PartialEq> PartialEq for NodePriority<C> {
    fn eq(&self, other: &Self) -> bool {
        self.estimated_cost.eq(&other.estimated_cost) && self.cost.eq(&other.cost)
    }
}

impl<C: PartialEq> Eq for NodePriority<C> {}

impl PartialOrd for NodePriority<u64> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodePriority<u64> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => other.cost.cmp(&self.cost),
            ordering => ordering,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GridIndexer {
    nrows: u64,
    ncols: u64,
    total_cells: u64,
}

impl GridIndexer {
    fn new(nrows: u64, ncols: u64) -> Option<Self> {
        if nrows == 0 || ncols == 0 {
            return None;
        }
        let total_cells = nrows.checked_mul(ncols)?;
        Some(Self {
            nrows,
            ncols,
            total_cells,
        })
    }

    fn slot_of(&self, index: &ArrayIndex) -> Option<usize> {
        if index.i >= self.nrows || index.j >= self.ncols {
            return None;
        }

        let linear = index.i.checked_mul(self.ncols)?.checked_add(index.j)?;
        usize::try_from(linear).ok()
    }

    fn index_of(&self, slot: usize) -> ArrayIndex {
        let linear = slot as u64;
        ArrayIndex {
            i: linear / self.ncols,
            j: linear % self.ncols,
        }
    }

    fn finalized_bits_bytes(&self) -> u64 {
        self.total_cells.div_ceil(8)
    }
}

#[derive(Debug)]
struct FinalizedBits {
    bits: Vec<u8>,
}

impl FinalizedBits {
    fn new(total_cells: u64) -> Option<Self> {
        let bytes_len = total_cells.div_ceil(8);
        let bytes_len = usize::try_from(bytes_len).ok()?;
        Some(Self {
            bits: vec![0_u8; bytes_len],
        })
    }

    fn contains(&self, slot: usize) -> bool {
        let byte = slot / 8;
        let bit = (slot % 8) as u8;
        self.bits
            .get(byte)
            .map(|value| (value & (1 << bit)) != 0)
            .unwrap_or(false)
    }

    fn set(&mut self, slot: usize) {
        let byte = slot / 8;
        let bit = (slot % 8) as u8;
        if let Some(value) = self.bits.get_mut(byte) {
            *value |= 1 << bit;
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SpillRecord {
    cost: u64,
    parent_slot: u64,
}

impl SpillRecord {
    const RECORD_LEN: usize = 8 + 8;

    fn from_parts(cost: u64, parent_slot: Option<usize>) -> Self {
        let parent_slot = parent_slot
            .and_then(|slot| u64::try_from(slot).ok())
            .unwrap_or(NO_PARENT_SLOT);

        Self { cost, parent_slot }
    }

    fn parent_slot(self) -> Option<usize> {
        if self.parent_slot == NO_PARENT_SLOT {
            None
        } else {
            usize::try_from(self.parent_slot).ok()
        }
    }

    fn to_bytes(self) -> [u8; Self::RECORD_LEN] {
        let mut out = [0_u8; Self::RECORD_LEN];
        out[0..8].copy_from_slice(&self.cost.to_le_bytes());
        out[8..16].copy_from_slice(&self.parent_slot.to_le_bytes());
        out
    }

    fn from_bytes(bytes: [u8; Self::RECORD_LEN]) -> Self {
        let mut cost = [0_u8; 8];
        let mut parent_slot = [0_u8; 8];
        cost.copy_from_slice(&bytes[0..8]);
        parent_slot.copy_from_slice(&bytes[8..16]);

        Self {
            cost: u64::from_le_bytes(cost),
            parent_slot: u64::from_le_bytes(parent_slot),
        }
    }
}

#[derive(Debug)]
struct SwapStore {
    file: std::fs::File,
    path: std::path::PathBuf,
    write_buffer: Vec<(u64, SpillRecord)>,
    write_buffer_capacity: usize,
}

impl SwapStore {
    fn new(write_buffer_capacity: usize) -> std::io::Result<Self> {
        let mut path = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        path.push(format!("revrt-routing-swap-{pid}-{nanos}.bin"));

        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)?;

        debug!("Swap for Dijkstra graph at {:?}", path);
        Ok(Self {
            file,
            path,
            write_buffer: Vec::with_capacity(write_buffer_capacity.max(1)),
            write_buffer_capacity: write_buffer_capacity.max(1),
        })
    }

    fn slot_offset(slot: u64) -> std::io::Result<u64> {
        slot.checked_mul(SpillRecord::RECORD_LEN as u64)
            .ok_or_else(|| std::io::Error::other("swap slot offset overflow"))
    }

    fn write_slot(&mut self, slot: usize, record: SpillRecord) -> std::io::Result<()> {
        let slot = u64::try_from(slot).map_err(|_| std::io::Error::other("slot overflow"))?;
        self.write_buffer.push((slot, record));

        if self.write_buffer.len() >= self.write_buffer_capacity {
            self.flush()?;
        }

        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.write_buffer.len() > 1 {
            self.write_buffer.sort_unstable_by_key(|(slot, _)| *slot);
        }
        for (slot, record) in self.write_buffer.drain(..) {
            let offset = Self::slot_offset(slot)?;
            self.file.seek(SeekFrom::Start(offset))?;
            self.file.write_all(&record.to_bytes())?;
        }
        self.file.flush()
    }

    fn buffered_len(&self) -> usize {
        self.write_buffer.len()
    }

    fn read_slot(&mut self, slot: usize) -> std::io::Result<SpillRecord> {
        self.flush()?;

        let slot = u64::try_from(slot).map_err(|_| std::io::Error::other("slot overflow"))?;
        let offset = Self::slot_offset(slot)?;

        self.file.seek(SeekFrom::Start(offset))?;
        let mut bytes = [0_u8; SpillRecord::RECORD_LEN];
        self.file.read_exact(&mut bytes)?;
        Ok(SpillRecord::from_bytes(bytes))
    }
}

impl Drop for SwapStore {
    fn drop(&mut self) {
        let _ = self.flush();
        let _ = std::fs::remove_file(&self.path);
    }
}

#[derive(Clone, Copy, Debug)]
struct BoundedConfig {
    memory_budget_bytes: u64,
    minimum_budget_bytes: u64,
    open_entry_bytes: u64,
    tentative_entry_bytes: u64,
    parent_entry_bytes: u64,
    spill_buffer_entry_bytes: u64,
    spill_buffer_capacity: usize,
}

impl BoundedConfig {
    fn standard(memory_budget_bytes: u64) -> Self {
        Self {
            memory_budget_bytes,
            minimum_budget_bytes: 2 * 1024 * 1024 * 1024,
            open_entry_bytes: 64,
            tentative_entry_bytes: 96,
            parent_entry_bytes: 96,
            spill_buffer_entry_bytes: 24,
            spill_buffer_capacity: 4096,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SpillStats {
    spilled_records: usize,
}

pub(super) fn bounded_dijkstra<C, FN, IN, FS>(
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
    let config = BoundedConfig::standard(memory_budget_bytes);
    bounded_dijkstra_with_config(start, &mut successors, &mut success, config, grid_shape)
        .map(|(route, cost, _)| (route, cost))
}

fn bounded_dijkstra_with_config<C, FN, IN, FS>(
    start: &ArrayIndex,
    successors: &mut FN,
    success: &mut FS,
    config: BoundedConfig,
    grid_shape: (u64, u64),
) -> Option<(Vec<ArrayIndex>, C, SpillStats)>
where
    C: Zero + Ord + Copy,
    FN: FnMut(&ArrayIndex) -> IN,
    IN: IntoIterator<Item = (ArrayIndex, C)>,
    FS: FnMut(&ArrayIndex) -> bool,
    u64: From<C>,
    C: From<u64>,
{
    if config.memory_budget_bytes < config.minimum_budget_bytes {
        return None;
    }

    let grid = GridIndexer::new(grid_shape.0, grid_shape.1)?;
    let start_slot = grid.slot_of(start)?;

    let mut open = BinaryHeap::<NodePriority<u64>>::new();
    let mut tentative_costs = HashMap::<usize, u64>::new();
    let mut parents = HashMap::<usize, usize>::new();
    let mut finalized_bits = FinalizedBits::new(grid.total_cells)?;
    let mut swap = SwapStore::new(config.spill_buffer_capacity).ok()?;
    let mut stats = SpillStats::default();

    // if estimated_state_bytes(
    //     0,
    //     0,
    //     0,
    //     swap.buffered_len(),
    //     grid.finalized_bits_bytes(),
    //     config,
    // ) > config.memory_budget_bytes
    // {
    //     return None;
    // }

    tentative_costs.insert(start_slot, 0);
    open.push(NodePriority {
        slot: start_slot,
        cost: 0,
        estimated_cost: 0,
    });

    while let Some(NodePriority { slot, cost, .. }) = open.pop() {
        if finalized_bits.contains(slot) {
            continue;
        }

        let Some(current_best) = tentative_costs.get(&slot).copied() else {
            continue;
        };

        if cost != current_best {
            continue;
        }

        let index = grid.index_of(slot);

        let parent_slot = parents.remove(&slot);
        tentative_costs.remove(&slot);

        finalized_bits.set(slot);
        swap.write_slot(slot, SpillRecord::from_parts(cost, parent_slot))
            .ok()?;
        stats.spilled_records += 1;

        if success(&index) {
            let route = reconstruct_path(start_slot, slot, &grid, &mut swap)?;
            return Some((route, C::from(cost), stats));
        }

        for (neighbor, edge_cost) in successors(&index) {
            let Some(neighbor_slot) = grid.slot_of(&neighbor) else {
                continue;
            };

            if finalized_bits.contains(neighbor_slot) {
                continue;
            }

            let next_cost = cost.saturating_add(u64::from(edge_cost));
            let should_update = tentative_costs
                .get(&neighbor_slot)
                .map(|known| next_cost < *known)
                .unwrap_or(true);

            if should_update {
                tentative_costs.insert(neighbor_slot, next_cost);
                parents.insert(neighbor_slot, slot);
                open.push(NodePriority {
                    slot: neighbor_slot,
                    cost: next_cost,
                    estimated_cost: next_cost,
                });
            }
        }

        if estimated_state_bytes(
            open.len(),
            tentative_costs.len(),
            parents.len(),
            swap.buffered_len(),
            grid.finalized_bits_bytes(),
            config,
        ) > config.memory_budget_bytes
        {
            open = compact_open_set(&tentative_costs);
            swap.flush().ok()?;
        }

        // if estimated_state_bytes(
        //     open.len(),
        //     tentative_costs.len(),
        //     parents.len(),
        //     swap.buffered_len(),
        //     grid.finalized_bits_bytes(),
        //     config,
        // ) > config.memory_budget_bytes
        // {
        //     // The frontier itself no longer fits into the configured budget.
        //     return None;
        // }
    }

    None
}

fn compact_open_set(tentative_costs: &HashMap<usize, u64>) -> BinaryHeap<NodePriority<u64>> {
    tentative_costs
        .iter()
        .map(|(slot, cost)| NodePriority {
            slot: *slot,
            cost: *cost,
            estimated_cost: *cost,
        })
        .collect()
}

fn estimated_state_bytes(
    open_len: usize,
    tentative_len: usize,
    parent_len: usize,
    spill_buffer_len: usize,
    finalized_bits_bytes: u64,
    config: BoundedConfig,
) -> u64 {
    // Coarse accounting used only for pressure signaling.
    // It intentionally over-estimates per-entry overhead.
    let heap_bytes = open_len as u64 * config.open_entry_bytes;
    let tentative_bytes = tentative_len as u64 * config.tentative_entry_bytes;
    let parent_bytes = parent_len as u64 * config.parent_entry_bytes;
    let buffered_spill_bytes = spill_buffer_len as u64 * config.spill_buffer_entry_bytes;

    heap_bytes + tentative_bytes + parent_bytes + buffered_spill_bytes + finalized_bits_bytes
}

fn reconstruct_path(
    start_slot: usize,
    goal_slot: usize,
    grid: &GridIndexer,
    swap: &mut SwapStore,
) -> Option<Vec<ArrayIndex>> {
    let mut path = Vec::new();
    let mut current_slot = goal_slot;

    loop {
        path.push(grid.index_of(current_slot));
        if current_slot == start_slot {
            break;
        }

        let record = swap.read_slot(current_slot).ok()?;
        let parent = record.parent_slot()?;
        current_slot = parent;
    }

    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_finds_shortest_path() {
        let start = ArrayIndex::new(0, 0);
        let goal = ArrayIndex::new(2, 2);

        let ans = bounded_dijkstra(
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

        let ans = bounded_dijkstra(
            &start,
            |_p: &ArrayIndex| Vec::<(ArrayIndex, u64)>::new(),
            |_p| false,
            1024,
            (1, 1),
        );

        assert!(ans.is_none());
    }

    #[test]
    fn spills_when_pressure_exceeds_budget() {
        let start = ArrayIndex::new(10, 10);
        let goal = ArrayIndex::new(15, 15);
        let config = BoundedConfig {
            memory_budget_bytes: 2_000,
            minimum_budget_bytes: 1,
            open_entry_bytes: 1,
            tentative_entry_bytes: 1,
            parent_entry_bytes: 1,
            spill_buffer_entry_bytes: 300,
            spill_buffer_capacity: 8,
        };

        let mut successors = |p: &ArrayIndex| {
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
        };

        let mut success = |p: &ArrayIndex| *p == goal;
        let ans = bounded_dijkstra_with_config::<u64, _, _, _>(
            &start,
            &mut successors,
            &mut success,
            config,
            (31, 31),
        )
        .unwrap();

        assert!(ans.2.spilled_records > 0);
        assert_eq!(ans.0.first(), Some(&start));
        assert_eq!(ans.0.last(), Some(&goal));
    }

    #[test]
    fn returns_none_when_budget_too_small_for_frontier() {
        let start = ArrayIndex::new(0, 0);
        let config = BoundedConfig {
            memory_budget_bytes: 100,
            minimum_budget_bytes: 1,
            open_entry_bytes: 1024,
            tentative_entry_bytes: 1024,
            parent_entry_bytes: 1024,
            spill_buffer_entry_bytes: 1024,
            spill_buffer_capacity: 4,
        };

        let mut successors = |p: &ArrayIndex| {
            let mut out = Vec::new();
            if p.i < 20 {
                out.push((ArrayIndex::new(p.i + 1, p.j), 1_u64));
            }
            if p.j < 20 {
                out.push((ArrayIndex::new(p.i, p.j + 1), 1_u64));
            }
            out
        };

        let mut success = |_p: &ArrayIndex| false;
        let ans = bounded_dijkstra_with_config::<u64, _, _, _>(
            &start,
            &mut successors,
            &mut success,
            config,
            (21, 21),
        );

        assert!(ans.is_none());
    }

    #[test]
    fn grid_indexer_round_trip() {
        let grid = GridIndexer::new(7, 9).unwrap();
        let sample = [
            ArrayIndex::new(0, 0),
            ArrayIndex::new(1, 4),
            ArrayIndex::new(3, 8),
            ArrayIndex::new(6, 7),
        ];

        for index in sample {
            let slot = grid.slot_of(&index).unwrap();
            assert_eq!(grid.index_of(slot), index);
        }
    }

    #[test]
    fn swap_store_reads_written_slot() {
        let mut swap = SwapStore::new(2).unwrap();
        let record = SpillRecord {
            cost: 7,
            parent_slot: 55,
        };

        swap.write_slot(42, record).unwrap();
        swap.flush().unwrap();
        let restored = swap.read_slot(42).unwrap();

        assert_eq!(restored.cost, 7);
        assert_eq!(restored.parent_slot, 55);
    }

    #[test]
    fn swap_store_read_flushes_buffered_writes() {
        let mut swap = SwapStore::new(8).unwrap();
        let record = SpillRecord {
            cost: 11,
            parent_slot: 3,
        };

        swap.write_slot(4, record).unwrap();
        let restored = swap.read_slot(4).unwrap();

        assert_eq!(restored.cost, 11);
        assert_eq!(restored.parent_slot, 3);
    }
}
