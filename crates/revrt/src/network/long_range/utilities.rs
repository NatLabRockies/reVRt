use crate::ArrayIndex;

#[derive(Clone, Copy, Debug)]
pub(super) struct GridIndexer {
    nrows: u64,
    ncols: u64,
    total_cells: u64,
}

impl GridIndexer {
    pub(super) fn new(nrows: u64, ncols: u64) -> Option<Self> {
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

    pub(super) fn slot_of(&self, index: &ArrayIndex) -> Option<usize> {
        if index.i >= self.nrows || index.j >= self.ncols {
            return None;
        }

        let linear = index.i.checked_mul(self.ncols)?.checked_add(index.j)?;
        usize::try_from(linear).ok()
    }

    pub(super) fn index_of(&self, slot: usize) -> ArrayIndex {
        let linear = slot as u64;
        ArrayIndex {
            i: planar / self.ncols,
            j: planar % self.ncols,
            option,
        }
    }

    pub(super) fn new_finalized_bits(&self) -> Option<FinalizedBits> {
        FinalizedBits::new(self.total_cells)
    }
}

#[derive(Debug)]
pub(super) struct FinalizedBits {
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

    pub(super) fn contains(&self, slot: usize) -> bool {
        let byte = slot / 8;
        let bit = (slot % 8) as u8;
        self.bits
            .get(byte)
            .map(|value| (value & (1 << bit)) != 0)
            .unwrap_or(false)
    }

    pub(super) fn set(&mut self, slot: usize) {
        let byte = slot / 8;
        let bit = (slot % 8) as u8;
        if let Some(value) = self.bits.get_mut(byte) {
            *value |= 1 << bit;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_indexer_round_trip() {
        let grid = GridIndexer::new(7, 9, 1).unwrap();
        let sample = [
            ArrayIndex::new_ij(0, 0),
            ArrayIndex::new_ij(1, 4),
            ArrayIndex::new_ij(3, 8),
            ArrayIndex::new_ij(6, 7),
        ];

        for index in sample {
            let slot = grid.slot_of(&index).unwrap();
            assert_eq!(grid.index_of(slot), index);
        }
    }

    #[test]
    fn grid_indexer_rejects_zero_dimensions_and_overflow() {
        assert!(GridIndexer::new(0, 4, 1).is_none());
        assert!(GridIndexer::new(4, 0, 1).is_none());
        assert!(GridIndexer::new(0, 0, 1).is_none());
        assert!(GridIndexer::new(u64::MAX, 2, 1).is_none());
    }

    #[test]
    fn grid_indexer_slot_of_rejects_out_of_bounds_indices() {
        let grid = GridIndexer::new(3, 4, 1).unwrap();

        assert_eq!(grid.slot_of(&ArrayIndex::new_ij(0, 0)), Some(0));
        assert_eq!(grid.slot_of(&ArrayIndex::new_ij(2, 3)), Some(11));
        assert_eq!(grid.slot_of(&ArrayIndex::new_ij(3, 0)), None);
        assert_eq!(grid.slot_of(&ArrayIndex::new_ij(0, 4)), None);
        assert_eq!(grid.slot_of(&ArrayIndex::new_ij(9, 9)), None);
    }

    #[test]
    fn grid_indexer_finalized_bits_size_matches_cell_count() {
        let single = GridIndexer::new(1, 1, 1).unwrap();
        let full_byte = GridIndexer::new(2, 4, 1).unwrap();
        let partial_byte = GridIndexer::new(3, 3, 1).unwrap();

        assert_eq!(single.new_finalized_bits().unwrap().bits.len(), 1);
        assert_eq!(full_byte.new_finalized_bits().unwrap().bits.len(), 1);
        assert_eq!(partial_byte.new_finalized_bits().unwrap().bits.len(), 2);
    }

    #[test]
    fn finalized_bits_new_handles_zero_cells() {
        let bits = FinalizedBits::new(0).unwrap();

        assert!(bits.bits.is_empty());
        assert!(!bits.contains(0));
    }

    #[test]
    fn finalized_bits_defaults_to_false_and_tracks_multiple_slots() {
        let mut bits = FinalizedBits::new(10).unwrap();

        for slot in 0..10 {
            assert!(!bits.contains(slot));
        }

        bits.set(0);
        bits.set(7);
        bits.set(8);
        bits.set(9);

        assert!(bits.contains(0));
        assert!(bits.contains(7));
        assert!(bits.contains(8));
        assert!(bits.contains(9));
        assert!(!bits.contains(1));
        assert!(!bits.contains(6));
    }

    #[test]
    fn finalized_bits_ignores_out_of_range_slots() {
        let mut bits = FinalizedBits::new(8).unwrap();

        bits.set(100);

        for slot in 0..8 {
            assert!(!bits.contains(slot));
        }
        assert!(!bits.contains(100));
    }
}
