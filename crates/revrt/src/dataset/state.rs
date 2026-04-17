//! Derived chunk materialization state
//!
//! This module tracks which swap chunks have already been derived so repeated
//! neighborhood reads can avoid recomputing the same cost and barrier data.

use std::sync::RwLock;

use tracing::{debug, trace};

use ndarray::Array2;

use super::swap::SourceLayout;

/// Track which swap chunks have already been materialized.
///
/// The internal boolean grid mirrors the source chunk layout. A value of
/// `true` means the corresponding derived swap chunk has already been written
/// and does not need to be recomputed.
pub(super) struct DerivedChunkState {
    /// Boolean materialization state indexed by chunk row and chunk column.
    swap_chunk_idx: RwLock<ndarray::Array2<bool>>,
}

impl DerivedChunkState {
    /// Create an empty materialization-state grid for a source layout.
    ///
    /// # Arguments
    /// `layout`: Source layout whose chunk-grid dimensions determine the size
    ///           of the internal tracking array.
    ///
    /// # Returns
    /// A `DerivedChunkState` initialized with all chunks marked as not yet
    /// materialized.
    pub(super) fn new(layout: &SourceLayout) -> Self {
        Self {
            swap_chunk_idx: Array2::from_elem(
                (layout.chunk_grid_rows, layout.chunk_grid_cols),
                false,
            )
            .into(),
        }
    }

    /// Materialize any missing derived chunks overlapping a subset.
    ///
    /// This method first determines which chunk-grid cells intersect the
    /// requested subset. Each chunk is checked under a read lock and, when
    /// still missing, rechecked under a write lock before invoking the
    /// provided `materialize_chunk` callback. This avoids duplicate work when
    /// multiple threads request the same chunk concurrently.
    ///
    /// # Arguments
    /// `array`: Swap array whose chunk grid is used to map the subset to
    ///          chunk indices.
    /// `subset`: Requested array subset that may span one or more chunks.
    /// `materialize_chunk`: Callback that computes and writes the derived data
    ///                      for a given chunk row and column index.
    pub(super) fn ensure_derived_data_for_subset<F>(
        &self,
        array: &zarrs::array::Array<dyn zarrs::storage::ReadableStorageTraits>,
        subset: &zarrs::array_subset::ArraySubset,
        materialize_chunk: F,
    ) where
        F: Fn(u64, u64),
    {
        let chunks = &array.chunks_in_array_subset(subset).unwrap().unwrap();
        trace!("Derived-data chunks: {:?}", chunks);
        trace!(
            "Derived-data subset extends to {:?} chunks",
            chunks.num_elements_usize()
        );

        for ci in chunks.start()[1]..(chunks.start()[1] + chunks.shape()[1]) {
            for cj in chunks.start()[2]..(chunks.start()[2] + chunks.shape()[2]) {
                trace!(
                    "Checking if derived data for chunk ({}, {}) has been calculated",
                    ci, cj
                );
                if self.swap_chunk_idx.read().unwrap()[[ci as usize, cj as usize]] {
                    trace!("Derived data for chunk ({}, {}) already calculated", ci, cj);
                    continue;
                }

                debug!("Requesting write lock for swap_chunk_idx ({}, {})", ci, cj);
                let mut chunk_idx = self
                    .swap_chunk_idx
                    .write()
                    .expect("Failed to acquire write lock");
                debug!("Acquired write lock for swap_chunk_idx ({}, {})", ci, cj);
                if chunk_idx[[ci as usize, cj as usize]] {
                    trace!(
                        "Derived data for chunk ({}, {}) already calculated while waiting for the lock",
                        ci, cj
                    );
                } else {
                    materialize_chunk(ci, cj);
                    chunk_idx[[ci as usize, cj as usize]] = true;
                    debug!(
                        "Recorded derived data for chunk ({}, {}) as calculated. Total number of computed chunks: {}",
                        ci,
                        cj,
                        chunk_idx.iter().filter(|&&value| value).count()
                    );
                }
                debug!("Released write lock for swap_chunk_idx ({}, {})", ci, cj);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use zarrs::array_subset::ArraySubset;
    use zarrs::filesystem::FilesystemStore;
    use zarrs::storage::ReadableListableStorage;

    use super::*;
    use crate::dataset::samples;
    use crate::dataset::swap::inspect_source_layout;

    #[test]
    fn new_initializes_all_chunks_as_not_materialized() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A"]);
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));
        let layout = inspect_source_layout(&source).expect("source layout inspection failed");

        let state = DerivedChunkState::new(&layout);
        let chunk_idx = state
            .swap_chunk_idx
            .read()
            .expect("failed to acquire read lock");

        assert_eq!(chunk_idx.dim(), (2, 2));
        assert!(chunk_idx.iter().all(|&value| !value));
    }

    #[test]
    fn ensure_derived_data_for_subset_only_materializes_missing_chunks() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A"]);
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));
        let layout = inspect_source_layout(&source).expect("source layout inspection failed");
        let readable_source: Arc<dyn zarrs::storage::ReadableStorageTraits> = Arc::new(
            FilesystemStore::new(tmp.path()).expect("could not reopen readable test store"),
        );
        let array =
            zarrs::array::Array::open(readable_source, "/A").expect("failed to open source array");
        let state = DerivedChunkState::new(&layout);
        let materialized = Mutex::new(Vec::new());

        let first_subset = ArraySubset::new_with_ranges(&[0..1, 1..7, 1..3]);
        state.ensure_derived_data_for_subset(&array, &first_subset, |ci, cj| {
            materialized
                .lock()
                .expect("failed to record materialized chunk")
                .push((ci, cj));
        });

        let second_subset = ArraySubset::new_with_ranges(&[0..1, 3..6, 2..7]);
        state.ensure_derived_data_for_subset(&array, &second_subset, |ci, cj| {
            materialized
                .lock()
                .expect("failed to record materialized chunk")
                .push((ci, cj));
        });

        state.ensure_derived_data_for_subset(&array, &second_subset, |ci, cj| {
            materialized
                .lock()
                .expect("failed to record materialized chunk")
                .push((ci, cj));
        });

        assert_eq!(
            *materialized
                .lock()
                .expect("failed to read materialized chunks"),
            vec![(0, 0), (1, 0), (0, 1), (1, 1)]
        );

        let chunk_idx = state
            .swap_chunk_idx
            .read()
            .expect("failed to acquire read lock");
        assert_eq!(*chunk_idx, Array2::from_elem((2, 2), true));
    }
}
