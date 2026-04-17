use std::sync::RwLock;

use tracing::{debug, trace};

use ndarray::Array2;

use super::swap::SourceLayout;

pub(super) struct DerivedChunkState {
    swap_chunk_idx: RwLock<ndarray::Array2<bool>>,
}

impl DerivedChunkState {
    pub(super) fn new(layout: &SourceLayout) -> Self {
        Self {
            swap_chunk_idx: Array2::from_elem(
                (layout.chunk_grid_rows, layout.chunk_grid_cols),
                false,
            )
            .into(),
        }
    }

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
