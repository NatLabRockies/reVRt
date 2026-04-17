use std::iter;
use std::sync::Arc;

use tracing::{debug, trace, warn};
use zarrs::array::codec::CodecOptions;
use zarrs::array::{ChunkCache, ChunkCacheDecodedLruSizeLimit};
use zarrs::storage::{ReadableStorageTraits, ReadableWritableListableStorage};

use super::swap::SourceLayout;
use super::swap::cumulative_soft_barrier_mask_name;
use crate::ArrayIndex;
use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy)]
/// Cache sizes assigned to each neighborhood reader dataset
struct CacheBudgets {
    /// Cache budget for the primary cost array
    per_cost_cache: u64,
    /// Cache budget for the hard barrier mask
    hard_barrier_cache: u64,
    /// Cache budget for each cumulative soft barrier mask
    per_soft_barrier_cache: u64,
}

/// Cached access to derived 3x3 neighborhoods from the swap dataset
pub(super) struct NeighborhoodReader {
    /// Decoded chunk cache for the main per-cell routing cost
    cost_cache: ChunkCacheDecodedLruSizeLimit,
    /// Decoded chunk cache for invariant movement costs
    cost_invariant_cache: ChunkCacheDecodedLruSizeLimit,
    /// Decoded chunk cache for the hard barrier mask
    hard_barrier_cache: ChunkCacheDecodedLruSizeLimit,
    /// Decoded chunk caches for cumulative soft barrier masks by retry state
    cumulative_soft_barrier_caches: Vec<ChunkCacheDecodedLruSizeLimit>,
    /// Number of rows in the routing grid
    grid_nrows: u64,
    /// Number of columns in the routing grid
    grid_ncols: u64,
}

impl NeighborhoodReader {
    /// Open cached readers for the derived swap arrays
    pub(super) fn open(
        swap: ReadableWritableListableStorage,
        cache_size: u64,
        soft_barrier_group_count: usize,
        layout: SourceLayout,
    ) -> Result<Self> {
        if cache_size < 1_000_000 {
            warn!("Cache size smaller than 1MB");
        }
        debug!(
            "Creating caches with total size {}MB",
            cache_size / 1_000_000
        );
        let cost_array_readable =
            Arc::new(zarrs::array::Array::open(swap.clone(), "/cost")?.readable());
        let cost_invariant_array_readable =
            Arc::new(zarrs::array::Array::open(swap.clone(), "/cost_invariant")?.readable());
        let hard_barrier_array_readable =
            Arc::new(zarrs::array::Array::open(swap.clone(), "/hard_barrier_mask")?.readable());
        let cumulative_soft_barrier_arrays = (0..=soft_barrier_group_count)
            .map(|retry_state| {
                let path = format!("/{}", cumulative_soft_barrier_mask_name(retry_state));
                zarrs::array::Array::open(swap.clone(), &path)
                    .map_err(|err| Error::IO(std::io::Error::other(err.to_string())))
                    .map(|array| Arc::new(array.readable()))
            })
            .collect::<Result<Vec<_>>>()?;

        let budgets = distribute_cache_budgets(cache_size, cumulative_soft_barrier_arrays.len());
        debug!("Cache budgets: {:?}", budgets);

        let cost_cache =
            ChunkCacheDecodedLruSizeLimit::new(cost_array_readable.clone(), budgets.per_cost_cache);
        let cost_invariant_cache = ChunkCacheDecodedLruSizeLimit::new(
            cost_invariant_array_readable.clone(),
            budgets.per_cost_cache,
        );
        let hard_barrier_cache = ChunkCacheDecodedLruSizeLimit::new(
            hard_barrier_array_readable.clone(),
            budgets.hard_barrier_cache,
        );
        let cumulative_soft_barrier_caches = cumulative_soft_barrier_arrays
            .into_iter()
            .map(|array| ChunkCacheDecodedLruSizeLimit::new(array, budgets.per_soft_barrier_cache))
            .collect();

        Ok(Self {
            cost_cache,
            cost_invariant_cache,
            hard_barrier_cache,
            cumulative_soft_barrier_caches,
            grid_nrows: layout.grid_nrows,
            grid_ncols: layout.grid_ncols,
        })
    }

    /// Read the valid 3x3 neighborhood movement costs around an index
    ///
    /// The returned costs combine the directional cost surface, the
    /// invariant movement penalty, diagonal scaling, and optional hard
    /// barrier filtering. If the center cell is itself a hard barrier,
    /// an empty vector is returned.
    pub(super) fn get_3x3<F>(
        &self,
        index: &ArrayIndex,
        has_hard_barriers: bool,
        ensure_derived_data_for_subset: F,
    ) -> Vec<(ArrayIndex, f32)>
    where
        F: Fn(&zarrs::array::Array<dyn ReadableStorageTraits>, &zarrs::array_subset::ArraySubset),
    {
        let &ArrayIndex { i, j } = index;

        trace!("Getting 3x3 neighborhood for (i={}, j={})", i, j);

        trace!("Opening cost dataset via cache");
        let cost_array = self.cost_cache.array();
        trace!("Cost dataset with shape: {:?}", cost_array.shape());

        let (i_range, j_range, subset) = self.neighborhood_subset(index);
        trace!("Cost subset: {:?}", subset);
        ensure_derived_data_for_subset(&cost_array, &subset);

        let neighbors = self.get_neighbor_costs(i_range.clone(), j_range.clone(), &subset, false);
        let invariant_neighbors =
            self.get_neighbor_costs(i_range.clone(), j_range.clone(), &subset, true);
        let hard_barrier_values: Vec<bool> = if has_hard_barriers {
            self.hard_barrier_cache
                .retrieve_array_subset_elements::<bool>(&subset, &CodecOptions::default())
                .unwrap()
        } else {
            std::iter::repeat_n(false, neighbors.len()).collect()
        };

        let center = neighbors
            .iter()
            .zip(hard_barrier_values.iter())
            .find(|(((ir, jr), _), _)| *ir == i && *jr == j)
            .map(|(((ir, jr), v), is_barrier)| {
                if *is_barrier {
                    ((ir, jr), &0_f32, true)
                } else if v.is_nan() {
                    ((ir, jr), &0_f32, false)
                } else {
                    ((ir, jr), v, false)
                }
            })
            .unwrap();
        if center.2 {
            return Vec::new();
        }
        trace!("Center point: {:?}", center);

        let cost_to_neighbors = neighbors
            .iter()
            .zip(invariant_neighbors.iter())
            .zip(hard_barrier_values.iter())
            .filter(|((((ir, jr), v), _), is_barrier)| {
                !(**is_barrier || v.is_nan() || (*ir == i && *jr == j))
            })
            .map(|((((ir, jr), v), ((inv_ir, inv_jr), inv_cost)), _)| {
                debug_assert_eq!((ir, jr), (inv_ir, inv_jr));
                ((ir, jr), 0.5 * (v + center.1), inv_cost)
            })
            .map(|((ir, jr), v, inv_cost)| {
                let scaled = if *ir != i && *jr != j {
                    v * f32::sqrt(2.0)
                } else {
                    v
                };
                (ArrayIndex { i: *ir, j: *jr }, scaled + inv_cost)
            })
            .collect::<Vec<_>>();

        trace!("Neighbors {:?}", cost_to_neighbors);
        cost_to_neighbors
    }

    /// Return soft barrier cells in the 3x3 neighborhood for a retry state
    pub(super) fn get_3x3_soft_barrier_cells<F>(
        &self,
        index: &ArrayIndex,
        retry_state: usize,
        ensure_derived_data_for_subset: F,
    ) -> Vec<ArrayIndex>
    where
        F: Fn(&zarrs::array::Array<dyn ReadableStorageTraits>, &zarrs::array_subset::ArraySubset),
    {
        self.get_3x3_cached_barrier_cells(
            index,
            &self.cumulative_soft_barrier_caches[retry_state],
            ensure_derived_data_for_subset,
        )
    }

    /// Return the grid shape backing this reader as `(rows, cols)`
    pub(super) fn grid_shape(&self) -> (u64, u64) {
        (self.grid_nrows, self.grid_ncols)
    }

    /// Build the row and column ranges for a clipped 3x3 neighborhood
    ///
    /// The returned subset includes the leading band dimension expected by
    /// the derived swap arrays.
    pub(super) fn neighborhood_subset(
        &self,
        index: &ArrayIndex,
    ) -> (
        std::ops::Range<u64>,
        std::ops::Range<u64>,
        zarrs::array_subset::ArraySubset,
    ) {
        let &ArrayIndex { i, j } = index;
        debug_assert!(self.grid_nrows > 0);
        debug_assert!(self.grid_ncols > 0);

        let max_i = self.grid_nrows - 1;
        let max_j = self.grid_ncols - 1;

        let i_range = match i {
            0 if max_i == 0 => 0..1,
            0 => 0..2,
            _ if i == max_i => i - 1..i + 1,
            _ => i - 1..i + 2,
        };
        let j_range = match j {
            0 if max_j == 0 => 0..1,
            0 => 0..2,
            _ if j == max_j => j - 1..j + 1,
            _ => j - 1..j + 2,
        };

        let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
            0..1,
            i_range.clone(),
            j_range.clone(),
        ]);

        (i_range, j_range, subset)
    }

    /// Read cost values for every cell in a neighborhood subset
    ///
    /// When `is_invariant` is true, values are read from the invariant cost
    /// array. Otherwise values are read from the primary cost array.
    pub(super) fn get_neighbor_costs(
        &self,
        i_range: std::ops::Range<u64>,
        j_range: std::ops::Range<u64>,
        subset: &zarrs::array_subset::ArraySubset,
        is_invariant: bool,
    ) -> Vec<((u64, u64), f32)> {
        trace!("Opening cost dataset (is_invariant={})", is_invariant);

        let cache = if is_invariant {
            &self.cost_invariant_cache
        } else {
            &self.cost_cache
        };
        let cost_array = cache.array();
        trace!(
            "Cost dataset (is_invariant={}) with shape: {:?}",
            is_invariant,
            cost_array.shape()
        );

        let cost_values: Vec<f32> = cache
            .retrieve_array_subset_elements::<f32>(subset, &CodecOptions::default())
            .unwrap();

        trace!("Read values {:?}", cost_values);

        let neighbor_costs = i_range
            .flat_map(|row| iter::repeat(row).zip(j_range.clone()))
            .zip(cost_values)
            .collect();

        trace!("Neighbors {:?}", neighbor_costs);
        neighbor_costs
    }

    /// Read barrier cells from a cached boolean neighborhood mask
    fn get_3x3_cached_barrier_cells<F>(
        &self,
        index: &ArrayIndex,
        cache: &ChunkCacheDecodedLruSizeLimit,
        ensure_derived_data_for_subset: F,
    ) -> Vec<ArrayIndex>
    where
        F: Fn(&zarrs::array::Array<dyn ReadableStorageTraits>, &zarrs::array_subset::ArraySubset),
    {
        let (i_range, j_range, subset) = self.neighborhood_subset(index);
        ensure_derived_data_for_subset(&cache.array(), &subset);
        let barrier_values = cache
            .retrieve_array_subset_elements::<bool>(&subset, &CodecOptions::default())
            .unwrap();
        let mut barrier_cells = Vec::new();

        for ((ir, jr), is_barrier) in i_range
            .flat_map(|row| iter::repeat(row).zip(j_range.clone()))
            .zip(barrier_values)
        {
            if is_barrier {
                barrier_cells.push(ArrayIndex { i: ir, j: jr });
            }
        }

        barrier_cells
    }
}

/// Split the requested cache size across all neighborhood reader caches
///
/// One third of `cache_size` is assigned to each of the two cost caches,
/// with a minimum of 1 byte per cache. The remaining budget is then split
/// in half: one half goes to the hard barrier cache and the other half is
/// reserved for all cumulative soft barrier caches. The soft barrier share
/// is divided evenly across `soft_barrier_cache_count`, again with a
/// minimum of 1 byte per cache. Saturating subtraction is used for the
/// remainder calculations so very small cache sizes still produce valid
/// nonzero budgets.
fn distribute_cache_budgets(cache_size: u64, soft_barrier_cache_count: usize) -> CacheBudgets {
    let per_cost_cache = (cache_size / 3).max(1);
    let remaining_cache = cache_size.saturating_sub(2 * per_cost_cache).max(1);
    let hard_barrier_cache = (remaining_cache / 2).max(1);
    let soft_cache_budget = remaining_cache.saturating_sub(hard_barrier_cache).max(1);

    let per_soft_barrier_cache = if soft_barrier_cache_count == 0 {
        1
    } else {
        (soft_cache_budget / soft_barrier_cache_count as u64).max(1)
    };

    CacheBudgets {
        per_cost_cache,
        hard_barrier_cache,
        per_soft_barrier_cache,
    }
}
