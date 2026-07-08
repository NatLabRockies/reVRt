//! Cached readers for derived neighborhood data
//!
//! This module provides read access to chunk-cached derived arrays stored in
//! the swap dataset. It focuses on retrieving clipped 3x3 neighborhoods for
//! routing, including cost surfaces, invariant penalties, and barrier masks.

use std::iter;
use std::sync::Arc;

use tracing::{debug, trace, warn};
use zarrs::array::CodecOptions;
use zarrs::array::chunk_cache::{ChunkCache, ChunkCacheDecodedLruSizeLimit};
use zarrs::storage::{ReadableStorageTraits, ReadableWritableListableStorage};

use super::swap::SourceLayout;
use super::swap::cumulative_soft_barrier_mask_name;
use super::{NeighborhoodGeometry, NeighborhoodPoint, RoutingOptionNeighborhood};
use crate::ArrayIndex;
use crate::error::{Error, Result};

/// Capability required to materialize derived swap data for a subset.
pub(super) trait DerivedDataMaterializer {
    /// Whether the derived dataset includes a hard barrier mask.
    fn has_hard_barriers(&self) -> bool;

    /// Ensure all derived swap data exists for a requested subset.
    fn ensure_derived_data_for_subset(
        &self,
        array: &zarrs::array::Array<dyn ReadableStorageTraits>,
        subset: &zarrs::array::ArraySubset,
    );
}

#[derive(Debug, Clone, Copy)]
/// Cache sizes assigned to each neighborhood reader dataset.
///
/// The cache budget is split between the two cost arrays, any optional driver
/// and hard barrier arrays, and the family of cumulative soft barrier masks
/// so neighborhood lookups can reuse decoded chunks efficiently.
struct CacheBudgets {
    /// Cache budget for the primary cost array.
    cost_cache: u64,
    /// Cache budget for the invariant cost array.
    invariant_cost_cache: Option<u64>,
    /// Cache budget for the driver multiplier array.
    driver_multiplier_cache: Option<u64>,
    /// Cache budget for the hard barrier mask.
    hard_barrier_cache: Option<u64>,
    /// Cache budget for each cumulative soft barrier mask.
    per_soft_barrier_cache: Option<u64>,
}

/// Cached access to derived data from the swap dataset.
///
/// The reader keeps decoded chunk caches for each derived array needed during
/// routing so repeated neighborhood lookups can avoid reopening and decoding
/// the same swap chunks.
pub(super) struct DerivedDataReader {
    /// Decoded chunk cache for the main per-cell routing cost.
    cost_cache: ChunkCacheDecodedLruSizeLimit,
    /// Decoded chunk cache for invariant movement costs.
    cost_invariant_cache: Option<ChunkCacheDecodedLruSizeLimit>,
    /// Decoded chunk cache for precomputed driver multipliers.
    driver_multiplier_cache: Option<ChunkCacheDecodedLruSizeLimit>,
    /// Decoded chunk cache for the hard barrier mask.
    hard_barrier_cache: Option<ChunkCacheDecodedLruSizeLimit>,
    /// Decoded chunk caches for cumulative soft barrier masks by retry state.
    cumulative_soft_barrier_caches: Vec<ChunkCacheDecodedLruSizeLimit>,
    /// Number of routing options on the leading band axis.
    grid_noptions: u32,
    /// Number of rows in the routing grid.
    grid_nrows: u64,
    /// Number of columns in the routing grid.
    grid_ncols: u64,
}

impl DerivedDataReader {
    /// Open cached readers for the derived swap arrays.
    ///
    /// This initializes one decoded chunk cache per derived array used during
    /// routing and records the grid dimensions needed to clip neighborhood
    /// lookups at dataset boundaries.
    ///
    /// # Arguments
    /// `swap`: Writable swap storage that already contains the derived arrays.
    /// `cache_size`: Total cache budget, in bytes, to distribute across all
    ///               internal chunk caches.
    /// `soft_barrier_group_count`: Number of soft barrier importance groups,
    ///                             used to determine how many cumulative mask
    ///                             caches are required.
    /// `layout`: Source grid layout metadata used to record dataset shape.
    ///
    /// # Returns
    /// A `DerivedDataReader` with initialized chunk caches for every derived
    /// neighborhood array.
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
        let cost_invariant_array = open_optional_readable_array(swap.clone(), "/cost_invariant")?;
        let cumulative_soft_barrier_arrays = if soft_barrier_group_count == 0 {
            Vec::new()
        } else {
            (0..=soft_barrier_group_count)
                .map(|retry_state| {
                    let path = format!("/{}", cumulative_soft_barrier_mask_name(retry_state));
                    zarrs::array::Array::open(swap.clone(), &path)
                        .map_err(|err| Error::IO(std::io::Error::other(err.to_string())))
                        .map(|array| Arc::new(array.readable()))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let driver_multiplier_array =
            open_optional_readable_array(swap.clone(), "/driver_multiplier")?;
        let hard_barrier_array = open_optional_readable_array(swap.clone(), "/hard_barrier_mask")?;

        let budgets = distribute_cache_budgets(
            cache_size,
            cumulative_soft_barrier_arrays.len(),
            cost_invariant_array.is_some(),
            driver_multiplier_array.is_some(),
            hard_barrier_array.is_some(),
        );
        debug!("Cache budgets: {:?}", budgets);

        let cost_cache =
            ChunkCacheDecodedLruSizeLimit::new(cost_array_readable.clone(), budgets.cost_cache);
        let cost_invariant_cache =
            cost_invariant_array
                .zip(budgets.invariant_cost_cache)
                .map(|(array, cache_budget)| {
                    ChunkCacheDecodedLruSizeLimit::new(Arc::new(array), cache_budget)
                });
        let driver_multiplier_cache = driver_multiplier_array
            .zip(budgets.driver_multiplier_cache)
            .map(|(array, cache_budget)| {
                ChunkCacheDecodedLruSizeLimit::new(Arc::new(array), cache_budget)
            });
        let hard_barrier_cache =
            hard_barrier_array
                .zip(budgets.hard_barrier_cache)
                .map(|(array, cache_budget)| {
                    ChunkCacheDecodedLruSizeLimit::new(Arc::new(array), cache_budget)
                });
        let cumulative_soft_barrier_caches = cumulative_soft_barrier_arrays
            .into_iter()
            .map(|array| {
                ChunkCacheDecodedLruSizeLimit::new(
                    array,
                    budgets
                        .per_soft_barrier_cache
                        .expect("soft barrier cache budget missing for soft barrier array"),
                )
            })
            .collect();

        Ok(Self {
            cost_cache,
            cost_invariant_cache,
            driver_multiplier_cache,
            hard_barrier_cache,
            cumulative_soft_barrier_caches,
            grid_noptions: layout.grid_noptions,
            grid_nrows: layout.grid_nrows,
            grid_ncols: layout.grid_ncols,
        })
    }

    /// Read the clipped 3x3 neighborhood data for every routing option.
    ///
    /// The returned neighborhoods include the directional cost surface,
    /// invariant movement penalty, and optional hard barrier state for
    /// each neighboring cell in the clipped 3x3 window. The result always
    /// contains one `RoutingOptionNeighborhood` per routing option. If a
    /// center cell is blocked or otherwise invalid for an option, that
    /// option's neighborhood is still returned with `center_primary_cost`
    /// set to `None`.
    ///
    /// # Arguments
    /// `index`: Grid index whose neighborhood should be read.
    /// `data_materializer`: Derived-data materializer responsible for
    ///                      ensuring the required swap chunks exist
    ///                      before the cached read occurs.
    ///
    /// # Returns
    /// A vector containing one `RoutingOptionNeighborhood` per routing option.
    pub(super) fn get_3x3_neighborhood_all_options(
        &self,
        index: &ArrayIndex,
        data_materializer: &impl DerivedDataMaterializer,
    ) -> Vec<RoutingOptionNeighborhood> {
        let _profiling_scope =
            crate::profiling::scope("dataset::DerivedDataReader::get_3x3_neighborhood_all_options");
        let &ArrayIndex { i: ci, j: cj, .. } = index;

        trace!(
            "Getting 3x3 neighborhood points for all options at (i={}, j={})",
            ci, cj
        );

        trace!("Opening cost dataset via cache");
        let cost_array = self.cost_cache.array();
        trace!("Cost dataset with shape: {:?}", cost_array.shape());

        let (i_range, j_range, subset) = self.neighborhood_subset_all_options(index);
        trace!("Cost subset: {:?}", subset);
        self.ensure_neighborhood_all_option_data(&cost_array, &subset, data_materializer);

        let primary_costs = self.retrieve_neighborhood_primary_costs(&subset);
        let invariant_costs =
            self.retrieve_neighborhood_invariant_costs(&subset, primary_costs.len());
        let driver_multipliers =
            self.retrieve_neighborhood_driver_multipliers(&subset, primary_costs.len());
        let hard_barrier_values = self.retrieve_neighborhood_hard_barriers(
            &subset,
            primary_costs.len(),
            data_materializer,
        );

        let ij_coordinates = Self::rebuild_neighborhood_coordinates(&i_range, &j_range);
        let cost_to_neighbors = self.rebuild_option_neighborhoods(
            index,
            &ij_coordinates,
            &primary_costs,
            &invariant_costs,
            &driver_multipliers,
            &hard_barrier_values,
        );

        trace!(
            "Center point: {:?} Neighbors {:?}",
            index, cost_to_neighbors
        );

        cost_to_neighbors
    }

    /// Ensure derived data exists for an all-option neighborhood read.
    fn ensure_neighborhood_all_option_data(
        &self,
        cost_array: &zarrs::array::Array<dyn ReadableStorageTraits>,
        subset: &zarrs::array::ArraySubset,
        data_materializer: &impl DerivedDataMaterializer,
    ) {
        let _profiling_scope = crate::profiling::scope(
            "dataset::DerivedDataReader::ensure_neighborhood_all_option_data",
        );
        data_materializer.ensure_derived_data_for_subset(cost_array, subset);
    }

    /// Read primary costs for an all-option neighborhood subset.
    fn retrieve_neighborhood_primary_costs(&self, subset: &zarrs::array::ArraySubset) -> Vec<f32> {
        let _profiling_scope = crate::profiling::scope(
            "dataset::DerivedDataReader::retrieve_neighborhood_primary_costs",
        );
        self.cost_cache
            .retrieve_array_subset::<Vec<f32>>(subset, &CodecOptions::default())
            .unwrap()
    }

    /// Read invariant costs for an all-option neighborhood subset.
    fn retrieve_neighborhood_invariant_costs(
        &self,
        subset: &zarrs::array::ArraySubset,
        len: usize,
    ) -> Vec<f32> {
        let _profiling_scope = crate::profiling::scope(
            "dataset::DerivedDataReader::retrieve_neighborhood_invariant_costs",
        );
        self.cost_invariant_cache
            .as_ref()
            .map(|cache| {
                cache
                    .retrieve_array_subset::<Vec<f32>>(subset, &CodecOptions::default())
                    .unwrap()
            })
            .unwrap_or_else(|| std::iter::repeat_n(0.0, len).collect())
    }

    /// Read driver multipliers for an all-option neighborhood subset.
    fn retrieve_neighborhood_driver_multipliers(
        &self,
        subset: &zarrs::array::ArraySubset,
        len: usize,
    ) -> Vec<f32> {
        let _profiling_scope = crate::profiling::scope(
            "dataset::DerivedDataReader::retrieve_neighborhood_driver_multipliers",
        );
        if let Some(cache) = &self.driver_multiplier_cache {
            cache
                .retrieve_array_subset::<Vec<f32>>(subset, &CodecOptions::default())
                .unwrap()
        } else {
            std::iter::repeat_n(1.0, len).collect()
        }
    }

    /// Read hard barrier flags for an all-option neighborhood subset.
    fn retrieve_neighborhood_hard_barriers(
        &self,
        subset: &zarrs::array::ArraySubset,
        len: usize,
        data_materializer: &impl DerivedDataMaterializer,
    ) -> Vec<bool> {
        let _profiling_scope = crate::profiling::scope(
            "dataset::DerivedDataReader::retrieve_neighborhood_hard_barriers",
        );
        if data_materializer.has_hard_barriers() {
            self.hard_barrier_cache
                .as_ref()
                .expect("hard barrier cache missing for materializer with hard barriers")
                .retrieve_array_subset::<Vec<bool>>(subset, &CodecOptions::default())
                .unwrap()
        } else {
            std::iter::repeat_n(false, len).collect()
        }
    }

    /// Rebuild coordinates for a clipped 3x3 neighborhood subset.
    fn rebuild_neighborhood_coordinates(
        i_range: &std::ops::Range<u64>,
        j_range: &std::ops::Range<u64>,
    ) -> Vec<(u64, u64)> {
        let _profiling_scope =
            crate::profiling::scope("dataset::DerivedDataReader::rebuild_neighborhood_coordinates");
        i_range
            .clone()
            .flat_map(|row| iter::repeat(row).zip(j_range.clone()))
            .collect()
    }

    /// Rebuild per-option neighborhoods from cached subset arrays.
    fn rebuild_option_neighborhoods(
        &self,
        index: &ArrayIndex,
        ij_coordinates: &[(u64, u64)],
        primary_costs: &[f32],
        invariant_costs: &[f32],
        driver_multipliers: &[f32],
        hard_barrier_values: &[bool],
    ) -> Vec<RoutingOptionNeighborhood> {
        let _profiling_scope =
            crate::profiling::scope("dataset::DerivedDataReader::rebuild_option_neighborhoods");
        let &ArrayIndex { i: ci, j: cj, .. } = index;
        let per_option_len = ij_coordinates.len();
        let center_index = ij_coordinates
            .iter()
            .position(|(row, col)| *row == ci && *col == cj)
            .unwrap();

        (0..self.grid_noptions as usize)
            .map(|option_idx| {
                let offset = option_idx * per_option_len;
                let primary_slice = &primary_costs[offset..offset + per_option_len];
                let invariant_slice = &invariant_costs[offset..offset + per_option_len];
                let driver_slice = &driver_multipliers[offset..offset + per_option_len];
                let hard_barrier_slice = &hard_barrier_values[offset..offset + per_option_len];
                let center_primary_cost = if hard_barrier_slice[center_index]
                    || primary_slice[center_index].is_nan()
                    || primary_slice[center_index] <= 0.0
                {
                    None
                } else {
                    Some(primary_slice[center_index])
                };
                let center_driver_multiplier = driver_slice[center_index]
                    .is_finite()
                    .then_some(driver_slice[center_index]);

                let points = ij_coordinates
                    .iter()
                    .zip(primary_slice.iter())
                    .zip(invariant_slice.iter())
                    .zip(driver_slice.iter())
                    .zip(hard_barrier_slice.iter())
                    .enumerate()
                    .filter(|(cell_index, _)| *cell_index != center_index)
                    .map(
                        |(
                            _,
                            (
                                ((((row, col), primary_cost), invariant_cost), driver_multiplier),
                                is_barrier,
                            ),
                        )| {
                            NeighborhoodPoint {
                                destination: ArrayIndex {
                                    i: *row,
                                    j: *col,
                                    option: option_idx as u32,
                                },
                                geometry: if *row != ci && *col != cj {
                                    NeighborhoodGeometry::Corner
                                } else {
                                    NeighborhoodGeometry::Side
                                },
                                destination_primary_cost: *primary_cost,
                                destination_invariant_cost: *invariant_cost,
                                destination_driver_multiplier: *driver_multiplier,
                                destination_is_hard_barrier: *is_barrier,
                            }
                        },
                    )
                    .collect::<Vec<_>>();

                RoutingOptionNeighborhood {
                    option: option_idx as u32,
                    center_primary_cost,
                    center_driver_multiplier,
                    points,
                }
            })
            .collect()
    }

    /// Return soft barrier cells in the 3x3 neighborhood for a retry state.
    ///
    /// The retry state selects which cumulative soft barrier mask should be
    /// consulted. Higher retry states correspond to progressively more relaxed
    /// soft barrier constraints.
    ///
    /// # Arguments
    /// `index`: Grid index whose neighborhood should be inspected.
    /// `retry_state`: Index into the cumulative soft barrier mask caches.
    /// `data_materializer`: Derived-data materializer responsible for ensuring
    ///                      the required swap chunks exist before the cached
    ///                      read occurs.
    ///
    /// # Returns
    /// A vector containing the neighborhood cells marked as soft barriers for
    /// the selected retry state.
    pub(super) fn get_3x3_soft_barrier_cells(
        &self,
        index: &ArrayIndex,
        retry_state: usize,
        data_materializer: &impl DerivedDataMaterializer,
    ) -> Vec<ArrayIndex> {
        let _profiling_scope =
            crate::profiling::scope("dataset::DerivedDataReader::get_3x3_soft_barrier_cells");
        if retry_state >= self.cumulative_soft_barrier_caches.len() {
            return Vec::new();
        }

        let (i_range, j_range, subset) = self.neighborhood_subset(index);
        let cache = &self.cumulative_soft_barrier_caches[retry_state];
        data_materializer.ensure_derived_data_for_subset(&cache.array(), &subset);
        let barrier_values = cache
            .retrieve_array_subset::<Vec<bool>>(&subset, &CodecOptions::default())
            .unwrap();
        let mut barrier_cells = Vec::new();

        for ((ir, jr), is_barrier) in i_range
            .flat_map(|row| iter::repeat(row).zip(j_range.clone()))
            .zip(barrier_values)
        {
            if is_barrier {
                barrier_cells.push(ArrayIndex {
                    i: ir,
                    j: jr,
                    option: index.option,
                });
            }
        }

        barrier_cells
    }

    /// Return the center-cell entry cost for a single option state.
    ///
    /// The returned value includes the length-dependent center-cell cost and
    /// the invariant adders for the destination option. Hard barriers and
    /// invalid costs return `None`.
    pub(super) fn get_cell_cost_components(
        &self,
        index: &ArrayIndex,
        data_materializer: &impl DerivedDataMaterializer,
    ) -> Option<(f32, f32)> {
        let _profiling_scope =
            crate::profiling::scope("dataset::DerivedDataReader::get_cell_cost_components");
        let subset = zarrs::array::ArraySubset::new_with_ranges(&[
            u64::from(index.option)..u64::from(index.option) + 1,
            index.i..index.i + 1,
            index.j..index.j + 1,
        ]);
        let cost_array = self.cost_cache.array();
        data_materializer.ensure_derived_data_for_subset(&cost_array, &subset);

        let cost = self
            .cost_cache
            .retrieve_array_subset::<Vec<f32>>(&subset, &CodecOptions::default())
            .ok()?
            .into_iter()
            .next()?;
        let is_hard_barrier = if data_materializer.has_hard_barriers() {
            self.hard_barrier_cache
                .as_ref()?
                .retrieve_array_subset::<Vec<bool>>(&subset, &CodecOptions::default())
                .ok()?
                .into_iter()
                .next()?
        } else {
            false
        };

        if is_hard_barrier || cost.is_nan() || cost <= 0.0 {
            return None;
        }

        let invariant = self
            .cost_invariant_cache
            .as_ref()
            .and_then(|cache| {
                cache
                    .retrieve_array_subset::<Vec<f32>>(&subset, &CodecOptions::default())
                    .ok()
            })
            .and_then(|values| values.into_iter().next())
            .unwrap_or(0.0);

        Some((cost, invariant))
    }

    /// Return the cached driver multiplier for a single option state.
    ///
    /// A non-finite stored value represents an excluded routing option and is
    /// returned as `None` to preserve existing routing behavior.
    pub(super) fn get_driver_multiplier(
        &self,
        index: &ArrayIndex,
        data_materializer: &impl DerivedDataMaterializer,
    ) -> Option<f32> {
        let _profiling_scope =
            crate::profiling::scope("dataset::DerivedDataReader::get_driver_multiplier");
        let Some(driver_multiplier_cache) = &self.driver_multiplier_cache else {
            return Some(1.0);
        };
        let subset = zarrs::array::ArraySubset::new_with_ranges(&[
            u64::from(index.option)..u64::from(index.option) + 1,
            index.i..index.i + 1,
            index.j..index.j + 1,
        ]);
        let driver_multiplier_array = driver_multiplier_cache.array();
        data_materializer.ensure_derived_data_for_subset(&driver_multiplier_array, &subset);

        driver_multiplier_cache
            .retrieve_array_subset::<Vec<f32>>(&subset, &CodecOptions::default())
            .ok()?
            .into_iter()
            .next()
            .filter(|value| value.is_finite())
    }

    /// Return the grid shape backing this reader as `(rows, cols, options)`.
    ///
    /// # Returns
    /// The routing grid dimensions recorded when the reader was opened.
    pub(super) fn grid_shape(&self) -> (u64, u64, u32) {
        (self.grid_nrows, self.grid_ncols, self.grid_noptions)
    }

    /// Build the row and column ranges for a clipped 3x3 neighborhood.
    ///
    /// The returned subset includes the leading band dimension expected by
    /// the derived swap arrays.
    ///
    /// # Arguments
    /// `index`: Center grid index for the requested neighborhood.
    ///
    /// # Returns
    /// A tuple containing the clipped row range, clipped column range, and
    /// the corresponding swap-array subset including the leading band axis.
    pub(super) fn neighborhood_subset(
        &self,
        index: &ArrayIndex,
    ) -> (
        std::ops::Range<u64>,
        std::ops::Range<u64>,
        zarrs::array::ArraySubset,
    ) {
        let &ArrayIndex { i, j, option } = index;
        debug_assert!(self.grid_nrows > 0);
        debug_assert!(self.grid_ncols > 0);
        debug_assert!(option < self.grid_noptions);

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

        let subset = zarrs::array::ArraySubset::new_with_ranges(&[
            u64::from(option)..u64::from(option) + 1,
            i_range.clone(),
            j_range.clone(),
        ]);

        (i_range, j_range, subset)
    }

    /// Build the row and column ranges for a clipped 3x3 neighborhood.
    ///
    /// Unlike `neighborhood_subset`, the returned subset spans every routing
    /// option on the leading band axis so callers can read all option bands in
    /// a single cached request.
    ///
    /// # Arguments
    /// `index`: Center grid index for the requested neighborhood.
    ///
    /// # Returns
    /// A tuple containing the clipped row range, clipped column range, and
    /// the corresponding swap-array subset including all option bands.
    fn neighborhood_subset_all_options(
        &self,
        index: &ArrayIndex,
    ) -> (
        std::ops::Range<u64>,
        std::ops::Range<u64>,
        zarrs::array::ArraySubset,
    ) {
        let &ArrayIndex { i, j, .. } = index;
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

        let subset = zarrs::array::ArraySubset::new_with_ranges(&[
            0..u64::from(self.grid_noptions),
            i_range.clone(),
            j_range.clone(),
        ]);

        (i_range, j_range, subset)
    }
}

fn open_optional_readable_array(
    swap: ReadableWritableListableStorage,
    path: &str,
) -> Result<Option<zarrs::array::Array<dyn ReadableStorageTraits>>> {
    match zarrs::array::Array::open(swap, path) {
        Ok(array) => Ok(Some(array.readable())),
        Err(zarrs::array::ArrayCreateError::MissingMetadata) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Split the requested cache size across all neighborhood reader caches.
///
/// The allocator divides `cache_size` evenly across the always-present cost
/// cache, any optional invariant-cost and driver-multiplier caches, and one
/// additional barrier bucket when at least one barrier cache exists. The
/// barrier bucket is the post-division remainder after the cost, invariant,
/// and driver budgets are assigned.
///
/// When a hard barrier cache exists, it receives half of that barrier bucket.
/// The rest is reserved for cumulative soft barrier caches and split evenly
/// across `soft_barrier_cache_count`. When no hard barrier cache exists, the
/// full barrier bucket is available to the soft barrier caches. When no hard
/// or soft barrier caches exist, no barrier bucket is reserved and the entire
/// budget is distributed across the non-barrier caches.
///
/// Every allocated cache gets at least 1 byte, and saturating subtraction
/// keeps tiny cache sizes valid.
///
/// # Arguments
/// `cache_size`: Total cache budget, in bytes.
/// `soft_barrier_cache_count`: Number of cumulative soft barrier caches that
///                             need a share of the remaining budget.
///
/// # Returns
/// A `CacheBudgets` value containing the per-cache allocations used to build
/// the neighborhood reader.
fn distribute_cache_budgets(
    cache_size: u64,
    soft_barrier_cache_count: usize,
    has_invariant_costs: bool,
    has_active_drivers: bool,
    has_hard_barriers: bool,
) -> CacheBudgets {
    let has_barrier_caches = has_hard_barriers || soft_barrier_cache_count > 0;
    let divisor = 1
        + u64::from(has_invariant_costs)
        + u64::from(has_active_drivers)
        + u64::from(has_barrier_caches);
    let cost_cache = (cache_size / divisor).max(1);
    let invariant_cost_cache = has_invariant_costs.then_some(cost_cache);
    let driver_multiplier_cache = has_active_drivers.then_some(cost_cache);
    let (hard_barrier_cache, per_soft_barrier_cache) = if has_barrier_caches {
        let remaining_cache = cache_size
            .saturating_sub(cost_cache)
            .saturating_sub(invariant_cost_cache.unwrap_or(0))
            .saturating_sub(driver_multiplier_cache.unwrap_or(0))
            .max(1);
        let hard_barrier_cache = has_hard_barriers.then_some((remaining_cache / 2).max(1));
        let soft_cache_budget = if let Some(hard_barrier_cache) = hard_barrier_cache {
            remaining_cache.saturating_sub(hard_barrier_cache).max(1)
        } else {
            remaining_cache
        };

        let per_soft_barrier_cache = if soft_barrier_cache_count == 0 {
            None
        } else {
            Some((soft_cache_budget / soft_barrier_cache_count as u64).max(1))
        };

        (hard_barrier_cache, per_soft_barrier_cache)
    } else {
        (None, None)
    };

    CacheBudgets {
        cost_cache,
        invariant_cost_cache,
        driver_multiplier_cache,
        hard_barrier_cache,
        per_soft_barrier_cache,
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::SQRT_2;
    use std::sync::Arc;

    use ndarray::Array3;
    use tempfile::TempDir;
    use test_case::test_case;
    use zarrs::array::Array;
    use zarrs::array::ArraySubset;
    use zarrs::filesystem::FilesystemStore;
    use zarrs::storage::ReadableListableStorage;

    use super::*;
    use crate::dataset::samples::{LayerConfig, ZarrTestBuilder};
    use crate::dataset::swap::{initialize_swap, inspect_source_layout};

    struct NoOpMaterializer {
        has_hard_barriers: bool,
    }

    impl DerivedDataMaterializer for NoOpMaterializer {
        fn has_hard_barriers(&self) -> bool {
            self.has_hard_barriers
        }

        fn ensure_derived_data_for_subset(
            &self,
            _array: &zarrs::array::Array<dyn ReadableStorageTraits>,
            _subset: &zarrs::array::ArraySubset,
        ) {
        }
    }

    #[test]
    fn distribute_cache_budgets_splits_budget_across_cache_types() {
        let budgets = distribute_cache_budgets(120, 4, true, true, true);

        assert_eq!(budgets.cost_cache, 30);
        assert_eq!(budgets.invariant_cost_cache, Some(30));
        assert_eq!(budgets.driver_multiplier_cache, Some(30));
        assert_eq!(budgets.hard_barrier_cache, Some(15));
        assert_eq!(budgets.per_soft_barrier_cache, Some(3));
    }

    #[test]
    fn distribute_cache_budgets_keeps_nonzero_budgets_for_tiny_cache_sizes() {
        let budgets = distribute_cache_budgets(1, 0, true, true, true);

        assert_eq!(budgets.cost_cache, 1);
        assert_eq!(budgets.invariant_cost_cache, Some(1));
        assert_eq!(budgets.driver_multiplier_cache, Some(1));
        assert_eq!(budgets.hard_barrier_cache, Some(1));
        assert_eq!(budgets.per_soft_barrier_cache, None);
    }

    #[test]
    fn distribute_cache_budgets_skips_driver_cache_for_identity_rules() {
        let budgets = distribute_cache_budgets(120, 4, true, false, true);

        assert_eq!(budgets.cost_cache, 40);
        assert_eq!(budgets.invariant_cost_cache, Some(40));
        assert_eq!(budgets.driver_multiplier_cache, None);
        assert_eq!(budgets.hard_barrier_cache, Some(20));
        assert_eq!(budgets.per_soft_barrier_cache, Some(5));
    }

    #[test]
    fn distribute_cache_budgets_skips_hard_barrier_cache_when_disabled() {
        let budgets = distribute_cache_budgets(120, 4, true, true, false);

        assert_eq!(budgets.cost_cache, 30);
        assert_eq!(budgets.invariant_cost_cache, Some(30));
        assert_eq!(budgets.driver_multiplier_cache, Some(30));
        assert_eq!(budgets.hard_barrier_cache, None);
        assert_eq!(budgets.per_soft_barrier_cache, Some(7));
    }

    #[test]
    fn distribute_cache_budgets_skips_invariant_cache_when_disabled() {
        let budgets = distribute_cache_budgets(120, 4, false, true, false);

        assert_eq!(budgets.cost_cache, 40);
        assert_eq!(budgets.invariant_cost_cache, None);
        assert_eq!(budgets.driver_multiplier_cache, Some(40));
        assert_eq!(budgets.hard_barrier_cache, None);
        assert_eq!(budgets.per_soft_barrier_cache, Some(10));
    }

    #[test]
    fn distribute_cache_budgets_skips_barrier_reserve_without_barriers() {
        let budgets = distribute_cache_budgets(120, 0, true, true, false);

        assert_eq!(budgets.cost_cache, 40);
        assert_eq!(budgets.invariant_cost_cache, Some(40));
        assert_eq!(budgets.driver_multiplier_cache, Some(40));
        assert_eq!(budgets.hard_barrier_cache, None);
        assert_eq!(budgets.per_soft_barrier_cache, None);
    }

    #[test_case(3, 3, 1, 1, 0..3, 0..3; "interior point")]
    #[test_case(3, 3, 0, 0, 0..2, 0..2; "top left corner")]
    #[test_case(3, 3, 2, 2, 1..3, 1..3; "bottom right corner")]
    #[test_case(1, 1, 0, 0, 0..1, 0..1; "single cell grid")]
    fn neighborhood_subset_clips_ranges_to_grid_bounds(
        grid_nrows: u64,
        grid_ncols: u64,
        i: u64,
        j: u64,
        expected_i_range: std::ops::Range<u64>,
        expected_j_range: std::ops::Range<u64>,
    ) {
        let reader = reader_for_grid(grid_nrows, grid_ncols);

        let (i_range, j_range, subset) = reader.neighborhood_subset(&ArrayIndex::new_ij(i, j));

        assert_eq!(i_range, expected_i_range.clone());
        assert_eq!(j_range, expected_j_range.clone());
        assert_eq!(
            subset.shape(),
            vec![
                1,
                expected_i_range.end - expected_i_range.start,
                expected_j_range.end - expected_j_range.start,
            ]
        );
    }

    #[test]
    fn get_3x3_combines_costs_invariant_costs_and_hard_barriers() {
        let fixture = reader_fixture(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, f32::NAN, 7.0, 8.0, 9.0],
            vec![1.0; 9],
            vec![false, true, false, false, false, false, false, false, false],
            vec![false; 9],
            vec![true, false, false, false, false, false, false, true, false],
        );

        let index = ArrayIndex::new_ij(1, 1);
        let neighborhoods = fixture.reader.get_3x3_neighborhood_all_options(
            &index,
            &NoOpMaterializer {
                has_hard_barriers: true,
            },
        );
        let neighbors = same_option_neighbors(&neighborhoods, index.option);

        let expected = [
            (ArrayIndex::new_ij(0, 0), 3.0 * SQRT_2 + 1.0),
            (ArrayIndex::new_ij(0, 2), 4.0 * SQRT_2 + 1.0),
            (ArrayIndex::new_ij(1, 0), 5.5),
            (ArrayIndex::new_ij(2, 0), 6.0 * SQRT_2 + 1.0),
            (ArrayIndex::new_ij(2, 1), 7.5),
            (ArrayIndex::new_ij(2, 2), 7.0 * SQRT_2 + 1.0),
        ];

        assert_eq!(neighbors.len(), expected.len());
        for ((index, value), (expected_index, expected_value)) in
            neighbors.iter().zip(expected.iter())
        {
            assert_eq!(index, expected_index);
            assert!((value - expected_value).abs() < 1e-6);
        }
    }

    #[test]
    fn get_3x3_returns_no_neighbors_when_center_cell_is_a_hard_barrier() {
        let fixture = reader_fixture(
            vec![1.0; 9],
            vec![0.0; 9],
            vec![false, false, false, false, true, false, false, false, false],
            vec![false; 9],
            vec![false; 9],
        );

        let index = ArrayIndex::new_ij(1, 1);
        let neighborhoods = fixture.reader.get_3x3_neighborhood_all_options(
            &index,
            &NoOpMaterializer {
                has_hard_barriers: true,
            },
        );
        let neighbors = same_option_neighbors(&neighborhoods, index.option);

        assert!(neighbors.is_empty());
        assert_eq!(
            neighborhoods[index.option as usize].center_primary_cost,
            None
        );
    }

    #[test]
    fn get_3x3_filters_hard_barriers_without_mutating_cached_costs() {
        let fixture = reader_fixture(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![0.0; 9],
            vec![false, true, false, true, false, true, false, true, false],
            vec![false; 9],
            vec![false; 9],
        );
        let index = ArrayIndex::new_ij(1, 1);

        let neighborhoods = fixture.reader.get_3x3_neighborhood_all_options(
            &index,
            &NoOpMaterializer {
                has_hard_barriers: true,
            },
        );
        let option_neighborhood = &neighborhoods[index.option as usize];
        let neighbors = same_option_neighbors(&neighborhoods, index.option);

        let raw_costs = option_neighborhood
            .points
            .iter()
            .map(|point| {
                (
                    (point.destination.i, point.destination.j),
                    point.destination_primary_cost,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(option_neighborhood.center_primary_cost, Some(5.0));
        assert_eq!(
            raw_costs,
            vec![
                ((0, 0), 1.0),
                ((0, 1), 2.0),
                ((0, 2), 3.0),
                ((1, 0), 4.0),
                ((1, 2), 6.0),
                ((2, 0), 7.0),
                ((2, 1), 8.0),
                ((2, 2), 9.0),
            ]
        );
        assert_eq!(
            neighbors,
            vec![
                (ArrayIndex::new_ij(0, 0), 3.0 * SQRT_2),
                (ArrayIndex::new_ij(0, 2), 4.0 * SQRT_2),
                (ArrayIndex::new_ij(2, 0), 6.0 * SQRT_2),
                (ArrayIndex::new_ij(2, 2), 7.0 * SQRT_2),
            ]
        );
    }

    #[test]
    fn get_3x3_soft_barrier_cells_reads_retry_state_specific_mask() {
        let fixture = reader_fixture(
            vec![1.0; 9],
            vec![0.0; 9],
            vec![false; 9],
            vec![false, true, false, false, false, false, true, false, false],
            vec![true, false, false, false, false, false, false, true, false],
        );

        let retry_zero = fixture.reader.get_3x3_soft_barrier_cells(
            &ArrayIndex::new_ij(1, 1),
            0,
            &NoOpMaterializer {
                has_hard_barriers: false,
            },
        );
        let retry_one = fixture.reader.get_3x3_soft_barrier_cells(
            &ArrayIndex::new_ij(1, 1),
            1,
            &NoOpMaterializer {
                has_hard_barriers: false,
            },
        );

        assert_eq!(
            retry_zero,
            vec![ArrayIndex::new_ij(0, 1), ArrayIndex::new_ij(2, 0)]
        );
        assert_eq!(
            retry_one,
            vec![ArrayIndex::new_ij(0, 0), ArrayIndex::new_ij(2, 1)]
        );
    }

    #[test]
    fn get_3x3_reads_from_requested_option_band() {
        let fixture = reader_fixture_with_shape(
            2,
            3,
            3,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0,
            ],
            vec![0.0; 18],
            vec![false; 18],
            vec![false; 18],
            vec![false; 18],
        );

        let index = ArrayIndex {
            i: 1,
            j: 1,
            option: 1,
        };
        let neighborhoods = fixture.reader.get_3x3_neighborhood_all_options(
            &index,
            &NoOpMaterializer {
                has_hard_barriers: false,
            },
        );
        let neighbors = same_option_neighbors(&neighborhoods, index.option);

        assert_eq!(neighborhoods.len(), 2);
        assert_eq!(neighborhoods[index.option as usize].option, 1);
        assert_eq!(
            neighborhoods[index.option as usize].center_primary_cost,
            Some(15.0)
        );
        assert!(
            neighbors
                .iter()
                .all(|(neighbor_index, _)| neighbor_index.option == 1)
        );
        assert!(neighbors.contains(&(
            ArrayIndex {
                i: 0,
                j: 1,
                option: 1
            },
            13.5
        )));
        assert!(neighbors.contains(&(
            ArrayIndex {
                i: 1,
                j: 2,
                option: 1,
            },
            15.5
        )));
    }

    #[test]
    fn get_3x3_neighborhood_all_options_reads_each_option_once_and_keeps_points_for_blocked_centers()
     {
        let fixture = reader_fixture_with_shape(
            2,
            3,
            3,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0, 14.0, -1.0, 16.0,
                17.0, 18.0, 19.0,
            ],
            vec![0.0; 18],
            vec![false; 18],
            vec![false; 18],
            vec![false; 18],
        );

        let neighborhoods = fixture.reader.get_3x3_neighborhood_all_options(
            &ArrayIndex::new_ij(1, 1),
            &NoOpMaterializer {
                has_hard_barriers: false,
            },
        );

        assert_eq!(neighborhoods.len(), 2);
        assert_eq!(neighborhoods[0].option, 0);
        assert_eq!(neighborhoods[1].option, 1);
        assert_eq!(neighborhoods[0].center_primary_cost, Some(5.0));
        assert_eq!(neighborhoods[1].center_primary_cost, None);
        assert_eq!(neighborhoods[1].points.len(), 8);
        assert!(neighborhoods[1].points.iter().any(|point| {
            point.destination
                == ArrayIndex {
                    i: 1,
                    j: 2,
                    option: 1,
                }
                && point.destination_primary_cost == 16.0
        }));
    }

    #[test]
    fn grid_shape_reports_option_count() {
        let fixture = reader_fixture_with_shape(
            2,
            3,
            4,
            vec![1.0; 24],
            vec![0.0; 24],
            vec![false; 24],
            vec![false; 24],
            vec![false; 24],
        );

        assert_eq!(fixture.reader.grid_shape(), (3, 4, 2));
    }

    #[test]
    fn get_cell_cost_reads_requested_option_band() {
        let fixture = reader_fixture_with_shape(
            2,
            2,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            vec![0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            vec![false; 8],
            vec![false; 8],
            vec![false; 8],
        );

        let cost = fixture
            .reader
            .get_cell_cost_components(
                &ArrayIndex {
                    i: 1,
                    j: 0,
                    option: 1,
                },
                &NoOpMaterializer {
                    has_hard_barriers: false,
                },
            )
            .unwrap();

        assert_eq!(cost.0 + cost.1, 31.0);
    }

    #[test]
    fn get_cell_cost_components_split_primary_and_invariant_costs() {
        let fixture = reader_fixture_with_shape(
            2,
            2,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            vec![0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            vec![false; 8],
            vec![false; 8],
            vec![false; 8],
        );

        let (primary_cost, invariant_cost) = fixture
            .reader
            .get_cell_cost_components(
                &ArrayIndex {
                    i: 1,
                    j: 0,
                    option: 1,
                },
                &NoOpMaterializer {
                    has_hard_barriers: false,
                },
            )
            .unwrap();

        assert_eq!(primary_cost, 30.0);
        assert_eq!(invariant_cost, 1.0);
    }

    fn reader_for_grid(grid_nrows: u64, grid_ncols: u64) -> DerivedDataReader {
        let fixture = reader_fixture_with_shape(
            1,
            grid_nrows,
            grid_ncols,
            vec![1.0; (grid_nrows * grid_ncols) as usize],
            vec![0.0; (grid_nrows * grid_ncols) as usize],
            vec![false; (grid_nrows * grid_ncols) as usize],
            vec![false; (grid_nrows * grid_ncols) as usize],
            vec![false; (grid_nrows * grid_ncols) as usize],
        );
        fixture.reader
    }

    fn reader_fixture(
        cost_values: Vec<f32>,
        invariant_values: Vec<f32>,
        hard_barrier_values: Vec<bool>,
        soft_retry_zero_values: Vec<bool>,
        soft_retry_one_values: Vec<bool>,
    ) -> ReaderFixture {
        reader_fixture_with_shape(
            1,
            3,
            3,
            cost_values,
            invariant_values,
            hard_barrier_values,
            soft_retry_zero_values,
            soft_retry_one_values,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reader_fixture_with_shape(
        grid_noptions: u64,
        grid_nrows: u64,
        grid_ncols: u64,
        cost_values: Vec<f32>,
        invariant_values: Vec<f32>,
        hard_barrier_values: Vec<bool>,
        soft_retry_zero_values: Vec<bool>,
        soft_retry_one_values: Vec<bool>,
    ) -> ReaderFixture {
        let source_tmp = ZarrTestBuilder::new()
            .dimensions(1, grid_nrows, grid_ncols)
            .chunks(1, grid_nrows, grid_ncols)
            .layer(LayerConfig::ones("source"))
            .build()
            .expect("failed to create source test dataset");
        let source: ReadableListableStorage = Arc::new(
            FilesystemStore::new(source_tmp.path()).expect("could not open source test store"),
        );
        let layout = inspect_source_layout(&source, grid_noptions as u32)
            .expect("source layout inspection should succeed");
        let has_hard_barriers = !hard_barrier_values.is_empty();

        let swap_tmp = TempDir::new().expect("could not create temporary swap");
        let swap = initialize_swap(swap_tmp.path(), &layout, 1, true, false, has_hard_barriers)
            .expect("swap initialization should succeed");

        store_f32_layer(
            swap.clone(),
            "/cost",
            grid_noptions,
            grid_nrows,
            grid_ncols,
            cost_values,
        );
        store_f32_layer(
            swap.clone(),
            "/cost_invariant",
            grid_noptions,
            grid_nrows,
            grid_ncols,
            invariant_values,
        );
        if has_hard_barriers {
            store_bool_layer(
                swap.clone(),
                "/hard_barrier_mask",
                grid_noptions,
                grid_nrows,
                grid_ncols,
                hard_barrier_values,
            );
        }
        store_bool_layer(
            swap.clone(),
            "/soft_barrier_mask_retry_0",
            grid_noptions,
            grid_nrows,
            grid_ncols,
            soft_retry_zero_values,
        );
        store_bool_layer(
            swap.clone(),
            "/soft_barrier_mask_retry_1",
            grid_noptions,
            grid_nrows,
            grid_ncols,
            soft_retry_one_values,
        );

        let reader = DerivedDataReader::open(swap, 90, 1, layout).expect("reader should open");

        ReaderFixture {
            _source_tmp: source_tmp,
            _swap_tmp: swap_tmp,
            reader,
        }
    }

    fn store_f32_layer(
        swap: ReadableWritableListableStorage,
        path: &str,
        grid_noptions: u64,
        grid_nrows: u64,
        grid_ncols: u64,
        values: Vec<f32>,
    ) {
        let data = Array3::from_shape_vec(
            (
                grid_noptions as usize,
                grid_nrows as usize,
                grid_ncols as usize,
            ),
            values,
        )
        .expect("f32 layer values should match requested shape");
        let array = Array::open(swap, path).expect("expected f32 layer to exist");
        let subset = chunk_subset(&array);

        array
            .store_chunks(&subset, &data)
            .expect("could not store f32 layer data");
    }

    fn store_bool_layer(
        swap: ReadableWritableListableStorage,
        path: &str,
        grid_noptions: u64,
        grid_nrows: u64,
        grid_ncols: u64,
        values: Vec<bool>,
    ) {
        let data = Array3::from_shape_vec(
            (
                grid_noptions as usize,
                grid_nrows as usize,
                grid_ncols as usize,
            ),
            values,
        )
        .expect("bool layer values should match requested shape");
        let array = Array::open(swap, path).expect("expected bool layer to exist");
        let subset = chunk_subset(&array);

        array
            .store_chunks(&subset, &data)
            .expect("could not store bool layer data");
    }

    #[test]
    fn open_succeeds_without_hard_barrier_layer() {
        let source_tmp = ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(LayerConfig::ones("source"))
            .build()
            .expect("failed to create source test dataset");
        let source: ReadableListableStorage = Arc::new(
            FilesystemStore::new(source_tmp.path()).expect("could not open source test store"),
        );
        let layout =
            inspect_source_layout(&source, 1).expect("source layout inspection should succeed");

        let swap_tmp = TempDir::new().expect("could not create temporary swap");
        let swap = initialize_swap(swap_tmp.path(), &layout, 1, false, false, false)
            .expect("swap initialization should succeed");

        let reader = DerivedDataReader::open(swap, 90, 1, layout);

        assert!(reader.is_ok());
    }

    #[test]
    fn open_succeeds_without_soft_barrier_layers() {
        let source_tmp = ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(LayerConfig::ones("source"))
            .build()
            .expect("failed to create source test dataset");
        let source: ReadableListableStorage = Arc::new(
            FilesystemStore::new(source_tmp.path()).expect("could not open source test store"),
        );
        let layout =
            inspect_source_layout(&source, 1).expect("source layout inspection should succeed");

        let swap_tmp = TempDir::new().expect("could not create temporary swap");
        let swap = initialize_swap(swap_tmp.path(), &layout, 0, false, false, false)
            .expect("swap initialization should succeed");

        let reader = DerivedDataReader::open(swap, 90, 0, layout);

        assert!(reader.is_ok());
    }

    #[test]
    fn get_3x3_soft_barrier_cells_returns_empty_without_soft_barrier_layers() {
        let source_tmp = ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(LayerConfig::ones("source"))
            .build()
            .expect("failed to create source test dataset");
        let source: ReadableListableStorage = Arc::new(
            FilesystemStore::new(source_tmp.path()).expect("could not open source test store"),
        );
        let layout =
            inspect_source_layout(&source, 1).expect("source layout inspection should succeed");

        let swap_tmp = TempDir::new().expect("could not create temporary swap");
        let swap = initialize_swap(swap_tmp.path(), &layout, 0, false, false, false)
            .expect("swap initialization should succeed");
        store_f32_layer(swap.clone(), "/cost", 1, 3, 3, vec![1.0; 9]);
        let reader = DerivedDataReader::open(swap, 90, 0, layout).expect("reader should open");

        let cells = reader.get_3x3_soft_barrier_cells(
            &ArrayIndex::new_ij(1, 1),
            0,
            &NoOpMaterializer {
                has_hard_barriers: false,
            },
        );

        assert!(cells.is_empty());
    }

    #[test]
    fn get_driver_multiplier_defaults_to_identity_without_driver_layer() {
        let source_tmp = ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(LayerConfig::ones("source"))
            .build()
            .expect("failed to create source test dataset");
        let source: ReadableListableStorage = Arc::new(
            FilesystemStore::new(source_tmp.path()).expect("could not open source test store"),
        );
        let layout =
            inspect_source_layout(&source, 1).expect("source layout inspection should succeed");

        let swap_tmp = TempDir::new().expect("could not create temporary swap");
        let swap = initialize_swap(swap_tmp.path(), &layout, 1, false, false, false)
            .expect("swap initialization should succeed");

        let reader = DerivedDataReader::open(swap, 90, 1, layout).expect("reader should open");

        assert_eq!(
            reader.get_driver_multiplier(
                &ArrayIndex::new_ij(0, 0),
                &NoOpMaterializer {
                    has_hard_barriers: false,
                },
            ),
            Some(1.0)
        );
    }

    #[test]
    fn get_cell_cost_components_defaults_invariant_cost_to_zero_without_layer() {
        let source_tmp = ZarrTestBuilder::new()
            .dimensions(1, 2, 2)
            .chunks(1, 2, 2)
            .layer(LayerConfig::ones("source"))
            .build()
            .expect("failed to create source test dataset");
        let source: ReadableListableStorage = Arc::new(
            FilesystemStore::new(source_tmp.path()).expect("could not open source test store"),
        );
        let layout =
            inspect_source_layout(&source, 1).expect("source layout inspection should succeed");

        let swap_tmp = TempDir::new().expect("could not create temporary swap");
        let swap = initialize_swap(swap_tmp.path(), &layout, 1, false, false, false)
            .expect("swap initialization should succeed");

        store_f32_layer(swap.clone(), "/cost", 1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        store_bool_layer(
            swap.clone(),
            "/soft_barrier_mask_retry_0",
            1,
            2,
            2,
            vec![false; 4],
        );
        store_bool_layer(
            swap.clone(),
            "/soft_barrier_mask_retry_1",
            1,
            2,
            2,
            vec![false; 4],
        );

        let reader = DerivedDataReader::open(swap, 90, 1, layout).expect("reader should open");

        assert_eq!(
            reader.get_cell_cost_components(
                &ArrayIndex::new_ij(1, 0),
                &NoOpMaterializer {
                    has_hard_barriers: false,
                },
            ),
            Some((3.0, 0.0))
        );
    }

    fn chunk_subset<T: ?Sized>(array: &Array<T>) -> ArraySubset {
        let chunk_grid_shape = array.chunk_grid_shape();

        ArraySubset::new_with_ranges(&[
            0..chunk_grid_shape[0],
            0..chunk_grid_shape[1],
            0..chunk_grid_shape[2],
        ])
    }

    fn same_option_neighbors(
        neighborhoods: &[RoutingOptionNeighborhood],
        option: u32,
    ) -> Vec<(ArrayIndex, f32)> {
        let Some(neighborhood) = neighborhoods.iter().find(|item| item.option == option) else {
            return Vec::new();
        };
        let Some(source_primary_cost) = neighborhood.center_primary_cost else {
            return Vec::new();
        };

        neighborhood
            .points
            .iter()
            .filter_map(|point| {
                point
                    .traversal_cost(source_primary_cost, 1.0, 1.0)
                    .map(|cost| (point.destination.clone(), cost))
            })
            .collect()
    }

    struct ReaderFixture {
        _source_tmp: TempDir,
        _swap_tmp: TempDir,
        reader: DerivedDataReader,
    }
}
