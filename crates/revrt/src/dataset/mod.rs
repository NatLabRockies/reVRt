//! Source Dataset Access
//!
//! This module focus on accessing the input features data that are used by
//! the cost function.
//!
//! We are currently based on Zarr but we might move this to a generalized
//! backend on the future to access different data formats.
//!
//! The main benefits of using Zarr in the moment are the natural support
//! to concurrent access to multiple chunks, async support, rich metadata,
//! and up to a 20x factor on size reduction (mostly due to efficient
//! compression and constant values within the same chunk).
//!
//! Note: This is currently in a transition state. The initial prototype
//! included the cost function, which was removed to the `Scenario` level.

mod lazy_subset;
#[cfg(test)]
pub(crate) mod samples;
mod swap;

use std::iter;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use tracing::{debug, info, trace, warn};
use zarrs::array::codec::CodecOptions;
use zarrs::array::{ChunkCache, ChunkCacheDecodedLruSizeLimit};
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableWritableListableStorage,
};

use crate::ArrayIndex;
use crate::cost::CostFunction;
use crate::error::{Error, Result};
pub(crate) use lazy_subset::LazySubset;
use swap::{cumulative_soft_barrier_mask_name, initialize_swap, inspect_source_layout};

#[derive(Debug, Clone, Copy)]
struct CacheBudgets {
    per_cost_cache: u64,
    hard_barrier_cache: u64,
    per_soft_barrier_cache: u64,
}

/// Manages the features datasets and calculated total cost
pub(super) struct Dataset {
    /// A Zarr storages with the features
    source: ReadableListableStorage,
    // Silly way to keep the tmp path alive
    #[allow(dead_code)]
    cost_path: Option<tempfile::TempDir>,
    /// Variables used to define cost
    /// Minimalist solution for the cost calculation. In the future
    /// it will be modified to include weights and other types of
    /// relations such as operations between features.
    /// At this point it just allows custom variables names and the
    /// cost is calculated from multiple variables.
    // cost_variables: Vec<String>,
    /// Storage for the calculated cost
    swap: ReadableWritableListableStorage,
    /// Index of cost chunks already calculated
    cost_chunk_idx: RwLock<ndarray::Array2<bool>>,
    /// Explicit barriers that are always active for this dataset
    hard_barrier_layers: Vec<crate::cost::BarrierLayer>,
    /// Soft barriers grouped by importance and ordered for retry states
    soft_barrier_groups: Vec<(u32, Vec<crate::cost::BarrierLayer>)>,
    /// Custom cost function definition
    cost_function: CostFunction,
    /// Cache for decoded cost chunks shared across calls
    cost_cache: ChunkCacheDecodedLruSizeLimit,
    /// Cache for decoded invariant cost chunks shared across calls
    cost_invariant_cache: ChunkCacheDecodedLruSizeLimit,
    /// Cache for decoded hard barrier chunks shared across calls
    hard_barrier_cache: ChunkCacheDecodedLruSizeLimit,
    /// Caches for decoded cumulative soft barrier chunks per retry state
    cumulative_soft_barrier_caches: Vec<ChunkCacheDecodedLruSizeLimit>,
    /// Number of rows in the routing grid
    grid_nrows: u64,
    /// Number of columns in the routing grid
    grid_ncols: u64,
}

impl Dataset {
    pub(super) fn open<P: AsRef<std::path::Path>>(
        path: P,
        cost_function: CostFunction,
        cache_size: u64,
    ) -> Result<Self> {
        let tmp_path = tempfile::TempDir::new()
            .expect("could not create temporary directory for swap dataset");
        let tmp = tmp_path.path().to_path_buf();
        info!("Initializing a temporary swap dataset at {:?}", tmp);
        let mut dataset = Self::open_with_path(path, cost_function, cache_size, tmp)?;
        dataset.cost_path = Some(tmp_path);
        Ok(dataset)
    }

    pub(super) fn open_with_swap<P: AsRef<std::path::Path>>(
        path: P,
        cost_function: CostFunction,
        cache_size: u64,
        swap_fp: PathBuf,
    ) -> Result<Self> {
        Self::open_with_path(path, cost_function, cache_size, swap_fp)
    }

    fn open_with_path<P: AsRef<std::path::Path>>(
        path: P,
        cost_function: CostFunction,
        cache_size: u64,
        swap_fp: PathBuf,
    ) -> Result<Self> {
        debug!("Opening dataset: {:?}", path.as_ref());
        let hard_barrier_layers = cost_function.hard_barrier_layers();
        let soft_barrier_groups = cost_function.soft_barrier_groups();
        let cost_function = cost_function.without_barriers();
        let filesystem =
            zarrs::filesystem::FilesystemStore::new(path).expect("could not open filesystem store");
        let source: ReadableListableStorage = std::sync::Arc::new(filesystem);
        let source_layout = inspect_source_layout(&source)?;
        let initialized_swap = initialize_swap(swap_fp, &source_layout, soft_barrier_groups.len())?;
        let swap = initialized_swap.storage;
        let cost_chunk_idx = initialized_swap.cost_chunk_idx.into();
        let grid_nrows = source_layout.grid_nrows;
        let grid_ncols = source_layout.grid_ncols;

        if cache_size < 1_000_000 {
            warn!("Cache size smaller than 1MB");
        }
        debug!(
            "Creating caches with total size {}MB",
            cache_size / 1_000_000
        );
        // TODO: tune cache_size against typical chunk sizes
        // (e.g. chunk_bytes * hot_chunks * safety_factor)
        let cost_array_readable =
            Arc::new(zarrs::array::Array::open(swap.clone(), "/cost")?.readable());
        let cost_invariant_array_readable =
            Arc::new(zarrs::array::Array::open(swap.clone(), "/cost_invariant")?.readable());
        let hard_barrier_array_readable =
            Arc::new(zarrs::array::Array::open(swap.clone(), "/hard_barrier_mask")?.readable());
        let cumulative_soft_barrier_arrays = (0..=soft_barrier_groups.len())
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

        trace!("Dataset opened successfully");
        Ok(Self {
            source,
            cost_path: None,
            swap,
            cost_chunk_idx,
            hard_barrier_layers,
            soft_barrier_groups,
            cost_function,
            cost_cache,
            cost_invariant_cache,
            hard_barrier_cache,
            cumulative_soft_barrier_caches,
            grid_nrows,
            grid_ncols,
        })
    }

    fn calculate_chunk_derived_data(&self, ci: u64, cj: u64) {
        trace!("Creating a LazySubset for ({}, {})", ci, cj);

        // cost variable is stored in the swap dataset
        let variable = zarrs::array::Array::open(self.swap.clone(), "/cost").unwrap();
        // Get the subset according to cost's chunk
        let subset = variable.chunk_subset(&[0, ci, cj]).unwrap();
        let chunk_subset =
            zarrs::array_subset::ArraySubset::new_with_ranges(&[0..1, ci..(ci + 1), cj..(cj + 1)]);
        let mut data = LazySubset::<f32>::new(self.source.clone(), subset.clone());

        self.calculate_chunk_cost_single_layer(ci, cj, &mut data, &chunk_subset, true);
        self.calculate_chunk_cost_single_layer(ci, cj, &mut data, &chunk_subset, false);
        self.calculate_chunk_hard_barrier_mask(&mut data, &subset, &chunk_subset);
        self.calculate_chunk_cumulative_soft_barrier_masks(&mut data, &subset, &chunk_subset);
    }

    fn calculate_chunk_cost_single_layer(
        &self,
        ci: u64,
        cj: u64,
        features: &mut LazySubset<f32>,
        chunk_subset: &zarrs::array_subset::ArraySubset,
        is_invariant: bool,
    ) {
        let output;
        let layer_name;
        if is_invariant {
            trace!("Calculating invariant cost for chunk ({}, {})", ci, cj);
            output = self.cost_function.compute(features, true);
            layer_name = "/cost_invariant";
        } else {
            trace!(
                "Calculating length-dependent cost for chunk ({}, {})",
                ci, cj
            );
            output = self.cost_function.compute(features, false);
            layer_name = "/cost";
        }

        trace!("Cost function: {:?}", self.cost_function);

        let cost = zarrs::array::Array::open(self.swap.clone(), layer_name).unwrap();
        cost.store_metadata().unwrap();
        let chunk_indices: Vec<u64> = vec![0, ci, cj];
        trace!("Storing chunk at {:?}", chunk_indices);
        trace!("Target chunk subset: {:?}", chunk_subset);
        cost.store_chunks_ndarray(chunk_subset, output).unwrap();
    }

    fn calculate_chunk_hard_barrier_mask(
        &self,
        features: &mut LazySubset<f32>,
        subset: &zarrs::array_subset::ArraySubset,
        chunk_subset: &zarrs::array_subset::ArraySubset,
    ) {
        trace!("Calculating hard barrier mask for subset {:?}", subset);

        let output = if self.hard_barrier_layers.is_empty() {
            ndarray::ArrayD::<bool>::from_elem(
                ndarray::IxDyn(
                    &subset
                        .shape()
                        .iter()
                        .map(|&dim| {
                            usize::try_from(dim).expect("subset dimension exceeds usize range")
                        })
                        .collect::<Vec<_>>(),
                ),
                false,
            )
        } else {
            let barrier_masks = self
                .hard_barrier_layers
                .iter()
                .map(|layer| crate::cost::build_single_barrier_layer(layer, features))
                .collect::<Vec<_>>();

            let mut output =
                ndarray::ArrayD::<bool>::from_elem(ndarray::IxDyn(barrier_masks[0].shape()), false);
            for mask in barrier_masks {
                ndarray::Zip::from(&mut output)
                    .and(mask.view())
                    .for_each(|out, value| *out = *out || *value);
            }
            output
        };

        let variable = zarrs::array::Array::open(self.swap.clone(), "/hard_barrier_mask").unwrap();
        variable.store_metadata().unwrap();
        variable.store_chunks_ndarray(chunk_subset, output).unwrap();
    }

    fn calculate_chunk_cumulative_soft_barrier_masks(
        &self,
        features: &mut LazySubset<f32>,
        subset: &zarrs::array_subset::ArraySubset,
        chunk_subset: &zarrs::array_subset::ArraySubset,
    ) {
        trace!(
            "Calculating cumulative soft barrier masks for subset {:?}",
            subset
        );

        let empty_mask = empty_bool_mask(subset);
        let group_masks = self
            .soft_barrier_groups
            .iter()
            .map(|(_, layers)| {
                combine_barrier_layers_for_subset(layers, features, subset)
                    .unwrap_or_else(|| empty_mask.clone())
            })
            .collect::<Vec<_>>();

        for retry_state in 0..=self.soft_barrier_groups.len() {
            let layer_name = cumulative_soft_barrier_mask_name(retry_state);
            let target =
                zarrs::array::Array::open(self.swap.clone(), &format!("/{layer_name}")).unwrap();

            let mut output = empty_mask.clone();
            for mask in group_masks.iter().skip(retry_state) {
                ndarray::Zip::from(&mut output)
                    .and(mask.view())
                    .for_each(|out, value| *out = *out || *value);
            }

            target.store_metadata().unwrap();
            target.store_chunks_ndarray(chunk_subset, output).unwrap();
        }
    }

    pub(super) fn get_3x3(&self, index: &ArrayIndex) -> Vec<(ArrayIndex, f32)> {
        let &ArrayIndex { i, j } = index;

        trace!("Getting 3x3 neighborhood for (i={}, j={})", i, j);

        trace!("Cost dataset contents: {:?}", self.swap.list().unwrap());
        trace!("Cost dataset size: {:?}", self.swap.size().unwrap());

        trace!("Opening cost dataset via cache");
        let cost_array = self.cost_cache.array();
        trace!("Cost dataset with shape: {:?}", cost_array.shape());

        let (i_range, j_range, subset) = self.neighborhood_subset(index);
        trace!("Cost subset: {:?}", subset);
        self.ensure_derived_data_for_subset(&cost_array, &subset);

        let neighbors = self.get_neighbor_costs(i_range.clone(), j_range.clone(), &subset, false);
        let invariant_neighbors =
            self.get_neighbor_costs(i_range.clone(), j_range.clone(), &subset, true);
        let hard_barrier_values: Vec<bool> = if self.hard_barrier_layers.is_empty() {
            std::iter::repeat_n(false, neighbors.len()).collect()
        } else {
            self.hard_barrier_cache
                .retrieve_array_subset_elements::<bool>(&subset, &CodecOptions::default())
                .unwrap()
        };

        // Extract the origin point.
        let center = neighbors
            .iter()
            .zip(hard_barrier_values.iter())
            .find(|(((ir, jr), _), _)| *ir == i && *jr == j)
            .map(|(((ir, jr), v), is_barrier)| {
                if *is_barrier {
                    ((ir, jr), &0_f32, true)
                } else if v.is_nan() {
                    ((ir, jr), &0_f32, false) // NaN's don't contribute to cost
                } else {
                    ((ir, jr), v, false)
                }
            })
            .unwrap();
        if center.2 {
            return Vec::new();
        }
        trace!("Center point: {:?}", center);

        /*
         * The transition between two gridpoint centers is along half the distance
         * on the original gridpoint, plus half the distance to the target gridpoint
         * (center). Therefore, the transition cost is the average between the origin
         * gridpoint cost and the target gridpoint cost.
         * Note that the same principle is valid for diagonals, it is still the average
         * of both values, but we have to scale for the longer distance along the
         * diagonal, thus a sqrt(2) factor along the diagonals.
         */
        // Calculate the average with center point (half grid + other half grid).
        // Also, apply the diagonal factor for the extra distance.
        // Finally, add any invariant costs.
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
                    // Diagonal factor for longer distance (hypotenuse)
                    v * f32::sqrt(2.0)
                } else {
                    v
                };
                (ArrayIndex { i: *ir, j: *jr }, scaled + inv_cost)
            })
            .collect::<Vec<_>>();

        trace!("Neighbors {:?}", cost_to_neighbors);

        cost_to_neighbors

        /*
        let mut data = array
            .load_chunks_ndarray(&zarrs::array_subset::ArraySubset::new_with_ranges(&[0..2, 0..2]))
            .unwrap();
        data[[x as usize, y as usize]] = 0.0;
        array
            .store_chunks_ndarray(
                &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..2, 0..2]),
                data,
            )
            .unwrap();
        */
    }

    fn neighborhood_subset(
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

    fn get_neighbor_costs(
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

        // Retrieve the 3x3 neighborhood values
        let cost_values: Vec<f32> = cache
            .retrieve_array_subset_elements::<f32>(subset, &CodecOptions::default())
            .unwrap();

        trace!("Read values {:?}", cost_values);

        // Match the indices
        let neighbor_costs = i_range
            .flat_map(|e| iter::repeat(e).zip(j_range.clone()))
            .zip(cost_values)
            .collect();

        trace!("Neighbors {:?}", neighbor_costs);
        neighbor_costs
    }

    fn get_3x3_cached_barrier_cells(
        &self,
        index: &ArrayIndex,
        cache: &ChunkCacheDecodedLruSizeLimit,
    ) -> Vec<ArrayIndex> {
        let (i_range, j_range, subset) = self.neighborhood_subset(index);
        self.ensure_derived_data_for_subset(&cache.array(), &subset);
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

    pub(super) fn get_3x3_soft_barrier_cells(
        &self,
        index: &ArrayIndex,
        dropped_soft_groups: usize,
    ) -> Vec<ArrayIndex> {
        let retry_state = dropped_soft_groups.min(self.soft_barrier_groups.len());
        self.get_3x3_cached_barrier_cells(index, &self.cumulative_soft_barrier_caches[retry_state])
    }

    pub(super) fn grid_shape(&self) -> (u64, u64) {
        (self.grid_nrows, self.grid_ncols)
    }

    fn ensure_derived_data_for_subset(
        &self,
        array: &zarrs::array::Array<dyn zarrs::storage::ReadableStorageTraits>,
        subset: &zarrs::array_subset::ArraySubset,
    ) {
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
                if self.cost_chunk_idx.read().unwrap()[[ci as usize, cj as usize]] {
                    trace!("Derived data for chunk ({}, {}) already calculated", ci, cj);
                    continue;
                }

                debug!("Requesting write lock for cost_chunk_idx ({}, {})", ci, cj);
                let mut chunk_idx = self
                    .cost_chunk_idx
                    .write()
                    .expect("Failed to acquire write lock");
                debug!("Acquired write lock for cost_chunk_idx ({}, {})", ci, cj);
                if chunk_idx[[ci as usize, cj as usize]] {
                    trace!(
                        "Derived data for chunk ({}, {}) already calculated while waiting for the lock",
                        ci, cj
                    );
                } else {
                    self.calculate_chunk_derived_data(ci, cj);
                    chunk_idx[[ci as usize, cj as usize]] = true;
                    debug!(
                        "Recorded derived data for chunk ({}, {}) as calculated. Total number of computed chunks: {}",
                        ci,
                        cj,
                        chunk_idx.iter().filter(|&&value| value).count()
                    );
                }
                debug!("Released write lock for cost_chunk_idx ({}, {})", ci, cj);
            }
        }
    }

    #[cfg(test)]
    pub(super) fn hard_barrier_layers(&self) -> &[crate::cost::BarrierLayer] {
        &self.hard_barrier_layers
    }

    #[cfg(test)]
    pub(super) fn soft_barrier_group_count(&self) -> usize {
        self.soft_barrier_groups.len()
    }

    #[cfg(test)]
    pub(super) fn get_3x3_barrier_cells(
        &self,
        index: &ArrayIndex,
        barrier_layers: &[crate::cost::BarrierLayer],
    ) -> Vec<ArrayIndex> {
        if barrier_layers.is_empty() {
            return Vec::new();
        }

        let (i_range, j_range, subset) = self.neighborhood_subset(index);
        let mut features = LazySubset::<f32>::new(self.source.clone(), subset);
        let barrier_masks = barrier_layers
            .iter()
            .map(|layer| crate::cost::build_single_barrier_layer(layer, &mut features))
            .collect::<Vec<_>>();
        let mut barrier_cells = Vec::new();

        for (row_offset, ir) in i_range.enumerate() {
            for (col_offset, jr) in j_range.clone().enumerate() {
                let is_barrier = barrier_masks
                    .iter()
                    .any(|mask| mask[[0, row_offset, col_offset]]);
                if is_barrier {
                    barrier_cells.push(ArrayIndex { i: ir, j: jr });
                }
            }
        }

        barrier_cells
    }
}

/// Split the decoded chunk cache budget across derived dataset layers.
///
/// One third of the total budget is assigned to the dynamic cost cache and
/// one third to the invariant cost cache. The remaining budget is then split
/// between the hard-barrier cache and the cumulative soft-barrier caches:
/// half goes to the hard-barrier cache and the rest is divided evenly across
/// each soft-barrier retry-state cache.
///
/// Every allocation is clamped to at least 1 so the cache setup remains
/// valid even when the total cache budget is very small.
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

fn empty_bool_mask(subset: &zarrs::array_subset::ArraySubset) -> ndarray::ArrayD<bool> {
    ndarray::ArrayD::<bool>::from_elem(
        ndarray::IxDyn(
            &subset
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim).expect("subset dimension exceeds usize range"))
                .collect::<Vec<_>>(),
        ),
        false,
    )
}

fn combine_barrier_layers_for_subset(
    barrier_layers: &[crate::cost::BarrierLayer],
    features: &mut LazySubset<f32>,
    subset: &zarrs::array_subset::ArraySubset,
) -> Option<ndarray::ArrayD<bool>> {
    if barrier_layers.is_empty() {
        return None;
    }

    let barrier_masks = barrier_layers
        .iter()
        .map(|layer| crate::cost::build_single_barrier_layer(layer, features))
        .collect::<Vec<_>>();
    let mut output = empty_bool_mask(subset);
    for mask in barrier_masks {
        ndarray::Zip::from(&mut output)
            .and(mask.view())
            .for_each(|out, value| *out = *out || *value);
    }

    Some(output)
}

#[cfg(test)]
/// Make a LazySubset from a source and array subset to be used in tests
///
/// # Returns
/// An initialized LazySubset<f32> instance.
pub(crate) fn make_lazy_subset_for_tests(
    source: ReadableListableStorage,
    subset: zarrs::array_subset::ArraySubset,
) -> LazySubset<f32> {
    LazySubset::new(source, subset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::SQRT_2;
    use std::sync::Arc;
    use test_case::test_case;
    use zarrs::array::{ArrayBuilder, DataType, FillValue};
    use zarrs::filesystem::FilesystemStore;
    use zarrs::group::GroupBuilder;
    use zarrs::storage::ReadableWritableListableStorage;

    #[test]
    fn test_simple_cost_function_get_3x3() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "B", "C", "cost"]);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "A"}]}"#).unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let test_points = [ArrayIndex { i: 3, j: 1 }, ArrayIndex { i: 2, j: 2 }];
        let array = zarrs::array::Array::open(dataset.source.clone(), "/A").unwrap();
        for point in test_points {
            let results = dataset.get_3x3(&point);

            // index 0, 0 has a cost of 0 and should therefore be filtered out
            assert!(
                !results
                    .iter()
                    .any(|(ArrayIndex { i, j }, _)| *i == 0 && *j == 0)
            );
            let ArrayIndex { i: ci, j: cj } = point;
            let center_subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
                0..1,
                ci..(ci + 1),
                cj..(cj + 1),
            ]);
            let center_cost: f32 = array
                .retrieve_array_subset_elements(&center_subset)
                .expect("Error reading zarr data")[0];

            for (ArrayIndex { i, j }, val) in results {
                let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
                    0..1,
                    i..(i + 1),
                    j..(j + 1),
                ]);
                let subset_elements: Vec<f32> = array
                    .retrieve_array_subset_elements(&subset)
                    .expect("Error reading zarr data");
                assert_eq!(subset_elements.len(), 1);

                let neighbor_cost: f32 = subset_elements[0];
                let mut averaged_cost: f32 = 0.5 * (neighbor_cost + center_cost);
                if i != ci && j != cj {
                    averaged_cost *= SQRT_2;
                }
                assert_eq!(averaged_cost, val)
            }
        }
    }

    #[test]
    fn test_open_rejects_representative_variable_with_too_few_dimensions() {
        // Cannot use `ZarrTestBuilder` here because we need to purposely
        // build an incorrectly formatted dataset
        let tmp_path = tempfile::TempDir::new().unwrap();
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(tmp_path.path()).unwrap());

        GroupBuilder::new()
            .build(store.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();

        ArrayBuilder::new(
            vec![3, 4],
            vec![3, 4],
            DataType::Float32,
            FillValue::from(zarrs::array::ZARR_NAN_F32),
        )
        .build(store, "/A")
        .unwrap()
        .store_metadata()
        .unwrap();

        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "A"}]}"#).unwrap();

        let error = Dataset::open(tmp_path.path(), cost_function, 1_000)
            .err()
            .expect("Expected Dataset::open to reject a 2D representative variable");

        assert!(matches!(
            error,
            Error::InvalidDatasetShape {
                variable,
                min_rank: 3,
                shape,
            } if variable == "A" && shape == vec![3, 4]
        ));
    }

    #[test]
    fn test_simple_invariant_cost_function_get_3x3() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "B", "C", "cost"]);
        let cost_function = CostFunction::from_json(
            r#"{"cost_layers": [{"layer_name": "A", "is_invariant": true}]}"#,
        )
        .unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let test_points = [ArrayIndex { i: 3, j: 1 }, ArrayIndex { i: 2, j: 2 }];
        let array = zarrs::array::Array::open(dataset.source.clone(), "/A").unwrap();
        for point in test_points {
            let results = dataset.get_3x3(&point);

            for (ArrayIndex { i, j }, val) in results {
                let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
                    0..1,
                    i..(i + 1),
                    j..(j + 1),
                ]);
                let subset_elements: Vec<f32> = array
                    .retrieve_array_subset_elements(&subset)
                    .expect("Error reading zarr data");
                assert_eq!(subset_elements.len(), 1);
                assert_eq!(subset_elements[0], val)
            }
        }
    }

    #[test]
    fn test_sample_cost_function_get_3x3() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "B", "C", "cost"]);
        let cost_function = crate::cost::sample::cost_function();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let test_points = [ArrayIndex { i: 3, j: 1 }, ArrayIndex { i: 2, j: 2 }];
        let array_a = zarrs::array::Array::open(dataset.source.clone(), "/A").unwrap();
        let array_b = zarrs::array::Array::open(dataset.source.clone(), "/B").unwrap();
        let array_c = zarrs::array::Array::open(dataset.source.clone(), "/C").unwrap();
        for point in test_points {
            let results = dataset.get_3x3(&point);

            // index 0, 0 has a cost of 0 and should therefore be filtered out
            assert!(
                !results
                    .iter()
                    .any(|(ArrayIndex { i, j }, _)| *i == 0 && *j == 0)
            );
            let ArrayIndex { i: ci, j: cj } = point;
            let center_subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
                0..1,
                ci..(ci + 1),
                cj..(cj + 1),
            ]);
            let center_a = array_a
                .retrieve_array_subset_elements::<f32>(&center_subset)
                .expect("Error reading zarr data")[0];
            let center_b = array_b
                .retrieve_array_subset_elements::<f32>(&center_subset)
                .expect("Error reading zarr data")[0];
            let center_c = array_c
                .retrieve_array_subset_elements::<f32>(&center_subset)
                .expect("Error reading zarr data")[0];

            let center_cost: f32 =
                center_a + center_b * 100. + center_a * center_b + center_c * center_a * 2.;

            for (ArrayIndex { i, j }, val) in results {
                let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
                    0..1,
                    i..(i + 1),
                    j..(j + 1),
                ]);
                let subset_elements_a: Vec<f32> = array_a
                    .retrieve_array_subset_elements(&subset)
                    .expect("Error reading zarr data");
                assert_eq!(subset_elements_a.len(), 1);

                let subset_elements_b: Vec<f32> = array_b
                    .retrieve_array_subset_elements(&subset)
                    .expect("Error reading zarr data");
                assert_eq!(subset_elements_b.len(), 1);

                let subset_elements_c: Vec<f32> = array_c
                    .retrieve_array_subset_elements(&subset)
                    .expect("Error reading zarr data");
                assert_eq!(subset_elements_c.len(), 1);

                // based on the const function definition
                let neighbor_cost: f32 = subset_elements_a[0]
                    + subset_elements_b[0] * 100.
                    + subset_elements_a[0] * subset_elements_b[0]
                    + subset_elements_c[0] * subset_elements_a[0] * 2.;
                let mut averaged_cost: f32 = 0.5 * (neighbor_cost + center_cost);
                if i != ci && j != cj {
                    averaged_cost *= SQRT_2;
                }
                // add invariant cost
                let expected: f32 = averaged_cost + subset_elements_c[0] * 100.;

                let diff: f32 = (expected - val).abs();
                assert!(
                    diff < 1e-4_f32,
                    "Unexpected cost for {:?}: {:?} (expected {:?}): ",
                    (i, j),
                    val,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_get_3x3_single_item_array() {
        let tmp = samples::cost_as_index_zarr(1, 1, 1, 1, 1, 1);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: 0, j: 0 });

        // index 0, 0 has a cost of 0 and should therefore be filtered out
        assert!(
            !results
                .iter()
                .any(|(ArrayIndex { i, j }, _)| *i == 0 && *j == 0)
        );

        assert_eq!(results, vec![]);
    }

    #[test_case((0, 0), vec![(0, 1, 0.5), (1, 0, 1.0), (1, 1, 1.5 * SQRT_2)] ; "top left corner")]
    #[test_case((0, 1), vec![(1, 0, 1.5 * SQRT_2), (1, 1, 2.)] ; "top right corner")]
    #[test_case((1, 0), vec![(0, 1, 1.5 * SQRT_2), (1, 1, 2.5)] ; "bottom left corner")]
    #[test_case((1, 1), vec![(0, 1, 2.), (1, 0, 2.5)] ; "bottom right corner")]
    fn test_get_3x3_two_by_two_array((si, sj): (u64, u64), expected_output: Vec<(u64, u64, f32)>) {
        let tmp = samples::cost_as_index_zarr(1, 2, 2, 1, 2, 2);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: si, j: sj });

        // index 0, 0 has a cost of 0 and should therefore be filtered out
        assert!(
            !results
                .iter()
                .any(|(ArrayIndex { i, j }, _)| *i == 0 && *j == 0)
        );

        assert_eq!(
            results,
            expected_output
                .into_iter()
                .map(|(i, j, v)| (ArrayIndex { i, j }, v))
                .collect::<Vec<_>>()
        );
    }

    #[test_case((0, 0), vec![(0, 1, 0.5), (1, 0, 1.5), (1, 1, 2.0 * SQRT_2)] ; "top left corner")]
    #[test_case((0, 1), vec![(0, 2, 1.5), (1, 0, 2.0 * SQRT_2), (1, 1, 2.5), (1, 2, 3. * SQRT_2)] ; "top middle")]
    #[test_case((0, 2), vec![(0, 1, 1.5), (1, 1, 3.0 * SQRT_2), (1, 2, 3.5)] ; "top right corner")]
    #[test_case((1, 0), vec![(0, 1, 2.0 * SQRT_2), (1, 1, 3.5), (2, 0, 4.5), (2, 1, 5.0 * SQRT_2)] ; "middle left")]
    #[test_case((1, 1), vec![(0, 1, 2.5), (0, 2, 3.0 * SQRT_2), (1, 0, 3.5), (1, 2, 4.5), (2, 0, 5.0 * SQRT_2), (2, 1, 5.5), (2, 2, 6.0 * SQRT_2)] ; "middle middle")]
    #[test_case((1, 2), vec![(0, 1, 3.0 * SQRT_2), (0, 2, 3.5), (1, 1, 4.5), (2, 1, 6.0 * SQRT_2), (2, 2, 6.5)] ; "middle right")]
    #[test_case((2, 0), vec![(1, 0, 4.5), (1, 1, 5.0 * SQRT_2), (2, 1, 6.5)] ; "bottom left corner")]
    #[test_case((2, 1), vec![(1, 0, 5.0 * SQRT_2), (1, 1, 5.5), (1, 2, 6.0 * SQRT_2), (2, 0, 6.5), (2, 2, 7.5)] ; "bottom middle")]
    #[test_case((2, 2), vec![(1, 1, 6.0 * SQRT_2), (1, 2, 6.5), (2, 1, 7.5)] ; "bottom right corner")]
    fn test_get_3x3_three_by_three_array(
        (si, sj): (u64, u64),
        expected_output: Vec<(u64, u64, f32)>,
    ) {
        let tmp = samples::cost_as_index_zarr(1, 3, 3, 1, 3, 3);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: si, j: sj });

        // index 0, 0 has a cost of 0 and should therefore be filtered out
        assert!(
            !results
                .iter()
                .any(|(ArrayIndex { i, j }, _)| *i == 0 && *j == 0)
        );

        assert_eq!(
            results,
            expected_output
                .into_iter()
                .map(|(i, j, v)| (ArrayIndex { i, j }, v))
                .collect::<Vec<_>>()
        );
    }

    #[test_case((0, 0), vec![(0, 1, 0.5), (1, 0, 2.), (1, 1, 2.5 * SQRT_2)] ; "top left corner")]
    #[test_case((0, 1), vec![(0, 2, 1.5), (1, 0, 2.5 * SQRT_2), (1, 1, 3.), (1, 2, 3.5 * SQRT_2)] ; "top left edge")]
    #[test_case((0, 2), vec![(0, 1, 1.5), (0, 3, 2.5), (1, 1, 3.5 * SQRT_2), (1, 2, 4.), (1, 3, 4.5 * SQRT_2)] ; "top right edge")]
    #[test_case((0, 3), vec![(0, 2, 2.5), (1, 2, 4.5 * SQRT_2), (1, 3, 5.)] ; "top right corner")]
    #[test_case((1, 0), vec![(0, 1, 2.5 * SQRT_2), (1, 1, 4.5), (2, 0, 6.), (2, 1, 6.5 * SQRT_2)] ; "left top edge")]
    #[test_case((1, 3), vec![(0, 2, 4.5 * SQRT_2), (0, 3, 5.), (1, 2, 6.5), (2, 2, 8.5 * SQRT_2), (2, 3, 9.)] ; "right top edge")]
    #[test_case((2, 0), vec![(1, 0, 6.), (1, 1, 6.5 * SQRT_2), (2, 1, 8.5), (3, 0, 10.), (3, 1, 10.5 * SQRT_2)] ; "left bottom edge")]
    #[test_case((2, 3), vec![(1, 2, 8.5 * SQRT_2), (1, 3, 9.), (2, 2, 10.5), (3, 2, 12.5 * SQRT_2), (3, 3, 13.)] ; "right bottom edge")]
    #[test_case((3, 0), vec![(2, 0, 10.), (2, 1, 10.5 * SQRT_2), (3, 1, 12.5)] ; "bottom left corner")]
    #[test_case((3, 1), vec![(2, 0, 10.5 * SQRT_2), (2, 1, 11.), (2, 2, 11.5 * SQRT_2), (3, 0, 12.5), (3, 2, 13.5)] ; "bottom left edge")]
    #[test_case((3, 2), vec![(2, 1, 11.5 * SQRT_2), (2, 2, 12.), (2, 3, 12.5 * SQRT_2), (3, 1, 13.5), (3, 3, 14.5)] ; "bottom right edge")]
    #[test_case((3, 3), vec![(2, 2, 12.5 * SQRT_2), (2, 3, 13.), (3, 2, 14.5)] ; "bottom right corner")]
    fn test_get_3x3_four_by_four_array(
        (si, sj): (u64, u64),
        expected_output: Vec<(u64, u64, f32)>,
    ) {
        let tmp = samples::cost_as_index_zarr(1, 4, 4, 1, 2, 2);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: si, j: sj });

        // index 0, 0 has a cost of 0 and should therefore be filtered out
        assert!(
            !results
                .iter()
                .any(|(ArrayIndex { i, j }, _)| *i == 0 && *j == 0)
        );

        assert_eq!(
            results,
            expected_output
                .into_iter()
                .map(|(i, j, v)| (ArrayIndex { i, j }, v))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_get_3x3_with_invariant_and_friction_layers() {
        // Define cost function: A normal, C invariant, friction from B * 0.5
        let json = r#"
        {
            "cost_layers": [
                {"layer_name": "A"},
                {"layer_name": "C", "is_invariant": true}
            ],
            "friction_layers": [
                {"multiplier_layer": "B", "multiplier_scalar": 0.5}
            ]
        }
        "#;

        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(samples::LayerConfig::sequential("A", 1))
            .layer(samples::LayerConfig::constant("B", 0.2_f32))
            .layer(samples::LayerConfig::constant("C", 10.0_f32))
            .build()
            .expect("Error creating test zarr");
        let cost_function = CostFunction::from_json(json).unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        // Request center neighbors
        let point = ArrayIndex { i: 1, j: 1 };
        let results = dataset.get_3x3(&point);

        // Build expected results: for each neighbor (excluding center),
        // averaged = 0.5 * (A_neighbor + A_center)
        // if diagonal => averaged *= sqrt(2)
        // total_before_friction = averaged + C_neighbor
        // friction = B_neighbor * 0.5
        // expected = total_before_friction * (1 + friction)

        let a_array = zarrs::array::Array::open(dataset.source.clone(), "/A").unwrap();
        let b_array = zarrs::array::Array::open(dataset.source.clone(), "/B").unwrap();
        let c_array = zarrs::array::Array::open(dataset.source.clone(), "/C").unwrap();

        let mut expected: Vec<(ArrayIndex, f32)> = vec![];
        let center_subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[0..1, 1..2, 1..2]);
        let center_a: f32 = a_array
            .retrieve_array_subset_elements(&center_subset)
            .unwrap()[0];

        for ir in 0..3u64 {
            for jr in 0..3u64 {
                if ir == 1 && jr == 1 {
                    continue; // skip center
                }
                let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
                    0..1,
                    ir..(ir + 1),
                    jr..(jr + 1),
                ]);
                let a_n: f32 = a_array.retrieve_array_subset_elements(&subset).unwrap()[0];
                let b_n: f32 = b_array.retrieve_array_subset_elements(&subset).unwrap()[0];
                let c_n: f32 = c_array.retrieve_array_subset_elements(&subset).unwrap()[0];

                let mut averaged = 0.5_f32 * (a_n + center_a);
                if ir != 1 && jr != 1 {
                    averaged *= std::f32::consts::SQRT_2;
                }
                let total_before = averaged + c_n;
                let friction = b_n * 0.5_f32;
                let expected_val = total_before * (1.0_f32 + friction);
                expected.push((ArrayIndex { i: ir, j: jr }, expected_val));
            }
        }

        // Compare results: lengths and per-item approx equality
        assert_eq!(results.len(), expected.len());
        for (idx, val) in expected {
            let found = results
                .iter()
                .find(|(ai, _)| ai.i == idx.i && ai.j == idx.j);
            assert!(found.is_some(), "Missing neighbor {:?} in results", idx);
            let actual = found.unwrap().1;
            let diff = (actual - val).abs();
            assert!(
                diff < 1e-5,
                "mismatch for {:?}: actual={} expected={} diff={}",
                idx,
                actual,
                val,
                diff
            );
        }
    }

    #[test_case(r#"{"cost_layers": [{"layer_name": "B"}], "ignore_invalid_costs": true}"# ; "zero layer")]
    #[test_case(r#"{"cost_layers": [{"layer_name": "C"}], "ignore_invalid_costs": true}"# ; "negative layer")]
    fn test_get_3x3_with_hard_barriered_layers(json: &str) {
        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(samples::LayerConfig::sequential("A", 1))
            .layer(samples::LayerConfig::constant("B", 0_f32))
            .layer(samples::LayerConfig::constant("C", -1_f32))
            .build()
            .expect("Error creating test zarr");
        let cost_function = CostFunction::from_json(json).unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: 1, j: 1 });
        assert!(
            results.is_empty(),
            "Found data with `ignore_invalid_costs=true`"
        );
    }

    #[test_case(r#"{"cost_layers": [{"layer_name": "B"}], "ignore_invalid_costs": false}"# ; "zero layer")]
    #[test_case(r#"{"cost_layers": [{"layer_name": "C"}], "ignore_invalid_costs": false}"# ; "negative layer")]
    fn test_get_3x3_with_soft_barrier_layers(json: &str) {
        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(samples::LayerConfig::sequential("A", 1))
            .layer(samples::LayerConfig::constant("B", 0_f32))
            .layer(samples::LayerConfig::constant("C", -1_f32))
            .build()
            .expect("Error creating test zarr");
        let cost_function = CostFunction::from_json(json).unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: 1, j: 1 });
        assert_eq!(results.len(), 8);

        let mut expected: Vec<(ArrayIndex, f32)> = vec![];
        for ir in 0..3u64 {
            for jr in 0..3u64 {
                if ir == 1 && jr == 1 {
                    continue; // skip center
                }

                let mut averaged = 1e10f32;
                if ir != 1 && jr != 1 {
                    averaged *= std::f32::consts::SQRT_2;
                }
                expected.push((ArrayIndex { i: ir, j: jr }, averaged));
            }
        }

        for (idx, val) in expected {
            let found = results
                .iter()
                .find(|(ai, _)| ai.i == idx.i && ai.j == idx.j);
            assert!(found.is_some(), "Missing neighbor {:?} in results", idx);
            let actual = found.unwrap().1;
            let diff = (actual - val).abs();
            assert!(
                diff < 1e-5,
                "mismatch for {:?}: actual={} expected={} diff={}",
                idx,
                actual,
                val,
                diff
            );
        }
    }

    #[test]
    fn test_get_3x3_keeps_explicit_barriers_out_of_cached_costs() {
        let json = r#"
        {
            "cost_layers": [{"layer_name": "A"}],
            "barrier_layers": [
                {
                    "layer_name": "B",
                    "barrier_operator": "eq",
                    "barrier_threshold": 1.0
                }
            ]
        }
        "#;

        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(samples::LayerConfig::sequential("A", 1))
            .layer(samples::LayerConfig::new(
                "B",
                samples::FillStrategy::Values(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            ))
            .build()
            .expect("Error creating test zarr");
        let cost_function = CostFunction::from_json(json).unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: 1, j: 1 });
        let (i_range, j_range, subset) = dataset.neighborhood_subset(&ArrayIndex { i: 1, j: 1 });
        let raw_costs = dataset.get_neighbor_costs(i_range, j_range, &subset, false);
        assert_eq!(raw_costs.len(), 9);
        assert_eq!(
            results,
            vec![
                (ArrayIndex { i: 0, j: 0 }, 3.0 * std::f32::consts::SQRT_2),
                (ArrayIndex { i: 0, j: 2 }, 4.0 * std::f32::consts::SQRT_2),
                (ArrayIndex { i: 2, j: 0 }, 6.0 * std::f32::consts::SQRT_2),
                (ArrayIndex { i: 2, j: 2 }, 7.0 * std::f32::consts::SQRT_2),
            ]
        );
    }

    #[test]
    fn test_explicit_barriers_do_not_modify_cached_costs_when_invalid_costs_are_soft() {
        let json = r#"
        {
            "cost_layers": [{"layer_name": "A"}],
            "barrier_layers": [
                {
                    "layer_name": "B",
                    "barrier_operator": "eq",
                    "barrier_threshold": 1.0,
                    "barrier_importance": 1
                }
            ],
            "ignore_invalid_costs": false
        }
        "#;

        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(samples::LayerConfig::constant("A", 1.0))
            .layer(samples::LayerConfig::new(
                "B",
                samples::FillStrategy::Values(vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
            ))
            .build()
            .expect("Error creating test zarr");
        let cost_function = CostFunction::from_json(json).unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: 1, j: 1 });
        assert_eq!(results.len(), 8);
    }

    #[test]
    fn test_cumulative_soft_barrier_masks_follow_retry_state() {
        let json = r#"
        {
            "cost_layers": [{"layer_name": "A"}],
            "barrier_layers": [
                {
                    "layer_name": "B",
                    "barrier_operator": "eq",
                    "barrier_threshold": 1.0,
                    "barrier_importance": 1
                },
                {
                    "layer_name": "C",
                    "barrier_operator": "eq",
                    "barrier_threshold": 1.0,
                    "barrier_importance": 2
                }
            ]
        }
        "#;

        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(samples::LayerConfig::constant("A", 1.0))
            .layer(samples::LayerConfig::new(
                "B",
                samples::FillStrategy::Values(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ))
            .layer(samples::LayerConfig::new(
                "C",
                samples::FillStrategy::Values(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ))
            .build()
            .expect("Error creating test zarr");
        let dataset = Dataset::open(tmp.path(), CostFunction::from_json(json).unwrap(), 1_000)
            .expect("Error opening dataset");

        assert_eq!(dataset.soft_barrier_group_count(), 2);
        let center = ArrayIndex { i: 1, j: 1 };
        dataset.get_3x3(&center);

        assert_eq!(
            dataset.get_3x3_soft_barrier_cells(&center, 0),
            vec![ArrayIndex { i: 0, j: 1 }, ArrayIndex { i: 1, j: 0 }]
        );
        assert_eq!(
            dataset.get_3x3_soft_barrier_cells(&center, 1),
            vec![ArrayIndex { i: 0, j: 1 }]
        );
        assert!(dataset.get_3x3_soft_barrier_cells(&center, 2).is_empty());
        assert!(dataset.get_3x3_soft_barrier_cells(&center, 99).is_empty());
    }

    #[test]
    fn test_cumulative_soft_barrier_masks_or_tied_importance_groups() {
        let json = r#"
        {
            "cost_layers": [{"layer_name": "A"}],
            "barrier_layers": [
                {
                    "layer_name": "B",
                    "barrier_operator": "eq",
                    "barrier_threshold": 1.0,
                    "barrier_importance": 1
                },
                {
                    "layer_name": "C",
                    "barrier_operator": "eq",
                    "barrier_threshold": 1.0,
                    "barrier_importance": 1
                }
            ]
        }
        "#;

        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(samples::LayerConfig::constant("A", 1.0))
            .layer(samples::LayerConfig::new(
                "B",
                samples::FillStrategy::Values(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ))
            .layer(samples::LayerConfig::new(
                "C",
                samples::FillStrategy::Values(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ))
            .build()
            .expect("Error creating test zarr");
        let dataset = Dataset::open(tmp.path(), CostFunction::from_json(json).unwrap(), 1_000)
            .expect("Error opening dataset");

        let center = ArrayIndex { i: 1, j: 1 };
        dataset.get_3x3(&center);

        assert_eq!(dataset.soft_barrier_group_count(), 1);
        assert_eq!(
            dataset.get_3x3_soft_barrier_cells(&center, 0),
            vec![ArrayIndex { i: 0, j: 1 }, ArrayIndex { i: 1, j: 0 }]
        );
        assert!(dataset.get_3x3_soft_barrier_cells(&center, 1).is_empty());
    }

    #[test]
    fn test_open_extracts_hard_barriers_while_preserving_barrier_free_costs() {
        let json = r#"
        {
            "cost_layers": [{"layer_name": "A"}],
            "barrier_layers": [
                {
                    "layer_name": "B",
                    "barrier_operator": "eq",
                    "barrier_threshold": 1.0
                }
            ]
        }
        "#;

        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(samples::LayerConfig::sequential("A", 1))
            .layer(samples::LayerConfig::new(
                "B",
                samples::FillStrategy::Values(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            ))
            .build()
            .expect("Error creating test zarr");
        let cost_function = CostFunction::from_json(json).unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        assert_eq!(dataset.hard_barrier_layers().len(), 1);

        let results = dataset.get_3x3(&ArrayIndex { i: 1, j: 1 });
        let (i_range, j_range, subset) = dataset.neighborhood_subset(&ArrayIndex { i: 1, j: 1 });
        let raw_costs = dataset.get_neighbor_costs(i_range, j_range, &subset, false);
        let barrier_cells = dataset
            .get_3x3_barrier_cells(&ArrayIndex { i: 1, j: 1 }, dataset.hard_barrier_layers());

        assert_eq!(raw_costs.len(), 9);
        assert_eq!(results.len(), 4);
        assert_eq!(
            barrier_cells,
            vec![
                ArrayIndex { i: 0, j: 1 },
                ArrayIndex { i: 1, j: 0 },
                ArrayIndex { i: 1, j: 2 },
                ArrayIndex { i: 2, j: 1 },
            ]
        );
    }
}
