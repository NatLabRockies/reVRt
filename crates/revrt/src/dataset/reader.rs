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

#[cfg(test)]
mod tests {
    use std::f32::consts::SQRT_2;
    use std::sync::Arc;

    use ndarray::Array3;
    use tempfile::TempDir;
    use test_case::test_case;
    use zarrs::array::Array;
    use zarrs::array_subset::ArraySubset;
    use zarrs::filesystem::FilesystemStore;
    use zarrs::storage::ReadableListableStorage;

    use super::*;
    use crate::dataset::samples::{LayerConfig, ZarrTestBuilder};
    use crate::dataset::swap::{initialize_swap, inspect_source_layout};

    #[test]
    fn distribute_cache_budgets_splits_budget_across_cache_types() {
        let budgets = distribute_cache_budgets(120, 4);

        assert_eq!(budgets.per_cost_cache, 40);
        assert_eq!(budgets.hard_barrier_cache, 20);
        assert_eq!(budgets.per_soft_barrier_cache, 5);
    }

    #[test]
    fn distribute_cache_budgets_keeps_nonzero_budgets_for_tiny_cache_sizes() {
        let budgets = distribute_cache_budgets(1, 0);

        assert_eq!(budgets.per_cost_cache, 1);
        assert_eq!(budgets.hard_barrier_cache, 1);
        assert_eq!(budgets.per_soft_barrier_cache, 1);
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

        let (i_range, j_range, subset) = reader.neighborhood_subset(&ArrayIndex { i, j });

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

        let neighbors =
            fixture
                .reader
                .get_3x3(&ArrayIndex { i: 1, j: 1 }, true, |_array, _subset| {});

        let expected = [
            (ArrayIndex { i: 0, j: 0 }, 3.0 * SQRT_2 + 1.0),
            (ArrayIndex { i: 0, j: 2 }, 4.0 * SQRT_2 + 1.0),
            (ArrayIndex { i: 1, j: 0 }, 5.5),
            (ArrayIndex { i: 2, j: 0 }, 6.0 * SQRT_2 + 1.0),
            (ArrayIndex { i: 2, j: 1 }, 7.5),
            (ArrayIndex { i: 2, j: 2 }, 7.0 * SQRT_2 + 1.0),
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

        let neighbors =
            fixture
                .reader
                .get_3x3(&ArrayIndex { i: 1, j: 1 }, true, |_array, _subset| {});

        assert!(neighbors.is_empty());
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
            &ArrayIndex { i: 1, j: 1 },
            0,
            |_array, _subset| {},
        );
        let retry_one = fixture.reader.get_3x3_soft_barrier_cells(
            &ArrayIndex { i: 1, j: 1 },
            1,
            |_array, _subset| {},
        );

        assert_eq!(
            retry_zero,
            vec![ArrayIndex { i: 0, j: 1 }, ArrayIndex { i: 2, j: 0 }]
        );
        assert_eq!(
            retry_one,
            vec![ArrayIndex { i: 0, j: 0 }, ArrayIndex { i: 2, j: 1 }]
        );
    }

    fn reader_for_grid(grid_nrows: u64, grid_ncols: u64) -> NeighborhoodReader {
        let fixture = reader_fixture_with_shape(
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
            3,
            3,
            cost_values,
            invariant_values,
            hard_barrier_values,
            soft_retry_zero_values,
            soft_retry_one_values,
        )
    }

    fn reader_fixture_with_shape(
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
        let layout =
            inspect_source_layout(&source).expect("source layout inspection should succeed");

        let swap_tmp = TempDir::new().expect("could not create temporary swap");
        let swap = initialize_swap(swap_tmp.path(), &layout, 1)
            .expect("swap initialization should succeed");

        store_f32_layer(swap.clone(), "/cost", grid_nrows, grid_ncols, cost_values);
        store_f32_layer(
            swap.clone(),
            "/cost_invariant",
            grid_nrows,
            grid_ncols,
            invariant_values,
        );
        store_bool_layer(
            swap.clone(),
            "/hard_barrier_mask",
            grid_nrows,
            grid_ncols,
            hard_barrier_values,
        );
        store_bool_layer(
            swap.clone(),
            "/soft_barrier_mask_retry_0",
            grid_nrows,
            grid_ncols,
            soft_retry_zero_values,
        );
        store_bool_layer(
            swap.clone(),
            "/soft_barrier_mask_retry_1",
            grid_nrows,
            grid_ncols,
            soft_retry_one_values,
        );

        let reader = NeighborhoodReader::open(swap, 90, 1, layout).expect("reader should open");

        ReaderFixture {
            _source_tmp: source_tmp,
            _swap_tmp: swap_tmp,
            reader,
        }
    }

    fn store_f32_layer(
        swap: ReadableWritableListableStorage,
        path: &str,
        grid_nrows: u64,
        grid_ncols: u64,
        values: Vec<f32>,
    ) {
        let data =
            Array3::from_shape_vec((1_usize, grid_nrows as usize, grid_ncols as usize), values)
                .expect("f32 layer values should match requested shape");
        let array = Array::open(swap, path).expect("expected f32 layer to exist");
        let subset = chunk_subset(&array);

        array
            .store_chunks_ndarray(&subset, data)
            .expect("could not store f32 layer data");
    }

    fn store_bool_layer(
        swap: ReadableWritableListableStorage,
        path: &str,
        grid_nrows: u64,
        grid_ncols: u64,
        values: Vec<bool>,
    ) {
        let data =
            Array3::from_shape_vec((1_usize, grid_nrows as usize, grid_ncols as usize), values)
                .expect("bool layer values should match requested shape");
        let array = Array::open(swap, path).expect("expected bool layer to exist");
        let subset = chunk_subset(&array);

        array
            .store_chunks_ndarray(&subset, data)
            .expect("could not store bool layer data");
    }

    fn chunk_subset<T: ?Sized>(array: &Array<T>) -> ArraySubset {
        let chunk_grid_shape = array.chunk_grid_shape();

        ArraySubset::new_with_ranges(&[
            0..chunk_grid_shape[0],
            0..chunk_grid_shape[1],
            0..chunk_grid_shape[2],
        ])
    }

    struct ReaderFixture {
        _source_tmp: TempDir,
        _swap_tmp: TempDir,
        reader: NeighborhoodReader,
    }
}
