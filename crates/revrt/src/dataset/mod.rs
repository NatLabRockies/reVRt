//! Source Dataset Access
//!
//! This module manages access to the input feature data used during routing.
//!
//! It is built around Zarr-backed source arrays together with a temporary
//! swap dataset that stores derived cost and barrier layers. The public
//! surface of the module focuses on opening datasets, materializing derived
//! chunks on demand, and returning 3x3 neighborhoods suitable for routing.
//!
//! Zarr is currently a good fit because it supports concurrent chunk access,
//! rich metadata, and efficient compression for large raster-like grids.
//!
//! Note: This module is still in a transition state. The initial prototype
//! included the cost function directly here, but cost-function ownership was
//! moved up to the `Scenario` level.

mod derived;
mod lazy_subset;
mod reader;
#[cfg(test)]
pub(crate) mod samples;
mod state;
mod swap;

use std::path::PathBuf;

use tracing::{debug, info, trace};
use zarrs::storage::ReadableListableStorage;

use crate::ArrayIndex;
use crate::cost::CostFunction;
use crate::error::Result;
use derived::DerivedDataWriter;
pub(crate) use lazy_subset::LazySubset;
use reader::NeighborhoodReader;
use state::DerivedChunkState;
use swap::{initialize_swap, inspect_source_layout};

/// Manage source features together with derived swap-backed routing data.
///
/// A `Dataset` owns access to the original feature store, the temporary swap
/// dataset, the chunk-materialization state, and the cached readers used to
/// serve routing neighborhoods. Derived arrays are created lazily, chunk by
/// chunk, the first time a neighborhood read requires them.
pub(super) struct Dataset {
    /// Zarr storage containing the original feature arrays.
    #[allow(dead_code)]
    source: ReadableListableStorage,
    /// Temporary directory backing the swap dataset when one is auto-created.
    ///
    /// This is stored only to keep the directory alive for the lifetime of
    /// the dataset handle.
    #[allow(dead_code)]
    cost_path: Option<tempfile::TempDir>,
    /// State tracking which swap chunks have already been materialized.
    ///
    /// Internally this mirrors the source chunk grid as a boolean array where
    /// `true` means the corresponding derived swap chunk has already been
    /// computed.
    derived_chunk_state: DerivedChunkState,
    /// Writer responsible for materializing derived chunk data into swap.
    derived_data_writer: DerivedDataWriter,
    /// Reader responsible for cached neighborhood access to derived data.
    neighborhood_reader: NeighborhoodReader,
    /// Shape of the source routing grid as `(rows, cols)`.
    pub(super) grid_shape: (u64, u64),
}

impl Dataset {
    /// Open a dataset using an automatically managed temporary swap directory.
    ///
    /// # Arguments
    /// `path`: Filesystem path to the source Zarr dataset.
    /// `cost_function`: Cost function definition used to derive cost and
    ///                  barrier arrays.
    /// `cache_size`: Total cache budget, in bytes, for neighborhood readers.
    ///
    /// # Returns
    /// A `Dataset` backed by the source store and a new temporary swap store.
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

    /// Open a dataset using an existing or caller-provided swap path.
    ///
    /// # Arguments
    /// `path`: Filesystem path to the source Zarr dataset.
    /// `cost_function`: Cost function definition used to derive cost and
    ///                  barrier arrays.
    /// `cache_size`: Total cache budget, in bytes, for neighborhood readers.
    /// `swap_fp`: Filesystem path where the swap dataset should be created.
    ///
    /// # Returns
    /// A `Dataset` backed by the source store and the specified swap path.
    pub(super) fn open_with_swap<P: AsRef<std::path::Path>>(
        path: P,
        cost_function: CostFunction,
        cache_size: u64,
        swap_fp: PathBuf,
    ) -> Result<Self> {
        Self::open_with_path(path, cost_function, cache_size, swap_fp)
    }

    /// Open a dataset using the provided swap path and initialize internals.
    ///
    /// This helper inspects the source layout, initializes the swap dataset,
    /// prepares chunk-derivation tracking, and builds the cached neighborhood
    /// readers.
    ///
    /// # Arguments
    /// `path`: Filesystem path to the source Zarr dataset.
    /// `cost_function`: Cost function definition used to derive cost and
    ///                  barrier arrays.
    /// `cache_size`: Total cache budget, in bytes, for neighborhood readers.
    /// `swap_fp`: Filesystem path where the swap dataset should be created.
    ///
    /// # Returns
    /// A fully initialized `Dataset` ready to serve routing neighborhoods.
    fn open_with_path<P: AsRef<std::path::Path>>(
        path: P,
        cost_function: CostFunction,
        cache_size: u64,
        swap_fp: PathBuf,
    ) -> Result<Self> {
        debug!("Opening dataset: {:?}", path.as_ref());
        let soft_barrier_group_count = cost_function.soft_barrier_groups().len();

        let filesystem =
            zarrs::filesystem::FilesystemStore::new(path).expect("could not open filesystem store");
        let source: ReadableListableStorage = std::sync::Arc::new(filesystem);

        let source_layout = inspect_source_layout(&source)?;
        let swap = initialize_swap(swap_fp, &source_layout, soft_barrier_group_count)?;

        let derived_chunk_state = DerivedChunkState::new(&source_layout);
        let derived_data_writer =
            DerivedDataWriter::new(source.clone(), swap.clone(), cost_function);

        let neighborhood_reader = NeighborhoodReader::open(
            swap.clone(),
            cache_size,
            soft_barrier_group_count,
            source_layout,
        )?;
        let grid_shape = neighborhood_reader.grid_shape();

        trace!("Dataset opened successfully");
        Ok(Self {
            source,
            cost_path: None,
            derived_chunk_state,
            derived_data_writer,
            neighborhood_reader,
            grid_shape,
        })
    }

    /// Return reachable movement costs in the 3x3 neighborhood of an index.
    ///
    /// Derived data is materialized on demand for the needed swap chunks
    /// before the neighborhood is read.
    ///
    /// # Arguments
    /// `index`: Center cell whose 3x3 neighborhood should be queried.
    ///
    /// # Returns
    /// A vector of neighboring indices paired with movement costs from the
    /// center cell.
    pub(super) fn get_3x3(&self, index: &ArrayIndex) -> Vec<(ArrayIndex, f32)> {
        self.neighborhood_reader.get_3x3(
            index,
            !self.derived_data_writer.hard_barrier_layers().is_empty(),
            |array, subset| self.ensure_derived_data_for_subset(array, subset),
        )
    }

    /// Return soft-barrier cells in the 3x3 neighborhood of an index.
    ///
    /// The number of dropped soft groups is clamped to the available retry
    /// states before the matching cumulative soft barrier mask is queried.
    ///
    /// # Arguments
    /// `index`: Center cell whose 3x3 neighborhood should be queried.
    /// `dropped_soft_groups`: Number of soft barrier groups that have already
    ///                        been relaxed for the current retry state.
    ///
    /// # Returns
    /// A vector of neighborhood cells that remain soft barriers for the
    /// selected retry state.
    pub(super) fn get_3x3_soft_barrier_cells(
        &self,
        index: &ArrayIndex,
        dropped_soft_groups: usize,
    ) -> Vec<ArrayIndex> {
        let retry_state =
            dropped_soft_groups.min(self.derived_data_writer.soft_barrier_group_count());
        self.neighborhood_reader
            .get_3x3_soft_barrier_cells(index, retry_state, |array, subset| {
                self.ensure_derived_data_for_subset(array, subset)
            })
    }

    /// Ensure all derived swap data exists for a requested subset.
    ///
    /// Any chunk overlapped by `subset` is materialized exactly once through
    /// the tracked chunk state before cached neighborhood reads proceed.
    ///
    /// # Arguments
    /// `array`: Swap array whose subset is about to be accessed.
    /// `subset`: Requested subset within the swap array.
    fn ensure_derived_data_for_subset(
        &self,
        array: &zarrs::array::Array<dyn zarrs::storage::ReadableStorageTraits>,
        subset: &zarrs::array_subset::ArraySubset,
    ) {
        self.derived_chunk_state
            .ensure_derived_data_for_subset(array, subset, |ci, cj| {
                self.derived_data_writer.materialize_chunk(ci, cj)
            });
    }

    #[cfg(test)]
    /// Return the extracted hard barrier layers for assertions in tests.
    ///
    /// # Returns
    /// A shared slice of hard barrier layers owned by the derived data writer.
    pub(super) fn hard_barrier_layers(&self) -> &[crate::cost::BarrierLayer] {
        self.derived_data_writer.hard_barrier_layers()
    }

    #[cfg(test)]
    /// Return the number of soft barrier importance groups in tests.
    ///
    /// # Returns
    /// The number of distinct retry states backed by cumulative soft barrier
    /// masks, excluding the final fully relaxed state calculation details.
    pub(super) fn soft_barrier_group_count(&self) -> usize {
        self.derived_data_writer.soft_barrier_group_count()
    }

    #[cfg(test)]
    /// Delegate neighborhood-subset construction for test assertions.
    ///
    /// # Arguments
    /// `index`: Center cell whose neighborhood bounds should be computed.
    ///
    /// # Returns
    /// The clipped row range, clipped column range, and matching swap-array
    /// subset for the requested neighborhood.
    fn neighborhood_subset(
        &self,
        index: &ArrayIndex,
    ) -> (
        std::ops::Range<u64>,
        std::ops::Range<u64>,
        zarrs::array_subset::ArraySubset,
    ) {
        self.neighborhood_reader.neighborhood_subset(index)
    }

    #[cfg(test)]
    /// Delegate neighborhood cost reads for test assertions.
    ///
    /// # Arguments
    /// `i_range`: Row range covering the neighborhood.
    /// `j_range`: Column range covering the neighborhood.
    /// `subset`: Swap-array subset to read.
    /// `is_invariant`: Whether to read the invariant cost array.
    ///
    /// # Returns
    /// A vector pairing neighborhood coordinates with their cached cost values.
    fn get_neighbor_costs(
        &self,
        i_range: std::ops::Range<u64>,
        j_range: std::ops::Range<u64>,
        subset: &zarrs::array_subset::ArraySubset,
        is_invariant: bool,
    ) -> Vec<((u64, u64), f32)> {
        self.neighborhood_reader
            .get_neighbor_costs(i_range, j_range, subset, is_invariant)
    }

    #[cfg(test)]
    /// Build explicit barrier cells directly from barrier layers for tests.
    ///
    /// This bypasses the cached swap masks so tests can compare the direct
    /// barrier interpretation of source features against derived swap output.
    ///
    /// # Arguments
    /// `index`: Center cell whose 3x3 neighborhood should be inspected.
    /// `barrier_layers`: Barrier layers to evaluate directly from source data.
    ///
    /// # Returns
    /// A vector of neighborhood cells where any provided barrier layer is true.
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

#[cfg(test)]
/// Construct a `LazySubset` helper for tests.
///
/// # Arguments
/// `source`: Source dataset storage to read from lazily.
/// `subset`: Array subset that bounds the lazy view.
///
/// # Returns
/// An initialized `LazySubset<f32>` instance.
pub(crate) fn make_lazy_subset_for_tests(
    source: ReadableListableStorage,
    subset: zarrs::array_subset::ArraySubset,
) -> LazySubset<f32> {
    LazySubset::new(source, subset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
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
