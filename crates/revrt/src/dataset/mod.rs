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
mod swap;

use std::path::PathBuf;

use num_traits::AsPrimitive;
use tracing::{debug, info, trace};
use zarrs::array::{Array, DataType, ElementOwned};
use zarrs::storage::ReadableListableStorage;

use crate::ArrayIndex;
use crate::cost::CostFunction;
use crate::cost::components::BarrierLayer;
use crate::error::Result;
use derived::DerivedDataWriter;
pub(crate) use lazy_subset::LazySubset;
use reader::DerivedDataReader;
use swap::{initialize_swap, inspect_source_layout};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NeighborhoodGeometry {
    Side,
    Corner,
}

impl NeighborhoodGeometry {
    pub(super) fn edge_scale(self) -> f32 {
        match self {
            Self::Side => 0.5,
            Self::Corner => f32::sqrt(2.0) / 2.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NeighborhoodPoint {
    pub(super) destination: ArrayIndex,
    pub(super) geometry: NeighborhoodGeometry,
    pub(super) destination_primary_cost: f32,
    pub(super) destination_invariant_cost: f32,
    pub(super) destination_is_hard_barrier: bool,
}

impl NeighborhoodPoint {
    pub(super) fn edge_scale(&self) -> f32 {
        self.geometry.edge_scale()
    }

    pub(super) fn traversal_cost(
        &self,
        source_primary_cost: f32,
        source_multiplier: f32,
        destination_multiplier: f32,
    ) -> Option<f32> {
        if self.destination_is_hard_barrier || self.destination_primary_cost.is_nan() {
            return None;
        }

        Some(
            self.edge_scale()
                * (source_primary_cost * source_multiplier
                    + self.destination_primary_cost * destination_multiplier)
                + self.destination_invariant_cost * destination_multiplier,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RoutingOptionNeighborhood {
    pub(super) center_primary_cost: Option<f32>,
    pub(super) points: Vec<NeighborhoodPoint>,
}

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
    /// Derived-data materializer responsible for chunk tracking and writes.
    derived_data_writer: DerivedDataWriter,
    /// Reader responsible for cached access to derived data.
    derived_data_reader: DerivedDataReader,
    /// Shape of the source routing grid as `(rows, cols, options)`.
    pub(super) grid_shape: (u64, u64, u32),
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
        let routing_option_count =
            u32::try_from(cost_function.routing_options.len()).map_err(|_| {
                crate::error::Error::IO(std::io::Error::other("routing option count exceeds u32"))
            })?;

        let filesystem =
            zarrs::filesystem::FilesystemStore::new(path).expect("could not open filesystem store");
        let source: ReadableListableStorage = std::sync::Arc::new(filesystem);

        let source_layout = inspect_source_layout(&source, routing_option_count)?;
        let swap = initialize_swap(swap_fp, &source_layout, soft_barrier_group_count)?;

        let derived_data_writer =
            DerivedDataWriter::new(&source_layout, source.clone(), swap.clone(), cost_function);

        let derived_data_reader = DerivedDataReader::open(
            swap.clone(),
            cache_size,
            soft_barrier_group_count,
            source_layout,
        )?;
        let grid_shape = derived_data_reader.grid_shape();

        trace!("Dataset opened successfully");
        Ok(Self {
            source,
            cost_path: None,
            derived_data_writer,
            derived_data_reader,
            grid_shape,
        })
    }

    /// Return 3x3 routing neighborhoods for all option states at an index.
    ///
    /// Derived data is materialized on demand for the needed swap chunks
    /// before the neighborhood is read.
    ///
    /// # Arguments
    /// `index`: Center cell whose 3x3 neighborhood should be queried.
    ///
    /// # Returns
    /// A vector containing one `RoutingOptionNeighborhood` per routing option,
    /// each with the center primary cost and the reachable neighboring
    /// cells for that option.
    pub(super) fn get_3x3_neighborhood_all_options(
        &self,
        index: &ArrayIndex,
    ) -> Vec<RoutingOptionNeighborhood> {
        self.derived_data_reader
            .get_3x3_neighborhood_all_options(index, &self.derived_data_writer)
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
            dropped_soft_groups.min(self.derived_data_writer.soft_barrier_groups.len());
        self.derived_data_reader.get_3x3_soft_barrier_cells(
            index,
            retry_state,
            &self.derived_data_writer,
        )
    }

    /// Return the center-cell entry cost for a single option state.
    pub(super) fn get_cell_cost_components(&self, index: &ArrayIndex) -> Option<(f32, f32)> {
        self.derived_data_reader
            .get_cell_cost_components(index, &self.derived_data_writer)
    }

    /// Return a single source-layer cell as `f32` for the requested index.
    /// Useful for reading non-cost (i.e. not derived) features
    pub(super) fn get_source_cell_value(
        &self,
        layer_name: &str,
        index: &ArrayIndex,
    ) -> Option<f32> {
        let array = Array::open(self.source.clone(), &format!("/{layer_name}")).ok()?;
        let subset = match array.shape().len() {
            2 => zarrs::array_subset::ArraySubset::new_with_ranges(&[
                index.i..(index.i + 1),
                index.j..(index.j + 1),
            ]),
            3 => zarrs::array_subset::ArraySubset::new_with_ranges(&[
                u64::from(index.option)..(u64::from(index.option) + 1),
                index.i..(index.i + 1),
                index.j..(index.j + 1),
            ]),
            _ => return None,
        };
        read_source_cell_as_f32(&array, &subset)
    }

    /// Return the number of soft barrier importance groups.
    ///
    /// # Returns
    /// The count of retry-state groups that can be progressively dropped
    /// during routing retries.
    pub(super) fn soft_barrier_groups(&self) -> &Vec<(u32, Vec<BarrierLayer>)> {
        &self.derived_data_writer.soft_barrier_groups
    }
}

fn read_source_cell_as_f32<TStorage>(
    array: &Array<TStorage>,
    subset: &zarrs::array_subset::ArraySubset,
) -> Option<f32>
where
    TStorage: ?Sized + zarrs::storage::ReadableListableStorageTraits + 'static,
{
    match array.data_type() {
        DataType::Float32 => read_typed_cell::<f32, _>(array, subset),
        DataType::Float64 => read_typed_cell::<f64, _>(array, subset),
        DataType::Int8 => read_typed_cell::<i8, _>(array, subset),
        DataType::Int16 => read_typed_cell::<i16, _>(array, subset),
        DataType::Int32 => read_typed_cell::<i32, _>(array, subset),
        DataType::Int64 => read_typed_cell::<i64, _>(array, subset),
        DataType::UInt8 => read_typed_cell::<u8, _>(array, subset),
        DataType::UInt16 => read_typed_cell::<u16, _>(array, subset),
        DataType::UInt32 => read_typed_cell::<u32, _>(array, subset),
        DataType::UInt64 => read_typed_cell::<u64, _>(array, subset),
        _ => None,
    }
}

fn read_typed_cell<T, TStorage>(
    array: &Array<TStorage>,
    subset: &zarrs::array_subset::ArraySubset,
) -> Option<f32>
where
    T: ElementOwned + Clone + AsPrimitive<f32>,
    TStorage: ?Sized + zarrs::storage::ReadableListableStorageTraits + 'static,
{
    array
        .retrieve_array_subset_elements::<T>(subset)
        .ok()?
        .into_iter()
        .next()
        .map(|value| value.as_())
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
        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"A"}]}}}"#,
        )
        .unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let test_points = [ArrayIndex::new_ij(3, 1), ArrayIndex::new_ij(2, 2)];
        let array = zarrs::array::Array::open(dataset.source.clone(), "/A").unwrap();
        for point in test_points {
            let neighborhoods = dataset.get_3x3_neighborhood_all_options(&point);

            // index 0, 0 has a cost of 0 and should therefore be filtered out
            assert!(
                !neighborhoods[point.option as usize]
                    .points
                    .iter()
                    .any(|point| point.destination.i == 0 && point.destination.j == 0)
            );
            let ArrayIndex { i: ci, j: cj, .. } = point;
            let center_subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
                0..1,
                ci..(ci + 1),
                cj..(cj + 1),
            ]);
            let center_cost: f32 = array
                .retrieve_array_subset_elements(&center_subset)
                .expect("Error reading zarr data")[0];
            let results = same_option_neighbors(&neighborhoods, point.option, Some(center_cost));

            for (ArrayIndex { i, j, .. }, val) in results {
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

        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"A"}]}}}"#,
        )
        .unwrap();

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
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"A","is_invariant":true}]}}}"#,
        )
        .unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let test_points = [ArrayIndex::new_ij(3, 1), ArrayIndex::new_ij(2, 2)];
        let array = zarrs::array::Array::open(dataset.source.clone(), "/A").unwrap();
        for point in test_points {
            let neighborhoods = dataset.get_3x3_neighborhood_all_options(&point);
            let results = &neighborhoods[point.option as usize].points;

            for neighborhood_point in results {
                let ArrayIndex { i, j, .. } = neighborhood_point.destination;
                let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
                    0..1,
                    i..(i + 1),
                    j..(j + 1),
                ]);
                let subset_elements: Vec<f32> = array
                    .retrieve_array_subset_elements(&subset)
                    .expect("Error reading zarr data");
                assert_eq!(subset_elements.len(), 1);
                assert_eq!(
                    subset_elements[0],
                    neighborhood_point.destination_invariant_cost
                )
            }
        }
    }

    #[test]
    fn test_sample_cost_function_get_3x3() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "B", "C", "cost"]);
        let cost_function = crate::cost::sample::cost_function();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let test_points = [ArrayIndex::new_ij(3, 1), ArrayIndex::new_ij(2, 2)];
        let array_a = zarrs::array::Array::open(dataset.source.clone(), "/A").unwrap();
        let array_b = zarrs::array::Array::open(dataset.source.clone(), "/B").unwrap();
        let array_c = zarrs::array::Array::open(dataset.source.clone(), "/C").unwrap();
        for point in test_points {
            let neighborhoods = dataset.get_3x3_neighborhood_all_options(&point);

            // index 0, 0 has a cost of 0 and should therefore be filtered out
            assert!(
                !neighborhoods[point.option as usize]
                    .points
                    .iter()
                    .any(|point| point.destination.i == 0 && point.destination.j == 0)
            );
            let ArrayIndex { i: ci, j: cj, .. } = point;
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
            let results = same_option_neighbors(&neighborhoods, point.option, Some(center_cost));

            for (ArrayIndex { i, j, .. }, val) in results {
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
        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"cost"}]}}}"#,
        )
        .unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let index = ArrayIndex::new_ij(0, 0);
        let neighborhoods = dataset.get_3x3_neighborhood_all_options(&index);
        let results = same_option_neighbors(
            &neighborhoods,
            index.option,
            dataset.get_source_cell_value("cost", &index),
        );

        // index 0, 0 has a cost of 0 and should therefore be filtered out
        assert!(
            !results
                .iter()
                .any(|(ArrayIndex { i, j, .. }, _)| *i == 0 && *j == 0)
        );

        assert_eq!(results, vec![]);
    }

    #[test_case((0, 0), vec![(0, 1, 0.5), (1, 0, 1.0), (1, 1, 1.5 * SQRT_2)] ; "top left corner")]
    #[test_case((0, 1), vec![(1, 0, 1.5 * SQRT_2), (1, 1, 2.)] ; "top right corner")]
    #[test_case((1, 0), vec![(0, 1, 1.5 * SQRT_2), (1, 1, 2.5)] ; "bottom left corner")]
    #[test_case((1, 1), vec![(0, 1, 2.), (1, 0, 2.5)] ; "bottom right corner")]
    fn test_get_3x3_two_by_two_array((si, sj): (u64, u64), expected_output: Vec<(u64, u64, f32)>) {
        let tmp = samples::cost_as_index_zarr(1, 2, 2, 1, 2, 2);
        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"cost"}]}}}"#,
        )
        .unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let index = ArrayIndex::new_ij(si, sj);
        let neighborhoods = dataset.get_3x3_neighborhood_all_options(&index);
        let results = same_option_neighbors(
            &neighborhoods,
            index.option,
            dataset.get_source_cell_value("cost", &index),
        );

        // index 0, 0 has a cost of 0 and should therefore be filtered out
        assert!(
            !results
                .iter()
                .any(|(ArrayIndex { i, j, .. }, _)| *i == 0 && *j == 0)
        );

        assert_eq!(
            results,
            expected_output
                .into_iter()
                .map(|(i, j, v)| (ArrayIndex::new_ij(i, j), v))
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
        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"cost"}]}}}"#,
        )
        .unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let index = ArrayIndex::new_ij(si, sj);
        let neighborhoods = dataset.get_3x3_neighborhood_all_options(&index);
        let results = same_option_neighbors(
            &neighborhoods,
            index.option,
            dataset.get_source_cell_value("cost", &index),
        );

        // index 0, 0 has a cost of 0 and should therefore be filtered out
        assert!(
            !results
                .iter()
                .any(|(ArrayIndex { i, j, .. }, _)| *i == 0 && *j == 0)
        );

        assert_eq!(
            results,
            expected_output
                .into_iter()
                .map(|(i, j, v)| (ArrayIndex::new_ij(i, j), v))
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
        let cost_function = CostFunction::from_json(
            r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"cost"}]}}}"#,
        )
        .unwrap();
        let dataset =
            Dataset::open(tmp.path(), cost_function, 1_000).expect("Error opening dataset");

        let index = ArrayIndex::new_ij(si, sj);
        let neighborhoods = dataset.get_3x3_neighborhood_all_options(&index);
        let results = same_option_neighbors(
            &neighborhoods,
            index.option,
            dataset.get_source_cell_value("cost", &index),
        );

        // index 0, 0 has a cost of 0 and should therefore be filtered out
        assert!(
            !results
                .iter()
                .any(|(ArrayIndex { i, j, .. }, _)| *i == 0 && *j == 0)
        );

        assert_eq!(
            results,
            expected_output
                .into_iter()
                .map(|(i, j, v)| (ArrayIndex::new_ij(i, j), v))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_get_3x3_with_invariant_and_friction_layers() {
        // Define cost function: A normal, C invariant, friction from B * 0.5
        let json = r#"
        {
            "routing_options": {
                "default": {
                    "cost_layers": [
                        {"layer_name": "A"},
                        {"layer_name": "C", "is_invariant": true}
                    ],
                    "friction_layers": [
                        {"multiplier_layer": "B", "multiplier_scalar": 0.5}
                    ]
                }
            }
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
        let point = ArrayIndex::new_ij(1, 1);
        let neighborhoods = dataset.get_3x3_neighborhood_all_options(&point);
        let results = same_option_neighbors(
            &neighborhoods,
            point.option,
            dataset.get_source_cell_value("A", &point),
        );

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
                expected.push((ArrayIndex::new_ij(ir, jr), expected_val));
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

    #[test_case(r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"B"}]}},"ignore_invalid_costs":true}"# ; "zero layer")]
    #[test_case(r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"C"}]}},"ignore_invalid_costs":true}"# ; "negative layer")]
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

        let index = ArrayIndex::new_ij(1, 1);
        let neighborhoods = dataset.get_3x3_neighborhood_all_options(&index);
        let results = same_option_neighbors(&neighborhoods, index.option, None);
        assert!(
            results.is_empty(),
            "Found data with `ignore_invalid_costs=true`"
        );
    }

    #[test_case(r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"B"}]}},"ignore_invalid_costs":false}"# ; "zero layer")]
    #[test_case(r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"C"}]}},"ignore_invalid_costs":false}"# ; "negative layer")]
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

        let index = ArrayIndex::new_ij(1, 1);
        let neighborhoods = dataset.get_3x3_neighborhood_all_options(&index);
        let results = same_option_neighbors(&neighborhoods, index.option, None);
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
                expected.push((ArrayIndex::new_ij(ir, jr), averaged));
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
            "routing_options": {
                "default": {
                    "cost_layers": [{"layer_name": "A"}],
                    "barrier_layers": [
                        {
                            "layer_name": "B",
                            "barrier_operator": "eq",
                            "barrier_threshold": 1.0
                        }
                    ]
                }
            }
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

        let index = ArrayIndex::new_ij(1, 1);
        let neighborhoods = dataset.get_3x3_neighborhood_all_options(&index);
        let results = same_option_neighbors(
            &neighborhoods,
            index.option,
            dataset.get_source_cell_value("A", &index),
        );
        assert_eq!(
            results,
            vec![
                (ArrayIndex::new_ij(0, 0), 3.0 * std::f32::consts::SQRT_2),
                (ArrayIndex::new_ij(0, 2), 4.0 * std::f32::consts::SQRT_2),
                (ArrayIndex::new_ij(2, 0), 6.0 * std::f32::consts::SQRT_2),
                (ArrayIndex::new_ij(2, 2), 7.0 * std::f32::consts::SQRT_2),
            ]
        );
    }

    #[test]
    fn test_explicit_barriers_do_not_modify_cached_costs_when_invalid_costs_are_soft() {
        let json = r#"
        {
            "routing_options": {
                "default": {
                    "cost_layers": [{"layer_name": "A"}],
                    "barrier_layers": [
                        {
                            "layer_name": "B",
                            "barrier_operator": "eq",
                            "barrier_threshold": 1.0,
                            "barrier_importance": 1
                        }
                    ]
                }
            },
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

        let index = ArrayIndex::new_ij(1, 1);
        let neighborhoods = dataset.get_3x3_neighborhood_all_options(&index);
        let results = same_option_neighbors(
            &neighborhoods,
            index.option,
            dataset.get_source_cell_value("A", &index),
        );
        assert_eq!(results.len(), 8);
    }

    #[test]
    fn test_cumulative_soft_barrier_masks_follow_retry_state() {
        let json = r#"
        {
            "routing_options": {
                "default": {
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
            }
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

        let center = ArrayIndex::new_ij(1, 1);
        dataset.get_3x3_neighborhood_all_options(&center);

        assert_eq!(
            dataset.get_3x3_soft_barrier_cells(&center, 0),
            vec![ArrayIndex::new_ij(0, 1), ArrayIndex::new_ij(1, 0)]
        );
        assert_eq!(
            dataset.get_3x3_soft_barrier_cells(&center, 1),
            vec![ArrayIndex::new_ij(0, 1)]
        );
        assert!(dataset.get_3x3_soft_barrier_cells(&center, 2).is_empty());
        assert!(dataset.get_3x3_soft_barrier_cells(&center, 99).is_empty());
    }

    #[test]
    fn test_cumulative_soft_barrier_masks_or_tied_importance_groups() {
        let json = r#"
        {
            "routing_options": {
                "default": {
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
            }
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

        let center = ArrayIndex::new_ij(1, 1);
        dataset.get_3x3_neighborhood_all_options(&center);

        assert_eq!(
            dataset.get_3x3_soft_barrier_cells(&center, 0),
            vec![ArrayIndex::new_ij(0, 1), ArrayIndex::new_ij(1, 0)]
        );
        assert!(dataset.get_3x3_soft_barrier_cells(&center, 1).is_empty());
    }

    fn same_option_neighbors(
        neighborhoods: &[RoutingOptionNeighborhood],
        option: u32,
        fallback_center_primary_cost: Option<f32>,
    ) -> Vec<(ArrayIndex, f32)> {
        let neighborhood = &neighborhoods[option as usize];
        let Some(source_primary_cost) = neighborhood
            .center_primary_cost
            .or(fallback_center_primary_cost)
        else {
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
}
