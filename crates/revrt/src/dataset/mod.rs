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

use std::iter;
use std::sync::{Arc, RwLock};

use tracing::{debug, info, trace, warn};
use zarrs::array::codec::CodecOptions;
use zarrs::array::{ChunkCache, ChunkCacheDecodedLruSizeLimit, ChunkGrid};
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableWritableListableStorage,
};

use crate::ArrayIndex;
use crate::cost::CostFunction;
use crate::error::{Error, Result};
pub(crate) use lazy_subset::LazySubset;

/// Manages the features datasets and calculated total cost
pub(super) struct Dataset {
    /// A Zarr storages with the features
    source: ReadableListableStorage,
    // Silly way to keep the tmp path alive
    #[allow(dead_code)]
    cost_path: tempfile::TempDir,
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
    /// Custom cost function definition
    cost_function: CostFunction,
    /// Cache for decoded cost chunks shared across calls
    cost_cache: ChunkCacheDecodedLruSizeLimit,
    /// Cache for decoded invariant cost chunks shared across calls
    cost_invariant_cache: ChunkCacheDecodedLruSizeLimit,
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
        swap_fp: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        debug!("Opening dataset: {:?}", path.as_ref());
        let filesystem =
            zarrs::filesystem::FilesystemStore::new(path).expect("could not open filesystem store");
        let source = std::sync::Arc::new(filesystem);

        // ==== Create the swap dataset ====
        let tmp_path = tempfile::TempDir::new()
            .expect("could not create temporary directory for swap dataset");
        let swap_fp = match swap_fp {
            Some(path) => {
                info!("Using provided swap dataset at {:?}", path);
                path
            }
            None => {
                let tmp = tmp_path.path().to_path_buf();
                info!("Initializing a temporary swap dataset at {:?}", tmp);
                tmp
            }
        };
        let swap: ReadableWritableListableStorage = std::sync::Arc::new(
            zarrs::filesystem::FilesystemStore::new(swap_fp)
                .expect("could not open filesystem store"),
        );

        trace!("Creating a new group for the cost dataset");
        zarrs::group::GroupBuilder::new()
            .build(swap.clone(), "/")?
            .store_metadata()?;

        let entries = source
            .list()
            .expect("failed to list variables in source dataset");
        let first_entry_opt = entries
            .into_iter()
            .map(|entry| entry.to_string())
            .find(|entry| {
                let name = entry.split('/').next().unwrap_or("").to_ascii_lowercase();
                // Skip coordinate axes when selecting a representative variable for cost storage.
                const EXCLUDES: [&str; 6] =
                    ["latitude", "longitude", "band", "x", "y", "spatial_ref"];
                !name.ends_with(".json") && !EXCLUDES.iter().any(|needle| name == *needle)
            });
        let first_entry = match first_entry_opt {
            Some(e) => e,
            None => {
                return Err(Error::IO(std::io::Error::other(format!(
                    "no non-coordinate variables found in source dataset: {:?}",
                    source.list().ok()
                ))));
            }
        };

        // Skip coordinate axes when selecting a representative variable for cost storage.
        let varname = match first_entry.split('/').next() {
            Some(name) => name,
            None => {
                return Err(Error::IO(std::io::Error::other(
                    "Could not determine any variable names from source dataset",
                )));
            }
        };
        debug!("Using '{}' to determine shape of cost data", varname);
        let tmp = zarrs::array::Array::open(source.clone(), &format!("/{varname}"))?;
        let shape = tmp.shape();
        if shape.len() < 3 {
            return Err(Error::InvalidDatasetShape {
                variable: varname.to_string(),
                min_rank: 3,
                shape: shape.to_vec(),
            });
        }
        let grid_nrows = shape[1];
        let grid_ncols = shape[2];
        let chunk_grid = tmp.chunk_grid();
        debug!("Chunk grid info: {:?}", &chunk_grid);

        add_layer_to_data("cost_invariant", chunk_grid, &swap)?;
        add_layer_to_data("cost", chunk_grid, &swap)?;

        let cost_chunk_idx = ndarray::Array2::from_elem(
            (
                tmp.chunk_grid_shape()[1] as usize,
                tmp.chunk_grid_shape()[2] as usize,
            ),
            false,
        )
        .into();

        if cache_size < 1_000_000 {
            warn!("Cache size smaller than 1MB");
        }
        debug!("Creating cache with size {}MB", cache_size / 1_000_000);
        // TODO: tune cache_size against typical chunk sizes
        // (e.g. chunk_bytes * hot_chunks * safety_factor)
        let cost_array_readable =
            Arc::new(zarrs::array::Array::open(swap.clone(), "/cost")?.readable());
        let cost_invariant_array_readable =
            Arc::new(zarrs::array::Array::open(swap.clone(), "/cost_invariant")?.readable());
        let cost_cache =
            ChunkCacheDecodedLruSizeLimit::new(cost_array_readable.clone(), cache_size / 2);
        let cost_invariant_cache = ChunkCacheDecodedLruSizeLimit::new(
            cost_invariant_array_readable.clone(),
            cache_size / 2,
        );

        trace!("Dataset opened successfully");
        Ok(Self {
            source,
            cost_path: tmp_path,
            swap,
            cost_chunk_idx,
            cost_function,
            cost_cache,
            cost_invariant_cache,
            grid_nrows,
            grid_ncols,
        })
    }

    fn calculate_chunk_cost(&self, ci: u64, cj: u64) {
        trace!("Creating a LazySubset for ({}, {})", ci, cj);

        // cost variable is stored in the swap dataset
        let variable = zarrs::array::Array::open(self.swap.clone(), "/cost").unwrap();
        // Get the subset according to cost's chunk
        let subset = variable.chunk_subset(&[0, ci, cj]).unwrap();
        let mut data = LazySubset::<f32>::new(self.source.clone(), subset);

        self.calculate_chunk_cost_single_layer(ci, cj, &mut data, true);
        self.calculate_chunk_cost_single_layer(ci, cj, &mut data, false);
    }

    fn calculate_chunk_cost_single_layer(
        &self,
        ci: u64,
        cj: u64,
        subset: &mut LazySubset<f32>,
        is_invariant: bool,
    ) {
        let output;
        let layer_name;
        if is_invariant {
            trace!("Calculating invariant cost for chunk ({}, {})", ci, cj);
            output = self.cost_function.compute(subset, true);
            layer_name = "/cost_invariant";
        } else {
            trace!(
                "Calculating length-dependent cost for chunk ({}, {})",
                ci, cj
            );
            output = self.cost_function.compute(subset, false);
            layer_name = "/cost";
        }

        trace!("Cost function: {:?}", self.cost_function);

        let cost = zarrs::array::Array::open(self.swap.clone(), layer_name).unwrap();
        cost.store_metadata().unwrap();
        let chunk_indices: Vec<u64> = vec![0, ci, cj];
        trace!("Storing chunk at {:?}", chunk_indices);
        let chunk_subset =
            &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..1, ci..(ci + 1), cj..(cj + 1)]);
        trace!("Target chunk subset: {:?}", chunk_subset);
        cost.store_chunks_ndarray(chunk_subset, output).unwrap();
    }

    pub(super) fn get_3x3(&self, index: &ArrayIndex) -> Vec<(ArrayIndex, f32)> {
        let &ArrayIndex { i, j } = index;

        trace!("Getting 3x3 neighborhood for (i={}, j={})", i, j);

        trace!("Cost dataset contents: {:?}", self.swap.list().unwrap());
        trace!("Cost dataset size: {:?}", self.swap.size().unwrap());

        trace!("Opening cost dataset via cache");
        let cost_array = self.cost_cache.array();
        trace!("Cost dataset with shape: {:?}", cost_array.shape());

        // Cutting off the edges for now.
        let shape = cost_array.shape();
        debug_assert!(!shape.contains(&0));

        let max_i = shape[1] - 1;
        let max_j = shape[2] - 1;

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

        // Capture the 3x3 neighborhood
        let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
            0..1,
            i_range.clone(),
            j_range.clone(),
        ]);
        trace!("Cost subset: {:?}", subset);

        // Find the chunks that intersect the subset
        let chunks = &cost_array.chunks_in_array_subset(&subset).unwrap().unwrap();
        trace!("Cost chunks: {:?}", chunks);
        trace!(
            "Cost subset extends to {:?} chunks",
            chunks.num_elements_usize()
        );

        for ci in chunks.start()[1]..(chunks.start()[1] + chunks.shape()[1]) {
            for cj in chunks.start()[2]..(chunks.start()[2] + chunks.shape()[2]) {
                trace!(
                    "Checking if cost for chunk ({}, {}) has been calculated",
                    ci, cj
                );
                if self.cost_chunk_idx.read().unwrap()[[ci as usize, cj as usize]] {
                    trace!("Cost for chunk ({}, {}) already calculated", ci, cj);
                } else {
                    debug!("Requesting write lock for cost_chunk_idx ({}, {})", ci, cj);
                    let mut chunk_idx = self
                        .cost_chunk_idx
                        .write()
                        .expect("Failed to acquire write lock");
                    debug!("Acquired write lock for cost_chunk_idx ({}, {})", ci, cj);
                    if chunk_idx[[ci as usize, cj as usize]] {
                        trace!(
                            "Cost for chunk ({}, {}) already calculated while waiting for the lock",
                            ci, cj
                        );
                    } else {
                        self.calculate_chunk_cost(ci, cj);
                        debug!("Recording chunk ({}, {}) as calculated", ci, cj);
                        chunk_idx[[ci as usize, cj as usize]] = true;
                    }
                    debug!("Released write lock for cost_chunk_idx ({}, {})", ci, cj);
                }
            }
        }

        let neighbors = self.get_neighbor_costs(i_range.clone(), j_range.clone(), &subset, false);
        let invariant_neighbors = self.get_neighbor_costs(i_range, j_range, &subset, true);

        /*
         * The transition between two gridpoint centers is along half the distance
         * on the original gridpoint, plus half the distance to the target gridpoint
         * (center). Therefore, the transition cost is the average between the origin
         * gridpoint cost and the target gridpoint cost.
         * Note that the same principle is valid for diagonals, it is still the average
         * of both values, but we have to scale for the longer distance along the
         * diagonal, thus a sqrt(2) factor along the diagonals.
         */

        // Extract the origin point.
        let center = neighbors
            .iter()
            .find(|((ir, jr), _)| *ir == i && *jr == j)
            .map(|((ir, jr), v)| {
                if v.is_nan() {
                    ((ir, jr), &0_f32) // NaN's don't contribute to cost
                } else {
                    ((ir, jr), v)
                }
            })
            .unwrap();
        trace!("Center point: {:?}", center);

        // Calculate the average with center point (half grid + other half grid).
        // Also, apply the diagonal factor for the extra distance.
        // Finally, add any invariant costs.
        let cost_to_neighbors = neighbors
            .iter()
            .zip(invariant_neighbors.iter())
            .filter(|(((ir, jr), v), _)| !(v.is_nan() || *ir == i && *jr == j)) // no center point and only valid costs
            .map(|(((ir, jr), v), ((inv_ir, inv_jr), inv_cost))| {
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

    pub(super) fn grid_shape(&self) -> (u64, u64) {
        (self.grid_nrows, self.grid_ncols)
    }
}

fn add_layer_to_data(
    layer_name: &str,
    chunk_shape: &ChunkGrid,
    swap: &ReadableWritableListableStorage,
) -> Result<()> {
    trace!("Creating an empty {} array", layer_name);
    let dataset_path = format!("/{layer_name}");
    let mut builder = zarrs::array::ArrayBuilder::new_with_chunk_grid(
        chunk_shape.clone(),
        zarrs::array::DataType::Float32,
        zarrs::array::FillValue::from(zarrs::array::ZARR_NAN_F32),
    );

    let built = builder
        .dimension_names(["band", "y", "x"].into())
        .build(swap.clone(), &dataset_path)?;
    built.store_metadata()?;

    let array = zarrs::array::Array::open(swap.clone(), &dataset_path)?;
    trace!("'{}' shape: {:?}", layer_name, array.shape().to_vec());
    trace!("'{}' chunk shape: {:?}", layer_name, array.chunk_grid());

    trace!(
        "Dataset contents after '{}' creation: {:?}",
        layer_name,
        swap.list()?
    );
    Ok(())
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
            Dataset::open(tmp.path(), cost_function, 1_000, None).expect("Error opening dataset");

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
            Dataset::open(tmp.path(), cost_function, 1_000, None).expect("Error opening dataset");

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
            Dataset::open(tmp.path(), cost_function, 1_000, None).expect("Error opening dataset");

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
            Dataset::open(tmp.path(), cost_function, 1_000, None).expect("Error opening dataset");

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
            Dataset::open(tmp.path(), cost_function, 1_000, None).expect("Error opening dataset");

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
            Dataset::open(tmp.path(), cost_function, 1_000, None).expect("Error opening dataset");

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
            Dataset::open(tmp.path(), cost_function, 1_000, None).expect("Error opening dataset");

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
            Dataset::open(tmp.path(), cost_function, 1_000, None).expect("Error opening dataset");

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
            Dataset::open(tmp.path(), cost_function, 1_000, None).expect("Error opening dataset");

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
            Dataset::open(tmp.path(), cost_function, 1_000, None).expect("Error opening dataset");

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
}
