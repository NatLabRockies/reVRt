//! Swap dataset initialization helpers
//!
//! This module inspects the source dataset to recover the grid and chunk
//! layout needed by routing, then creates a temporary swap dataset with the
//! derived arrays used during neighborhood expansion.

use std::path::Path;

use tracing::{debug, trace};
use zarrs::array::ChunkGrid;
use zarrs::array::chunk_grid::regular::RegularChunkGrid;
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableWritableListableStorage,
};

use crate::error::{Error, Result};

/// Grid and chunk metadata derived from the source dataset.
///
/// The swap dataset mirrors the representative source array layout so all
/// derived cost and barrier arrays align with the original feature data.
pub(super) struct SourceLayout {
    /// Chunk grid definition copied from the representative source array.
    pub(super) chunk_grid: ChunkGrid,
    /// Number of chunks along the band axis.
    pub(super) chunk_grid_bands: usize,
    /// Number of chunk rows in the source grid.
    pub(super) chunk_grid_rows: usize,
    /// Number of chunk columns in the source grid.
    pub(super) chunk_grid_cols: usize,
    /// Number of routing options encoded on the leading band axis.
    pub(super) grid_noptions: u32,
    /// Number of rows in the full source grid.
    pub(super) grid_nrows: u64,
    /// Number of columns in the full source grid.
    pub(super) grid_ncols: u64,
}

/// Inspect the source dataset and recover the layout used for swap storage.
///
/// Every non-coordinate source layer is expected to be a 3D array with a
/// single input band. Routing options are represented only in the derived
/// swap dataset, not in the source inputs. A representative source layer is
/// then used to infer the full grid shape and chunk grid. Coordinate-like
/// arrays such as latitude, longitude, and spatial reference metadata are
/// ignored during selection.
///
/// # Arguments
/// `source`: Source dataset storage containing the input feature arrays.
/// `routing_option_count`: Number of routing options that will be encoded
///                         on the leading band dimension of the swap arrays.
///
/// # Returns
/// A `SourceLayout` describing the representative array's spatial shape and
/// chunking, combined with the caller-provided routing-option count used for
/// swap arrays.
pub(super) fn inspect_source_layout(
    source: &ReadableListableStorage,
    routing_option_count: u32,
) -> Result<SourceLayout> {
    let variable_names = source_variable_names(source)?;
    let varname = &variable_names[0];
    debug!("Using '{}' to determine shape of cost data", varname);

    let representative = zarrs::array::Array::open(source.clone(), &format!("/{varname}"))?;
    let shape = representative.shape();
    validate_source_variable_shape(varname, shape)?;

    for variable_name in variable_names.iter().skip(1) {
        let variable = zarrs::array::Array::open(source.clone(), &format!("/{variable_name}"))?;
        validate_source_variable_shape(variable_name, variable.shape())?;
    }

    let chunk_shape = representative
        .chunk_grid()
        .chunk_shape_u64(&[0, 0, 0])
        .map_err(|error| {
            Error::IO(std::io::Error::other(format!(
                "failed to inspect source chunk shape: {error}"
            )))
        })?
        .ok_or_else(|| {
            Error::IO(std::io::Error::other(
                "source chunk shape was unavailable for representative array",
            ))
        })?;
    let chunk_grid = ChunkGrid::new(
        RegularChunkGrid::new(
            vec![u64::from(routing_option_count), shape[1], shape[2]],
            chunk_shape.try_into().map_err(|error| {
                Error::IO(std::io::Error::other(format!(
                    "failed to convert source chunk shape: {error}"
                )))
            })?,
        )
        .map_err(|error| {
            Error::IO(std::io::Error::other(format!(
                "failed to create swap chunk grid: {error}"
            )))
        })?,
    );
    let chunk_grid_shape = chunk_grid.grid_shape().to_vec();
    let layout = SourceLayout {
        chunk_grid,
        chunk_grid_bands: chunk_grid_shape[0] as usize,
        chunk_grid_rows: chunk_grid_shape[1] as usize,
        chunk_grid_cols: chunk_grid_shape[2] as usize,
        grid_noptions: routing_option_count,
        grid_nrows: shape[1],
        grid_ncols: shape[2],
    };
    debug!("Chunk grid info: {:?}", &layout.chunk_grid);

    Ok(layout)
}

fn source_variable_names(source: &ReadableListableStorage) -> Result<Vec<String>> {
    let entries = source.list().map_err(|err| {
        Error::IO(std::io::Error::other(format!(
            "failed to list variables in source dataset: {err}"
        )))
    })?;
    let mut variable_names = Vec::new();

    for entry in entries.into_iter().map(|entry| entry.to_string()) {
        let Some(name) = entry.split('/').next() else {
            continue;
        };
        let normalized = name.to_ascii_lowercase();
        const EXCLUDES: [&str; 6] = ["latitude", "longitude", "band", "x", "y", "spatial_ref"];
        if normalized.ends_with(".json")
            || EXCLUDES.iter().any(|needle| normalized == *needle)
            || variable_names.iter().any(|existing| existing == name)
        {
            continue;
        }

        variable_names.push(name.to_string());
    }

    if variable_names.is_empty() {
        return Err(Error::IO(std::io::Error::other(format!(
            "no non-coordinate variables found in source dataset: {:?}",
            source.list().ok()
        ))));
    }

    Ok(variable_names)
}

fn validate_source_variable_shape(variable: &str, shape: &[u64]) -> Result<()> {
    if shape.len() < 3 {
        return Err(Error::InvalidDatasetShape {
            variable: variable.to_string(),
            min_rank: 3,
            shape: shape.to_vec(),
        });
    }

    if shape[0] != 1 {
        return Err(Error::InvalidSourceBandCount {
            variable: variable.to_string(),
            expected: 1,
            found: shape[0],
        });
    }

    Ok(())
}

/// Create and initialize the derived swap dataset.
///
/// The swap dataset contains floating-point cost arrays and boolean barrier
/// masks with the same chunk structure as the representative source array.
/// One cumulative soft barrier mask is created for each retry state, including
/// the initial state where no soft barrier groups have been dropped.
///
/// # Arguments
/// `swap_path`: Filesystem location where the swap dataset should be created.
/// `layout`: Grid and chunk metadata copied from the source dataset.
/// `soft_barrier_group_count`: Number of soft barrier importance groups,
///                             which determines how many cumulative retry
///                             masks must be created.
/// `has_invariant_costs`: Whether invariant cost data should be stored.
/// `has_active_drivers`: Whether driver multiplier data should be stored.
/// `has_hard_barriers`: Whether a hard barrier mask should be stored.
///
/// # Returns
/// A readable and writable storage handle for the initialized swap dataset.
pub(super) fn initialize_swap<P: AsRef<Path>>(
    swap_path: P,
    layout: &SourceLayout,
    soft_barrier_group_count: usize,
    has_invariant_costs: bool,
    has_active_drivers: bool,
    has_hard_barriers: bool,
) -> Result<ReadableWritableListableStorage> {
    let swap: ReadableWritableListableStorage = std::sync::Arc::new(
        zarrs::filesystem::FilesystemStore::new(swap_path).map_err(|error| {
            Error::IO(std::io::Error::other(format!(
                "could not open filesystem store: {error}"
            )))
        })?,
    );

    debug!("Creating a new group for the cost dataset");
    zarrs::group::GroupBuilder::new()
        .build(swap.clone(), "/")?
        .store_metadata()?;

    add_layer_to_data("cost", &layout.chunk_grid, &swap)?;
    if has_invariant_costs {
        add_layer_to_data("cost_invariant", &layout.chunk_grid, &swap)?;
    }
    if has_active_drivers {
        add_layer_to_data("driver_multiplier", &layout.chunk_grid, &swap)?;
    }
    if has_hard_barriers {
        add_bool_layer_to_data("hard_barrier_mask", &layout.chunk_grid, &swap)?;
    }
    if soft_barrier_group_count > 0 {
        for retry_state in 0..=soft_barrier_group_count {
            add_bool_layer_to_data(
                &cumulative_soft_barrier_mask_name(retry_state),
                &layout.chunk_grid,
                &swap,
            )?;
        }
    }

    match swap.list() {
        Ok(list) => debug!("Swap dataset contents: {:?}", list),
        Err(error) => trace!("Could not inspect swap dataset contents: {error}"),
    }
    match swap.size() {
        Ok(size) => debug!("Swap dataset size: {:?}", size),
        Err(error) => trace!("Could not inspect swap dataset size: {error}"),
    }

    Ok(swap)
}

/// Build the array name for a cumulative soft barrier retry mask.
///
/// # Arguments
/// `retry_state`: Retry-state index whose cumulative mask name should be
///                constructed.
///
/// # Returns
/// The swap-array name used to store the cumulative soft barrier mask for the
/// requested retry state.
pub(super) fn cumulative_soft_barrier_mask_name(retry_state: usize) -> String {
    format!("soft_barrier_mask_retry_{retry_state}")
}

/// Add a floating-point derived layer to the swap dataset.
///
/// The created array uses the source chunk grid, a leading `band` dimension,
/// and a `NaN` fill value so unread cells default to an invalid routing cost.
///
/// # Arguments
/// `layer_name`: Name of the derived layer to create.
/// `chunk_shape`: Chunk grid copied from the representative source array.
/// `swap`: Swap dataset storage that will receive the new array.
///
/// # Returns
/// `Ok(())` after the array metadata has been written successfully.
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

/// Add a boolean derived layer to the swap dataset.
///
/// The created array uses the source chunk grid, a leading `band` dimension,
/// and an all-`false` fill value so barriers are absent until explicitly
/// materialized for a chunk.
///
/// # Arguments
/// `layer_name`: Name of the boolean mask layer to create.
/// `chunk_shape`: Chunk grid copied from the representative source array.
/// `swap`: Swap dataset storage that will receive the new array.
///
/// # Returns
/// `Ok(())` after the array metadata has been written successfully.
fn add_bool_layer_to_data(
    layer_name: &str,
    chunk_shape: &ChunkGrid,
    swap: &ReadableWritableListableStorage,
) -> Result<()> {
    trace!("Creating an empty {} array", layer_name);
    let dataset_path = format!("/{layer_name}");
    let mut builder = zarrs::array::ArrayBuilder::new_with_chunk_grid(
        chunk_shape.clone(),
        zarrs::array::DataType::Bool,
        zarrs::array::FillValue::from(false),
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
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;
    use zarrs::array::{ArrayBuilder, DataType, FillValue};
    use zarrs::filesystem::FilesystemStore;
    use zarrs::group::GroupBuilder;

    use super::*;
    use crate::dataset::samples::{self, LayerConfig, ZarrTestBuilder};

    #[test]
    fn inspect_source_layout_returns_expected_grid_metadata() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "B", "cost"]);
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));

        let layout = inspect_source_layout(&source, 2).expect("source layout inspection failed");

        assert_eq!(layout.grid_noptions, 2);
        assert_eq!(layout.chunk_grid_bands, 2);
        assert_eq!(layout.grid_nrows, 8);
        assert_eq!(layout.grid_ncols, 8);
        assert_eq!(layout.chunk_grid_rows, 2);
        assert_eq!(layout.chunk_grid_cols, 2);
    }

    #[test]
    fn inspect_source_layout_rejects_multi_band_source_variables() {
        let tmp = samples::multi_variable_random(2, 8, 8, 1, 4, 4, &["A"]);
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));

        let error = inspect_source_layout(&source, 2)
            .err()
            .expect("expected multi-band source dataset to be rejected");

        assert!(matches!(
            error,
            Error::InvalidSourceBandCount {
                variable,
                expected: 1,
                found: 2,
            } if variable == "A"
        ));
    }

    #[test]
    fn inspect_source_layout_rejects_sources_without_non_coordinate_layers() {
        let tmp = ZarrTestBuilder::new()
            .dimensions(1, 8, 8)
            .chunks(1, 4, 4)
            .layer(LayerConfig::ones("latitude"))
            .layer(LayerConfig::ones("longitude"))
            .layer(LayerConfig::ones("band"))
            .build()
            .expect("failed to create coordinate-only dataset");
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));

        let error = inspect_source_layout(&source, 1)
            .err()
            .expect("expected coordinate-only dataset to be rejected");

        assert!(matches!(error, Error::IO(_)));
    }

    #[test]
    fn inspect_source_layout_rejects_representative_variable_with_too_few_dimensions() {
        let tmp = malformed_two_dimensional_dataset();
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));

        let error = inspect_source_layout(&source, 1)
            .err()
            .expect("expected 2D representative variable to be rejected");

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
    fn initialize_swap_creates_expected_layers_and_chunk_index() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "cost"]);
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));
        let layout = inspect_source_layout(&source, 3).expect("source layout inspection failed");
        let swap_dir = TempDir::new().expect("could not create temporary swap directory");

        let initialized_swap = initialize_swap(swap_dir.path(), &layout, 2, true, true, true)
            .expect("swap initialization failed");

        let expected_layers = [
            ("/cost", DataType::Float32),
            ("/cost_invariant", DataType::Float32),
            ("/driver_multiplier", DataType::Float32),
            ("/hard_barrier_mask", DataType::Bool),
            ("/soft_barrier_mask_retry_0", DataType::Bool),
            ("/soft_barrier_mask_retry_1", DataType::Bool),
            ("/soft_barrier_mask_retry_2", DataType::Bool),
        ];

        for (layer_name, expected_dtype) in expected_layers {
            let array = zarrs::array::Array::open(initialized_swap.clone(), layer_name)
                .unwrap_or_else(|_| panic!("expected layer {layer_name} to exist"));
            assert_eq!(array.shape(), &[3, 8, 8], "wrong shape for {layer_name}");
            assert_eq!(
                array.chunk_grid_shape(),
                &[3, 2, 2],
                "wrong chunk grid shape for {layer_name}"
            );
            assert_eq!(*array.data_type(), expected_dtype);
        }
    }

    #[test]
    fn initialize_swap_omits_hard_barrier_layer_when_disabled() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "cost"]);
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));
        let layout = inspect_source_layout(&source, 3).expect("source layout inspection failed");
        let swap_dir = TempDir::new().expect("could not create temporary swap directory");

        let initialized_swap = initialize_swap(swap_dir.path(), &layout, 2, true, true, false)
            .expect("swap initialization failed");

        assert!(zarrs::array::Array::open(initialized_swap.clone(), "/hard_barrier_mask").is_err());

        for (layer_name, expected_dtype) in [
            ("/cost", DataType::Float32),
            ("/cost_invariant", DataType::Float32),
            ("/driver_multiplier", DataType::Float32),
            ("/soft_barrier_mask_retry_0", DataType::Bool),
            ("/soft_barrier_mask_retry_1", DataType::Bool),
            ("/soft_barrier_mask_retry_2", DataType::Bool),
        ] {
            let array = zarrs::array::Array::open(initialized_swap.clone(), layer_name)
                .unwrap_or_else(|_| panic!("expected layer {layer_name} to exist"));
            assert_eq!(array.shape(), &[3, 8, 8], "wrong shape for {layer_name}");
            assert_eq!(*array.data_type(), expected_dtype);
        }
    }

    #[test]
    fn initialize_swap_omits_driver_multiplier_layer_when_disabled() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "cost"]);
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));
        let layout = inspect_source_layout(&source, 3).expect("source layout inspection failed");
        let swap_dir = TempDir::new().expect("could not create temporary swap directory");

        let initialized_swap = initialize_swap(swap_dir.path(), &layout, 2, true, false, false)
            .expect("swap initialization failed");

        assert!(zarrs::array::Array::open(initialized_swap.clone(), "/driver_multiplier").is_err());

        for (layer_name, expected_dtype) in [
            ("/cost", DataType::Float32),
            ("/cost_invariant", DataType::Float32),
            ("/soft_barrier_mask_retry_0", DataType::Bool),
            ("/soft_barrier_mask_retry_1", DataType::Bool),
            ("/soft_barrier_mask_retry_2", DataType::Bool),
        ] {
            let array = zarrs::array::Array::open(initialized_swap.clone(), layer_name)
                .unwrap_or_else(|_| panic!("expected layer {layer_name} to exist"));
            assert_eq!(array.shape(), &[3, 8, 8], "wrong shape for {layer_name}");
            assert_eq!(*array.data_type(), expected_dtype);
        }
    }

    #[test]
    fn initialize_swap_omits_invariant_layer_when_disabled() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "cost"]);
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));
        let layout = inspect_source_layout(&source, 3).expect("source layout inspection failed");
        let swap_dir = TempDir::new().expect("could not create temporary swap directory");

        let initialized_swap = initialize_swap(swap_dir.path(), &layout, 2, false, false, false)
            .expect("swap initialization failed");

        assert!(zarrs::array::Array::open(initialized_swap.clone(), "/cost_invariant").is_err());

        for (layer_name, expected_dtype) in [
            ("/cost", DataType::Float32),
            ("/soft_barrier_mask_retry_0", DataType::Bool),
            ("/soft_barrier_mask_retry_1", DataType::Bool),
            ("/soft_barrier_mask_retry_2", DataType::Bool),
        ] {
            let array = zarrs::array::Array::open(initialized_swap.clone(), layer_name)
                .unwrap_or_else(|_| panic!("expected layer {layer_name} to exist"));
            assert_eq!(array.shape(), &[3, 8, 8], "wrong shape for {layer_name}");
            assert_eq!(*array.data_type(), expected_dtype);
        }
    }

    #[test]
    fn initialize_swap_omits_soft_barrier_layers_when_none_configured() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "cost"]);
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));
        let layout = inspect_source_layout(&source, 3).expect("source layout inspection failed");
        let swap_dir = TempDir::new().expect("could not create temporary swap directory");

        let initialized_swap = initialize_swap(swap_dir.path(), &layout, 0, true, true, true)
            .expect("swap initialization failed");

        for layer_name in [
            "/soft_barrier_mask_retry_0",
            "/soft_barrier_mask_retry_1",
            "/soft_barrier_mask_retry_2",
        ] {
            assert!(zarrs::array::Array::open(initialized_swap.clone(), layer_name).is_err());
        }

        for (layer_name, expected_dtype) in [
            ("/cost", DataType::Float32),
            ("/cost_invariant", DataType::Float32),
            ("/driver_multiplier", DataType::Float32),
            ("/hard_barrier_mask", DataType::Bool),
        ] {
            let array = zarrs::array::Array::open(initialized_swap.clone(), layer_name)
                .unwrap_or_else(|_| panic!("expected layer {layer_name} to exist"));
            assert_eq!(array.shape(), &[3, 8, 8], "wrong shape for {layer_name}");
            assert_eq!(*array.data_type(), expected_dtype);
        }
    }

    fn malformed_two_dimensional_dataset() -> TempDir {
        let tmp = TempDir::new().expect("could not create temporary directory");
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));

        GroupBuilder::new()
            .build(store.clone(), "/")
            .expect("could not create root group")
            .store_metadata()
            .expect("could not store root metadata");

        ArrayBuilder::new(
            vec![3, 4],
            vec![3, 4],
            DataType::Float32,
            FillValue::from(zarrs::array::ZARR_NAN_F32),
        )
        .build(store, "/A")
        .expect("could not create malformed array")
        .store_metadata()
        .expect("could not store malformed array metadata");

        tmp
    }
}
