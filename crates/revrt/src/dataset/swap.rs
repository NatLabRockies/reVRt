use std::path::Path;

use ndarray::Array2;
use tracing::{debug, trace};
use zarrs::array::ChunkGrid;
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableWritableListableStorage,
};

use crate::error::{Error, Result};

pub(super) struct SourceLayout {
    pub(super) chunk_grid: ChunkGrid,
    pub(super) chunk_grid_rows: usize,
    pub(super) chunk_grid_cols: usize,
    pub(super) grid_nrows: u64,
    pub(super) grid_ncols: u64,
}

pub(super) struct InitializedSwap {
    pub(super) storage: ReadableWritableListableStorage,
    pub(super) swap_chunk_idx: Array2<bool>,
}

pub(super) fn inspect_source_layout(source: &ReadableListableStorage) -> Result<SourceLayout> {
    let entries = source
        .list()
        .expect("failed to list variables in source dataset");
    let first_entry_opt = entries
        .into_iter()
        .map(|entry| entry.to_string())
        .find(|entry| {
            let name = entry.split('/').next().unwrap_or("").to_ascii_lowercase();
            // Skip coordinate axes when selecting a representative variable for cost storage.
            const EXCLUDES: [&str; 6] = ["latitude", "longitude", "band", "x", "y", "spatial_ref"];
            !name.ends_with(".json") && !EXCLUDES.iter().any(|needle| name == *needle)
        });
    let first_entry = match first_entry_opt {
        Some(entry) => entry,
        None => {
            return Err(Error::IO(std::io::Error::other(format!(
                "no non-coordinate variables found in source dataset: {:?}",
                source.list().ok()
            ))));
        }
    };

    let varname = match first_entry.split('/').next() {
        Some(name) => name,
        None => {
            return Err(Error::IO(std::io::Error::other(
                "Could not determine any variable names from source dataset",
            )));
        }
    };
    debug!("Using '{}' to determine shape of cost data", varname);

    let representative = zarrs::array::Array::open(source.clone(), &format!("/{varname}"))?;
    let shape = representative.shape();
    if shape.len() < 3 {
        return Err(Error::InvalidDatasetShape {
            variable: varname.to_string(),
            min_rank: 3,
            shape: shape.to_vec(),
        });
    }

    let chunk_grid_shape = representative.chunk_grid_shape();
    let layout = SourceLayout {
        chunk_grid: representative.chunk_grid().clone(),
        chunk_grid_rows: chunk_grid_shape[1] as usize,
        chunk_grid_cols: chunk_grid_shape[2] as usize,
        grid_nrows: shape[1],
        grid_ncols: shape[2],
    };
    debug!("Chunk grid info: {:?}", &layout.chunk_grid);

    Ok(layout)
}

pub(super) fn initialize_swap<P: AsRef<Path>>(
    swap_path: P,
    layout: &SourceLayout,
    soft_barrier_group_count: usize,
) -> Result<InitializedSwap> {
    let swap: ReadableWritableListableStorage = std::sync::Arc::new(
        zarrs::filesystem::FilesystemStore::new(swap_path)
            .expect("could not open filesystem store"),
    );

    trace!("Creating a new group for the cost dataset");
    zarrs::group::GroupBuilder::new()
        .build(swap.clone(), "/")?
        .store_metadata()?;

    add_layer_to_data("cost_invariant", &layout.chunk_grid, &swap)?;
    add_layer_to_data("cost", &layout.chunk_grid, &swap)?;
    add_bool_layer_to_data("hard_barrier_mask", &layout.chunk_grid, &swap)?;
    for retry_state in 0..=soft_barrier_group_count {
        add_bool_layer_to_data(
            &cumulative_soft_barrier_mask_name(retry_state),
            &layout.chunk_grid,
            &swap,
        )?;
    }

    Ok(InitializedSwap {
        storage: swap,
        swap_chunk_idx: Array2::from_elem((layout.chunk_grid_rows, layout.chunk_grid_cols), false),
    })
}

pub(super) fn cumulative_soft_barrier_mask_name(retry_state: usize) -> String {
    format!("soft_barrier_mask_retry_{retry_state}")
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

        let layout = inspect_source_layout(&source).expect("source layout inspection failed");

        assert_eq!(layout.grid_nrows, 8);
        assert_eq!(layout.grid_ncols, 8);
        assert_eq!(layout.chunk_grid_rows, 2);
        assert_eq!(layout.chunk_grid_cols, 2);
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

        let error = inspect_source_layout(&source)
            .err()
            .expect("expected coordinate-only dataset to be rejected");

        assert!(matches!(error, Error::IO(_)));
    }

    #[test]
    fn inspect_source_layout_rejects_representative_variable_with_too_few_dimensions() {
        let tmp = malformed_two_dimensional_dataset();
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));

        let error = inspect_source_layout(&source)
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
        let layout = inspect_source_layout(&source).expect("source layout inspection failed");
        let swap_dir = TempDir::new().expect("could not create temporary swap directory");

        let initialized_swap =
            initialize_swap(swap_dir.path(), &layout, 2).expect("swap initialization failed");

        assert_eq!(initialized_swap.swap_chunk_idx.shape(), &[2, 2]);
        assert!(initialized_swap.swap_chunk_idx.iter().all(|value| !*value));

        let expected_layers = [
            ("/cost", DataType::Float32),
            ("/cost_invariant", DataType::Float32),
            ("/hard_barrier_mask", DataType::Bool),
            ("/soft_barrier_mask_retry_0", DataType::Bool),
            ("/soft_barrier_mask_retry_1", DataType::Bool),
            ("/soft_barrier_mask_retry_2", DataType::Bool),
        ];

        for (layer_name, expected_dtype) in expected_layers {
            let array = zarrs::array::Array::open(initialized_swap.storage.clone(), layer_name)
                .unwrap_or_else(|_| panic!("expected layer {layer_name} to exist"));
            assert_eq!(array.shape(), &[1, 8, 8], "wrong shape for {layer_name}");
            assert_eq!(
                array.chunk_grid_shape(),
                &[1, 2, 2],
                "wrong chunk grid shape for {layer_name}"
            );
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
