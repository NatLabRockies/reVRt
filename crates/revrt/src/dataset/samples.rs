//! Builder pattern for creating test Zarr datasets
//!
//! This module provides support for creating Zarr test datasets with
//! various configurations.

use std::sync::Arc;

use ndarray::Array3;
#[cfg(test)]
use object_store::local::LocalFileSystem;
use rand::RngExt;
use tempfile::TempDir;
use zarrs::array::{ArrayBuilder, DataType, FillValue};
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;
#[cfg(test)]
use zarrs::storage::AsyncReadableListableStorage;
use zarrs::storage::ReadableWritableListableStorage;
#[cfg(test)]
use zarrs_object_store::AsyncObjectStore;

/// Fill strategy for layer data
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) enum FillStrategy {
    /// Fill with constant value
    Constant(f32),
    /// Fill with sequential values starting from the input value
    Sequential(u64),
    /// Fill with random values in range [min, max]
    Random(f32, f32),
    /// Fill with custom function: (row, col) -> value
    Custom(fn(u64, u64) -> f32),
    /// Fill with provided vector
    Values(Vec<f32>),
}

/// Configuration for a single layer
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct LayerConfig {
    /// Layer name (e.g., "A", "temperature", "cost")
    name: String,
    /// Fill strategy for this layer
    fill: FillStrategy,
}

#[allow(dead_code)]
impl LayerConfig {
    /// Create a new layer configuration
    pub(crate) fn new(name: impl Into<String>, fill: FillStrategy) -> Self {
        Self {
            name: name.into(),
            fill,
        }
    }

    /// Create a constant-filled layer
    pub(crate) fn constant(name: impl Into<String>, value: f32) -> Self {
        Self::new(name, FillStrategy::Constant(value))
    }

    /// Create a sequentially-filled layer
    pub(crate) fn sequential(name: impl Into<String>, start: u64) -> Self {
        Self::new(name, FillStrategy::Sequential(start))
    }

    /// Create a randomly-filled layer
    pub(crate) fn random(name: impl Into<String>, min: f32, max: f32) -> Self {
        Self::new(name, FillStrategy::Random(min, max))
    }

    /// Create a custom-filled layer
    pub(crate) fn custom(name: impl Into<String>, fill_fn: fn(u64, u64) -> f32) -> Self {
        Self::new(name, FillStrategy::Custom(fill_fn))
    }

    /// Create a layer with all ones
    pub(crate) fn ones(name: impl Into<String>) -> Self {
        Self::constant(name, 1.0)
    }

    /// Create a layer with all zeros
    pub(crate) fn zeros(name: impl Into<String>) -> Self {
        Self::constant(name, 0.0)
    }
}

/// Builder for creating test Zarr datasets
///
/// # Example
/// ```
/// use dataset::samples::{ZarrTestBuilder, LayerConfig, FillStrategy};
///
/// let store = ZarrTestBuilder::new()
///     .dimensions(8, 8)
///     .chunks(4, 4)
///     .layer(LayerConfig::ones("A"))
///     .layer(LayerConfig::sequential("B"))
///     .layer(LayerConfig::constant("C", 5.0))
///     .build()
///     .unwrap();
/// ```
pub(crate) struct ZarrTestBuilder {
    /// Number of bands
    nb: u64,
    /// Number of rows
    ni: u64,
    /// Number of columns
    nj: u64,
    /// Chunk bands
    cb: u64,
    /// Chunk rows
    ci: u64,
    /// Chunk columns
    cj: u64,
    /// Layer configurations
    layers: Vec<LayerConfig>,
    /// Data type for all layers
    dtype: DataType,
    /// Fill value for NaN
    fill_value: FillValue,
}

impl Default for ZarrTestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl ZarrTestBuilder {
    /// Create a new builder with default settings
    pub(crate) fn new() -> Self {
        Self {
            nb: 1,
            ni: 8,
            nj: 8,
            cb: 1,
            ci: 4,
            cj: 4,
            layers: Vec::new(),
            dtype: DataType::Float32,
            fill_value: FillValue::from(zarrs::array::ZARR_NAN_F32),
        }
    }

    /// Set array dimensions (rows, columns)
    pub(crate) fn dimensions(mut self, nb: u64, ni: u64, nj: u64) -> Self {
        self.nb = nb;
        self.ni = ni;
        self.nj = nj;
        self
    }

    /// Set chunk dimensions (rows, columns)
    pub(crate) fn chunks(mut self, cb: u64, ci: u64, cj: u64) -> Self {
        self.cb = cb;
        self.ci = ci;
        self.cj = cj;
        self
    }

    /// Set both array and chunk dimensions
    pub(crate) fn shape(mut self, nb: u64, ni: u64, nj: u64, cb: u64, ci: u64, cj: u64) -> Self {
        self.nb = nb;
        self.ni = ni;
        self.nj = nj;
        self.cb = cb;
        self.ci = ci;
        self.cj = cj;
        self
    }

    /// Add a layer configuration
    pub(crate) fn layer(mut self, layer: LayerConfig) -> Self {
        self.layers.push(layer);
        self
    }

    /// Add multiple layers at once
    pub(crate) fn layers(mut self, layers: Vec<LayerConfig>) -> Self {
        self.layers.extend(layers);
        self
    }

    /// Set data type for all layers (default: Float32)
    pub(crate) fn data_type(mut self, dtype: DataType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Build the Zarr store with configured layers
    pub(crate) fn build(self) -> Result<TempDir, Box<dyn std::error::Error>> {
        let tmp_path = TempDir::new()?;

        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(tmp_path.path()).unwrap());

        // Create root group
        GroupBuilder::new()
            .build(store.clone(), "/")?
            .store_metadata()?;

        // Create each layer
        for layer_config in &self.layers {
            self.create_layer(&store, layer_config)?;
        }

        Ok(tmp_path)
    }

    /// Create a single layer with its data
    fn create_layer(
        &self,
        store: &ReadableWritableListableStorage,
        config: &LayerConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create array
        let array = ArrayBuilder::new(
            vec![self.nb, self.ni, self.nj],
            vec![self.cb, self.ci, self.cj],
            self.dtype.clone(),
            self.fill_value.clone(),
        )
        .dimension_names(["band", "y", "x"].into())
        .build(store.clone(), &format!("/{}", config.name))?;

        array.store_metadata()?;

        // Generate data based on fill strategy
        let data = self.generate_data(&config.fill)?;

        // Write data
        let subset = ArraySubset::new_with_ranges(&[
            0..self.nb,
            0..(self.ni / self.ci),
            0..(self.nj / self.cj),
        ]);

        array.store_chunks_ndarray(&subset, data)?;

        Ok(())
    }

    /// Generate data based on fill strategy
    fn generate_data(
        &self,
        fill: &FillStrategy,
    ) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
        let size = (self.ni * self.nj) as usize;
        let values = match fill {
            FillStrategy::Constant(val) => vec![*val; size],

            FillStrategy::Sequential(offset) => (*offset..(size as u64 + offset))
                .map(|x| x as f32)
                .collect(),

            FillStrategy::Random(min, max) => {
                let mut rng = rand::rng();
                (0..size).map(|_| rng.random_range(*min..=*max)).collect()
            }

            FillStrategy::Custom(func) => {
                let mut values = Vec::with_capacity(size);
                for i in 0..self.ni {
                    for j in 0..self.nj {
                        values.push(func(i, j));
                    }
                }
                values
            }

            FillStrategy::Values(vals) => {
                if vals.len() != size {
                    return Err(format!(
                        "Values vector length {} doesn't match array size {}",
                        vals.len(),
                        size
                    )
                    .into());
                }
                vals.clone()
            }
        };

        let data = Array3::from_shape_vec(
            (self.nb as usize, self.ni as usize, self.nj as usize),
            values,
        )?;

        Ok(data)
    }
}

// ============================================================================
// Convenience builders for common patterns
// ============================================================================

/// Quick builder for uniform cost surfaces (all ones)
pub(crate) fn uniform_ones_cost_zarr(
    nb: u64,
    ni: u64,
    nj: u64,
    cb: u64,
    ci: u64,
    cj: u64,
) -> TempDir {
    ZarrTestBuilder::new()
        .shape(nb, ni, nj, cb, ci, cj)
        .layer(LayerConfig::ones("cost"))
        .build()
        .expect("Failed to create uniform cost zarr")
}

/// Quick builder for uniform cost surfaces (custom cost)
pub(crate) fn cost_as_index_zarr(nb: u64, ni: u64, nj: u64, cb: u64, ci: u64, cj: u64) -> TempDir {
    ZarrTestBuilder::new()
        .shape(nb, ni, nj, cb, ci, cj)
        .layer(LayerConfig::sequential("cost", 0))
        .build()
        .expect("Failed to create uniform cost zarr")
}

/// Quick builder for uniform cost surfaces (custom cost)
pub(crate) fn uniform_cost_zarr(
    nb: u64,
    ni: u64,
    nj: u64,
    cb: u64,
    ci: u64,
    cj: u64,
    cost_value: f32,
) -> TempDir {
    ZarrTestBuilder::new()
        .shape(nb, ni, nj, cb, ci, cj)
        .layer(LayerConfig::constant("cost", cost_value))
        .build()
        .expect("Failed to create uniform cost zarr")
}

/// Quick builder for three-layer test (A, B, C with ones)
pub(crate) fn three_layer_ones(nb: u64, ni: u64, nj: u64, cb: u64, ci: u64, cj: u64) -> TempDir {
    ZarrTestBuilder::new()
        .shape(nb, ni, nj, cb, ci, cj)
        .layer(LayerConfig::ones("A"))
        .layer(LayerConfig::ones("B"))
        .layer(LayerConfig::ones("C"))
        .build()
        .expect("Failed to create three-layer zarr")
}

/// Quick builder for multi-variable random data
pub(crate) fn multi_variable_random(
    nb: u64,
    ni: u64,
    nj: u64,
    cb: u64,
    ci: u64,
    cj: u64,
    layers: &[&str],
) -> TempDir {
    let mut builder = ZarrTestBuilder::new().shape(nb, ni, nj, cb, ci, cj);

    for &layer_name in layers {
        builder = builder.layer(LayerConfig::random(layer_name, 0.0, 1.0));
    }

    builder
        .build()
        .expect("Failed to create multi-variable zarr")
}

// ============================================================================
// Preset configurations
// ============================================================================

/// Preset: Simple 4x4 grid for quick unit tests
pub(crate) fn preset_small() -> ZarrTestBuilder {
    ZarrTestBuilder::new().dimensions(1, 4, 4).chunks(1, 2, 2)
}

/// Preset: Medium 16x16 grid for integration tests
#[allow(dead_code)]
pub(crate) fn preset_medium() -> ZarrTestBuilder {
    ZarrTestBuilder::new().dimensions(1, 16, 16).chunks(1, 4, 4)
}

/// Preset: Large 128x128 grid for performance tests
#[allow(dead_code)]
pub(crate) fn preset_large() -> ZarrTestBuilder {
    ZarrTestBuilder::new()
        .dimensions(1, 128, 128)
        .chunks(1, 32, 32)
}

/// Preset: Standard cost surface setup (A, B, C layers)
pub(crate) fn preset_cost_surface() -> ZarrTestBuilder {
    ZarrTestBuilder::new()
        .layer(LayerConfig::sequential("A", 1))
        .layer(LayerConfig::constant("B", 2.0))
        .layer(LayerConfig::ones("C"))
}

/// Wrap any on-disk sample path in an `AsyncReadableListableStorage`.
///
/// # Example
/// ```rust
/// let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "B", "C", "cost"]);
/// let source = samples::async_storage_for(tmp.path());
/// ```
#[cfg(test)]
pub(crate) fn async_storage_for(path: &std::path::Path) -> AsyncReadableListableStorage {
    let store =
        LocalFileSystem::new_with_prefix(path).expect("could not open local filesystem store");
    std::sync::Arc::new(AsyncObjectStore::new(store))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let sample = ZarrTestBuilder::new()
            .dimensions(1, 4, 4)
            .chunks(1, 2, 2)
            .layer(LayerConfig::ones("test"))
            .build()
            .unwrap();

        assert!(sample.path().exists());
    }

    #[test]
    fn test_builder_multiple_layers() {
        let sample = ZarrTestBuilder::new()
            .dimensions(1, 8, 8)
            .chunks(1, 4, 4)
            .layer(LayerConfig::ones("A"))
            .layer(LayerConfig::sequential("B", 1))
            .layer(LayerConfig::constant("C", 5.0))
            .build()
            .unwrap();
        let path = sample.path();

        assert!(path.exists());

        // Verify layers exist
        let store = Arc::new(FilesystemStore::new(path).unwrap());
        for layer_name in ["A", "B", "C"] {
            let array = zarrs::array::Array::open(store.clone(), &format!("/{}", layer_name));
            assert!(array.is_ok(), "Layer {} should exist", layer_name);
        }
    }

    #[test]
    fn test_custom_fill() {
        let sample = ZarrTestBuilder::new()
            .dimensions(1, 4, 4)
            .chunks(1, 2, 2)
            .layer(LayerConfig::custom("custom", |i, j| (i * 10 + j) as f32))
            .build()
            .unwrap();

        assert!(sample.path().exists());
    }

    #[test]
    fn test_uniform_cost_helper() {
        let sample = uniform_ones_cost_zarr(1, 4, 4, 1, 2, 2);
        let path = sample.path();
        assert!(path.exists());

        let store = Arc::new(FilesystemStore::new(path).unwrap());
        let array = zarrs::array::Array::open(store, "/cost");
        assert!(array.is_ok());
    }

    #[test]
    fn test_three_layer_ones_helper() {
        let sample = three_layer_ones(1, 4, 4, 1, 2, 2);
        let path = sample.path();
        assert!(path.exists());

        let store = Arc::new(FilesystemStore::new(path).unwrap());
        for layer in ["A", "B", "C"] {
            let array = zarrs::array::Array::open(store.clone(), &format!("/{}", layer));
            assert!(array.is_ok(), "Layer {} should exist", layer);
        }
    }

    #[test]
    fn test_preset_small() {
        let sample = preset_small()
            .layer(LayerConfig::ones("test"))
            .build()
            .unwrap();

        assert!(sample.path().exists());
    }

    #[test]
    fn test_preset_cost_surface() {
        let sample = preset_cost_surface()
            .dimensions(1, 8, 8)
            .chunks(1, 4, 4)
            .build()
            .unwrap();
        let path = sample.path();

        assert!(path.exists());

        let store = Arc::new(FilesystemStore::new(path).unwrap());
        for layer in ["A", "B", "C"] {
            let array = zarrs::array::Array::open(store.clone(), &format!("/{}", layer));
            assert!(array.is_ok(), "Layer {} should exist", layer);
        }
    }
}
