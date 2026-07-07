//! Samples to support tests that require Features
//!
//! All arrays are 2-D with shape `[ni, nj]` and dimension names `["y", "x"]`.
use std::sync::Arc;

use ndarray::Array2;
use object_store::local::LocalFileSystem;
use rand::RngExt;
use tempfile::TempDir;
use zarrs::array::data_type;
use zarrs::array::{ArrayBuilder, ArraySubset, DataType, FillValue};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;
use zarrs::storage::{AsyncReadableListableStorage, ReadableWritableListableStorage};
use zarrs_object_store::AsyncObjectStore;

// -----------------------------------------------------------------------
// Fill strategy
// -----------------------------------------------------------------------

/// How to populate the values of a single feature layer.
#[derive(Debug, Clone)]
pub(crate) enum FillStrategy {
    /// Every cell gets the same value.
    Constant(f32),
    /// Cells are numbered `1, 2, …, ni*nj` in row-major order.
    Sequential,
    /// Each cell receives an independent uniform random draw from `[min, max]`.
    Random(f32, f32),
    /// Values are derived from `(row, col)` via a user-supplied function.
    Custom(fn(u64, u64) -> f32),
    /// Use an explicit flat vector (must have length `ni * nj`).
    Values(Vec<f32>),
}

// -----------------------------------------------------------------------
// Data type
// -----------------------------------------------------------------------

/// Zarr data type to use when writing a feature layer.
///
/// Fill strategies are always expressed as `f32` values; this enum
/// controls only how those values are cast and stored on disk.
/// Floating-point NaN is used as the Zarr fill value for float types;
/// the type's maximum value (`i8::MAX`, `u32::MAX`, etc.) is used for
/// integer types, making unwritten cells easy to distinguish from valid data.
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
pub(crate) enum FeatureDataType {
    #[default]
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl FeatureDataType {
    fn zarrs_dtype(self) -> DataType {
        match self {
            Self::Float32 => data_type::float32(),
            Self::Float64 => data_type::float64(),
            Self::Int8 => data_type::int8(),
            Self::Int16 => data_type::int16(),
            Self::Int32 => data_type::int32(),
            Self::Int64 => data_type::int64(),
            Self::UInt8 => data_type::uint8(),
            Self::UInt16 => data_type::uint16(),
            Self::UInt32 => data_type::uint32(),
            Self::UInt64 => data_type::uint64(),
        }
    }

    fn fill_value(self) -> FillValue {
        match self {
            Self::Float32 => FillValue::from(zarrs::array::ZARR_NAN_F32),
            Self::Float64 => FillValue::from(zarrs::array::ZARR_NAN_F64),
            Self::Int8 => FillValue::from(i8::MAX),
            Self::Int16 => FillValue::from(i16::MAX),
            Self::Int32 => FillValue::from(i32::MAX),
            Self::Int64 => FillValue::from(i64::MAX),
            Self::UInt8 => FillValue::from(u8::MAX),
            Self::UInt16 => FillValue::from(u16::MAX),
            Self::UInt32 => FillValue::from(u32::MAX),
            Self::UInt64 => FillValue::from(u64::MAX),
        }
    }
}

// -----------------------------------------------------------------------
// Layer configuration
// -----------------------------------------------------------------------

/// Configuration for a single named feature layer.
#[derive(Debug, Clone)]
pub(crate) struct LayerConfig {
    /// Layer name, e.g. `"A"`, `"elevation"`, `"land_cover"`.
    pub(crate) name: String,
    /// Strategy used to fill this layer's values.
    pub(crate) fill: FillStrategy,
    /// Zarr element type to use when writing this layer (default: `Float32`).
    pub(crate) dtype: FeatureDataType,
}

impl LayerConfig {
    /// Create a layer with an arbitrary fill strategy and the default data type (`Float32`).
    pub(crate) fn new(name: impl Into<String>, fill: FillStrategy) -> Self {
        Self {
            name: name.into(),
            fill,
            dtype: FeatureDataType::default(),
        }
    }

    /// Override the Zarr data type for this layer.
    ///
    /// The fill strategy values are cast to the requested type at write
    /// time.  For integer types the conversion truncates toward zero, so
    /// keep fill values within the target type's range.
    pub(crate) fn with_dtype(mut self, dtype: FeatureDataType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Constant-valued layer.
    pub(crate) fn constant(name: impl Into<String>, value: f32) -> Self {
        Self::new(name, FillStrategy::Constant(value))
    }

    /// Layer filled with `1.0` everywhere.
    pub(crate) fn ones(name: impl Into<String>) -> Self {
        Self::constant(name, 1.0)
    }

    /// Layer filled with `0.0` everywhere.
    #[allow(dead_code)]
    pub(crate) fn zeros(name: impl Into<String>) -> Self {
        Self::constant(name, 0.0)
    }

    /// Layer whose cells are numbered `1, 2, …, ni*nj` in row-major order.
    pub(crate) fn sequential(name: impl Into<String>) -> Self {
        Self::new(name, FillStrategy::Sequential)
    }

    /// Layer filled with independent uniform draws from `[min, max]`.
    pub(crate) fn random(name: impl Into<String>, min: f32, max: f32) -> Self {
        Self::new(name, FillStrategy::Random(min, max))
    }

    /// Layer whose values are computed cell-by-cell with `f(row, col)`.
    pub(crate) fn custom(name: impl Into<String>, f: fn(u64, u64) -> f32) -> Self {
        Self::new(name, FillStrategy::Custom(f))
    }
}

// -----------------------------------------------------------------------
// Builder
// -----------------------------------------------------------------------

/// Builds an in-memory-backed (temp-dir) Zarr store suitable for use with
/// [`Features`](super::Features).
///
/// Each layer is stored as a 2-D array with shape `[ni, nj]` and chunk
/// shape `[ci, cj]`, with dimension names `["y", "x"]`.  The element
/// type defaults to `Float32` but can be overridden per layer via
/// [`LayerConfig::with_dtype`].
///
/// # Example
/// ```rust
/// let (tmp, storage) = FeaturesTestBuilder::new()
///     .dimensions(8, 8)
///     .chunks(4, 4)
///     .layer(LayerConfig::ones("elevation"))
///     .layer(LayerConfig::sequential("land_cover"))
///     .build()
///     .unwrap();
/// ```
pub(crate) struct FeaturesTestBuilder {
    ni: u64,
    nj: u64,
    ci: u64,
    cj: u64,
    layers: Vec<LayerConfig>,
}

impl Default for FeaturesTestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FeaturesTestBuilder {
    /// Create a new builder with default dimensions `8×8` and chunk size `4×4`.
    pub(crate) fn new() -> Self {
        Self {
            ni: 8,
            nj: 8,
            ci: 4,
            cj: 4,
            layers: Vec::new(),
        }
    }

    /// Set array dimensions (rows, columns).
    pub(crate) fn dimensions(mut self, ni: u64, nj: u64) -> Self {
        self.ni = ni;
        self.nj = nj;
        self
    }

    /// Set chunk dimensions (rows, columns).
    pub(crate) fn chunks(mut self, ci: u64, cj: u64) -> Self {
        self.ci = ci;
        self.cj = cj;
        self
    }

    /// Add a single layer.
    pub(crate) fn layer(mut self, layer: LayerConfig) -> Self {
        self.layers.push(layer);
        self
    }

    /// Build the store.
    ///
    /// Returns the [`TempDir`] (keeping it alive keeps the files on disk)
    /// and an [`AsyncReadableListableStorage`] pointing at it.
    pub(crate) fn build(
        self,
    ) -> Result<(TempDir, AsyncReadableListableStorage), Box<dyn std::error::Error>> {
        let tmp = TempDir::new()?;

        let sync_store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path())?);

        GroupBuilder::new()
            .build(sync_store.clone(), "/")?
            .store_metadata()?;

        for layer_config in &self.layers {
            self.write_layer(&sync_store, layer_config)?;
        }

        let async_store: AsyncReadableListableStorage = Arc::new(AsyncObjectStore::new(
            LocalFileSystem::new_with_prefix(tmp.path())?,
        ));

        Ok((tmp, async_store))
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn write_layer(
        &self,
        store: &ReadableWritableListableStorage,
        config: &LayerConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let array = ArrayBuilder::new(
            vec![self.ni, self.nj],
            vec![self.ci, self.cj],
            config.dtype.zarrs_dtype(),
            config.dtype.fill_value(),
        )
        .dimension_names(["y", "x"].into())
        .build(store.clone(), &format!("/{}", config.name))?;

        array.store_metadata()?;

        let flat = self.generate_flat(&config.fill)?;
        let shape = (self.ni as usize, self.nj as usize);
        let subset = ArraySubset::new_with_ranges(&[
            0..self.ni.div_ceil(self.ci),
            0..self.nj.div_ceil(self.cj),
        ]);

        macro_rules! store_as {
            ($T:ty) => {{
                let data: Array2<$T> =
                    Array2::from_shape_vec(shape, flat.into_iter().map(|v| v as $T).collect())?;
                array.store_chunks_ndarray(&subset, data)?;
            }};
        }

        match config.dtype {
            Featuredata_type::float32() => store_as!(f32),
            Featuredata_type::float64() => store_as!(f64),
            Featuredata_type::int8() => store_as!(i8),
            Featuredata_type::int16() => store_as!(i16),
            Featuredata_type::int32() => store_as!(i32),
            Featuredata_type::int64() => store_as!(i64),
            Featuredata_type::uint8() => store_as!(u8),
            Featuredata_type::uint16() => store_as!(u16),
            Featuredata_type::uint32() => store_as!(u32),
            Featuredata_type::uint64() => store_as!(u64),
        }

        Ok(())
    }

    /// Generate the flat `f32` values described by `fill`.
    ///
    /// All fill strategies produce `f32` values.  The caller is
    /// responsible for casting to the target element type before writing.
    fn generate_flat(&self, fill: &FillStrategy) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let size = (self.ni * self.nj) as usize;

        let flat = match fill {
            FillStrategy::Constant(v) => vec![*v; size],

            FillStrategy::Sequential => (1..=size).map(|x| x as f32).collect(),

            FillStrategy::Random(min, max) => {
                let mut rng = rand::rng();
                (0..size).map(|_| rng.random_range(*min..=*max)).collect()
            }

            FillStrategy::Custom(f) => {
                let mut values = Vec::with_capacity(size);
                for i in 0..self.ni {
                    for j in 0..self.nj {
                        values.push(f(i, j));
                    }
                }
                values
            }

            FillStrategy::Values(vals) => {
                if vals.len() != size {
                    return Err(format!(
                        "Values length {} does not match array size {}",
                        vals.len(),
                        size
                    )
                    .into());
                }
                vals.clone()
            }
        };

        Ok(flat)
    }
}

// -----------------------------------------------------------------------
// Preset builders
// -----------------------------------------------------------------------

/// Preset: small 4×4 grid (2×2 chunks). Good for quick unit tests.
pub(crate) fn preset_small() -> FeaturesTestBuilder {
    FeaturesTestBuilder::new().dimensions(4, 4).chunks(2, 2)
}

/// Preset: medium 16×16 grid (4×4 chunks). Good for integration tests.
#[allow(dead_code)]
pub(crate) fn preset_medium() -> FeaturesTestBuilder {
    FeaturesTestBuilder::new().dimensions(16, 16).chunks(4, 4)
}

/// Preset: large 128×128 grid (32×32 chunks). Good for performance tests.
#[allow(dead_code)]
pub(crate) fn preset_large() -> FeaturesTestBuilder {
    FeaturesTestBuilder::new()
        .dimensions(128, 128)
        .chunks(32, 32)
}

// -----------------------------------------------------------------------
// Convenience functions
// -----------------------------------------------------------------------

/// Single layer of all-ones, 8×8 with 4×4 chunks.
///
/// Useful when the actual feature values are irrelevant to the test.
#[allow(dead_code)]
pub(crate) fn single_ones_layer(
    name: &str,
) -> Result<(TempDir, AsyncReadableListableStorage), Box<dyn std::error::Error>> {
    FeaturesTestBuilder::new()
        .layer(LayerConfig::ones(name))
        .build()
}

/// Classic three-layer setup (`A`, `B`, `C`) filled with sequential values.
///
/// Mirrors the spirit of `multi_variable_random` from the dataset samples,
/// but uses predictable deterministic values.
pub(crate) fn multi_variable_sequential(
    ni: u64,
    nj: u64,
    ci: u64,
    cj: u64,
) -> Result<(TempDir, AsyncReadableListableStorage), Box<dyn std::error::Error>> {
    FeaturesTestBuilder::new()
        .dimensions(ni, nj)
        .chunks(ci, cj)
        .layer(LayerConfig::sequential("A"))
        .layer(LayerConfig::sequential("B"))
        .layer(LayerConfig::sequential("C"))
        .build()
}

/// Three-layer setup (`A`, `B`, `C`) filled with independent random values.
///
/// Each call produces a different dataset. Prefer
/// [`multi_variable_sequential`] when reproducibility matters.
pub(crate) fn multi_variable_random(
    ni: u64,
    nj: u64,
    ci: u64,
    cj: u64,
) -> Result<(TempDir, AsyncReadableListableStorage), Box<dyn std::error::Error>> {
    FeaturesTestBuilder::new()
        .dimensions(ni, nj)
        .chunks(ci, cj)
        .layer(LayerConfig::random("A", 0.0, 1.0))
        .layer(LayerConfig::random("B", 0.0, 1.0))
        .layer(LayerConfig::random("C", 0.0, 1.0))
        .build()
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_creates_store() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::ones("test"))
            .build()
            .unwrap();

        assert!(tmp.path().exists());
    }

    #[test]
    fn builder_multiple_layers_exist_on_disk() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(8, 8)
            .chunks(4, 4)
            .layer(LayerConfig::ones("A"))
            .layer(LayerConfig::sequential("B"))
            .layer(LayerConfig::constant("C", 5.0))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        for name in ["A", "B", "C"] {
            assert!(
                zarrs::array::Array::open(sync_store.clone(), &format!("/{name}")).is_ok(),
                "layer {name} should exist on disk"
            );
        }
    }

    #[test]
    fn builder_array_shape_is_2d() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(4, 6)
            .chunks(2, 3)
            .layer(LayerConfig::ones("elevation"))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let array = zarrs::array::Array::open(sync_store, "/elevation").unwrap();
        assert_eq!(array.shape(), &[4, 6]);
    }

    #[test]
    fn builder_sequential_values() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(2, 3)
            .chunks(2, 3)
            .layer(LayerConfig::sequential("seq"))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let array = zarrs::array::Array::open(sync_store, "/seq").unwrap();
        let subset = zarrs::array::ArraySubset::new_with_ranges(&[0..2, 0..3]);
        let vals: Vec<f32> = array.retrieve_array_subset_elements(&subset).unwrap();
        // Sequential starts at 1
        let expected: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        assert_eq!(vals, expected);
    }

    #[test]
    fn builder_custom_fill() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(3, 3)
            .chunks(3, 3)
            .layer(LayerConfig::custom("idx", |i, j| (i * 10 + j) as f32))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let array = zarrs::array::Array::open(sync_store, "/idx").unwrap();
        let subset = zarrs::array::ArraySubset::new_with_ranges(&[0..3, 0..3]);
        let vals: Vec<f32> = array.retrieve_array_subset_elements(&subset).unwrap();

        for i in 0..3u64 {
            for j in 0..3u64 {
                let expected = (i * 10 + j) as f32;
                assert_eq!(
                    vals[(i * 3 + j) as usize],
                    expected,
                    "mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn builder_values_wrong_length_errors() {
        let result = FeaturesTestBuilder::new()
            .dimensions(2, 2)
            .chunks(2, 2)
            .layer(LayerConfig::new(
                "bad",
                FillStrategy::Values(vec![1.0, 2.0]),
            )) // needs 4
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn preset_small_produces_correct_shape() {
        let (tmp, _storage) = preset_small()
            .layer(LayerConfig::ones("A"))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let array = zarrs::array::Array::open(sync_store, "/A").unwrap();
        assert_eq!(array.shape(), &[4, 4]);
    }

    #[tokio::test]
    async fn async_storage_opens_array() {
        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::constant("temperature", 273.15))
            .build()
            .unwrap();

        let array = zarrs::array::Array::async_open(storage, "/temperature")
            .await
            .unwrap();
        assert_eq!(array.shape(), &[4, 4]);
    }

    #[tokio::test]
    async fn async_storage_retrieves_correct_values() {
        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::constant("elev", 42.0))
            .build()
            .unwrap();

        let array = zarrs::array::Array::async_open(storage, "/elev")
            .await
            .unwrap();
        let chunk: Vec<f32> = array
            .async_retrieve_chunk_elements::<f32>(&[0, 0])
            .await
            .unwrap();
        assert!(chunk.iter().all(|&v| v == 42.0));
    }

    #[tokio::test]
    async fn multi_variable_sequential_has_three_layers() {
        let (_tmp, storage) = multi_variable_sequential(4, 4, 2, 2).unwrap();

        for name in ["A", "B", "C"] {
            let result =
                zarrs::array::Array::async_open(storage.clone(), &format!("/{name}")).await;
            assert!(result.is_ok(), "layer {name} should be openable");
        }
    }

    #[tokio::test]
    async fn multi_variable_random_layers_are_non_empty() {
        let (_tmp, storage) = multi_variable_random(4, 4, 2, 2).unwrap();

        let array = zarrs::array::Array::async_open(storage, "/A")
            .await
            .unwrap();
        let chunk: Vec<f32> = array
            .async_retrieve_chunk_elements::<f32>(&[0, 0])
            .await
            .unwrap();
        assert!(!chunk.is_empty());
    }

    // -----------------------------------------------------------------------
    // Data-type tests
    // -----------------------------------------------------------------------

    #[test]
    fn layer_dtype_float64_has_correct_zarr_type() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::ones("elev").with_dtype(Featuredata_type::float64()))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let array = zarrs::array::Array::open(sync_store, "/elev").unwrap();
        assert_eq!(array.data_type(), &data_type::float64());
    }

    #[test]
    fn layer_dtype_int32_has_correct_zarr_type() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::sequential("band").with_dtype(Featuredata_type::int32()))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let array = zarrs::array::Array::open(sync_store, "/band").unwrap();
        assert_eq!(array.data_type(), &data_type::int32());
    }

    #[test]
    fn layer_dtype_uint8_has_correct_zarr_type() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::constant("mask", 1.0).with_dtype(Featuredata_type::uint8()))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let array = zarrs::array::Array::open(sync_store, "/mask").unwrap();
        assert_eq!(array.data_type(), &data_type::uint8());
    }

    #[test]
    fn layer_dtype_float64_stores_correct_values() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(2, 2)
            .chunks(2, 2)
            .layer(LayerConfig::constant("temp", 1.62).with_dtype(Featuredata_type::float64()))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let array = zarrs::array::Array::open(sync_store, "/temp").unwrap();
        let subset = zarrs::array::ArraySubset::new_with_ranges(&[0..2, 0..2]);
        let vals: Vec<f64> = array.retrieve_array_subset_elements(&subset).unwrap();
        for v in vals {
            let diff = (v - 1.62_f64).abs();
            // The fill value is specified as f32 (1.62f32 ≈ 1.6200001e0),
            // so the round-trip error vs the true f64 value is ~1e-7.
            assert!(diff < 1e-6, "expected ~1.62, got {v}");
        }
    }

    #[test]
    fn layer_dtype_int16_stores_correct_values() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(2, 3)
            .chunks(2, 3)
            .layer(LayerConfig::sequential("idx").with_dtype(Featuredata_type::int16()))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let array = zarrs::array::Array::open(sync_store, "/idx").unwrap();
        let subset = zarrs::array::ArraySubset::new_with_ranges(&[0..2, 0..3]);
        let vals: Vec<i16> = array.retrieve_array_subset_elements(&subset).unwrap();
        let expected: Vec<i16> = (1..=6).collect();
        assert_eq!(vals, expected);
    }

    #[test]
    fn layer_dtype_uint32_stores_correct_values() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(2, 2)
            .chunks(2, 2)
            .layer(LayerConfig::constant("cls", 255.0).with_dtype(Featuredata_type::uint32()))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let array = zarrs::array::Array::open(sync_store, "/cls").unwrap();
        let subset = zarrs::array::ArraySubset::new_with_ranges(&[0..2, 0..2]);
        let vals: Vec<u32> = array.retrieve_array_subset_elements(&subset).unwrap();
        assert!(vals.iter().all(|&v| v == 255));
    }

    #[test]
    fn mixed_dtypes_in_same_store() {
        // A realistic scenario: float cost layer + integer classification layer.
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::random("cost", 0.0, 100.0)) // Float32 (default)
            .layer(LayerConfig::constant("land_cover", 3.0).with_dtype(Featuredata_type::uint8()))
            .layer(LayerConfig::sequential("elevation").with_dtype(Featuredata_type::int16()))
            .build()
            .unwrap();

        let sync_store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let cost_array = zarrs::array::Array::open(sync_store.clone(), "/cost").unwrap();
        let lc_array = zarrs::array::Array::open(sync_store.clone(), "/land_cover").unwrap();
        let elev_array = zarrs::array::Array::open(sync_store.clone(), "/elevation").unwrap();

        assert_eq!(cost_array.data_type(), &data_type::float32());
        assert_eq!(lc_array.data_type(), &data_type::uint8());
        assert_eq!(elev_array.data_type(), &data_type::int16());
    }
}
