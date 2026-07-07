//! Asynchronous lazy-loaded subset of a Features store
//!
//! [`AsyncLazySubset`] provides cached, async access to a rectangular region
//! of a [`Features`](super::Features) store. Each variable is loaded at most
//! once; subsequent calls to [`get`](AsyncLazySubset::get) return a clone from
//! an internal cache protected by a [`tokio::sync::RwLock`], making concurrent
//! access from multiple tasks safe and efficient.
//!
//! If the requested [`ArraySubset`] extends beyond the source array's shape,
//! out-of-bounds grid points are filled with a type-specific sentinel:
//! `NaN` for floating-point types.
//!
//! We only support lazy subsets of f32 and f64. An initial prototype included
//! integer types, but that was dropped and there is no intention to bring
//! it back. For the pathfidning algoritythms we need a high ceiling to allow
//! long distances, and some precision to disntiguish between two neigbor grid
//! points, so a float type is the correct choice here. Note that the input
//! features can be of any numeric type, and the lazy subset is the point where
//! we normalize to a single data type so all variables can be operated together
//! and reduce to a single conversion calculation.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use ndarray::{ArrayD, IxDyn, SliceInfoElem};
use tokio::sync::RwLock;
use tracing::{error, trace};
use zarrs::array::{Array, ArraySubset, DataType, ElementOwned};
use zarrs::array::data_type;
use zarrs::array::ArraySubset;
use zarrs::storage::{AsyncReadableListableStorage, AsyncReadableListableStorageTraits};

use crate::error::{Error, Result};

/// Marker trait for element types that [`AsyncLazySubset`] can store.
///
/// Implementors must provide a *sentinel* value that represents
/// out-of-bounds or missing data. For IEEE floats this is `NaN`.
pub(crate) trait AsyncLazyElement: ElementOwned + Clone + Send + Sync + 'static {
    /// Value used to fill grid points that fall outside the source array.
    const NAN_VALUE: Self;
    /// Convert from f64 (used as the universal intermediary for type conversion).
    fn from_f64(v: f64) -> Self;
}

impl AsyncLazyElement for f32 {
    const NAN_VALUE: f32 = f32::NAN;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl AsyncLazyElement for f64 {
    const NAN_VALUE: f64 = f64::NAN;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
}

// ---------------------------------------------------------------------------
// AsyncLazySubset
// ---------------------------------------------------------------------------

/// Async, cached view into a rectangular region of a Zarr-backed feature
/// store.
///
/// The subset is fixed at construction time (via an [`ArraySubset`]) so that
/// every variable returned by [`get`](Self::get) covers the same spatial
/// domain. Data is loaded lazily on the first access and cached for the
/// lifetime of this struct.
///
/// # Concurrency
///
/// The internal cache is protected by a [`tokio::sync::RwLock`], so
/// multiple tasks may call [`get`](Self::get) concurrently. Only the
/// first caller for a given variable will trigger I/O; later callers
/// either wait for the write lock to be released or read directly from
/// the cache.
///
/// # Notes
///
/// - Consider using Arc<ArrayD<T>> in `data` to avoid cloning the full array
///   on every call to `get()`. This will be mostly used by the cost function
///   and assuming that most of the variables are used only once, the impact
///   of cloning should be negligible.
pub(crate) struct AsyncLazySubset<T: AsyncLazyElement> {
    /// Async readable storage backing the feature layers.
    source: AsyncReadableListableStorage,
    /// The index region this subset covers.
    subset: ArraySubset,
    /// Per-variable cache.
    data: Arc<RwLock<HashMap<String, ArrayD<T>>>>,
}

impl<T: AsyncLazyElement> fmt::Display for AsyncLazySubset<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "AsyncLazySubset<{}> {{ subset: {:?} }}",
            std::any::type_name::<T>(),
            self.subset,
        )
    }
}

impl<T: AsyncLazyElement> fmt::Debug for AsyncLazySubset<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let cached_vars = match self.data.try_read() {
            Ok(cache) => {
                let mut keys: Vec<&str> = cache.keys().map(|k| k.as_str()).collect();
                keys.sort();
                format!("{:?}", keys)
            }
            Err(_) => "<locked>".to_string(),
        };

        f.debug_struct("AsyncLazySubset")
            .field("subset", &self.subset)
            .field("element_type", &std::any::type_name::<T>())
            .field("cached_variables", &cached_vars)
            .finish()
    }
}

impl<T: AsyncLazyElement> AsyncLazySubset<T> {
    /// Create a new `AsyncLazySubset` tied to the given storage and region.
    ///
    /// No I/O is performed at construction time; data is fetched lazily on
    /// the first call to [`get`](Self::get).
    ///
    /// # Arguments
    ///
    /// * `source` – Async readable Zarr storage (typically obtained from
    ///   [`Features::storage`](super::Features::storage)).
    /// * `subset` – The array region to load. May extend beyond the source
    ///   array boundaries; out-of-bounds cells will be filled with
    ///   [`AsyncLazyElement::NAN_VALUE`].
    #[allow(dead_code)]
    pub(crate) fn new(source: AsyncReadableListableStorage, subset: ArraySubset) -> Self {
        trace!("Creating AsyncLazySubset for subset: {:?}", subset);
        Self {
            source,
            subset,
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// The subset that defines this view's index region.
    #[allow(dead_code)]
    pub(crate) fn subset(&self) -> &ArraySubset {
        &self.subset
    }

    /// Retrieve (or load) the data for `varname` within this subset.
    ///
    /// On the first call for a given variable the data is fetched from the
    /// underlying Zarr store. Subsequent calls return a clone from the
    /// cache without any I/O.
    ///
    /// If the subset extends beyond the source array's shape, the
    /// out-of-bounds cells are filled with [`AsyncLazyElement::NAN_VALUE`].
    ///
    /// # Errors
    ///
    /// * [`Error::VariableNotFound`] – the variable does not exist in the
    ///   store.
    /// * [`Error::IO`] – an I/O error occurred while reading the data.
    ///
    /// # Notes
    ///
    /// - Consider split the behavior into a strict and a flexible one padding
    ///   NaNs beyond the bounds, such as `get()` + `get_padded()`.
    #[allow(dead_code)]
    pub(crate) async fn get(&self, varname: &str) -> Result<ArrayD<T>> {
        trace!("AsyncLazySubset::get(\"{}\")", varname);

        // Fast path: check read-lock cache.
        {
            let cache = self.data.read().await;
            if let Some(cached) = cache.get(varname) {
                trace!("Cache hit for variable \"{}\"", varname);
                return Ok(cached.clone());
            }
        }

        // Slow path: acquire write lock, double-check, then load.
        let mut cache = self.data.write().await;

        // Another task may have populated the cache while we waited.
        if let Some(cached) = cache.get(varname) {
            trace!(
                "Cache hit for variable \"{}\" (populated while waiting for write lock)",
                varname
            );
            return Ok(cached.clone());
        }

        trace!(
            "Loading variable \"{}\" for subset {:?}",
            varname, self.subset
        );

        let data = self.load_variable(varname).await?;

        cache.insert(varname.to_string(), data.clone());
        Ok(data)
    }

    /// Read a variable's data from the Zarr store and convert it to the
    /// subset's element type `T`.
    ///
    /// The on-disk data type is inspected at runtime and the raw values are
    /// read as their native type, then converted to `T` via `f64` as an
    /// intermediary. This allows a single `AsyncLazySubset<f32>` (or `<f64>`)
    /// to transparently consume variables stored as any of the supported
    /// numeric types: f32, f64, i16, i32, i64, u8, u16, u32.
    ///
    /// Using `f64` as the intermediary is exact for all integer types up to
    /// 53 bits of magnitude (i.e. i16, i32, u8, u16, u32). For i64 and u64,
    /// values exceeding 2^53 may lose precision during the conversion.
    /// The final `f64 → f32` step (when `T` is f32) further truncates to
    /// 23 bits of mantissa, so large i32 or u32 values may also round.
    ///
    /// **Precondition**: `subset` must not extend beyond the array's shape.
    /// Violating this will surface a Zarr-level error, not silent corruption.
    ///
    /// # Arguments
    ///
    /// * `array`: An opened Zarr array pointing to the variable to read.
    /// * `varname`: The variable name, used only for error messages.
    /// * `subset`: The spatial region to retrieve, expressed as an
    ///   [`ArraySubset`] that must lie within the array's bounds.
    ///
    /// # Errors
    ///
    /// * [`Error::IO`]: an I/O error occurred while reading the data.
    /// * [`Error::UnsupportedDataType`]: the on-disk type is not one of the
    ///   supported numeric types (e.g. complex, string, bool).
    async fn retrieve_subset<S: AsyncReadableListableStorageTraits + 'static + ?Sized>(
        &self,
        array: &Array<S>,
        varname: &str,
        subset: &ArraySubset,
    ) -> Result<ArrayD<T>> {
        trace!(
            "Retrieving in-bounds subset {:?} for \"{}\" (on-disk type: {:?})",
            subset,
            varname,
            array.data_type()
        );

        macro_rules! read_and_convert {
            ($native:ty) => {{
                let native_data = array
                    .async_retrieve_array_subset_ndarray::<$native>(subset)
                    .await
                    .map_err(|err| {
                        Error::IO(std::io::Error::other(format!(
                            "Failed to retrieve subset for variable '{}': {}",
                            varname, err
                        )))
                    })?;
                Ok(native_data.mapv(|v| T::from_f64(v as f64)))
            }};
        }

        match array.data_type() {
            DataType::Float32 => read_and_convert!(f32),
            DataType::Float64 => read_and_convert!(f64),
            DataType::Int16 => read_and_convert!(i16),
            DataType::Int32 => read_and_convert!(i32),
            DataType::Int64 => read_and_convert!(i64),
            DataType::UInt8 => read_and_convert!(u8),
            DataType::UInt16 => read_and_convert!(u16),
            DataType::UInt32 => read_and_convert!(u32),
            other => Err(Error::UnsupportedDataType(
                format!("{:?}", other),
                varname.to_string(),
            )),
        }
    }

    /// Load a variable for this view's subset, padding with
    /// [`AsyncLazyElement::NAN_VALUE`] where the subset exceeds the source
    /// array boundaries.
    ///
    /// Delegates the actual I/O to [`retrieve_subset`](Self::retrieve_subset),
    /// which handles only the in-bounds portion.
    async fn load_variable(&self, varname: &str) -> Result<ArrayD<T>> {
        let array = Array::async_open(self.source.clone(), &format!("/{varname}"))
            .await
            .inspect_err(|err| error!("Failed to open variable '{}': {}", varname, err))?;

        let source_shape = array.shape().to_vec();
        let ndim = source_shape.len();
        let subset_start = self.subset.start();
        let subset_shape: Vec<u64> = self.subset.shape().to_vec();

        // Determine the intersection between the requested subset and the
        // source array, per dimension.
        let mut clipped_start: Vec<u64> = Vec::with_capacity(ndim);
        let mut clipped_shape: Vec<u64> = Vec::with_capacity(ndim);
        let mut fully_outside = false;

        for dim in 0..ndim {
            let req_start = subset_start[dim];
            let req_end = req_start + subset_shape[dim]; // exclusive
            let src_end = source_shape[dim]; // exclusive

            if req_start >= src_end || req_end == 0 {
                fully_outside = true;
                break;
            }

            let eff_start = req_start; // already >= 0 since u64
            let eff_end = req_end.min(src_end);

            clipped_start.push(eff_start);
            clipped_shape.push(eff_end - eff_start);
        }

        // Fully outside: return an array of sentinels immediately.
        if fully_outside {
            let out_shape: Vec<usize> = subset_shape.iter().map(|&d| d as usize).collect();
            trace!(
                "Subset for \"{}\" is fully outside source bounds; returning all-NaN",
                varname
            );
            return Ok(ArrayD::<T>::from_elem(IxDyn(&out_shape), T::NAN_VALUE));
        }

        // Fast path: the requested subset fits entirely inside the source.
        let needs_padding = clipped_shape != subset_shape;

        if !needs_padding {
            trace!("No padding needed for \"{}\"", varname);
            return self.retrieve_subset(&array, varname, &self.subset).await;
        }

        // Partial overlap: load only the valid region, then embed it into
        // a sentinel-filled output array.
        trace!(
            "Padding needed for \"{}\": clipped {:?} vs requested {:?}",
            varname, clipped_shape, subset_shape
        );

        let clipped_subset =
            ArraySubset::new_with_start_shape(clipped_start, clipped_shape.clone()).map_err(
                |err| {
                    Error::IO(std::io::Error::other(format!(
                        "Failed to build clipped subset for '{}': {}",
                        varname, err
                    )))
                },
            )?;

        let loaded = self
            .retrieve_subset(&array, varname, &clipped_subset)
            .await?;

        // Build the output, pre-filled with the sentinel.
        let out_shape: Vec<usize> = subset_shape.iter().map(|&d| d as usize).collect();
        let mut output = ArrayD::<T>::from_elem(IxDyn(&out_shape), T::NAN_VALUE);

        // Place loaded data into the top-left corner of the output
        // (offset is always [0, …, 0] since ArraySubset start is u64).
        let slice_info: Vec<SliceInfoElem> = clipped_shape
            .iter()
            .map(|&len| SliceInfoElem::Slice {
                start: 0,
                end: Some(len as isize),
                step: 1,
            })
            .collect();

        output
            .slice_mut(slice_info.as_slice())
            .assign(&loaded.view());

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::features::samples::{FeaturesTestBuilder, LayerConfig};

    /// Basic: loads correct shape and values for an in-bounds subset.
    #[tokio::test]
    async fn get_returns_correct_shape_and_values() {
        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::constant("A", 3.0))
            .build()
            .unwrap();

        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![4, 4]).unwrap();
        let lazy = AsyncLazySubset::<f32>::new(Arc::clone(&storage), subset);

        let data = lazy.get("A").await.unwrap();
        assert_eq!(data.shape(), &[4, 4]);
        assert!(data.iter().all(|&v| v == 3.0));
    }

    /// Requesting the same variable twice returns identical data (cache hit).
    #[tokio::test]
    async fn get_same_variable_twice_is_identical() {
        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::sequential("A"))
            .build()
            .unwrap();

        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![4, 4]).unwrap();
        let lazy = AsyncLazySubset::<f32>::new(Arc::clone(&storage), subset);

        let first = lazy.get("A").await.unwrap();
        let second = lazy.get("A").await.unwrap();
        assert_eq!(first, second);
    }

    /// Requesting a variable that does not exist returns VariableNotFound.
    #[tokio::test]
    async fn get_missing_variable_returns_error() {
        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::ones("A"))
            .build()
            .unwrap();

        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![4, 4]).unwrap();
        let lazy = AsyncLazySubset::<f32>::new(Arc::clone(&storage), subset);

        let result = lazy.get("NONEXISTENT").await;
        assert!(result.is_err(), "Expected an error for missing variable");
    }

    /// Subset that extends beyond the source array is padded with NaN.
    #[tokio::test]
    async fn get_pads_out_of_bounds_with_nan_f32() {
        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::constant("A", 7.0))
            .build()
            .unwrap();

        // Request a 6×6 region from a 4×4 array.
        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![6, 6]).unwrap();
        let lazy = AsyncLazySubset::<f32>::new(Arc::clone(&storage), subset);

        let data = lazy.get("A").await.unwrap();
        assert_eq!(data.shape(), &[6, 6]);

        // In-bounds portion should be 7.0
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(data[[i, j]], 7.0, "in-bounds cell [{i},{j}]");
            }
        }
        // Out-of-bounds portion should be NaN
        for i in 4..6 {
            for j in 0..6 {
                assert!(
                    data[[i, j]].is_nan(),
                    "out-of-bounds cell [{i},{j}] should be NaN, got {}",
                    data[[i, j]]
                );
            }
        }
        for i in 0..4 {
            for j in 4..6 {
                assert!(
                    data[[i, j]].is_nan(),
                    "out-of-bounds cell [{i},{j}] should be NaN, got {}",
                    data[[i, j]]
                );
            }
        }
    }

    /// f64 variant works.
    #[tokio::test]
    async fn get_f64_works() {
        use super::super::samples::FeatureDataType;

        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::constant("elev", 1.5).with_dtype(FeatureDataType::Float64))
            .build()
            .unwrap();

        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![4, 4]).unwrap();
        let lazy = AsyncLazySubset::<f64>::new(Arc::clone(&storage), subset);

        let data = lazy.get("elev").await.unwrap();
        assert_eq!(data.shape(), &[4, 4]);
        // f32 → f64 round-trip tolerance
        assert!(data.iter().all(|&v| (v - 1.5).abs() < 1e-6));
    }

    /// Subset completely outside the source returns all-NaN / sentinel.
    #[tokio::test]
    async fn fully_outside_returns_all_nan() {
        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::ones("A"))
            .build()
            .unwrap();

        // Start at row 10 — entirely outside a 4×4 array.
        let subset = ArraySubset::new_with_start_shape(vec![10, 10], vec![3, 3]).unwrap();
        let lazy = AsyncLazySubset::<f32>::new(Arc::clone(&storage), subset);

        let data = lazy.get("A").await.unwrap();
        assert_eq!(data.shape(), &[3, 3]);
        assert!(data.iter().all(|v| v.is_nan()));
    }

    /// Two concurrent tasks sharing a source both get correct results.
    #[tokio::test]
    async fn concurrent_get_different_variables() {
        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::constant("A", 1.0))
            .layer(LayerConfig::constant("B", 2.0))
            .build()
            .unwrap();

        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![4, 4]).unwrap();
        let lazy = AsyncLazySubset::<f32>::new(Arc::clone(&storage), subset);

        let (ra, rb) = tokio::join!(lazy.get("A"), lazy.get("B"));

        assert!(ra.unwrap().iter().all(|&v| v == 1.0));
        assert!(rb.unwrap().iter().all(|&v| v == 2.0));
    }

    /// Parallel tasks all get byte-identical results.
    #[tokio::test(flavor = "multi_thread")]
    async fn parallel_loads_match() {
        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::sequential("A"))
            .build()
            .unwrap();

        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![4, 4]).unwrap();

        // Sequential reference
        let reference = {
            let lazy = AsyncLazySubset::<f32>::new(Arc::clone(&storage), subset.clone());
            lazy.get("A").await.unwrap()
        };

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let src = Arc::clone(&storage);
                let sub = subset.clone();
                tokio::spawn(async move {
                    let lazy = AsyncLazySubset::<f32>::new(src, sub);
                    lazy.get("A").await
                })
            })
            .collect();

        for h in handles {
            let data = h.await.expect("task panicked").unwrap();
            assert_eq!(data, reference);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn shared_instance_across_tasks() {
        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::sequential("A"))
            .build()
            .unwrap();

        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![4, 4]).unwrap();
        let lazy = Arc::new(AsyncLazySubset::<f32>::new(Arc::clone(&storage), subset));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let lazy = Arc::clone(&lazy);
                tokio::spawn(async move { lazy.get("A").await })
            })
            .collect();

        let mut results = vec![];
        for h in handles {
            results.push(h.await.expect("task panicked").unwrap());
        }

        let first = &results[0];
        for result in &results[1..] {
            assert_eq!(result, first);
        }
    }

    /// Validates that variables stored with different Zarr data types (f32, f64,
    /// i16, i32, u8, u32) are all converted to the subset's element type (f32)
    /// when accessed through `AsyncLazySubset::get`.
    #[tokio::test]
    async fn get_converts_mixed_dtypes_to_common_type() {
        use super::super::samples::FeatureDataType;

        let (_tmp, storage) = FeaturesTestBuilder::new()
            .dimensions(2, 3)
            .chunks(2, 3)
            .layer(LayerConfig::sequential("from_f32"))
            .layer(LayerConfig::sequential("from_f64").with_dtype(FeatureDataType::Float64))
            .layer(LayerConfig::sequential("from_i16").with_dtype(FeatureDataType::Int16))
            .layer(LayerConfig::sequential("from_i32").with_dtype(FeatureDataType::Int32))
            .layer(LayerConfig::sequential("from_u8").with_dtype(FeatureDataType::UInt8))
            .layer(LayerConfig::sequential("from_u32").with_dtype(FeatureDataType::UInt32))
            .build()
            .unwrap();

        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![2, 3]).unwrap();
        let lazy = AsyncLazySubset::<f32>::new(Arc::clone(&storage), subset);

        let expected: Vec<f32> = (1..=6).map(|x| x as f32).collect();

        for varname in [
            "from_f32", "from_f64", "from_i16", "from_i32", "from_u8", "from_u32",
        ] {
            let data = lazy.get(varname).await.unwrap();
            assert_eq!(data.shape(), &[2, 3], "wrong shape for {varname}");
            let flat: Vec<f32> = data.iter().copied().collect();
            assert_eq!(flat, expected, "wrong values for {varname}");
        }
    }
}
