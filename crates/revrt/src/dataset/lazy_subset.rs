//! Lazy load a subset of the source Dataset
//!
//! This was originally developed to support the cost calculation, where the
//! variables that will be used are not known until the cost is actually
//! computed, and the same variable may be used multiple times. Thus the goal
//! is to load each variable only once, don't load unnecessary variables.
//!
//! The subset is fixed at the time of creation, so all variables are
//! consistent for the same domain.
//!
//! A `LazySubset` is tied to an `ArraySubset`, thus it has no assumptions
//! on the source's chunk. Therefore, the source can have variable chunk shapes,
//! one for each variable, and don't need to match the desired cost chunk shape.
//!
//! Note that we could have used Zarrs' intrinsic cache here, but a common
//! use for LazySubset is to load the features to compute cost for a chunk.
//! Therefore, those chunks of features are loaded only once and we don't
//! expect to use that anymore since we save the resulted cost. Using Zarrs'
//! cache would lead to unnecessary memory usage. Another problem is how
//! large should be that cache? It gets more difficult to estimate once we
//! consider the possibility of multiple threads working on different chunks.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::trace;
use zarrs::array::{Array, DataType, ElementOwned};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::AsyncReadableListableStorage;
use zarrs::storage::{ReadableListableStorage, ReadableListableStorageTraits};

use crate::error::{Error, Result};

/// Lazy loaded subset of a Zarr Dataset.
///
/// This struct is intended to work as a cache for a subset of a Zarr
/// Dataset.
pub(crate) struct LazySubset<T> {
    /// Source Zarr storage
    source: ReadableListableStorage,
    /// Subset of the source to be lazily loaded
    subset: ArraySubset,
    /// Data
    data: HashMap<
        String,
        ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>>,
    >,
}

impl<T> fmt::Display for LazySubset<T> {
    /// Display a LazySubset.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Add information on the source and the data HashMap.
        write!(f, "LazySubset {{ subset: {:?}, ... }}", self.subset,)
    }
}
impl<T: ElementOwned> LazySubset<T> {
    /// Create a new LazySubset for a given source and subset.
    pub(super) fn new(source: ReadableListableStorage, subset: ArraySubset) -> Self {
        trace!("Creating LazySubset for subset: {:?}", subset);

        LazySubset {
            source,
            subset,
            data: HashMap::new(),
        }
    }

    /// Show the subset used by this LazySubset.
    pub(crate) fn subset(&self) -> &ArraySubset {
        &self.subset
    }
}

impl LazySubset<f32> {
    /// Get a data for a specific variable.
    pub(crate) fn get(
        &mut self,
        varname: &str,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>> {
        trace!("Getting data subset for variable: {}", varname);

        let data = match self.data.get(varname) {
            Some(v) => {
                trace!("Data for variable {} already loaded", varname);
                v.clone()
            }
            None => {
                trace!(
                    "Loading data subset ({:?}) for variable: {}",
                    self.subset, varname
                );

                let variable =
                    Array::open(self.source.clone(), &format!("/{varname}")).map_err(|err| {
                        Error::IO(std::io::Error::other(format!(
                            "Failed to open layer '{varname}': {err}"
                        )))
                    })?;

                let values = self.load_as_f32(&variable, varname)?;

                self.data.insert(varname.to_string(), values.clone());

                values
            }
        };

        Ok(data)
    }

    fn load_as_f32<TStorage: ?Sized + ReadableListableStorageTraits + 'static>(
        &self,
        variable: &Array<TStorage>,
        varname: &str,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>> {
        let dtype = variable.data_type();

        match dtype {
            DataType::Float32 => {
                self.retrieve_and_convert::<f32, TStorage, _>(variable, varname, |v| v)
            }
            DataType::Float64 => {
                self.retrieve_and_convert::<f64, TStorage, _>(variable, varname, |v| v as f32)
            }
            DataType::Int8 => {
                self.retrieve_and_convert::<i8, TStorage, _>(variable, varname, |v| v as f32)
            }
            DataType::Int16 => {
                self.retrieve_and_convert::<i16, TStorage, _>(variable, varname, |v| v as f32)
            }
            DataType::Int32 => {
                self.retrieve_and_convert::<i32, TStorage, _>(variable, varname, |v| v as f32)
            }
            DataType::Int64 => {
                self.retrieve_and_convert::<i64, TStorage, _>(variable, varname, |v| v as f32)
            }
            DataType::UInt8 => {
                self.retrieve_and_convert::<u8, TStorage, _>(variable, varname, |v| v as f32)
            }
            DataType::UInt16 => {
                self.retrieve_and_convert::<u16, TStorage, _>(variable, varname, |v| v as f32)
            }
            DataType::UInt32 => {
                self.retrieve_and_convert::<u32, TStorage, _>(variable, varname, |v| v as f32)
            }
            DataType::UInt64 => {
                self.retrieve_and_convert::<u64, TStorage, _>(variable, varname, |v| v as f32)
            }
            other => Err(Error::IO(std::io::Error::other(format!(
                "Unsupported data type {:?} for layer '{varname}'",
                other
            )))),
        }
    }

    fn retrieve_and_convert<T, TStorage, F>(
        &self,
        variable: &Array<TStorage>,
        varname: &str,
        converter: F,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>>
    where
        T: ElementOwned + Clone,
        TStorage: ?Sized + ReadableListableStorageTraits + 'static,
        F: Fn(T) -> f32 + Copy,
    {
        let raw = variable
            .retrieve_array_subset_ndarray::<T>(&self.subset)
            .map_err(|err| {
                Error::IO(std::io::Error::other(format!(
                    "Failed to retrieve array subset for layer '{varname}': {err}"
                )))
            })?;
        Ok(raw.mapv(converter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::samples;
    use std::sync::Arc;
    // use zarrs::storage::store::MemoryStore;
    use zarrs::storage::ReadableListableStorage;

    #[test]
    fn sample() {
        let path = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A", "B", "C", "cost"]);
        let store: ReadableListableStorage =
            Arc::new(zarrs::filesystem::FilesystemStore::new(&path).unwrap());

        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 2, 2]).unwrap();
        let mut dataset = LazySubset::<f32>::new(store, subset);
        let tmp = dataset.get("A").unwrap();
        assert_eq!(tmp.shape(), &[1, 2, 2]);
    }

    /*
    #[test]
    fn test_lazy_dataset() {
        let storage = MemoryStore::new();
        let subset = ArraySubset::default();
        let mut lazy_dataset = LazySubset::<f32>::new(Arc::new(storage), subset);

        if let Some(data) = lazy_dataset.get("test_var") {
            assert!(!data.is_empty());
        } else {
            panic!("Failed to retrieve data for 'test_var'");
        }
    }
    */
}

#[allow(dead_code)]
/// Trait defining types that can be used as LazySubset element types
trait LazySubsetElement: ElementOwned + Clone + Send + Sync {
    /// Convert from f32
    fn from_f32(value: f32) -> Self;
    /// Convert from f64
    fn from_f64(value: f64) -> Self;
}

impl LazySubsetElement for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }
    // A lossy cast.
    // The value is rounded, if needed, and overflow results in infinity.
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl LazySubsetElement for f64 {
    fn from_f32(value: f32) -> Self {
        value as f64
    }
    fn from_f64(value: f64) -> Self {
        value
    }
}

#[allow(dead_code)]
/// Asynchronous lazy loaded subset of a Zarr Dataset.
///
/// Work as an async cache for a consistent subset (same indices range) for
/// multiple variables of a Zarr Dataset.
// pub struct AsyncLazySubset<T: LazySubsetElement> {
struct AsyncLazySubset<T: LazySubsetElement> {
    /// Source Zarr storage
    source: AsyncReadableListableStorage,
    /// Subset of the source to be lazily loaded
    subset: ArraySubset,
    /// Cached data with RwLock for concurrent access
    // Eventually refactor this into a new type: cache
    #[allow(clippy::type_complexity)]
    data: Arc<
        RwLock<
            HashMap<
                String,
                ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>>,
            >,
        >,
    >,
}

impl<T: LazySubsetElement> fmt::Display for AsyncLazySubset<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "AsyncLazySubset {{ subset: {:?}, ... }}", self.subset)
    }
}

impl<T: LazySubsetElement> fmt::Debug for AsyncLazySubset<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("AsyncLazySubset")
            .field("subset", &self.subset)
            .field("element_type", &std::any::type_name::<T>())
            .finish()
    }
}

#[allow(dead_code)]
impl<T: LazySubsetElement> AsyncLazySubset<T> {
    /// Create a new AsyncLazySubset for a given source and subset.
    ///
    /// # Arguments
    /// * `source` - Async readable Zarr storage
    /// * `subset` - The array subset to load
    ///
    /// # Returns
    /// A new AsyncLazySubset instance with an empty cache
    fn new(source: AsyncReadableListableStorage, subset: ArraySubset) -> Self {
        trace!("Creating AsyncLazySubset for subset: {:?}", subset);

        AsyncLazySubset {
            source,
            subset,
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the subset used by this AsyncLazySubset.
    fn subset(&self) -> &ArraySubset {
        &self.subset
    }

    /// Get data for a specific variable asynchronously.
    ///
    /// This method will check the cache first. If the variable is not cached,
    /// it will load the data from storage, convert it to the target type,
    /// cache it, and return it.
    ///
    /// # Arguments
    /// * `varname` - Name of the variable to load
    ///
    /// # Returns
    /// The array data as an ndarray with the target element type
    ///
    /// # Errors
    /// Returns an error if the variable cannot be opened or loaded
    async fn get(
        &self,
        varname: &str,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>>> {
        trace!("Getting data subset for variable: {}", varname);

        // Check if already cached (read lock)
        {
            let data_read = self.data.read().await;
            if let Some(cached) = data_read.get(varname) {
                trace!("Data for variable {} already loaded", varname);
                return Ok(cached.clone());
            }
        }

        // Not cached, need to load (write lock)
        trace!(
            "Loading data subset ({:?}) for variable: {}",
            self.subset, varname
        );

        let variable = Array::async_open(self.source.clone(), &format!("/{varname}")).await?;

        let values = variable
            .async_retrieve_array_subset_ndarray(&self.subset)
            .await
            .expect("Failed to retrieve array subset");

        // Cache the loaded data
        let mut data_write = self.data.write().await;
        data_write.insert(varname.to_string(), values.clone());

        Ok(values)
    }
}

#[cfg(test)]
mod tests_async {
    use super::*;
    use crate::dataset::samples::{self, LayerConfig, ZarrTestBuilder};

    // -------------------------------------------------------------------------
    // Baseline
    // -------------------------------------------------------------------------

    /// A single instance loads the correct shape and values.
    #[tokio::test]
    async fn get_returns_correct_shape_and_values() {
        let tmp = ZarrTestBuilder::new()
            .dimensions(1, 4, 4)
            .chunks(1, 2, 2)
            .layer(LayerConfig::constant("A", 3.0))
            .build()
            .unwrap();
        let source = samples::async_storage_for(tmp.path());

        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 4, 4]).unwrap();
        let lazy = AsyncLazySubset::<f32>::new(Arc::clone(&source), subset);

        let data = lazy.get("A").await.unwrap();
        assert_eq!(data.shape(), &[1, 4, 4]);
        assert!(data.iter().all(|&v| v == 3.0));
    }

    /// Requesting the same variable twice returns identical data (cache
    /// must not corrupt on repeated access).
    #[tokio::test]
    async fn get_same_variable_twice_is_identical() {
        let tmp = ZarrTestBuilder::new()
            .dimensions(1, 4, 4)
            .chunks(1, 2, 2)
            .layer(LayerConfig::sequential("A", 1))
            .build()
            .unwrap();
        let source = samples::async_storage_for(tmp.path());

        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 4, 4]).unwrap();
        let lazy = AsyncLazySubset::<f32>::new(Arc::clone(&source), subset);

        let first = lazy.get("A").await.unwrap();
        let second = lazy.get("A").await.unwrap();
        assert_eq!(first, second);
    }

    // -------------------------------------------------------------------------
    // Concurrent access via tokio::join!
    //
    // tokio::join! interleaves both futures on the same task. This is the
    // minimal concurrency test: it exercises shared-Arc access without
    // spawning extra threads.
    // -------------------------------------------------------------------------

    /// Two instances sharing one Arc, each loading a different layer,
    /// run concurrently and return the correct values for their layer.
    #[tokio::test]
    async fn two_instances_shared_source_join() {
        let tmp = ZarrTestBuilder::new()
            .dimensions(1, 4, 4)
            .chunks(1, 2, 2)
            .layer(LayerConfig::constant("A", 1.0))
            .layer(LayerConfig::constant("B", 2.0))
            .build()
            .unwrap();
        let source = samples::async_storage_for(tmp.path());

        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 4, 4]).unwrap();
        let lazy_a = AsyncLazySubset::<f32>::new(Arc::clone(&source), subset.clone());
        let lazy_b = AsyncLazySubset::<f32>::new(Arc::clone(&source), subset);

        let (result_a, result_b) = tokio::join!(lazy_a.get("A"), lazy_b.get("B"));

        assert!(result_a.unwrap().iter().all(|&v| v == 1.0));
        assert!(result_b.unwrap().iter().all(|&v| v == 2.0));
    }

    /// Two instances loading the same layer concurrently must return
    /// byte-identical data — no torn reads or cache interference.
    #[tokio::test]
    async fn two_instances_same_layer_join_are_identical() {
        let tmp = ZarrTestBuilder::new()
            .dimensions(1, 4, 4)
            .chunks(1, 2, 2)
            .layer(LayerConfig::sequential("A", 1))
            .build()
            .unwrap();
        let source = samples::async_storage_for(tmp.path());

        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 4, 4]).unwrap();
        let lazy_a = AsyncLazySubset::<f32>::new(Arc::clone(&source), subset.clone());
        let lazy_b = AsyncLazySubset::<f32>::new(Arc::clone(&source), subset);

        let (result_a, result_b) = tokio::join!(lazy_a.get("A"), lazy_b.get("A"));

        assert_eq!(result_a.unwrap(), result_b.unwrap());
    }

    // -------------------------------------------------------------------------
    // Parallel access via tokio::spawn (multi-thread runtime)
    //
    // Each spawned task runs on a separate thread-pool thread, so all tasks
    // hit the shared Arc simultaneously. This is the strongest test: it
    // catches data races in the storage implementation itself.
    // -------------------------------------------------------------------------

    /// Four tasks, each owning its own AsyncLazySubset but all sharing one
    /// Arc, run in parallel and each returns the correct constant value.
    #[tokio::test(flavor = "multi_thread")]
    async fn many_instances_parallel_spawn_shared_source() {
        let tmp = ZarrTestBuilder::new()
            .dimensions(1, 4, 4)
            .chunks(1, 2, 2)
            .layer(LayerConfig::constant("A", 5.0))
            .build()
            .unwrap();
        let source = samples::async_storage_for(tmp.path());

        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 4, 4]).unwrap();

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let source = Arc::clone(&source);
                let subset = subset.clone();
                tokio::spawn(
                    async move { AsyncLazySubset::<f32>::new(source, subset).get("A").await },
                )
            })
            .collect();

        for handle in handles {
            let data = handle.await.expect("task panicked").unwrap();
            assert_eq!(data.shape(), &[1, 4, 4]);
            assert!(data.iter().all(|&v| v == 5.0));
        }
    }

    /// Sequential reference load followed by parallel loads: all parallel
    /// results must be byte-identical to the reference. This rules out
    /// data races that produce wrong values without panicking.
    #[tokio::test(flavor = "multi_thread")]
    async fn parallel_results_match_sequential_reference() {
        let tmp = ZarrTestBuilder::new()
            .dimensions(1, 4, 4)
            .chunks(1, 2, 2)
            .layer(LayerConfig::sequential("A", 1))
            .build()
            .unwrap();
        let source = samples::async_storage_for(tmp.path());

        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 4, 4]).unwrap();

        // Sequential reference.
        let reference = AsyncLazySubset::<f32>::new(Arc::clone(&source), subset.clone())
            .get("A")
            .await
            .unwrap();

        // Four parallel tasks loading the same data.
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let source = Arc::clone(&source);
                let subset = subset.clone();
                tokio::spawn(
                    async move { AsyncLazySubset::<f32>::new(source, subset).get("A").await },
                )
            })
            .collect();

        for handle in handles {
            let data = handle.await.expect("task panicked").unwrap();
            assert_eq!(data, reference);
        }
    }
}
