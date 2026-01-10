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
        let path = samples::multi_variable_zarr();
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
    data: Arc<
        RwLock<
            HashMap<
                String,
                ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>>,
            >,
        >,
    >,
}
