//! Input features used by the cost function
//!
//! Provides asynchronous access to all features used to calculate the
//! cost of a path. Features provides read-only support and is intended
//! to be shared among multiple route searches. It is async to allow
//! multiple routes to move concurrently while minimizing the impact
//! of waiting on reading data.
//!
//! We currently only support Zarr store, but it might have use to
//! expand for other storage types in the future.
//!
//! We also provide the Lazy Subset resource to minimize redundant
//! I/O. See the module documentation for more details.

#[cfg(test)]
mod samples;

mod lazy_subset;

use std::sync::Arc;

use object_store::local::LocalFileSystem;
use tracing::debug;
use zarrs::array_subset::ArraySubset;
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::AsyncObjectStore;

use crate::error::Result;
use lazy_subset::AsyncLazySubset;

/// Input features used by the cost function.
///
/// Provides access to all input features used by the cost function.
/// This is intended to be shared among multiple routes, so it is
/// read-only and async.
///
/// Currently works with Zarrs storage only.
pub(super) struct Features {
    #[allow(dead_code)]
    /// Async readable storage holding the features.
    storage: AsyncReadableListableStorage,
}

impl Features {
    pub(super) fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        debug!("Opening features at {:?}", path.as_ref());

        let store = LocalFileSystem::new_with_prefix(path).unwrap();
        let storage: AsyncReadableListableStorage = Arc::new(AsyncObjectStore::new(store));

        Ok(Self { storage })
    }

    /// Creates an AsyncLazySubset of Features
    ///
    /// Intended to support efficient access of [`Features`] such
    /// as when calculating cost functions.
    pub(super) async fn lazy_subset(&self, subset: ArraySubset) -> AsyncLazySubset<f32> {
        AsyncLazySubset::<f32>::new(Arc::clone(&self.storage), subset)
    }
}

#[cfg(test)]
mod test {
    use super::samples::{FeaturesTestBuilder, LayerConfig};
    use super::*;

    /// Verify that `Features::lazy_subset()` produces an `AsyncLazySubset`
    /// that correctly loads data from the underlying store. Tests both a
    /// constant-filled layer and a sequentially-filled layer to confirm
    /// that the subset reads from the right variable and spatial region.
    #[tokio::test]
    async fn lazy_subset_returns_correct_data() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(8, 8)
            .chunks(4, 4)
            .layer(LayerConfig::constant("A", 5.0))
            .layer(LayerConfig::sequential("B"))
            .build()
            .unwrap();

        let features = Features::open(tmp.path()).unwrap();
        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![4, 4]).unwrap();
        let lazy = features.lazy_subset(subset).await;

        let a = lazy.get("A").await.unwrap();
        assert_eq!(a.shape(), &[4, 4]);
        assert!(a.iter().all(|&v| v == 5.0));

        let b = lazy.get("B").await.unwrap();
        assert_eq!(b.shape(), &[4, 4]);
        assert_eq!(b[[0, 0]], 1.0);
        assert_eq!(b[[0, 3]], 4.0);
    }

    /// Ensure that when `Features::lazy_subset()` is given a region that
    /// extends beyond the source array boundaries, out-of-bounds cells are
    /// filled with NaN while in-bounds cells retain their original values.
    /// This confirms that the padding logic in `AsyncLazySubset` is
    /// correctly wired through the `Features` API.
    #[tokio::test]
    async fn lazy_subset_pads_out_of_bounds() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::ones("A"))
            .build()
            .unwrap();

        let features = Features::open(tmp.path()).unwrap();
        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![6, 6]).unwrap();
        let lazy = features.lazy_subset(subset).await;

        let data = lazy.get("A").await.unwrap();
        assert_eq!(data.shape(), &[6, 6]);
        assert_eq!(data[[0, 0]], 1.0);
        assert!(data[[5, 5]].is_nan());
    }

    #[tokio::test]
    async fn dev() {
        let (tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(8, 8)
            .chunks(4, 4)
            .layer(LayerConfig::random("A", 0.0, 1.0))
            .build()
            .unwrap();
        let features = Features::open(tmp.path()).unwrap();
        let array = zarrs::array::Array::async_open(features.storage, "/A")
            .await
            .unwrap();
        let _data = array
            .async_retrieve_chunk_elements::<f32>(&[0, 0])
            .await
            .unwrap();
    }
}
