//! Input features used by the cost function
//!
//! Provides asynchronous access to all features used to calculate the
//! cost of a path. Features provides read-only support and is intended
//! to be shared among multiple route searches. It is async to allow
//! multiple routes to move concurrently while minimizing the impact
//! of waiting on reading data.
//!
//! We currently only support Zarr store, but we might expand for other
//! storage types in the future.
//!
//! We also provide the Lazy Subset resource to minimize redundant
//! I/O. See the module documentation for more details.
//!
//! Notes: In the future, migrate lazy_subset to return only valid
//! arrays and error if requested out of bounds. Then add a support
//! function that allows out of bounds and pad with NaNs if needed.

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
use lazy_subset::{AsyncLazyElement, AsyncLazySubset};

/// Input features used by the cost function.
///
/// Provides access to all input features used by the cost function.
/// This is intended to be shared among multiple routes, so it is
/// read-only and async.
///
/// Currently works with Zarrs storage only.
pub(super) struct Features {
    /// Async readable storage holding the features.
    #[allow(dead_code)]
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
    /// as for calculating cost functions based on multiple variabels.
    #[allow(dead_code)]
    pub(super) async fn lazy_subset<T: AsyncLazyElement>(
        &self,
        subset: ArraySubset,
    ) -> AsyncLazySubset<T> {
        AsyncLazySubset::<T>::new(Arc::clone(&self.storage), subset)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;
    use zarrs::array_subset::ArraySubset;

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
        let lazy = features.lazy_subset::<f32>(subset).await;

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
        let lazy = features.lazy_subset::<f32>(subset).await;

        let data = lazy.get("A").await.unwrap();
        assert_eq!(data.shape(), &[6, 6]);
        assert_eq!(data[[0, 0]], 1.0);
        assert!(data[[5, 5]].is_nan());
    }

    /// Validates that an `AsyncLazySubset` obtained from `Features::lazy_subset()`
    /// can be safely shared across multiple concurrent tokio tasks.
    ///
    /// Spawns 8 tasks that simultaneously load two different variables ("A" and "B")
    /// from the same `AsyncLazySubset` instance, verifying that every task receives
    /// the correct, identical result for its variable regardless of scheduling order.
    #[tokio::test(flavor = "multi_thread")]
    async fn features_lazy_subset_concurrent_access() {
        use super::samples::{FeaturesTestBuilder, LayerConfig};
        use std::sync::Arc;
        use zarrs::array_subset::ArraySubset;

        let (_tmp, _storage) = FeaturesTestBuilder::new()
            .dimensions(4, 4)
            .chunks(2, 2)
            .layer(LayerConfig::constant("A", 5.0))
            .layer(LayerConfig::sequential("B"))
            .build()
            .unwrap();

        let features = Features::open(_tmp.path()).unwrap();
        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![4, 4]).unwrap();
        let lazy = Arc::new(features.lazy_subset(subset).await);

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let lazy = Arc::clone(&lazy);
                let var = if i % 2 == 0 { "A" } else { "B" };
                let var_owned = var.to_string();
                tokio::spawn(async move {
                    let data = lazy.get(&var_owned).await;
                    (var_owned, data)
                })
            })
            .collect();

        let mut a_results = vec![];
        let mut b_results = vec![];
        for h in handles {
            let (var, data) = h.await.expect("task panicked");
            let data = data.unwrap();
            match var.as_str() {
                "A" => a_results.push(data),
                "B" => b_results.push(data),
                _ => panic!("unexpected variable {var}"),
            }
        }

        assert_eq!(a_results.len(), 4);
        assert_eq!(b_results.len(), 4);

        // All A results should be constant 5.0
        for r in &a_results {
            assert!(r.iter().all(|&v| v == 5.0));
        }
        for r in &a_results[1..] {
            assert_eq!(r, &a_results[0]);
        }

        // All B results should be sequential 1..=16
        let expected_b: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        for r in &b_results {
            let flat: Vec<f32> = r.iter().copied().collect();
            assert_eq!(flat, expected_b);
        }
        for r in &b_results[1..] {
            assert_eq!(r, &b_results[0]);
        }
    }
}
