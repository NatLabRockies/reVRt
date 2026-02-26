//! Input features used by the cost function
//!
//! Support for asynchronous reading of features from a Zarr store
//! to be used by the cost function.

#[cfg(test)]
mod samples;

use std::sync::Arc;

use object_store::local::LocalFileSystem;
use tracing::debug;
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::AsyncObjectStore;

use crate::error::Result;

/// Input features used by the cost function.
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
}

#[cfg(test)]
mod test {
    use super::samples::{FeaturesTestBuilder, LayerConfig};
    use super::*;

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
