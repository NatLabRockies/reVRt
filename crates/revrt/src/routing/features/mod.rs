//! Input features used by the cost function
//!
//! Support for asynchronous reading of features from a Zarr store
//! to be used by the cost function.

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
    storage: zarrs::storage::AsyncReadableListableStorage,
}

impl Features {
    pub(super) fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        debug!("Opening features at {:?}", path.as_ref());

        let store = LocalFileSystem::new_with_prefix(path).unwrap();
        let storage: AsyncReadableListableStorage = Arc::new(AsyncObjectStore::new(store));

        Ok(Self { storage })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dataset::samples::multi_variable_zarr;

    #[tokio::test]
    async fn dev() {
        let path = multi_variable_zarr();
        let features = Features::new(&path).unwrap();
        let array = zarrs::array::Array::async_open(features.storage, "/A")
            .await
            .unwrap();
        let _data = array.async_retrieve_chunk(&[0, 0, 0]).await.unwrap();
    }
}
