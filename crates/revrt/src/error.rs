/// Possible errors

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    IO(#[from] std::io::Error),

    #[error(transparent)]
    ObjectStore(#[from] object_store::Error),

    #[error(transparent)]
    ZarrsArrayCreate(#[from] zarrs::array::ArrayCreateError),

    #[error(transparent)]
    ZarrsGroupCreate(#[from] zarrs::group::GroupCreateError),

    #[error(transparent)]
    ZarrsArray(#[from] zarrs::array::ArrayError),

    #[error(transparent)]
    ZarrsStorage(#[from] zarrs::storage::StorageError),

    /// Tried to read data, such as input features, with unsupported
    /// data type.
    #[error("Unsupported data type '{0}' for variable '{1}'")]
    UnsupportedDataType(String, String),

    // #[error("All route end points are invalid: {0}")]
    // InvalidRouteEnd(String),

    // #[error("All route start points are invalid: {0}")]
    // InvalidRouteStart(String),
    #[allow(dead_code)]
    #[error("Undefined error")]
    // Used during development while it is not clear a category of error
    // or when it is not worth to create a new error type.
    /// Undefined error
    Undefined(String),
}

pub(crate) type Result<T> = core::result::Result<T, Error>;
