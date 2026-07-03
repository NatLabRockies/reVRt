//! Performance-related APIs

#[cfg(feature = "profiling")]
#[path = "./profiling.rs"]
pub mod profiling;

#[cfg(not(feature = "profiling"))]
#[path = "./profiling_disabled.rs"]
pub mod profiling;
