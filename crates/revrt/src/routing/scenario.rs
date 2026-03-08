//! A scenario for routing
//!
//! A `Scenario` encapsulates the cost surface and everything required to
//! define that, such as the input features and the cost function.
//!
//! A typical scenario uses a cost function that weights several
//! features to determine the cost per grid point.
//!
//! Given the relatively high resolution and the desire on routing long
//! distances, the data involved can be relatively large. A major
//! accomplishment of this crate is in this module, by working in chunks
//! with asynchronous I/O we keep the memory footprint low while
//! sustaining high performance as possible.

use tracing::trace;

use super::{Features, cost_as_u64};
use crate::{ArrayIndex, Result};

pub(super) struct Scenario {
    pub dataset: crate::dataset::Dataset,
    #[allow(dead_code)]
    features: Features,
    #[allow(dead_code)]
    cost_function: crate::CostFunction,
}

impl Scenario {
    pub(super) fn new<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        cache_size: u64,
    ) -> Result<Self> {
        trace!("Opening scenario with: {:?}", store_path.as_ref());

        let features = Features::new(&store_path)?;
        let dataset = crate::dataset::Dataset::open(store_path, cost_function.clone(), cache_size)?;

        Ok(Self {
            dataset,
            features,
            cost_function,
        })
    }

    pub(super) fn get_3x3(&self, position: &ArrayIndex) -> Vec<(ArrayIndex, f32)> {
        self.dataset.get_3x3(position)
    }

    /// Determine the successors of a position.
    ///
    /// ToDo:
    /// - Handle the edges of the array.
    /// - Weight the cost. Remember that the cost is for a side,
    ///   thus a diagonal move has to calculate consider the longer
    ///   distance.
    /// - Add starting cell cost by adding a is_start parameter and
    ///   passing it down to the get_3x3 function so that it can add
    ///   the center pixel to all successor cost values
    pub(super) fn successors(&self, position: &ArrayIndex) -> Vec<(ArrayIndex, u64)> {
        let neighbors = self.get_3x3(position);
        let neighbors = neighbors
            .into_iter()
            .map(|(p, c)| (p, cost_as_u64(c))) // ToDo: Maybe it's better to have get_3x3 return a u64 - then we can skip this map altogether
            .collect();
        trace!("Adjusting neighbors' types: {:?}", neighbors);
        neighbors
    }

    pub(super) fn grid_shape(&self) -> (u64, u64) {
        self.dataset.grid_shape()
    }
}
