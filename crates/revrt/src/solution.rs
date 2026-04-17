//! Routing solution representation

#[derive(Debug)]
/// Routing result for a single solved case
///
/// Stores the ordered route indices, the aggregate route cost, and any
/// barrier layers that had to be dropped before a valid path could be found.
/// The generic type `I` represents the index type used for route entries,
/// while `C` represents the accumulated cost type.
pub struct Solution<I, C> {
    /// Route indices for the solved path.
    pub(crate) route: Vec<I>,
    /// Aggregate cost of the solved path.
    pub(crate) total_cost: C,
    /// Barrier layers dropped to obtain a valid route.
    /// Will be empty if no barriers were dropped, or if the scenario had no barriers.
    pub(crate) dropped_barrier_layers: Vec<String>,
}

impl<I, C> Solution<I, C> {
    /// Create a solution from a route and its total cost.
    pub(crate) fn new(route: Vec<I>, total_cost: C) -> Self {
        Self {
            route,
            total_cost,
            dropped_barrier_layers: Vec::new(),
        }
    }

    /// Record the barrier layers that were dropped for this solution.
    pub(crate) fn record_dropped_barriers(mut self, dropped_barrier_layers: Vec<String>) -> Self {
        self.dropped_barrier_layers = dropped_barrier_layers;
        self
    }

    #[cfg(any(test, feature = "test-integration"))]
    /// Borrow the route indices for this solution.
    pub fn route(&self) -> &Vec<I> {
        &self.route
    }

    #[cfg(any(test, feature = "test-integration"))]
    /// Borrow the total cost for this solution.
    pub fn total_cost(&self) -> &C {
        &self.total_cost
    }

    #[cfg(any(test, feature = "test-integration"))]
    /// Borrow the names of dropped barrier layers for this solution.
    pub fn dropped_barrier_layers(&self) -> &Vec<String> {
        &self.dropped_barrier_layers
    }
}

/// Collection of routing solutions returned by reVRt.
pub type RevrtRoutingSolutions = Vec<Solution<crate::ArrayIndex, f32>>;
