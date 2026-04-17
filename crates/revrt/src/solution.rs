//! Routing solution representation

#[allow(missing_docs, dead_code)]
#[derive(Debug)]
/// Solution for one single routing case
///
pub struct Solution<I, C> {
    pub(crate) route: Vec<I>,
    pub(crate) total_cost: C,
    pub(crate) dropped_barrier_layers: Vec<String>,
}

impl<I, C> Solution<I, C> {
    #[allow(missing_docs)]
    pub(crate) fn new(route: Vec<I>, total_cost: C) -> Self {
        Self {
            route,
            total_cost,
            dropped_barrier_layers: Vec::new(),
        }
    }

    #[allow(missing_docs)]
    pub(crate) fn record_dropped_barriers(mut self, dropped_barrier_layers: Vec<String>) -> Self {
        self.dropped_barrier_layers = dropped_barrier_layers;
        self
    }

    #[cfg(any(test, feature = "test-integration"))]
    #[allow(missing_docs)]
    pub fn route(&self) -> &Vec<I> {
        &self.route
    }

    #[cfg(any(test, feature = "test-integration"))]
    #[allow(missing_docs)]
    pub fn total_cost(&self) -> &C {
        &self.total_cost
    }

    #[cfg(any(test, feature = "test-integration"))]
    #[allow(missing_docs)]
    pub fn dropped_barrier_layers(&self) -> &Vec<String> {
        &self.dropped_barrier_layers
    }
}

pub type RevrtRoutingSolutions = Vec<Solution<crate::ArrayIndex, f32>>;
