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

    #[allow(dead_code, missing_docs)]
    pub fn route(&self) -> &Vec<I> {
        &self.route
    }

    #[allow(dead_code, missing_docs)]
    pub fn total_cost(&self) -> &C {
        &self.total_cost
    }
}

pub type RevrtRoutingSolutions = Vec<Solution<crate::ArrayIndex, f32>>;
