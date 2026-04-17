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

use std::collections::HashSet;
use std::path::PathBuf;

use tracing::trace;

use super::cost_as_u64;
use crate::routing::features::Features;
use crate::{ArrayIndex, Result};

pub(super) struct Scenario {
    pub dataset: crate::dataset::Dataset,
    #[allow(dead_code)]
    features: Features,
}

impl Scenario {
    pub(super) fn new_with_swap<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        cache_size: u64,
        swap_fp: PathBuf,
    ) -> Result<Self> {
        trace!("Opening scenario with: {:?}", store_path.as_ref());

        let features = Features::open(&store_path)?;
        let dataset = crate::dataset::Dataset::open_with_swap(
            store_path,
            cost_function,
            cache_size,
            swap_fp,
        )?;

        Ok(Self { dataset, features })
    }

    pub(super) fn new<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        cache_size: u64,
    ) -> Result<Self> {
        trace!("Opening scenario with: {:?}", store_path.as_ref());

        let features = Features::open(&store_path)?;
        let dataset = crate::dataset::Dataset::open(store_path, cost_function, cache_size)?;

        Ok(Self { dataset, features })
    }

    pub(super) fn get_3x3(&self, position: &ArrayIndex) -> Vec<(ArrayIndex, f32)> {
        self.dataset.get_3x3(position)
    }

    pub(super) fn successors_for_attempt(
        &self,
        position: &ArrayIndex,
        dropped_soft_groups: usize,
    ) -> Vec<(ArrayIndex, u64)> {
        let neighbors = self.get_3x3(position);
        let soft_barrier_cells: HashSet<_> = self
            .dataset
            .get_3x3_soft_barrier_cells(position, dropped_soft_groups)
            .into_iter()
            .collect();

        if soft_barrier_cells.contains(position) {
            return Vec::new();
        }

        let neighbors = neighbors
            .into_iter()
            .filter(|(p, c)| c.is_finite() && *c > 0.0 && !soft_barrier_cells.contains(p))
            .map(|(p, c)| (p, cost_as_u64(c)))
            .collect();
        trace!("Adjusting neighbors' types: {:?}", neighbors);
        neighbors
    }

    pub(super) fn soft_barrier_group_count(&self) -> usize {
        self.dataset.soft_barrier_groups().len()
    }

    pub(super) fn dropped_barrier_layers(&self, dropped_soft_groups: usize) -> Vec<String> {
        let mut dropped_barrier_layers = Vec::new();

        for (__, layers) in self
            .dataset
            .soft_barrier_groups()
            .iter()
            .take(dropped_soft_groups)
        {
            for layer in layers {
                dropped_barrier_layers.push(layer.layer_name().to_string());
            }
        }

        dropped_barrier_layers
    }

    pub(super) fn grid_shape(&self) -> (u64, u64) {
        self.dataset.grid_shape
    }
}

#[cfg(test)]
mod tests {
    use super::Scenario;
    use crate::ArrayIndex;

    #[test]
    fn successors_keep_hard_barriers_after_soft_groups_drop() {
        let store = crate::dataset::samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(crate::dataset::samples::LayerConfig::constant("cost", 1.0))
            .layer(crate::dataset::samples::LayerConfig::new(
                "hard_barrier",
                crate::dataset::samples::FillStrategy::Values(vec![
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]),
            ))
            .layer(crate::dataset::samples::LayerConfig::new(
                "soft_barrier",
                crate::dataset::samples::FillStrategy::Values(vec![
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]),
            ))
            .build()
            .unwrap();
        let cost_function = crate::cost::CostFunction::from_json(
            r#"{
                "cost_layers": [{"layer_name": "cost"}],
                "barrier_layers": [
                    {
                        "layer_name": "hard_barrier",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0
                    },
                    {
                        "layer_name": "soft_barrier",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0,
                        "barrier_importance": 1
                    }
                ],
                "ignore_invalid_costs": false
            }"#,
        )
        .unwrap();
        let scenario = Scenario::new(store.path(), cost_function, 1_000).unwrap();

        let start = ArrayIndex { i: 1, j: 1 };
        let initial_successors = scenario.successors_for_attempt(&start, 0);
        let relaxed_successors = scenario.successors_for_attempt(&start, 1);

        assert!(
            !initial_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex { i: 0, j: 1 })
        );
        assert!(
            !initial_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex { i: 1, j: 0 })
        );

        assert!(
            !relaxed_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex { i: 0, j: 1 })
        );
        assert!(
            relaxed_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex { i: 1, j: 0 })
        );
    }

    #[test]
    fn successors_return_empty_when_start_is_hard_barrier() {
        let store = crate::dataset::samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(crate::dataset::samples::LayerConfig::constant("cost", 1.0))
            .layer(crate::dataset::samples::LayerConfig::new(
                "hard_barrier",
                crate::dataset::samples::FillStrategy::Values(vec![
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                ]),
            ))
            .build()
            .unwrap();
        let cost_function = crate::cost::CostFunction::from_json(
            r#"{
                "cost_layers": [{"layer_name": "cost"}],
                "barrier_layers": [
                    {
                        "layer_name": "hard_barrier",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0
                    }
                ],
                "ignore_invalid_costs": false
            }"#,
        )
        .unwrap();
        let scenario = Scenario::new(store.path(), cost_function, 1_000).unwrap();

        let successors = scenario.successors_for_attempt(&ArrayIndex { i: 1, j: 1 }, 0);

        assert!(successors.is_empty());
    }

    #[test]
    fn successors_use_cumulative_soft_masks_by_retry_state() {
        let store = crate::dataset::samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(crate::dataset::samples::LayerConfig::constant("cost", 1.0))
            .layer(crate::dataset::samples::LayerConfig::new(
                "soft_barrier_low",
                crate::dataset::samples::FillStrategy::Values(vec![
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]),
            ))
            .layer(crate::dataset::samples::LayerConfig::new(
                "soft_barrier_high",
                crate::dataset::samples::FillStrategy::Values(vec![
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]),
            ))
            .build()
            .unwrap();
        let cost_function = crate::cost::CostFunction::from_json(
            r#"{
                "cost_layers": [{"layer_name": "cost"}],
                "barrier_layers": [
                    {
                        "layer_name": "soft_barrier_low",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0,
                        "barrier_importance": 1
                    },
                    {
                        "layer_name": "soft_barrier_high",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0,
                        "barrier_importance": 2
                    }
                ],
                "ignore_invalid_costs": false
            }"#,
        )
        .unwrap();
        let scenario = Scenario::new(store.path(), cost_function, 1_000).unwrap();

        let start = ArrayIndex { i: 1, j: 1 };
        let initial_successors = scenario.successors_for_attempt(&start, 0);
        let retry_one_successors = scenario.successors_for_attempt(&start, 1);
        let retry_two_successors = scenario.successors_for_attempt(&start, 2);

        assert_eq!(scenario.soft_barrier_group_count(), 2);
        assert!(
            !initial_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex { i: 1, j: 0 })
        );
        assert!(
            !initial_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex { i: 0, j: 1 })
        );
        assert!(
            retry_one_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex { i: 1, j: 0 })
        );
        assert!(
            !retry_one_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex { i: 0, j: 1 })
        );
        assert!(
            retry_two_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex { i: 1, j: 0 })
        );
        assert!(
            retry_two_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex { i: 0, j: 1 })
        );
    }

    #[test]
    fn successors_return_empty_when_start_is_in_active_soft_mask() {
        let store = crate::dataset::samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(crate::dataset::samples::LayerConfig::constant("cost", 1.0))
            .layer(crate::dataset::samples::LayerConfig::new(
                "soft_barrier",
                crate::dataset::samples::FillStrategy::Values(vec![
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                ]),
            ))
            .build()
            .unwrap();
        let cost_function = crate::cost::CostFunction::from_json(
            r#"{
                "cost_layers": [{"layer_name": "cost"}],
                "barrier_layers": [
                    {
                        "layer_name": "soft_barrier",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0,
                        "barrier_importance": 1
                    }
                ],
                "ignore_invalid_costs": false
            }"#,
        )
        .unwrap();
        let scenario = Scenario::new(store.path(), cost_function, 1_000).unwrap();

        assert!(
            scenario
                .successors_for_attempt(&ArrayIndex { i: 1, j: 1 }, 0)
                .is_empty()
        );
        assert!(
            !scenario
                .successors_for_attempt(&ArrayIndex { i: 1, j: 1 }, 1)
                .is_empty()
        );
    }

    #[test]
    fn dropped_barrier_layers_follows_soft_barrier_groups() {
        let store = crate::dataset::samples::ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(crate::dataset::samples::LayerConfig::constant("cost", 1.0))
            .layer(crate::dataset::samples::LayerConfig::new(
                "soft_barrier_low_a",
                crate::dataset::samples::FillStrategy::Constant(0.0),
            ))
            .layer(crate::dataset::samples::LayerConfig::new(
                "soft_barrier_low_b",
                crate::dataset::samples::FillStrategy::Constant(0.0),
            ))
            .layer(crate::dataset::samples::LayerConfig::new(
                "soft_barrier_high",
                crate::dataset::samples::FillStrategy::Constant(0.0),
            ))
            .build()
            .unwrap();
        let cost_function = crate::cost::CostFunction::from_json(
            r#"{
                "cost_layers": [{"layer_name": "cost"}],
                "barrier_layers": [
                    {
                        "layer_name": "soft_barrier_low_a",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0,
                        "barrier_importance": 1
                    },
                    {
                        "layer_name": "soft_barrier_low_b",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0,
                        "barrier_importance": 1
                    },
                    {
                        "layer_name": "soft_barrier_high",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0,
                        "barrier_importance": 2
                    }
                ],
                "ignore_invalid_costs": false
            }"#,
        )
        .unwrap();
        let scenario = Scenario::new(store.path(), cost_function, 1_000).unwrap();

        assert!(scenario.dropped_barrier_layers(0).is_empty());
        assert_eq!(
            scenario.dropped_barrier_layers(1),
            vec!["soft_barrier_low_a", "soft_barrier_low_b"]
        );
        assert_eq!(
            scenario.dropped_barrier_layers(2),
            vec![
                "soft_barrier_low_a",
                "soft_barrier_low_b",
                "soft_barrier_high"
            ]
        );
    }
}
