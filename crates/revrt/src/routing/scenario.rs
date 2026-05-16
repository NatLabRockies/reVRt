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
use crate::cost::{DriverRuleSet, TransitionCostTable};
use crate::routing::features::Features;
use crate::{ArrayIndex, Result};

/// Scenario state required to evaluate routing moves.
///
/// A scenario owns the derived routing dataset together with the feature
/// metadata used to open and validate the underlying store. It provides
/// helpers for neighborhood lookups, retry-aware soft barrier filtering,
/// and reporting information about the active barrier groups.
pub(super) struct Scenario {
    /// Derived dataset containing cost arrays and barrier masks.
    pub dataset: crate::dataset::Dataset,
    transition_costs: TransitionCostTable,
    driver_rules: DriverRuleSet,
    #[allow(dead_code)]
    /// Source feature metadata kept alive for scenario lifetime management.
    features: Features,
}

impl Scenario {
    /// Open a scenario backed by a swap dataset.
    ///
    /// # Arguments
    /// `store_path`: Path to the source dataset store.
    /// `cost_function`: Cost function used to derive routing costs and
    ///                  barrier masks.
    /// `cache_size`: Maximum cache size for dataset-backed reads.
    /// `swap_fp`: Filesystem path where the swap dataset should be created.
    ///
    /// # Returns
    /// A `Scenario` whose dataset materializes derived arrays in the swap
    /// location while reusing the source feature definitions.
    pub(super) fn new_with_swap<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        cache_size: u64,
        swap_fp: PathBuf,
    ) -> Result<Self> {
        trace!("Opening scenario with: {:?}", store_path.as_ref());

        let driver_rules = cost_function.driver_rule_set()?;
        let transition_costs = cost_function.transition_cost_table()?;
        let features = Features::open(&store_path)?;
        let dataset = crate::dataset::Dataset::open_with_swap(
            store_path,
            cost_function,
            cache_size,
            swap_fp,
        )?;

        Ok(Self {
            dataset,
            transition_costs,
            driver_rules,
            features,
        })
    }

    /// Open a scenario that derives routing data directly from the source.
    ///
    /// # Arguments
    /// `store_path`: Path to the source dataset store.
    /// `cost_function`: Cost function used to derive routing costs and
    ///                  barrier masks.
    /// `cache_size`: Maximum cache size for dataset-backed reads.
    ///
    /// # Returns
    /// A `Scenario` ready to serve neighborhood and barrier queries.
    pub(super) fn new<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        cache_size: u64,
    ) -> Result<Self> {
        trace!("Opening scenario with: {:?}", store_path.as_ref());

        let driver_rules = cost_function.driver_rule_set()?;
        let transition_costs = cost_function.transition_cost_table()?;
        let features = Features::open(&store_path)?;
        let dataset = crate::dataset::Dataset::open(store_path, cost_function, cache_size)?;

        Ok(Self {
            dataset,
            transition_costs,
            driver_rules,
            features,
        })
    }

    /// Return retry-aware successor cells for a routing attempt.
    ///
    /// The returned successors exclude non-finite or non-positive costs and
    /// any cells covered by currently active soft barrier groups. If the
    /// starting cell itself is masked by an active soft barrier, no
    /// successors are returned.
    ///
    /// # Arguments
    /// `position`: Current grid position whose neighbors should be explored.
    /// `dropped_soft_groups`: Number of lowest-importance soft barrier
    ///                        groups that should be ignored for this retry.
    ///
    /// # Returns
    /// Neighbor positions and integer-encoded traversal costs suitable for
    /// routing expansion.
    pub(super) fn successors_for_attempt(
        &self,
        position: &ArrayIndex,
        dropped_soft_groups: usize,
    ) -> Vec<(ArrayIndex, u64)> {
        let spatial_neighbors = self.dataset.get_3x3(position);
        let soft_barrier_cells: HashSet<_> = self
            .dataset
            .get_3x3_soft_barrier_cells(position, dropped_soft_groups)
            .into_iter()
            .collect();

        if soft_barrier_cells.contains(position) {
            return Vec::new();
        }

        if self.driver_multiplier(position).is_none() {
            return Vec::new();
        }

        let mut neighbors = spatial_neighbors
            .into_iter()
            .filter(|(p, c)| c.is_finite() && *c > 0.0 && !soft_barrier_cells.contains(p))
            .filter_map(|(p, c)| {
                self.driver_multiplier(&p)
                    .map(|multiplier| (p, cost_as_u64(c * multiplier)))
            })
            .collect::<Vec<_>>();

        let (_, _, noptions) = self.grid_shape();
        for option in 0..noptions {
            if option == position.option {
                continue;
            }

            let destination = ArrayIndex {
                i: position.i,
                j: position.j,
                option,
            };
            let destination_soft_barriers: HashSet<_> = self
                .dataset
                .get_3x3_soft_barrier_cells(&destination, dropped_soft_groups)
                .into_iter()
                .collect();
            if destination_soft_barriers.contains(&destination) {
                continue;
            }

            let Some(destination_center_cost) = self.dataset.get_cell_cost(&destination) else {
                continue; // destination is hard barriered
            };
            let Some(driver_multiplier) = self.driver_multiplier(&destination) else {
                continue; // destination is excluded by the driver
            };

            let transition_cost = self.transition_cost(position.option, option);
            neighbors.push((
                destination,
                transition_cost
                    .saturating_add(cost_as_u64(destination_center_cost * driver_multiplier)),
            ));
        }

        trace!("Adjusting neighbors' types: {:?}", neighbors);
        neighbors
    }

    /// Count the configured soft barrier groups.
    ///
    /// # Returns
    /// Number of retry stages available for progressively dropping soft
    /// barrier groups.
    pub(super) fn soft_barrier_group_count(&self) -> usize {
        self.dataset.soft_barrier_groups().len()
    }

    /// List barrier layer names dropped for a retry state.
    ///
    /// # Arguments
    /// `dropped_soft_groups`: Number of soft barrier groups dropped in
    ///                        ascending importance order.
    ///
    /// # Returns
    /// Barrier layer names belonging to the dropped groups.
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

    /// Return the routing states allowed at a grid position.
    ///
    /// # Arguments
    /// `position`: Grid position whose routing options should be checked.
    /// `dropped_soft_groups`: Number of lowest-importance soft barrier
    ///                        groups that should be ignored for this retry.
    ///
    /// # Returns
    /// All routing-option states at the given cell that are not blocked by
    /// active soft barriers, hard barriers, or driver exclusions.
    pub(super) fn allowed_states_at(
        &self,
        position: &ArrayIndex,
        dropped_soft_groups: usize,
    ) -> impl Iterator<Item = ArrayIndex> + '_ {
        let (_, _, noptions) = self.grid_shape();
        let row = position.i;
        let col = position.j;

        (0..noptions).filter_map(move |option| {
            let candidate = ArrayIndex {
                i: row,
                j: col,
                option,
            };
            let soft_barrier_cells: HashSet<_> = self
                .dataset
                .get_3x3_soft_barrier_cells(&candidate, dropped_soft_groups)
                .into_iter()
                .collect();
            if soft_barrier_cells.contains(&candidate) {
                return None;
            }

            self.dataset
                .get_cell_cost(&candidate) // checks for hard barriers
                .filter(|_| self.driver_multiplier(&candidate).is_some()) // drops `None` values from `get_cell_cost`
                .map(|_| candidate) // drops `None` values from `driver_multiplier`
        })
    }

    /// Resolve the driver multiplier for a cell state.
    ///
    /// # Arguments
    /// `position`: Grid index whose routing option and source-layer values
    ///             should be evaluated against the configured driver rules.
    ///
    /// # Returns
    /// The multiplier for the state's routing option, or `None` when that
    /// option is excluded at the given position.
    fn driver_multiplier(&self, position: &ArrayIndex) -> Option<f32> {
        self.driver_rules.multiplier(position.option, |layer_name| {
            self.dataset.get_source_cell_value(layer_name, position)
        })
    }

    /// Return the integer-encoded cost of changing routing options.
    ///
    /// # Arguments
    /// `from_option`: Source routing option index.
    /// `to_option`: Destination routing option index.
    ///
    /// # Returns
    /// The configured transition cost between the two options, converted to
    /// the internal integer routing-cost representation.
    fn transition_cost(&self, from_option: u32, to_option: u32) -> u64 {
        cost_as_u64(self.transition_costs.cost(from_option, to_option))
    }

    /// Return the scenario grid dimensions.
    ///
    /// # Returns
    /// Tuple of `(rows, cols, options)` describing the routing grid shape.
    pub(super) fn grid_shape(&self) -> (u64, u64, u32) {
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

        let start = ArrayIndex::new_ij(1, 1);
        let initial_successors = scenario.successors_for_attempt(&start, 0);
        let relaxed_successors = scenario.successors_for_attempt(&start, 1);

        assert!(
            !initial_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex::new_ij(0, 1))
        );
        assert!(
            !initial_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex::new_ij(1, 0))
        );

        assert!(
            !relaxed_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex::new_ij(0, 1))
        );
        assert!(
            relaxed_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex::new_ij(1, 0))
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

        let successors = scenario.successors_for_attempt(&ArrayIndex::new_ij(1, 1), 0);

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

        let start = ArrayIndex::new_ij(1, 1);
        let initial_successors = scenario.successors_for_attempt(&start, 0);
        let retry_one_successors = scenario.successors_for_attempt(&start, 1);
        let retry_two_successors = scenario.successors_for_attempt(&start, 2);

        assert_eq!(scenario.soft_barrier_group_count(), 2);
        assert!(
            !initial_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex::new_ij(1, 0))
        );
        assert!(
            !initial_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex::new_ij(0, 1))
        );
        assert!(
            retry_one_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex::new_ij(1, 0))
        );
        assert!(
            !retry_one_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex::new_ij(0, 1))
        );
        assert!(
            retry_two_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex::new_ij(1, 0))
        );
        assert!(
            retry_two_successors
                .iter()
                .any(|(p, _)| *p == ArrayIndex::new_ij(0, 1))
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
                .successors_for_attempt(&ArrayIndex::new_ij(1, 1), 0)
                .is_empty()
        );
        assert!(
            !scenario
                .successors_for_attempt(&ArrayIndex::new_ij(1, 1), 1)
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
