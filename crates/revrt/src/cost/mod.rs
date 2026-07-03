pub(crate) mod components;
mod inputs;

use core::f32;
use ndarray::{ArrayD, Axis, IxDyn, stack};
use std::convert::TryFrom;
use tracing::{debug, trace};

use crate::cost::components::{
    BarrierLayer, BarrierOperator, CostLayer, DriverRuleSet, FrictionLayer, TransitionCostTable,
};
use crate::cost::inputs::CostFunctionInput;
use crate::dataset::LazySubset;
use crate::error::Result;

/// A multi-dimensional array representing cost data
type CostArray = ndarray::Array<f32, ndarray::Dim<ndarray::IxDynImpl>>;
type BarrierArray = ndarray::Array<bool, ndarray::Dim<ndarray::IxDynImpl>>;

/// Large friction value to use for invalid costs that can be routed through
const HIGH_FRICTION_INVALID_COST: f32 = 1e10;

#[derive(Clone, Debug)]
/// A cost function definition
///
/// `cost_layers`: A collection of cost layers with equal weight.
/// `friction_layers`: A collection of friction layers that scale the cost layer.
/// `barrier_layers`: A collection of layers that create impassable cells.
/// `invalid_costs_block_routing`: If true, cells with <=0 or NaN costs are skipped completely.
///
/// This was based on the original transmission router and is composed of
/// layers that are summed together (per grid point) to give the total cost.
pub(crate) struct CostFunction {
    cost_layers: Vec<CostLayer>,
    option_cost_multiplier_layers: Vec<Option<String>>,
    friction_layers: Option<Vec<FrictionLayer>>,
    barrier_layers: Option<Vec<BarrierLayer>>,
    pub(crate) routing_options: Vec<String>,
    pub(crate) drivers: DriverRuleSet,
    pub(crate) transition_costs: TransitionCostTable,
    /// Option to completely ignore <=0 cost cells
    pub(crate) invalid_costs_block_routing: bool,
}

impl CostFunction {
    #[allow(clippy::too_many_arguments)]
    fn from_input_parts(
        cost_layers: Vec<CostLayer>,
        option_cost_multiplier_layers: Vec<Option<String>>,
        friction_layers: Vec<FrictionLayer>,
        barrier_layers: Vec<BarrierLayer>,
        routing_options: Vec<String>,
        drivers: DriverRuleSet,
        transition_costs: TransitionCostTable,
        invalid_costs_block_routing: bool,
    ) -> Self {
        Self {
            cost_layers,
            option_cost_multiplier_layers,
            friction_layers: (!friction_layers.is_empty()).then_some(friction_layers),
            barrier_layers: (!barrier_layers.is_empty()).then_some(barrier_layers),
            routing_options,
            drivers,
            transition_costs,
            invalid_costs_block_routing,
        }
    }

    /// Create a new cost function from a JSON string (reVX format)
    ///
    /// # Arguments
    /// `json`: A JSON string representing the cost function with the format
    ///         used by reVX.
    ///
    /// # Returns
    /// A `CostFunction` object.
    ///
    /// Layer definitions must be nested under `routing_options`.
    /// ```json
    /// {
    ///   "routing_options": {
    ///     "default": {
    ///       "cost_layers": [
    ///         {"layer_name": "A"},
    ///         {
    ///           "layer_name": "A",
    ///           "multiplier_scalar": 2,
    ///           "multiplier_layer": "B"
    ///         }
    ///       ],
    ///       "barrier_layers": [
    ///         {
    ///           "layer_name": "barrier_mask",
    ///           "barrier_operator": "eq",
    ///           "barrier_threshold": 1.0
    ///         }
    ///       ]
    ///     }
    ///   }
    /// }
    /// ```
    pub(super) fn from_json(json: &str) -> Result<Self> {
        trace!("Parsing cost definition from json: {}", json);
        let cost_input: CostFunctionInput = serde_json::from_str(json)
            .map_err(|error| crate::error::Error::Undefined(error.to_string()))?;
        Self::try_from(cost_input)
    }

    /// Return a copy of this cost function with all barrier layers removed.
    pub(crate) fn without_barriers(&self) -> Self {
        let mut cost_function = self.clone();
        cost_function.barrier_layers = None;
        cost_function
    }

    /// Collect all barrier layers that act as hard barriers.
    ///
    /// Hard barriers are layers with no assigned importance, so they are
    /// always treated as impassable.
    pub(crate) fn hard_barrier_layers(&self) -> Vec<BarrierLayer> {
        self.barrier_layers
            .clone()
            .unwrap_or_default()
            .into_iter()
            .filter(|layer| layer.importance().is_none())
            .collect()
    }

    /// Group soft barrier layers by their importance.
    ///
    /// Only layers with an assigned importance are included in the output,
    /// and the returned groups are ordered by importance.
    pub(crate) fn soft_barrier_groups(&self) -> Vec<(u32, Vec<BarrierLayer>)> {
        let mut groups = std::collections::BTreeMap::<u32, Vec<BarrierLayer>>::new();

        for layer in self.barrier_layers.clone().unwrap_or_default() {
            if let Some(importance) = layer.importance() {
                groups.entry(importance).or_default().push(layer);
            }
        }

        groups.into_iter().collect()
    }

    /// Whether any configured cost layers contribute invariant costs.
    pub(crate) fn has_invariant_layers(&self) -> bool {
        self.cost_layers
            .iter()
            .any(|layer| layer.is_invariant.unwrap_or(false))
    }

    /// Calculate the cost from a given collection of input features
    ///
    /// Applies the cost function to a collection of input features, which
    /// is typically a subset of a larger dataset, such as a chunk from a
    /// Zarr dataset. The cost function is defined by a series of layers,
    /// each of which may have a multiplier scalar or a multiplier layer.
    ///
    /// # Arguments
    /// `features`: A lazy collection of input features.
    /// `is_invariant`: If true, only invariant layers contribute.
    ///
    /// # Returns
    /// A 2D array containing the cost for the subset covered by the input
    /// features.
    pub(crate) fn compute(&self, features: &mut LazySubset<f32>, is_invariant: bool) -> CostArray {
        debug!(
            "Calculating (is_invariant={}) cost for ({})",
            is_invariant,
            features.subset()
        );

        let cost_layers: Vec<&CostLayer> = self
            .cost_layers
            .iter()
            .filter(|layer| layer.is_invariant.unwrap_or(false) == is_invariant)
            .collect();

        if cost_layers.is_empty() {
            return empty_cost_array(features);
        }

        let cost_data = cost_layers
            .into_iter()
            .map(|layer| build_single_cost_layer(layer, features))
            .collect::<Vec<_>>();

        let mut final_cost_layer = reduce_layers(cost_data);
        self.apply_option_cost_multiplier_layers(&mut final_cost_layer, features);
        final_cost_layer.mapv_inplace(|v| {
            if v <= 0_f32 {
                if self.invalid_costs_block_routing {
                    f32::NAN
                } else {
                    HIGH_FRICTION_INVALID_COST
                }
            } else {
                v
            }
        });

        let friction_data = match &self.friction_layers {
            None => vec![],
            Some(layers) => layers
                .iter()
                .map(|layer| build_single_friction_layer(layer, features))
                .collect::<Vec<_>>(),
        };

        let mut final_friction_layer = match friction_data.is_empty() {
            true => ArrayD::<f32>::zeros(IxDyn(final_cost_layer.shape())),
            false => reduce_layers(friction_data),
        };

        // Ensure friction does not go below -1. If any values are below -1,
        // emit a warning and clamp them to -1 so the routing surface
        // calculation (1 + friction) does not produce negative cost values
        if final_friction_layer.iter().any(|v| *v <= -1.0) {
            tracing::warn!("Friction layer contains values <= -1; clamping to -1");
            final_friction_layer.mapv_inplace(|v| if v <= -1.0 { -1.0 + 1e-7 } else { v });
        }

        // routing surface is: final_cost_layer * (1 + final_friction_layer)
        final_cost_layer
            * (ArrayD::<f32>::ones(IxDyn(final_friction_layer.shape())) + final_friction_layer)
    }

    fn apply_option_cost_multiplier_layers(
        &self,
        final_cost_layer: &mut CostArray,
        features: &mut LazySubset<f32>,
    ) {
        for (option, multiplier_layer_name) in self.option_cost_multiplier_layers.iter().enumerate()
        {
            let Some(multiplier_layer_name) = multiplier_layer_name else {
                continue;
            };

            let multiplier =
                build_option_cost_multiplier_layer(multiplier_layer_name, option as u32, features);
            *final_cost_layer *= &multiplier;
        }
    }
}

fn empty_cost_array(features: &LazySubset<f32>) -> CostArray {
    let shape: Vec<usize> = features
        .subset()
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).expect("subset dimension exceeds usize range"))
        .collect();

    ArrayD::<f32>::zeros(IxDyn(&shape))
}

fn build_single_cost_layer(layer: &CostLayer, features: &mut LazySubset<f32>) -> CostArray {
    let layer_name = &layer.layer_name;
    trace!("Layer name: {}", layer_name);

    let mut cost = features
        .get(layer_name)
        .expect("Layer not found in features");

    if let Some(multiplier_scalar) = layer.multiplier_scalar {
        trace!(
            "Layer {} has multiplier scalar {}",
            layer_name, multiplier_scalar
        );
        // Apply the multiplier scalar to the value
        cost *= multiplier_scalar;
        // trace!( "Cost for chunk ({}, {}) in layer {}: {}", ci, cj, layer_name, cost);
    }

    if let Some(multiplier_layer) = &layer.multiplier_layer {
        trace!(
            "Layer {} has multiplier layer {}",
            layer_name, multiplier_layer
        );
        let multiplier_value = features
            .get(multiplier_layer)
            .expect("Multiplier layer not found in features");

        // Apply the multiplier layer to the value
        cost = cost * multiplier_value;
        // trace!( "Cost for chunk ({}, {}) in layer {}: {}", ci, cj, layer_name, cost);
    }

    cost.mapv_inplace(|v| if v > 0.0_f32 { v } else { 0.0_f32 });

    select_option_for_subset(cost, layer.option, features)
}

fn build_single_friction_layer(layer: &FrictionLayer, features: &mut LazySubset<f32>) -> CostArray {
    trace!("Building friction layer: {:?}", layer);

    let multiplier_layer_name = &layer.multiplier_layer;

    let mut friction = features
        .get(multiplier_layer_name)
        .expect("Multiplier layer not found in features");

    if let Some(multiplier_scalar) = layer.multiplier_scalar {
        trace!("\t- Layer has multiplier scalar {}", multiplier_scalar);
        friction *= multiplier_scalar;
    }

    select_option_for_subset(friction, layer.option, features)
}

fn build_option_cost_multiplier_layer(
    layer_name: &str,
    option: u32,
    features: &mut LazySubset<f32>,
) -> CostArray {
    let multiplier = features
        .get(layer_name)
        .expect("Cost multiplier layer not found in features");

    select_option_for_subset_with_fill(multiplier, option, features, 1.0_f32)
}

pub(crate) fn build_single_barrier_layer(
    layer: &BarrierLayer,
    features: &mut LazySubset<f32>,
) -> BarrierArray {
    trace!("Building barrier layer: {:?}", layer);

    let barrier_values = features
        .get(&layer.layer_name)
        .expect("Barrier layer not found in features")
        .mapv(|value| match layer.barrier_operator {
            BarrierOperator::NotEqual => value != layer.barrier_threshold,
            BarrierOperator::GreaterThan => value > layer.barrier_threshold,
            BarrierOperator::GreaterThanOrEqual => value >= layer.barrier_threshold,
            BarrierOperator::LessThan => value < layer.barrier_threshold,
            BarrierOperator::LessThanOrEqual => value <= layer.barrier_threshold,
            BarrierOperator::Equal => value == layer.barrier_threshold,
        });

    select_option_for_subset(barrier_values, layer.option, features)
}

fn select_option_for_subset<T>(
    values: ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>,
    option: u32,
    features: &LazySubset<f32>,
) -> ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>
where
    T: Clone + Default,
{
    select_option_for_subset_with_fill(values, option, features, T::default())
}

fn select_option_for_subset_with_fill<T>(
    values: ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>,
    option: u32,
    features: &LazySubset<f32>,
    fill: T,
) -> ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>
where
    T: Clone,
{
    let band_start = features.subset().start()[0];
    let band_end = band_start + features.subset().shape()[0];
    let option = u64::from(option);

    let output_shape = features
        .subset()
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).expect("subset dimension exceeds usize range"))
        .collect::<Vec<_>>();

    if option < band_start || option >= band_end {
        return ndarray::ArrayD::<T>::from_elem(ndarray::IxDyn(&output_shape), fill);
    }

    let local_option = (option - band_start) as usize;
    let mut selected = ndarray::ArrayD::<T>::from_elem(ndarray::IxDyn(&output_shape), fill);
    let source_option = if values.shape()[0] == 1 {
        0
    } else {
        local_option
    };

    if source_option >= values.shape()[0] {
        return selected;
    }

    selected
        .index_axis_mut(Axis(0), local_option)
        .assign(&values.index_axis(Axis(0), source_option));
    selected
}

fn reduce_layers(data: Vec<CostArray>) -> CostArray {
    let views: Vec<_> = data.iter().map(|a| a.view()).collect();
    let stack = stack(Axis(0), &views).unwrap();
    trace!("Stack shape: {:?}", stack.shape());
    let final_layer = stack.sum_axis(Axis(0));
    trace!("Stack shape: {:?}", stack.shape());
    final_layer
}

#[cfg(test)]
pub(crate) mod sample {
    use super::*;

    /// Sample cost definition
    pub(crate) fn as_text_v1() -> String {
        r#"
        {
            "routing_options": {
                "default": {
                    "cost_layers": [
                        {"layer_name": "A"},
                        {"layer_name": "B", "multiplier_scalar": 100},
                        {"layer_name": "A",
                            "multiplier_layer": "B"},
                        {"layer_name": "C", "multiplier_scalar": 2,
                            "multiplier_layer": "A"},
                        {"layer_name": "C", "multiplier_scalar": 100,
                            "is_invariant": true}
                    ]
                }
            }
        }
        "#
        .to_string()
    }

    pub(crate) fn cost_function() -> CostFunction {
        let json = as_text_v1();
        CostFunction::from_json(&json).unwrap()
    }
}

#[cfg(test)]
mod test_builder {
    use crate::cost::components::CostLayerBuilder;

    #[test]
    fn costlayer() {
        let layer = CostLayerBuilder::default()
            .layer_name("A".to_string())
            .multiplier_scalar(2.0)
            .multiplier_layer("B")
            .is_invariant(false)
            .build()
            .unwrap();

        assert_eq!(layer.layer_name, "A".to_string());
        assert_eq!(layer.multiplier_scalar, Some(2.0));
        assert_eq!(layer.multiplier_layer, Some("B".to_string()));
        assert_eq!(layer.is_invariant, Some(false));
        assert_eq!(layer.option, 0);
    }

    #[test]
    fn defaults() {
        let layer = CostLayerBuilder::default()
            .layer_name("A".to_string())
            .build()
            .unwrap();

        assert_eq!(layer.layer_name, "A".to_string());
        assert_eq!(layer.multiplier_scalar, None);
        assert_eq!(layer.multiplier_layer, None);
        assert_eq!(layer.is_invariant, None);
        assert_eq!(layer.option, 0);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dataset::{make_lazy_subset_for_tests, samples};
    use ndarray::ArrayD;
    use std::sync::Arc;
    use zarrs::array_subset::ArraySubset;
    use zarrs::filesystem::FilesystemStore;
    use zarrs::storage::ReadableListableStorage;

    fn make_features_for_costs_tests() -> (tempfile::TempDir, LazySubset<f32>) {
        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 8, 8)
            .chunks(1, 4, 4)
            .layer(samples::LayerConfig::random("A", 0.0, 1.0))
            .layer(samples::LayerConfig::random("B", 0.0, 1.0))
            .layer(samples::LayerConfig::random("C", 0.0, 1.0))
            .layer(samples::LayerConfig::random("cost", 0.0, 1.0))
            .build()
            .expect("Failed to create multi-variable zarr");
        let store: ReadableListableStorage = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 2, 2]).unwrap();
        (tmp, make_lazy_subset_for_tests(store, subset))
    }

    #[test]
    fn test_cost() {
        let json = sample::as_text_v1();
        let cost = CostFunction::from_json(&json).unwrap();

        assert_eq!(cost.cost_layers.len(), 5);
        assert_eq!(cost.cost_layers[0].layer_name, "A".to_string());
        assert_eq!(cost.cost_layers[0].is_invariant, None);
        assert_eq!(cost.cost_layers[0].option, 0);
        assert_eq!(cost.cost_layers[1].layer_name, "B".to_string());
        assert_eq!(cost.cost_layers[1].multiplier_scalar, Some(100.0));
        assert_eq!(cost.cost_layers[1].is_invariant, None);
        assert_eq!(cost.cost_layers[1].option, 0);
        assert_eq!(cost.cost_layers[2].layer_name, "A".to_string());
        assert_eq!(cost.cost_layers[2].multiplier_layer, Some("B".to_string()));
        assert_eq!(cost.cost_layers[2].is_invariant, None);
        assert_eq!(cost.cost_layers[2].option, 0);
        assert_eq!(cost.cost_layers[3].layer_name, "C".to_string());
        assert_eq!(cost.cost_layers[3].multiplier_layer, Some("A".to_string()));
        assert_eq!(cost.cost_layers[3].multiplier_scalar, Some(2.0));
        assert_eq!(cost.cost_layers[3].is_invariant, None);
        assert_eq!(cost.cost_layers[3].option, 0);
        assert_eq!(cost.cost_layers[4].layer_name, "C".to_string());
        assert_eq!(cost.cost_layers[4].multiplier_layer, None);
        assert_eq!(cost.cost_layers[4].multiplier_scalar, Some(100.0));
        assert_eq!(cost.cost_layers[4].is_invariant, Some(true));
        assert_eq!(cost.cost_layers[4].option, 0);
    }

    #[test]
    fn test_build_single_barrier_layer_supports_not_equal() {
        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 2, 2)
            .chunks(1, 2, 2)
            .layer(samples::LayerConfig::new(
                "barrier",
                samples::FillStrategy::Values(vec![0.0, 1.0, 2.0, 0.0]),
            ))
            .build()
            .expect("Failed to create barrier zarr");
        let store: ReadableListableStorage = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 2, 2]).unwrap();
        let mut features = make_lazy_subset_for_tests(store, subset);
        let layer = BarrierLayer {
            layer_name: "barrier".to_string(),
            barrier_operator: BarrierOperator::NotEqual,
            barrier_threshold: 0.0,
            barrier_importance: None,
            option: 0,
        };

        let barrier = build_single_barrier_layer(&layer, &mut features);

        assert_eq!(
            barrier,
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![false, true, true, false]).unwrap()
        );
    }

    #[test]
    fn test_friction_only_returns_zeros() {
        let (_tmp, mut features) = make_features_for_costs_tests();

        // friction-only (no `layer_name`) should return an empty cost array (zeros)
        let json = r#"
        {
            "routing_options": {
                "default": {
                    "cost_layers": [],
                    "friction_layers": [
                        {"multiplier_layer": "B", "multiplier_scalar": -3.0}
                    ]
                }
            }
        }
        "#;

        let cost_fn = CostFunction::from_json(json).unwrap();
        let result = cost_fn.compute(&mut features, false);

        assert_eq!(result.shape(), &[1, 2, 2]);
        for v in result.iter() {
            assert_eq!(*v, 0.0_f32);
        }
    }

    #[test]
    fn test_friction_clamp_with_cost_layer() {
        use ndarray::Zip;

        let (_tmp, mut features) = make_features_for_costs_tests();

        // cost layer A with a friction layer defined by B * -3.0
        let json = r#"
        {
            "routing_options": {
                "default": {
                    "cost_layers": [
                        {"layer_name": "A"}
                    ],
                    "friction_layers": [
                        {"multiplier_layer": "B", "multiplier_scalar": -3.0}
                    ]
                }
            }
        }
        "#;

        let cost_fn = CostFunction::from_json(json).unwrap();
        let result = cost_fn.compute(&mut features, false);

        let a = features.get("A").unwrap();
        let b = features.get("B").unwrap();
        Zip::from(&result)
            .and(&a)
            .and(&b)
            .for_each(|r, a_item, b_item| {
                // Build expected result: for each cell, friction = B * -3.0
                // if friction < -1 => clamp to -1+1e-12
                // result = A * (1 + friction_clamped)
                let mut friction = b_item * -3.0;
                if friction < -1.0 {
                    friction = -1.0 + 1e-7;
                }
                let truth = a_item * (1.0 + friction);

                if *a_item > 0.0_f32 {
                    dbg!(r, a_item, b_item);
                    assert!(*r > 0.0_f32);
                }
                let diff = (*r - truth).abs();
                assert!(diff < 1e-6, "mismatch {} vs {} (diff={})", r, truth, diff);
            });
    }

    #[test]
    fn routing_options_object_builds_ordered_names_and_band_specific_costs() {
        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 2, 2)
            .chunks(1, 2, 2)
            .layer(samples::LayerConfig::constant("overhead_cost", 1.0))
            .layer(samples::LayerConfig::constant("underground_cost", 2.0))
            .build()
            .expect("Failed to create routing option zarr");
        let store: ReadableListableStorage = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![2, 2, 2]).unwrap();
        let mut features = make_lazy_subset_for_tests(store, subset);
        let cost_fn = CostFunction::from_json(
            r#"{
                "routing_options": {
                    "overhead": {
                        "cost_layers": [{"layer_name": "overhead_cost"}]
                    },
                    "underground": {
                        "cost_layers": [{"layer_name": "underground_cost"}]
                    }
                }
            }"#,
        )
        .unwrap();

        let result = cost_fn.compute(&mut features, false);

        assert_eq!(cost_fn.routing_options, ["overhead", "underground"]);
        assert_eq!(
            result.index_axis(Axis(0), 0).to_owned(),
            ArrayD::from_elem(IxDyn(&[2, 2]), 1.0)
        );
        assert_eq!(
            result.index_axis(Axis(0), 1).to_owned(),
            ArrayD::from_elem(IxDyn(&[2, 2]), 2.0)
        );
    }

    #[test]
    fn routing_options_object_reuses_same_source_layer_across_options() {
        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 2, 2)
            .chunks(1, 2, 2)
            .layer(samples::LayerConfig::constant("shared_cost", 3.0))
            .build()
            .expect("Failed to create shared routing option zarr");
        let store: ReadableListableStorage = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![2, 2, 2]).unwrap();
        let mut features = make_lazy_subset_for_tests(store, subset);
        let cost_fn = CostFunction::from_json(
            r#"{
                "routing_options": {
                    "overhead": {
                        "cost_layers": [{"layer_name": "shared_cost"}]
                    },
                    "underground": {
                        "cost_layers": [{"layer_name": "shared_cost", "multiplier_scalar": 2.0}]
                    }
                }
            }"#,
        )
        .unwrap();

        let result = cost_fn.compute(&mut features, false);

        assert_eq!(cost_fn.routing_options, ["overhead", "underground"]);
        assert_eq!(
            result.index_axis(Axis(0), 0).to_owned(),
            ArrayD::from_elem(IxDyn(&[2, 2]), 3.0)
        );
        assert_eq!(
            result.index_axis(Axis(0), 1).to_owned(),
            ArrayD::from_elem(IxDyn(&[2, 2]), 6.0)
        );
    }

    #[test]
    fn routing_options_apply_option_cost_multiplier_layers() {
        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 2, 2)
            .chunks(1, 2, 2)
            .layer(samples::LayerConfig::constant("overhead_cost", 1.0))
            .layer(samples::LayerConfig::constant("underground_cost", 2.0))
            .layer(samples::LayerConfig::constant("overhead_li", 3.0))
            .layer(samples::LayerConfig::constant("underground_li", 4.0))
            .layer(samples::LayerConfig::constant("overhead_multiplier", 10.0))
            .layer(samples::LayerConfig::constant(
                "underground_multiplier",
                20.0,
            ))
            .build()
            .expect("Failed to create routing option multiplier zarr");
        let store: ReadableListableStorage = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![2, 2, 2]).unwrap();
        let mut features = make_lazy_subset_for_tests(store, subset);
        let cost_fn = CostFunction::from_json(
            r#"{
                "routing_options": {
                    "overhead": {
                        "cost_layers": [
                            {"layer_name": "overhead_cost"},
                            {"layer_name": "overhead_li", "is_invariant": true}
                        ],
                        "cost_multiplier_layer": "overhead_multiplier"
                    },
                    "underground": {
                        "cost_layers": [
                            {"layer_name": "underground_cost"},
                            {"layer_name": "underground_li", "is_invariant": true}
                        ],
                        "cost_multiplier_layer": "underground_multiplier"
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            cost_fn.option_cost_multiplier_layers,
            vec![
                Some("overhead_multiplier".to_string()),
                Some("underground_multiplier".to_string()),
            ]
        );

        let primary = cost_fn.compute(&mut features, false);
        let invariant = cost_fn.compute(&mut features, true);

        assert_eq!(
            primary.index_axis(Axis(0), 0).to_owned(),
            ArrayD::from_elem(IxDyn(&[2, 2]), 10.0)
        );
        assert_eq!(
            primary.index_axis(Axis(0), 1).to_owned(),
            ArrayD::from_elem(IxDyn(&[2, 2]), 40.0)
        );
        assert_eq!(
            invariant.index_axis(Axis(0), 0).to_owned(),
            ArrayD::from_elem(IxDyn(&[2, 2]), 30.0)
        );
        assert_eq!(
            invariant.index_axis(Axis(0), 1).to_owned(),
            ArrayD::from_elem(IxDyn(&[2, 2]), 80.0)
        );
    }

    #[test]
    fn option_cost_multiplier_does_not_revive_invalid_cost_cells() {
        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 2, 2)
            .chunks(1, 2, 2)
            .layer(samples::LayerConfig::new(
                "cost",
                samples::FillStrategy::Values(vec![-1.0, 1.0, 1.0, 1.0]),
            ))
            .layer(samples::LayerConfig::new(
                "multiplier",
                samples::FillStrategy::Values(vec![-1.0, 1.0, 1.0, 1.0]),
            ))
            .build()
            .expect("Failed to create invalid-cost multiplier zarr");
        let store: ReadableListableStorage = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 2, 2]).unwrap();
        let mut features = make_lazy_subset_for_tests(store, subset);
        let cost_fn = CostFunction::from_json(
            r#"{
                "invalid_costs_block_routing": true,
                "routing_options": {
                    "default": {
                        "cost_layers": [{"layer_name": "cost"}],
                        "cost_multiplier_layer": "multiplier"
                    }
                }
            }"#,
        )
        .unwrap();

        let result = cost_fn.compute(&mut features, false);

        assert!(result[[0, 0, 0]].is_nan());
        assert_eq!(result[[0, 0, 1]], 1.0);
        assert_eq!(result[[0, 1, 0]], 1.0);
        assert_eq!(result[[0, 1, 1]], 1.0);
    }

    #[test]
    fn routing_option_specific_single_band_layers_use_local_band_zero() {
        let tmp = samples::ZarrTestBuilder::new()
            .dimensions(1, 2, 2)
            .chunks(1, 2, 2)
            .layer(samples::LayerConfig::constant("overhead_cost", 1.0))
            .layer(samples::LayerConfig::constant("underground_cost", 2.0))
            .build()
            .expect("Failed to create single-band routing option zarr");
        let store: ReadableListableStorage = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let subset = ArraySubset::new_with_start_shape(vec![1, 0, 0], vec![1, 2, 2]).unwrap();
        let mut features = make_lazy_subset_for_tests(store, subset);
        let cost_fn = CostFunction::from_json(
            r#"{
                "routing_options": {
                    "overhead": {
                        "cost_layers": [{"layer_name": "overhead_cost"}]
                    },
                    "underground": {
                        "cost_layers": [{"layer_name": "underground_cost"}]
                    }
                }
            }"#,
        )
        .unwrap();

        let result = cost_fn.compute(&mut features, false);

        assert_eq!(result.shape(), &[1, 2, 2]);
        assert_eq!(result, ArrayD::from_elem(IxDyn(&[1, 2, 2]), 2.0));
    }

    #[test]
    fn routing_options_object_supports_sample_barrier_and_friction_syntax() {
        let cost_fn = CostFunction::from_json(
            r#"{
                "routing_options": {
                    "overhead": {
                        "cost_layers": [{"layer_name": "A"}],
                        "friction_layers": [{"layer_name": "wet_friction", "multiplier_scalar": 1.1}],
                        "barrier_layers": [{"layer_name": "barrier_mask", "barrier_operator": "eq", "barrier_threshold": 1.0}]
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(cost_fn.routing_options, ["overhead"]);
        assert_eq!(
            cost_fn.friction_layers.as_ref().unwrap()[0].multiplier_layer,
            "wet_friction"
        );
        assert_eq!(
            cost_fn.hard_barrier_layers()[0].barrier_operator as u8,
            BarrierOperator::Equal as u8
        );
        assert_eq!(cost_fn.hard_barrier_layers()[0].barrier_threshold, 1.0);
    }

    #[test]
    fn barrier_layers_require_split_operator_and_threshold_inputs() {
        let error = CostFunction::from_json(
            r#"{
                "routing_options": {
                    "overhead": {
                        "cost_layers": [{"layer_name": "A"}],
                        "barrier_layers": [{"layer_name": "barrier_mask", "where": "==1"}]
                    }
                }
            }"#,
        )
        .unwrap_err();

        assert!(matches!(error, crate::error::Error::Undefined(_)));
    }
}
