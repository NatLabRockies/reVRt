//! Cost function

use core::f32;
use derive_builder::Builder;
use ndarray::{ArrayD, Axis, IxDyn, stack};
use std::convert::TryFrom;
use tracing::{debug, trace};

use crate::dataset::LazySubset;
use crate::error::Result;

/// A multi-dimensional array representing cost data
type CostArray = ndarray::Array<f32, ndarray::Dim<ndarray::IxDynImpl>>;
type BarrierArray = ndarray::Array<bool, ndarray::Dim<ndarray::IxDynImpl>>;

/// Large friction value to use for invalid costs that can be routed through
const HIGH_FRICTION_INVALID_COST: f32 = 1e10;

fn true_option() -> bool {
    true
}

#[derive(Clone, Debug, serde::Deserialize)]
/// A cost function definition
///
/// `cost_layers`: A collection of cost layers with equal weight.
/// `friction_layers`: A collection of friction layers that scale the cost layer.
/// `barrier_layers`: A collection of layers that create impassable cells.
/// `ignore_invalid_costs`: If true, cells with <=0 or NaN costs are skipped completely.
///
/// This was based on the original transmission router and is composed of
/// layers that are summed together (per grid point) to give the total cost.
pub(crate) struct CostFunction {
    cost_layers: Vec<CostLayer>,
    friction_layers: Option<Vec<FrictionLayer>>,
    barrier_layers: Option<Vec<BarrierLayer>>,
    /// Option to completely ignore <=0 cost cells
    #[serde(default = "true_option")]
    pub(crate) ignore_invalid_costs: bool,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
pub(crate) enum BarrierOperator {
    #[serde(rename = "gt")]
    GreaterThan,
    #[serde(rename = "ge")]
    GreaterThanOrEqual,
    #[serde(rename = "lt")]
    LessThan,
    #[serde(rename = "le")]
    LessThanOrEqual,
    #[serde(rename = "eq")]
    Equal,
}

#[derive(Builder, Clone, Debug, serde::Deserialize)]
/// A cost layer
///
/// Each cost layer is a raster dataset, i.e. a regular grid, composed by
/// operating on input features. Following the original `revX` structure,
/// the possible compositions are limited to combinations of the relation
/// `weight * layer_name * multiplier_layer`, where the `weight` and the
/// `multiplier_layer` are optional. Each layer can also be marked as invariant,
/// meaning that its value does not get scaled by the distance traveled
/// through the cell. Instead, the value of the layer is added once, right
/// when the path enters the cell.
struct CostLayer {
    layer_name: String,
    #[builder(setter(strip_option), default)]
    multiplier_scalar: Option<f32>,
    #[builder(setter(strip_option, into), default)]
    multiplier_layer: Option<String>,
    #[builder(setter(strip_option), default)]
    is_invariant: Option<bool>,
}

#[derive(Builder, Clone, Debug, serde::Deserialize)]
/// A friction layer
///
/// Each friction layer is a raster dataset, i.e. a regular grid, that
/// represents multipliers that should be applied to the cost routing
/// layer. These multipliers affect the output route but will not be
/// reported in the output cost. Each friction layer is defined by a
/// `multiplier_layer` and an optional `multiplier_scalar`. The friction
/// value at each cell is computed as `multiplier_layer * multiplier_scalar`.
/// If the `multiplier_scalar` is not provided, it defaults to 1.0.
/// Friction layers are summed together to produce the final friction
/// layer that is applied to the cost layer. A clamp is applied to the
/// final friction layer to ensure that no values are below -1.0, which
/// would lead to negative routing costs.
struct FrictionLayer {
    multiplier_layer: String,
    #[builder(setter(strip_option), default)]
    multiplier_scalar: Option<f32>,
}

impl CostFunction {
    /// Create a new cost function from a JSON string (reVX format)
    ///
    /// # Arguments
    /// `json`: A JSON string representing the cost function with the format
    ///         used by reVX.
    ///
    /// # Returns
    /// A `CostFunction` object.
    ///
    /// The JSON pattern used by reVX was the following:
    /// ```json
    /// {"cost_layers": [
    ///   {"layer_name": "A"},
    ///   {"layer_name": "A", "multiplier_scalar": 2, "multiplier_layer": "B"}
    ///   ]}
    /// ```
    pub(super) fn from_json(json: &str) -> Result<Self> {
        trace!("Parsing cost definition from json: {}", json);
        let cost = serde_json::from_str(json).unwrap();
        Ok(cost)
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
        final_cost_layer.mapv_inplace(|v| {
            if v <= 0_f32 {
                if self.ignore_invalid_costs {
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
    cost
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

    friction
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
    use super::*;

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
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dataset::{make_lazy_subset_for_tests, samples};
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
        assert_eq!(cost.cost_layers[1].layer_name, "B".to_string());
        assert_eq!(cost.cost_layers[1].multiplier_scalar, Some(100.0));
        assert_eq!(cost.cost_layers[1].is_invariant, None);
        assert_eq!(cost.cost_layers[2].layer_name, "A".to_string());
        assert_eq!(cost.cost_layers[2].multiplier_layer, Some("B".to_string()));
        assert_eq!(cost.cost_layers[2].is_invariant, None);
        assert_eq!(cost.cost_layers[3].layer_name, "C".to_string());
        assert_eq!(cost.cost_layers[3].multiplier_layer, Some("A".to_string()));
        assert_eq!(cost.cost_layers[3].multiplier_scalar, Some(2.0));
        assert_eq!(cost.cost_layers[3].is_invariant, None);
        assert_eq!(cost.cost_layers[4].layer_name, "C".to_string());
        assert_eq!(cost.cost_layers[4].multiplier_layer, None);
        assert_eq!(cost.cost_layers[4].multiplier_scalar, Some(100.0));
        assert_eq!(cost.cost_layers[4].is_invariant, Some(true));
    }

    #[test]
    fn test_friction_only_returns_zeros() {
        let (_tmp, mut features) = make_features_for_costs_tests();

        // friction-only (no `layer_name`) should return an empty cost array (zeros)
        let json = r#"
        {
            "cost_layers": [],
            "friction_layers": [
                {"multiplier_layer": "B", "multiplier_scalar": -3.0}
            ]
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
            "cost_layers": [
                {"layer_name": "A"}
            ],
            "friction_layers": [
                {"multiplier_layer": "B", "multiplier_scalar": -3.0}
            ]
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
}
