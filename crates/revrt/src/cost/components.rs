//! Cost function components

use core::f32;
use derive_builder::Builder;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub(crate) struct TransitionCostTable {
    pub(super) default: f32,
    pub(super) pairwise: HashMap<(u32, u32), f32>,
}

impl TransitionCostTable {
    pub(super) fn new(default: f32, pairwise: HashMap<(u32, u32), f32>) -> Self {
        Self { default, pairwise }
    }

    pub(crate) fn cost(&self, from: u32, to: u32) -> f32 {
        self.pairwise
            .get(&(from, to))
            .copied()
            .unwrap_or(self.default)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct DriverRuleSet {
    pub(super) default: Vec<Option<f32>>,
    pub(super) zones: Vec<DriverZoneRule>,
}

impl DriverRuleSet {
    pub(super) fn new(default: Vec<Option<f32>>, zones: Vec<DriverZoneRule>) -> Self {
        Self { default, zones }
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.zones.is_empty()
            && self
                .default
                .iter()
                .all(|multiplier| matches!(multiplier, Some(value) if *value == 1.0))
    }

    pub(crate) fn multiplier<F>(&self, option: u32, mut layer_value: F) -> Option<f32>
    where
        F: FnMut(&str) -> Option<f32>,
    {
        let mut multiplier = self
            .default
            .get(option as usize)
            .copied()
            .unwrap_or(Some(1.0));

        for zone in &self.zones {
            let Some(value) = layer_value(&zone.layer_name) else {
                continue;
            };

            if zone.matches(value)
                && let Some(zone_multiplier) = zone.options.get(&option)
            {
                multiplier = *zone_multiplier; // TODO: This is not quite right. We want replacement. Maybe error if two zones overlap?
            }
        }

        multiplier
    }
}

#[derive(Clone, Debug)]
pub(super) struct DriverZoneRule {
    pub(super) layer_name: String,
    pub(super) operator: BarrierOperator,
    pub(super) threshold: f32,
    pub(super) options: HashMap<u32, Option<f32>>,
}

impl DriverZoneRule {
    pub(super) fn new(
        layer_name: String,
        operator: BarrierOperator,
        threshold: f32,
        options: HashMap<u32, Option<f32>>,
    ) -> Self {
        Self {
            layer_name,
            operator,
            threshold,
            options,
        }
    }

    fn matches(&self, value: f32) -> bool {
        match self.operator {
            BarrierOperator::NotEqual => value != self.threshold,
            BarrierOperator::GreaterThan => value > self.threshold,
            BarrierOperator::GreaterThanOrEqual => value >= self.threshold,
            BarrierOperator::LessThan => value < self.threshold,
            BarrierOperator::LessThanOrEqual => value <= self.threshold,
            BarrierOperator::Equal => value == self.threshold,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
pub(super) enum BarrierOperator {
    #[serde(rename = "ne")]
    NotEqual,
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

pub(super) type MultiplierLayers = Vec<String>;

#[derive(Builder, Clone, Debug, serde::Deserialize)]
/// A cost layer
///
/// Each cost layer is a raster dataset, i.e. a regular grid, composed by
/// operating on input features. Following the original `revX` structure,
/// the possible compositions are limited to combinations of the relation
/// `weight * layer_name * product(multiplier_layers)`, where the `weight` and
/// the `multiplier_layers` are optional. Each layer can also be marked as invariant,
/// meaning that its value does not get scaled by the distance traveled
/// through the cell. Instead, the value of the layer is added once, right
/// when the path enters the cell.
pub(super) struct CostLayer {
    pub(crate) layer_name: String,
    #[builder(setter(strip_option), default)]
    pub(super) multiplier_scalar: Option<f32>,
    #[builder(setter(strip_option, into), default)]
    pub(super) multiplier_layer: Option<MultiplierLayers>,
    #[builder(setter(strip_option), default)]
    pub(super) is_invariant: Option<bool>,
    #[builder(default, setter(skip))]
    #[serde(skip)]
    pub(super) option: u32,
}

impl CostLayer {
    pub(super) fn with_option(mut self, option: u32) -> Self {
        self.option = option;
        self
    }
}

#[derive(Builder, Clone, Debug, serde::Deserialize)]
/// A friction layer
///
/// Each friction layer is a raster dataset, i.e. a regular grid, that
/// represents multipliers that should be applied to the cost routing
/// layer. These multipliers affect the output route but will not be
/// reported in the output cost. Each friction layer is defined by a
/// one or more `multiplier_layer` inputs and an optional `multiplier_scalar`.
/// The friction value at each cell is computed as
/// `product(multiplier_layers) * multiplier_scalar`.
/// If the `multiplier_scalar` is not provided, it defaults to 1.0.
/// Friction layers are summed together to produce the final friction
/// layer that is applied to the cost layer. A clamp is applied to the
/// final friction layer to ensure that no values are below -1.0, which
/// would lead to negative routing costs.
pub(super) struct FrictionLayer {
    pub(super) multiplier_layers: MultiplierLayers,
    #[builder(setter(strip_option), default)]
    pub(super) multiplier_scalar: Option<f32>,
    #[serde(skip)]
    pub(super) option: u32,
}

impl FrictionLayer {
    pub(super) fn new(
        multiplier_layers: MultiplierLayers,
        multiplier_scalar: Option<f32>,
        option: u32,
    ) -> Self {
        Self {
            multiplier_layers,
            multiplier_scalar,
            option,
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
pub(crate) struct BarrierLayer {
    pub(crate) layer_name: String,
    pub(super) barrier_operator: BarrierOperator,
    pub(super) barrier_threshold: f32,
    pub(super) barrier_importance: Option<u32>,
    #[serde(skip)]
    pub(super) option: u32,
}

impl BarrierLayer {
    pub(super) fn importance(&self) -> Option<u32> {
        self.barrier_importance
    }

    pub(super) fn with_option(mut self, option: u32) -> Self {
        self.option = option;
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cost_layer_builder_sets_configured_values() {
        let layer = CostLayerBuilder::default()
            .layer_name("A".to_string())
            .multiplier_scalar(2.0)
            .multiplier_layer(vec!["B".to_string()])
            .is_invariant(false)
            .build()
            .unwrap();

        assert_eq!(layer.layer_name, "A".to_string());
        assert_eq!(layer.multiplier_scalar, Some(2.0));
        assert_eq!(layer.multiplier_layer, Some(vec!["B".to_string()]));
        assert_eq!(layer.is_invariant, Some(false));
        assert_eq!(layer.option, 0);
    }

    #[test]
    fn cost_layer_builder_applies_defaults() {
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

    #[test]
    fn driver_rule_set_supports_zone_overrides_and_exclusions() {
        let driver_rules = DriverRuleSet::new(
            vec![Some(1.0), None],
            vec![DriverZoneRule::new(
                "zone".to_string(),
                BarrierOperator::Equal,
                1.0,
                HashMap::from([(0, Some(10.0)), (1, Some(1.0))]),
            )],
        );

        assert_eq!(driver_rules.multiplier(0, |_| Some(0.0)), Some(1.0));
        assert_eq!(driver_rules.multiplier(1, |_| Some(0.0)), None);
        assert_eq!(driver_rules.multiplier(0, |_| Some(1.0)), Some(10.0));
        assert_eq!(driver_rules.multiplier(1, |_| Some(1.0)), Some(1.0));
    }
}
