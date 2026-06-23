//! Cost function inputs and parsing

use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use std::collections::HashMap;
use tracing::trace;

use crate::cost::CostFunction;
use crate::cost::components::{
    BarrierLayer, BarrierOperator, CostLayer, DriverRuleSet, DriverZoneRule, FrictionLayer,
    TransitionCostTable,
};
use crate::error::Result;

fn true_option() -> bool {
    true
}

#[derive(Clone, Debug, serde::Deserialize)]
pub(super) struct CostFunctionInput {
    #[serde(default)]
    routing_options: RoutingOptionsInput,
    #[serde(default)]
    drivers: DriversConfig,
    #[serde(default)]
    transition_costs: TransitionCostsConfig,
    #[serde(default = "true_option")]
    invalid_costs_block_routing: bool,
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(untagged)]
pub(super) enum RoutingOptionsInput {
    Definitions(Map<String, Value>),
    #[default]
    Missing,
}

#[derive(Clone, Debug)]
pub(super) struct RoutingOptionEntry<TDefinition> {
    pub(super) name: String,
    pub(super) index: u32,
    pub(super) definition: TDefinition,
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
pub(super) struct RoutingOptionDefinition {
    #[serde(default)]
    cost_layers: Vec<CostLayer>,
    #[serde(default)]
    friction_layers: Vec<FrictionLayerInput>,
    #[serde(default)]
    barrier_layers: Vec<BarrierLayer>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct RoutingOptionLayerSet {
    pub(super) cost_layers: Vec<CostLayer>,
    pub(super) friction_layers: Vec<FrictionLayer>,
    pub(super) barrier_layers: Vec<BarrierLayer>,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct FrictionLayerInput {
    #[serde(default)]
    layer_name: Option<String>,
    #[serde(default)]
    multiplier_layer: Option<String>,
    #[serde(default)]
    multiplier_scalar: Option<f32>,
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
pub(super) struct TransitionCostsConfig {
    #[serde(default)]
    pub(super) default: f32,
    #[serde(default)]
    pub(super) pairwise: Vec<TransitionCostRule>,
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
pub(super) struct DriversConfig {
    #[serde(default)]
    pub(super) default: HashMap<String, DriverRuleValue>,
    #[serde(default)]
    pub(super) zones: Vec<DriverZoneConfig>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub(super) struct DriverZoneConfig {
    pub(super) layer_name: String,
    pub(super) mask_operator: BarrierOperator,
    pub(super) mask_threshold: f32,
    #[serde(flatten)]
    pub(super) options: HashMap<String, DriverRuleValue>,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(untagged)]
pub(super) enum DriverRuleValue {
    Keyword(String),
    Multiplier(f32),
}

#[derive(Clone, Debug, serde::Deserialize)]
pub(super) struct TransitionCostRule {
    pub(super) from: TransitionOptionRef,
    pub(super) to: TransitionOptionRef,
    pub(super) cost: f32,
    #[serde(default)]
    pub(super) applies_bidirectionally: bool,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(untagged)]
pub(super) enum TransitionOptionRef {
    Index(u32),
    Name(String),
}

impl RoutingOptionsInput {
    pub(super) fn into_entries<TDefinition>(self) -> Result<Vec<RoutingOptionEntry<TDefinition>>>
    where
        TDefinition: DeserializeOwned,
    {
        match self {
            Self::Definitions(definitions) => {
                if definitions.is_empty() {
                    return Err(crate::error::Error::Undefined(
                        "routing_options must define at least one routing option".to_string(),
                    ));
                }

                let mut entries = Vec::with_capacity(definitions.len());
                for (name, value) in definitions {
                    let definition = serde_json::from_value(value)
                        .map_err(|error| crate::error::Error::Undefined(error.to_string()))?;
                    entries.push(RoutingOptionEntry {
                        name,
                        index: entries.len() as u32,
                        definition,
                    });
                }

                Ok(entries)
            }
            Self::Missing => Err(crate::error::Error::Undefined(
                "routing_options must be provided and all layer definitions must be nested under a routing option"
                    .to_string(),
            )),
        }
    }
}

impl TryFrom<CostFunctionInput> for CostFunction {
    type Error = crate::error::Error;

    fn try_from(input: CostFunctionInput) -> Result<Self> {
        let CostFunctionInput {
            routing_options,
            drivers,
            transition_costs,
            invalid_costs_block_routing,
        } = input;
        let mut cost_layers = Vec::new();
        let mut friction_layers = Vec::new();
        let mut barrier_layers = Vec::new();
        let mut routing_option_names = Vec::new();

        for RoutingOptionEntry {
            name,
            index,
            definition,
        } in routing_options.into_entries::<RoutingOptionDefinition>()?
        {
            let RoutingOptionLayerSet {
                cost_layers: option_cost_layers,
                friction_layers: option_friction_layers,
                barrier_layers: option_barrier_layers,
            } = definition.into_layers(index)?;
            routing_option_names.push(name);
            cost_layers.extend(option_cost_layers);
            friction_layers.extend(option_friction_layers);
            barrier_layers.extend(option_barrier_layers);
        }

        let drivers = drivers.into_rule_set(&routing_option_names)?;
        let transition_costs = transition_costs.into_table(&routing_option_names)?;

        Ok(CostFunction::from_input_parts(
            cost_layers,
            friction_layers,
            barrier_layers,
            routing_option_names,
            drivers,
            transition_costs,
            invalid_costs_block_routing,
        ))
    }
}

impl RoutingOptionDefinition {
    pub(super) fn into_layers(self, option: u32) -> Result<RoutingOptionLayerSet> {
        Ok(RoutingOptionLayerSet {
            cost_layers: self
                .cost_layers
                .into_iter()
                .map(|layer| layer.with_option(option))
                .collect(),
            friction_layers: self
                .friction_layers
                .into_iter()
                .map(|layer| layer.into_layer(option))
                .collect::<Result<_>>()?,
            barrier_layers: self
                .barrier_layers
                .into_iter()
                .map(|layer| layer.with_option(option))
                .collect(),
        })
    }
}

impl DriversConfig {
    pub(super) fn into_rule_set(self, routing_options: &[String]) -> Result<DriverRuleSet> {
        let mut default = vec![Some(1.0); routing_options.len()];

        for (name, value) in self.default {
            let option = resolve_routing_option(&name, routing_options, "drivers.default")?;
            default[option as usize] = resolve_driver_rule_value(&value)?;
        }

        let mut zones = Vec::with_capacity(self.zones.len());
        for zone in self.zones {
            let mut options = HashMap::new();

            for (name, value) in zone.options {
                let option = resolve_routing_option(&name, routing_options, "drivers.zones")?;
                options.insert(option, resolve_driver_rule_value(&value)?);
            }

            zones.push(DriverZoneRule::new(
                zone.layer_name,
                zone.mask_operator,
                zone.mask_threshold,
                options,
            ));
        }

        Ok(DriverRuleSet::new(default, zones))
    }
}

impl TransitionCostsConfig {
    pub(super) fn into_table(self, routing_options: &[String]) -> Result<TransitionCostTable> {
        let mut pairwise = HashMap::new();

        for rule in self.pairwise {
            let from = resolve_transition_option(&rule.from, routing_options)?;
            let to = resolve_transition_option(&rule.to, routing_options)?;
            pairwise.insert((from, to), rule.cost);
            if rule.applies_bidirectionally {
                pairwise.insert((to, from), rule.cost);
            }
        }

        trace!(
            "resolved transition cost table with default cost {} and pairwise costs: {:#?}",
            self.default, pairwise
        );

        Ok(TransitionCostTable::new(self.default, pairwise))
    }
}

impl FrictionLayerInput {
    fn into_layer(self, option: u32) -> Result<FrictionLayer> {
        let multiplier_layer = self.layer_name.or(self.multiplier_layer).ok_or_else(|| {
            crate::error::Error::Undefined(
                "friction layer requires layer_name or multiplier_layer".to_string(),
            )
        })?;

        Ok(FrictionLayer::new(
            multiplier_layer,
            self.multiplier_scalar,
            option,
        ))
    }
}

fn resolve_routing_option(name: &str, routing_options: &[String], context: &str) -> Result<u32> {
    routing_options
        .iter()
        .position(|option_name| option_name == name)
        .map(|index| index as u32)
        .ok_or_else(|| {
            crate::error::Error::Undefined(format!("unknown routing option {name:?} in {context}"))
        })
}

fn resolve_transition_option(
    option: &TransitionOptionRef,
    routing_options: &[String],
) -> Result<u32> {
    match option {
        TransitionOptionRef::Index(index) => Ok(*index),
        TransitionOptionRef::Name(name) => {
            resolve_routing_option(name, routing_options, "transition_costs")
        }
    }
}

fn resolve_driver_rule_value(value: &DriverRuleValue) -> Result<Option<f32>> {
    match value {
        DriverRuleValue::Multiplier(multiplier) => Ok(Some(*multiplier)),
        DriverRuleValue::Keyword(keyword) if keyword == "excluded" => Ok(None),
        DriverRuleValue::Keyword(keyword) => Err(crate::error::Error::Undefined(format!(
            "unsupported driver rule value {keyword:?}"
        ))),
    }
}
