//! # Support to benchmark ReVRt
//!

use crate::ArrayIndex;
use crate::cost::CostFunction;
use crate::routing::Routing;

#[inline]
/// A public interface to run benchmarks
///
/// This function is intended for use during development only. It will
/// eventually be replaced by a builder, thus more flexible and usable
/// for other purposes.
pub fn bench_minimalist(
    features_path: std::path::PathBuf,
    start: Vec<ArrayIndex>,
    end: Vec<ArrayIndex>,
) {
    // temporary solution for a cost function until we have a builder
    let cost_json = r#"{
        "cost_layers": [
            {"layer_name": "A"},
            {"layer_name": "B", "multiplier_scalar": 100},
            {"layer_name": "A", "multiplier_layer": "B"},
            {"layer_name": "C", "multiplier_layer": "A", "multiplier_scalar": 2}
        ]
    }"#
    .to_string();
    let cost_function = CostFunction::from_json(&cost_json).unwrap();

    let mut simulation: Routing = Routing::new(&features_path, cost_function, 1_000).unwrap();
    let solutions = simulation.compute(&start, end).collect::<Vec<_>>();
    assert!(!solutions.is_empty(), "No solutions found");
}
