use revrt::resolve;
use std::path::PathBuf;
use test_case::test_case;

const TEST_DATA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/data/utilities/transmission_layers.zarr"
);

#[test_case("dijkstra"; "dijkstra")]
#[test_case("long-range-astar"; "long-range-astar")]
#[test_case("long-range-dijkstra"; "long-range")]
#[test_case("bidirectional-long-range-dijkstra"; "bidirectional-long-range")]
fn basic_routing_in_data(algorithm: &str) {
    let layers_path = PathBuf::from(TEST_DATA);
    let start = &revrt::ArrayIndex::new(10, 10);
    let end = vec![revrt::ArrayIndex::new(20, 20)];
    let result = resolve(
        layers_path.to_str().expect("test data path is valid UTF-8"),
        r#"{"cost_layers": [{"layer_name": "tie_line_costs_102MW"}]}"#,
        algorithm,
        std::slice::from_ref(start),
        end,
        None,
        250_000_000,
    )
    .unwrap();
    dbg!(&result);
    assert_eq!(result.len(), 1);
    assert!(result[0].route().len() > 1);
    assert!(result[0].total_cost() > &0.);
}

#[test_case("dijkstra"; "dijkstra")]
#[test_case("long-range-astar"; "long-range-astar")]
#[test_case("long-range-dijkstra"; "long-range")]
#[test_case("bidirectional-long-range-dijkstra"; "bidirectional-long-range")]
fn basic_routing_in_data_with_friction(algorithm: &str) {
    let layers_path = PathBuf::from(TEST_DATA);
    let start = &revrt::ArrayIndex::new(10, 10);
    let end = vec![revrt::ArrayIndex::new(20, 20)];
    let result = resolve(
        layers_path.to_str().expect("test data path is valid UTF-8"),
        r#"{
            "cost_layers": [{"layer_name": "tie_line_costs_102MW"}],
            "friction_layers": [
                {"multiplier_layer": "transmission_barrier", "multiplier_scalar": 100}
            ]
        }"#,
        algorithm,
        std::slice::from_ref(start),
        end,
        None,
        250_000_000,
    )
    .unwrap();
    dbg!(&result);
    assert_eq!(result.len(), 1);
    assert!(result[0].route().len() > 1);
    assert!(result[0].total_cost() > &0.);
}
