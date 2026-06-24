"""reVrt tests for routing one point to many endpoints"""

import json
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
from rasterio.transform import from_origin

from revrt.utilities import LayeredFile
from revrt.routing.base import (
    RouteMetrics,
    RoutingLayerManager,
    RoutingScenario,
    _friction_layers_for_rust,
    _transition_cost_lookup,
)
from revrt.routing.utilities import compute_lens
from revrt.exceptions import revrtKeyError
from revrt.warn import revrtWarning, revrtDeprecationWarning


@pytest.fixture(scope="module")
def sample_layered_data(tmp_path_factory):
    """Sample layered data files to use across tests"""
    data_dir = tmp_path_factory.mktemp("routing_data")

    layered_fp = data_dir / "test_layered.zarr"
    layer_file = LayeredFile(layered_fp)

    height, width = (7, 8)
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    layer_1 = np.array(
        [
            [
                [7, 7, 8, 0, 9, 9, 9, 0],
                [8, 1, 2, 2, 9, 9, 9, 0],
                [9, 1, 3, 3, 9, 1, 2, 3],
                [9, 1, 2, 1, 9, 1, 9, 0],
                [9, 9, 9, 1, 9, 1, 9, 0],
                [9, 9, 9, 1, 1, 1, 9, 0],
                [9, 9, 9, 9, 9, 9, 9, 0],
            ]
        ],
        dtype=np.float32,
    )

    layer_2 = np.array(
        [
            [
                [8, 7, 6, 5, 5, 6, 7, 9],
                [7, 1, 1, 2, 3, 3, 2, 8],
                [6, 2, 9, 6, 5, 2, 1, 7],
                [7, 3, 8, 1, 2, 3, 2, 6],
                [8, 4, 7, 2, 8, 4, 3, 5],
                [9, 5, 6, 3, 4, 4, 3, 4],
                [9, 6, 7, 4, 5, 5, 4, 3],
            ]
        ],
        dtype=np.float32,
    )

    layer_3 = np.array(
        [
            [
                [6, 6, 6, 6, 6, 7, 8, 9],
                [5, 2, 2, 3, 4, 5, 6, 8],
                [4, 3, 7, 7, 6, 4, 5, 7],
                [5, 4, 6, 2, 3, 4, 4, 6],
                [6, 5, 5, 3, 7, 5, 5, 5],
                [7, 6, 6, 4, 5, 5, 4, 4],
                [8, 7, 7, 5, 6, 5, 4, 3],
            ]
        ],
        dtype=np.float32,
    )

    layer_4 = np.array(
        [
            [
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ],
        dtype=np.float32,
    )

    layer_5 = np.array(
        [
            [
                [0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ],
        dtype=np.float32,
    )

    # fmt: off
    layer_6 = np.array(
        [
            [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1, 0],
                [0, -1, -1, -1, -1, -1, -1, 3],
                [1, -1, -1, -1, -1, -1, -1, 1],
                [1, -1, -1, -1, -1, -1, -1, 1],
                [1, -1, -1, -1, -1, -1, -1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1],
            ]
        ],
        dtype=np.float32,
    )

    # fmt: off
    layer_7 = np.array(
        [
            [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [ 8, -1, -1, -1, -1,  4, -1, -1],  # noqa: E201, E241
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        ],
        dtype=np.float32,
    )

    for ind, routing_layer in enumerate(
        [
            layer_1,
            layer_2,
            layer_3,
            layer_4,
            layer_5,
            layer_6,
            layer_7,
        ],
        start=1,
    ):
        da = xr.DataArray(
            routing_layer,
            dims=("band", "y", "x"),
            coords={"y": y_coords, "x": x_coords},
        )
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.write_transform(transform)

        geotiff_fp = data_dir / f"layer_{ind}.tif"
        da.rio.to_raster(geotiff_fp, driver="GTiff")

        layer_file.write_geotiff_to_file(
            geotiff_fp, f"layer_{ind}", overwrite=True
        )
    return layered_fp


def test_routing_scenario_serializes_multi_option_config(sample_layered_data):
    """RoutingScenario emits the Rust multi-option schema"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "overhead": {
                "cost_layers": [
                    {
                        "layer_name": "layer_1",
                        "multiplier_scalar": 2,
                        "include_in_report": False,
                        "include_in_final_cost": False,
                    }
                ],
                "cost_multiplier_layer": "layer_6",
                "friction_layers": [
                    {
                        "mask": "layer_4",
                        "multiplier_scalar": 1.1,
                        "include_in_report": False,
                    }
                ],
                "barrier_layers": [
                    {
                        "layer_name": "layer_5",
                        "where": "==1",
                    }
                ],
            },
            "underground": {
                "cost_layers": [{"layer_name": "layer_2"}],
            },
        },
        drivers={
            "default": {"overhead": 1, "underground": "excluded"},
            "zones": [
                {
                    "layer_name": "layer_5",
                    "where": "==1",
                    "overhead": "excluded",
                    "underground": 2,
                }
            ],
        },
        transition_costs={
            "default": 0,
            "pairwise": [
                {
                    "between": ["overhead", "underground"],
                    "cost": 3,
                }
            ],
        },
        invalid_costs_block_routing=False,
    )

    payload = json.loads(scenario.cost_function_json)

    assert payload["invalid_costs_block_routing"] is False
    assert payload["drivers"] == {
        "default": {"overhead": 1, "underground": "excluded"},
        "zones": [
            {
                "layer_name": "layer_5",
                "mask_operator": "eq",
                "mask_threshold": 1.0,
                "overhead": "excluded",
                "underground": 2,
            }
        ],
    }
    assert payload["transition_costs"] == scenario.transition_costs
    assert set(payload["routing_options"]) == {"overhead", "underground"}
    assert payload["routing_options"]["overhead"]["cost_layers"] == [
        {"layer_name": "layer_1", "multiplier_scalar": 2}
    ]
    assert (
        payload["routing_options"]["overhead"]["cost_multiplier_layer"]
        == "layer_6"
    )
    assert payload["routing_options"]["overhead"]["friction_layers"] == [
        {"multiplier_layer": "layer_4", "multiplier_scalar": 1.1}
    ]
    assert payload["routing_options"]["overhead"]["barrier_layers"] == [
        {
            "layer_name": "layer_5",
            "barrier_operator": "eq",
            "barrier_threshold": 1.0,
        }
    ]
    assert payload["routing_options"]["underground"]["cost_layers"] == [
        {"layer_name": "layer_2"}
    ]


def test_routing_scenario_preserves_default_only_drivers(sample_layered_data):
    """Default-only driver rules are serialized unchanged"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "overhead": {
                "cost_layers": [{"layer_name": "layer_1"}],
            }
        },
        drivers={"default": {"overhead": 1}},
    )

    payload = json.loads(scenario.cost_function_json)

    assert payload["drivers"] == {"default": {"overhead": 1}}


def test_multi_option_route_metrics_use_option_layers(
    sample_layered_data,
):
    """Multi-option routes report per-option and transition costs"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "overhead": {
                "cost_layers": [
                    {"layer_name": "layer_1"},
                    {"layer_name": "layer_3", "is_invariant": True},
                ],
            },
            "underground": {
                "cost_layers": [
                    {"layer_name": "layer_2"},
                    {"layer_name": "layer_3", "is_invariant": True},
                ],
            },
        },
        drivers={"default": {"overhead": 1, "underground": 10}},
        transition_costs={
            "default": 0,
            "pairwise": [
                {
                    "between": ["overhead", "underground"],
                    "cost": 3.5,
                }
            ],
        },
        invalid_costs_block_routing=True,
    )
    routing_layers = RoutingLayerManager(scenario).build()
    try:
        metrics = RouteMetrics(
            routing_layers,
            route=[
                (1, 1, "overhead"),
                (1, 2, "overhead"),
                (1, 2, "underground"),
                (1, 3, "underground"),
            ],
            optimized_objective=42.5,
        )
        result = metrics.compute()

        # expected_cost = 3.0 + 9.0 + 3.5
        expected_cost = (
            # layer 1 costs
            1.5
            # layer 1 inv costs
            + 2
            + 2
            # transition costs
            + 3.5
            # layer 2 costs
            + 1.5
            # layer 2 inv costs
            + 2
            + 3
        )
        assert metrics.cost == pytest.approx(expected_cost)
        assert result["cost"] == pytest.approx(expected_cost)
        assert result["optimized_objective"] == pytest.approx(42.5)
    finally:
        routing_layers.close()


def test_transition_cost_lookup_uses_between_rules_bidirectionally():
    """Pairwise transition rules apply in both directions"""

    default_cost, pairwise_costs = _transition_cost_lookup(
        {
            "default": 1,
            "pairwise": [
                {
                    "between": ["overhead", "underground"],
                    "cost": 3.5,
                }
            ],
        }
    )

    assert default_cost == 1
    assert pairwise_costs[("overhead", "overhead")] == 0
    assert pairwise_costs[("underground", "underground")] == 0
    assert pairwise_costs[("overhead", "underground")] == pytest.approx(3.5)
    assert pairwise_costs[("underground", "overhead")] == pytest.approx(3.5)


def test_routing_scenario_normalizes_algorithm(sample_layered_data):
    """RoutingScenario normalizes supported algorithm aliases"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm="long-range-dijkstra",
    )

    # assert scenario.algorithm is RoutingAlgorithm.LONG_RANGE
    assert "algorithm: long-range-dijkstra" in repr(scenario)


def test_routing_scenario_forwards_a_star_alias(sample_layered_data):
    """RoutingScenario preserves the A* alias for Rust parsing"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm="a-star",
    )

    assert "algorithm: a-star" in repr(scenario)


def test_tracked_layers_apply_multiplier_scalar_and_layer(
    sample_layered_data,
):
    """Tracked layer aggregates use scaled values before aggregation"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        tracked_layers=[
            {
                "layer_name": "layer_1",
                "multiplier_scalar": 2,
                "agg_method": "max",
            },
            {
                "layer_name": "layer_2",
                "multiplier_layer": "layer_3",
                "agg_method": "mean",
            },
        ],
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        result = RouteMetrics(
            routing_layers,
            route=[(1, 1, "default"), (2, 1, "default")],
            optimized_objective=0.0,
        ).compute()

        assert result["layer_1_default_max"] == pytest.approx(2.0)
        assert result["layer_2_default_mean"] == pytest.approx(4.0)
    finally:
        routing_layers.close()


def test_user_tracked_layers_are_built_and_scoped_per_option(
    sample_layered_data,
):
    """User tracked layers are added once per option and filtered"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]},
            "alt": {"cost_layers": [{"layer_name": "layer_1"}]},
        },
        tracked_layers=[{"layer_name": "layer_1", "agg_method": "mean"}],
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        tracked_names = {layer.name for layer in routing_layers.tracked_layers}
        assert "layer_1_default_mean" not in tracked_names
        assert {"layer_1_default", "layer_1_alt"}.issubset(tracked_names)

        result = RouteMetrics(
            routing_layers,
            route=[
                (1, 1, "default"),
                (1, 2, "default"),
                (1, 3, "alt"),
            ],
            optimized_objective=0.0,
        ).compute()

        assert result["layer_1_default_mean"] == pytest.approx(1.5)
        assert result["layer_1_alt_mean"] == pytest.approx(2.0)
    finally:
        routing_layers.close()


def test_option_bound_characterized_layers_only_use_matching_segments(
    sample_layered_data,
):
    """Per-option trackers reuse only their matching route cells"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]},
            "alt": {"cost_layers": [{"layer_name": "layer_1"}]},
        },
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        result = RouteMetrics(
            routing_layers,
            route=[
                (1, 1, "default"),
                (1, 2, "default"),
                (1, 3, "alt"),
            ],
            optimized_objective=0.0,
        ).compute()

        assert result["layer_1_default_cost"] == pytest.approx(2.5)
        assert result["layer_1_alt_cost"] == pytest.approx(1.0)
        assert result["layer_1_default_length_km"] == pytest.approx(0.0015)
        assert result["layer_1_alt_length_km"] == pytest.approx(0.0005)
    finally:
        routing_layers.close()


def test_routing_scenario_repr_contains_fields(sample_layered_data):
    """RoutingScenario repr surfaces configured layer metadata"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {
                        "layer_name": "layer_1",
                        "multiplier_layer": "layer_3",
                        "multiplier_scalar": 1.5,
                    }
                ],
                "friction_layers": [{"mask": "layer_2"}],
            }
        },
    )

    representation = repr(scenario)

    assert "layer_1" in representation
    assert "layer_2" in representation
    assert "'multiplier_scalar': 1.5" in representation


def test_cost_multiplier_layer_and_scalar_applied(sample_layered_data):
    """Cost multipliers scale base costs before routing aggregation"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {
                        "layer_name": "layer_1",
                        "multiplier_layer": "layer_3",
                        "multiplier_scalar": 2.0,
                    }
                ]
            }
        },
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        cost_val = (
            routing_layers.costs["default"].isel(y=1, x=1).compute().item()
        )
        layer_one = (
            routing_layers._layer_fh["layer_1"]
            .isel(band=0, y=1, x=1)
            .compute()
            .item()
        )
        layer_three = (
            routing_layers._layer_fh["layer_3"]
            .isel(band=0, y=1, x=1)
            .compute()
            .item()
        )
        expected = (
            layer_one
            * layer_three
            * scenario.routing_options["default"]["cost_layers"][0][
                "multiplier_scalar"
            ]
        )

        assert cost_val == pytest.approx(expected)
    finally:
        routing_layers.close()


def test_option_cost_multiplier_layer_applies_to_invariant_costs(
    sample_layered_data,
):
    """Per-option multiplier layers scale invariant costs too"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {"layer_name": "layer_1"},
                    {"layer_name": "layer_2", "is_invariant": True},
                ],
                "cost_multiplier_layer": "layer_3",
            }
        },
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        cost_val = (
            routing_layers.costs["default"].isel(y=1, x=1).compute().item()
        )
        li_cost_val = (
            routing_layers.li_costs["default"].isel(y=1, x=1).compute().item()
        )
        layer_one = (
            routing_layers._layer_fh["layer_1"]
            .isel(band=0, y=1, x=1)
            .compute()
            .item()
        )
        layer_two = (
            routing_layers._layer_fh["layer_2"]
            .isel(band=0, y=1, x=1)
            .compute()
            .item()
        )
        multiplier = (
            routing_layers._layer_fh["layer_3"]
            .isel(band=0, y=1, x=1)
            .compute()
            .item()
        )

        assert cost_val == pytest.approx(layer_one * multiplier)
        assert li_cost_val == pytest.approx(layer_two * multiplier)
    finally:
        routing_layers.close()


def test_option_cost_multipliers_apply_to_all_cost_buckets(
    sample_layered_data,
):
    """Option multipliers scale tracked, invariant, and untracked costs"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {"layer_name": "layer_1"},
                    {"layer_name": "layer_2", "is_invariant": True},
                    {
                        "layer_name": "layer_4",
                        "include_in_final_cost": False,
                    },
                ],
                "cost_multiplier_scalar": 2.0,
                "cost_multiplier_layer": "layer_3",
            }
        },
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        row = 1
        col = 3
        cost_val = (
            routing_layers.costs["default"].isel(y=row, x=col).compute().item()
        )
        li_cost_val = (
            routing_layers.li_costs["default"]
            .isel(y=row, x=col)
            .compute()
            .item()
        )
        final_cost_val = (
            routing_layers.final_routing_layers["default"]
            .isel(y=row, x=col)
            .compute()
            .item()
        )

        multiplier = (
            routing_layers._layer_fh["layer_3"]
            .isel(band=0, y=row, x=col)
            .compute()
            .item()
        ) * scenario.routing_options["default"]["cost_multiplier_scalar"]

        expected_cost = (
            routing_layers._layer_fh["layer_1"]
            .isel(band=0, y=row, x=col)
            .compute()
            .item()
            * multiplier
        )
        expected_li_cost = (
            routing_layers._layer_fh["layer_2"]
            .isel(band=0, y=row, x=col)
            .compute()
            .item()
            * multiplier
        )
        expected_untracked_cost = (
            routing_layers._layer_fh["layer_4"]
            .isel(band=0, y=row, x=col)
            .compute()
            .item()
            * multiplier
        )

        assert cost_val == pytest.approx(expected_cost)
        assert li_cost_val == pytest.approx(expected_li_cost)
        assert final_cost_val == pytest.approx(
            expected_cost + expected_li_cost + expected_untracked_cost
        )
        assert final_cost_val - cost_val - li_cost_val == pytest.approx(
            expected_untracked_cost
        )
    finally:
        routing_layers.close()


def test_length_invariant_layer_costs_ignore_path_length(sample_layered_data):
    """Length invariant cost layers ignore per-cell distances"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {"layer_name": "layer_1"},
                    {"layer_name": "layer_2", "is_invariant": True},
                ]
            }
        },
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        route = [(1, 1), (1, 2)]
        result = RouteMetrics(
            routing_layers,
            [(1, 1, "default"), (1, 2, "default")],
            optimized_objective=0.0,
        ).compute()

        layer_two = (
            routing_layers._layer_fh["layer_2"]
            .isel(band=0)
            .compute()
            .to_numpy()
        )
        expected = sum(layer_two[row, col] for row, col in route)

        assert result["layer_2_default_cost"] == pytest.approx(expected)
    finally:
        routing_layers.close()


def test_tracked_layers_invalid_configs_warn(
    sample_layered_data, assert_message_was_logged
):
    """Tracked layer config issues emit revrtWarning messages"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        tracked_layers=[
            {"layer_name": "layer_1", "agg_method": "does_not_exist"},
            {"layer_name": "missing_layer", "agg_method": "mean"},
            {"layer_name": "layer_2"},
        ],
    )

    with pytest.warns(revrtWarning) as warning_record:
        routing_layers = RoutingLayerManager(scenario).build()

    assert_message_was_logged("Did not find layer", "WARNING")
    assert_message_was_logged("Did not find method", "WARNING")
    assert_message_was_logged("must specify an 'agg_method' key", "WARNING")

    try:
        assert len(warning_record) == 3
    finally:
        routing_layers.close()


def test_tracked_layer_missing_multiplier_layer_raises_key_error(
    sample_layered_data,
):
    """Tracked layers raise when referenced multiplier layers are missing"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        tracked_layers=[
            {
                "layer_name": "layer_1",
                "multiplier_layer": "missing_layer",
                "agg_method": "mean",
            }
        ],
    )

    with pytest.raises(
        revrtKeyError, match="Did not find layer 'missing_layer' in cost file"
    ):
        routing_layers = RoutingLayerManager(scenario).build()
        routing_layers.close()


def test_friction_layers_and_lcp_agg_costs(sample_layered_data):
    """Friction layers may include cost stack and tracked layer toggles"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {
                        "layer_name": "layer_1",
                        "include_in_report": False,
                    },
                    {
                        "layer_name": "layer_2",
                        "multiplier_scalar": 0.5,
                        "include_in_report": True,
                        "include_in_final_cost": False,
                    },
                ],
                "friction_layers": [
                    {
                        "mask": "layer_3",
                        "multiplier_scalar": 0.1,
                    },
                ],
            }
        },
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        tracked_names = {layer.name for layer in routing_layers.tracked_layers}
        assert "layer_1_default" not in tracked_names
        assert "layer_2_default" in tracked_names

        base_value = (
            routing_layers.costs["default"].isel(y=1, x=1).compute().item()
        )
        final_value = (
            routing_layers.final_routing_layers["default"]
            .isel(y=1, x=1)
            .compute()
            .item()
        )

        assert final_value > base_value
    finally:
        routing_layers.close()


def test_friction_layer_include_in_report_adds_tracker(sample_layered_data):
    """Friction layers flagged for reports extend tracked layers"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
                "friction_layers": [
                    {
                        "mask": "layer_4",
                        "multiplier_scalar": 0.5,
                        "include_in_report": True,
                    }
                ],
            }
        },
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        tracked_names = {layer.name for layer in routing_layers.tracked_layers}
        assert "layer_4_default" in tracked_names
    finally:
        routing_layers.close()


def test_route_result_geom_returns_point_for_single_cell(sample_layered_data):
    """RouteMetrics.geom returns a Point geometry for single-cell routes"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        route = [(1, 1, "default")]
        result = RouteMetrics(
            routing_layers,
            route,
            optimized_objective=0.0,
        )

        assert result.geom.geom_type == "Point"
    finally:
        routing_layers.close()


def test_characterized_layer_length_metric_uses_positive_mask(
    sample_layered_data,
):
    """CharacterizedLayer uses positive-value mask when summing lengths"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        layer = next(
            tracked
            for tracked in routing_layers.tracked_layers
            if tracked.name == "layer_1_default"
        )
        route = [
            (1, 1, "default"),
            (1, 2, "default"),
            (2, 3, "default"),
        ]
        point_lens, __ = compute_lens(
            [point[:2] for point in route],
            abs(routing_layers.transform.a),
        )
        metrics = layer.compute(
            route,
            abs(routing_layers.transform.a),
            point_lens,
        )

        assert metrics["layer_1_default_length_km"] >= 0
    finally:
        routing_layers.close()


def test_route_result_cached_properties_reuse_computed_values(
    sample_layered_data,
):
    """RouteMetrics caches per-route lengths after first computation"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        route = [(1, 1, "default"), (1, 2, "default"), (2, 3, "default")]
        result = RouteMetrics(
            routing_layers,
            route,
            optimized_objective=0.0,
        )

        first_length = result.total_path_length
        assert isinstance(first_length, float)
        second_length = result.total_path_length
        assert second_length == first_length

        first_lens = result._lens
        assert np.allclose(result._lens, first_lens)
    finally:
        routing_layers.close()


def test_route_result_cost_property_returns_value(sample_layered_data):
    """RouteMetrics.cost multiplies cell costs by cached travel lengths"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        route = [(1, 1, "default"), (1, 2, "default"), (2, 3, "default")]
        result = RouteMetrics(
            routing_layers,
            route,
            optimized_objective=0.0,
        )

        assert result.cost > 0
    finally:
        routing_layers.close()


def test_characterized_layer_total_length_computation(sample_layered_data):
    """CharacterizedLayer computes length-weighted costs for eager data"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
    )

    routing_layers = RoutingLayerManager(scenario, chunks=None).build()
    try:
        layer = next(
            tracked
            for tracked in routing_layers.tracked_layers
            if tracked.name == "layer_1_default"
        )
        route = [
            (1, 1, "default"),
            (1, 2, "default"),
            (2, 3, "default"),
        ]
        point_lens, __ = compute_lens(
            [point[:2] for point in route],
            abs(routing_layers.transform.a),
        )
        metrics = layer.compute(
            route,
            abs(routing_layers.transform.a),
            point_lens,
        )

        assert metrics["layer_1_default_cost"] > 0
        assert metrics["layer_1_default_length_km"] >= 0
    finally:
        routing_layers.close()


def test_friction_layer_with_layer_name_warns(sample_layered_data):
    """Layer name on friction layer drops with deprecation warning"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
                "friction_layers": [
                    {
                        "layer_name": "legacy_friction",
                        "mask": "layer_4",
                    }
                ],
            }
        },
    )

    with pytest.warns(revrtDeprecationWarning) as warning_record:
        layers_for_rust = list(
            _friction_layers_for_rust(
                scenario.routing_options["default"]["friction_layers"]
            )
        )

    assert len(warning_record) == 1
    friction_payload = layers_for_rust[-1]
    assert "layer_name" not in friction_payload
    assert friction_payload["multiplier_layer"] == "layer_4"


def test_friction_layer_with_multiplier_layer_only(sample_layered_data):
    """Friction layers support multiplier layer without mask"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
                "friction_layers": [
                    {
                        "multiplier_layer": "layer_4",
                        "multiplier_scalar": 0.25,
                    }
                ],
            }
        },
    )

    layers_for_rust = list(
        _friction_layers_for_rust(
            scenario.routing_options["default"]["friction_layers"]
        )
    )
    friction_payload = layers_for_rust[-1]
    assert friction_payload["multiplier_layer"] == "layer_4"
    assert "mask" not in friction_payload


def test_friction_layer_requires_mask(sample_layered_data):
    """Friction layer build enforces presence of mask metadata"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
                "friction_layers": [{"multiplier_scalar": 5}],
            }
        },
    )

    routing_layers = RoutingLayerManager(scenario)
    try:
        with pytest.raises(
            revrtKeyError,
            match=(
                "Friction layers must specify a 'mask' or "
                "'multiplier_layer' key!"
            ),
        ):
            routing_layers.build()
    finally:
        routing_layers.close()


def test_barrier_layers_are_normalized_for_rust(sample_layered_data):
    """Barrier layers are passed through for Rust-side parsing"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
                "barrier_layers": [
                    {
                        "layer_name": "layer_4",
                        "where": "==1",
                        "barrier_importance": 2,
                    },
                    {"layer_name": "layer_6", "where": "<0"},
                ],
            }
        },
    )

    cost_function = json.loads(scenario.cost_function_json)
    assert cost_function["routing_options"]["default"]["barrier_layers"] == [
        {
            "layer_name": "layer_4",
            "barrier_operator": "eq",
            "barrier_threshold": 1.0,
            "barrier_importance": 2,
        },
        {
            "layer_name": "layer_6",
            "barrier_operator": "lt",
            # "barrier_threshold": 0,
            # "barrier_importance": None,
            "barrier_threshold": 0.0,
        },
    ]


def test_barrier_layers_normalize_not_equal_for_rust(sample_layered_data):
    """Barrier layers normalize the not-equal operator for Rust"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
                "barrier_layers": [
                    {
                        "layer_name": "layer_4",
                        "where": "!=0",
                        "barrier_importance": 1,
                    }
                ],
            }
        },
    )

    cost_function = json.loads(scenario.cost_function_json)
    assert cost_function["routing_options"]["default"]["barrier_layers"] == [
        {
            "layer_name": "layer_4",
            "barrier_operator": "ne",
            "barrier_threshold": 0.0,
            "barrier_importance": 1,
        }
    ]


def test_invalid_barrier_values_raise(sample_layered_data):
    """Barrier layers reject malformed comparison expressions"""

    with pytest.raises(ValueError, match="Barrier values must use"):
        __ = RoutingScenario(
            cost_fpath=sample_layered_data,
            routing_options={
                "default": {
                    "cost_layers": [{"layer_name": "layer_1"}],
                    "barrier_layers": [
                        {"layer_name": "layer_4", "where": "~1"}
                    ],
                }
            },
        ).cost_function_json


def test_barrier_importance_must_be_positive(sample_layered_data):
    """Barrier layers reject non-positive soft barrier ranks"""

    with pytest.raises(ValueError, match="positive integer"):
        __ = RoutingScenario(
            cost_fpath=sample_layered_data,
            routing_options={
                "default": {
                    "cost_layers": [{"layer_name": "layer_1"}],
                    "barrier_layers": [
                        {
                            "layer_name": "layer_4",
                            "where": "==1",
                            "barrier_importance": 0,
                        }
                    ],
                }
            },
        ).cost_function_json


def test_explicit_barriers_remain_hard(sample_layered_data):
    """Explicit barriers stay impassable even with soft invalid costs"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_2"}],
                "barrier_layers": [{"layer_name": "layer_4", "where": "==1"}],
            }
        },
        invalid_costs_block_routing=False,
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        barrier_value = (
            routing_layers.final_routing_layers["default"]
            .isel(y=0, x=3)
            .compute()
            .item()
        )
        free_value = (
            routing_layers.final_routing_layers["default"]
            .isel(y=0, x=2)
            .compute()
            .item()
        )
    finally:
        routing_layers.close()

    assert np.isnan(barrier_value)
    assert free_value > 0


def test_not_equal_barriers_remain_hard(sample_layered_data):
    """Not-equal barriers stay impassable even with soft invalid costs"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_2"}],
                "barrier_layers": [{"layer_name": "layer_4", "where": "!=0"}],
            }
        },
        invalid_costs_block_routing=False,
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        barrier_value = (
            routing_layers.final_routing_layers["default"]
            .isel(y=0, x=3)
            .compute()
            .item()
        )
        free_value = (
            routing_layers.final_routing_layers["default"]
            .isel(y=0, x=2)
            .compute()
            .item()
        )
    finally:
        routing_layers.close()

    assert np.isnan(barrier_value)
    assert free_value > 0


def test_soft_barrier_setting_controls_barrier_value(sample_layered_data):
    """Soft barriers convert impassable cells to large positive costs"""

    hard_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        invalid_costs_block_routing=True,
    )
    hard_layers = RoutingLayerManager(hard_scenario).build()
    try:
        hard_value = (
            hard_layers.final_routing_layers["default"]
            .isel(y=0, x=3)
            .compute()
            .item()
        )
    finally:
        hard_layers.close()

    soft_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        invalid_costs_block_routing=False,
    )
    soft_layers = RoutingLayerManager(soft_scenario).build()
    try:
        soft_value = (
            soft_layers.final_routing_layers["default"]
            .isel(y=0, x=3)
            .compute()
            .item()
        )
        assert hard_value == -1
        assert soft_value > 0
        assert soft_value > abs(hard_value)
    finally:
        soft_layers.close()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
