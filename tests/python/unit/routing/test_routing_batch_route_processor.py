"""reVrt tests for routing one point to many endpoints"""

from pathlib import Path

import pytest
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.transform import from_origin

from revrt.utilities import LayeredFile
from revrt.routing.base import RouteMetrics, RoutingScenario
from revrt.routing.processing import (
    BatchRouteProcessor,
    _RouteDefinitionFormatter,
    _RouteResultWriter,
    _validate_out_fp,
)
from revrt.exceptions import revrtKeyError
from revrt.warn import revrtWarning


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

    # ruff:ignore[whitespace-after-open-bracket, multiple-spaces-after-comma]
    # fmt: off
    layer_7 = np.array(
        [
            [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [ 8, -1, -1, -1, -1,  4, -1, -1],
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


@pytest.mark.parametrize(
    "algorithm",
    [
        "astar",
        "dijkstra",
        "long-range-dijkstra",
        "bidirectional-long-range-dijkstra",
    ],
)
def test_basic_single_route_layered_file_short_path(
    sample_layered_data, tmp_path, algorithm
):
    """Test routing using a LayeredFile-generated cost surface"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(1, 2, "default")]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 1
    route = output.iloc[0]
    assert route["cost"] == pytest.approx((1 + 2) / 2)
    assert route["length_km"] == pytest.approx(1 / 1000)
    assert route["cost"] == route["optimized_objective"]


@pytest.mark.parametrize(
    "algorithm",
    [
        "astar",
        "dijkstra",
        "long-range-dijkstra",
        "bidirectional-long-range-dijkstra",
    ],
)
def test_basic_single_route_applies_cost_multiplier_layer_to_objective(
    sample_layered_data, tmp_path, algorithm
):
    """Cost multiplier layers scale route cost and Rust objective"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
                "cost_multiplier_layer": ["layer_3"],
            }
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(1, 3, "default")]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 1
    route = output.iloc[0]

    expected_cost = ((1 * 2) + (2 * 2)) / 2 + ((2 * 2) + (2 * 3)) / 2

    assert route["cost"] == pytest.approx(expected_cost)
    assert route["length_km"] == pytest.approx(2 / 1000)
    assert route["optimized_objective"] == pytest.approx(expected_cost)


def test_route_results_passes_routing_layer_out_fp(
    sample_layered_data, tmp_path, monkeypatch
):
    """routing_layer_out_fp should be passed through to RouteFinder"""

    recorded_kwargs = {}

    class FakeRouteFinder:
        def __init__(self, *_args, **kwargs):
            recorded_kwargs.update(kwargs)

        def __iter__(self):
            return iter([])

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm="long-range-dijkstra",
    )

    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
        ],
    )

    monkeypatch.setattr(
        "revrt.routing.processing.RouteFinder", FakeRouteFinder
    )

    routing_layer_out_fp = tmp_path / "routing_layer.zarr"
    list(
        route_computer._route_results(
            routing_layer_out_fp=routing_layer_out_fp
        )
    )
    assert recorded_kwargs["routing_layer_out_fp"] == routing_layer_out_fp


@pytest.mark.parametrize("save_paths", [False, True])
def test_multi_option_routes_write_companion_output(
    sample_layered_data, tmp_path, monkeypatch, save_paths
):
    """Multi-option routes should emit a companion routing-option file"""

    class FakeRouteFinder:
        def __init__(self, *_args, **_kwargs):
            pass

        def __iter__(self):
            return iter(
                [
                    (
                        7,
                        [
                            (
                                [
                                    (1, 1, "overhead"),
                                    (1, 2, "overhead"),
                                    (1, 2, "underground"),
                                    (1, 3, "underground"),
                                ],
                                12.0,
                                [],
                            )
                        ],
                    )
                ]
            )

    monkeypatch.setattr(
        "revrt.routing.processing.RouteFinder", FakeRouteFinder
    )

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "overhead": {"cost_layers": [{"layer_name": "layer_1"}]},
            "underground": {"cost_layers": [{"layer_name": "layer_2"}]},
        },
        drivers={"default": {"overhead": 1, "underground": 10}},
        transition_costs={"default": 3.5},
        tracked_layers=[{"layer_name": "layer_3", "agg_method": "mean"}],
        invalid_costs_block_routing=True,
    )
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[(7, [(1, 1, "overhead")], [(1, 3, "underground")])],
        route_attrs={
            (7, (1, 1, "overhead")): {
                "route_id": "route_7",
                "polarity": "shared",
                "voltage": 0,
                "polarity_overhead": "ac",
                "voltage_overhead": 500,
                "polarity_underground": "dc",
                "voltage_underground": 345,
            }
        },
    )

    out_fp = tmp_path / ("routes.gpkg" if save_paths else "routes.csv")
    route_computer.process(out_fp=out_fp, save_paths=save_paths)

    if save_paths:
        full_routes = gpd.read_file(out_fp)
        option_routes = gpd.read_file(tmp_path / "routes_routing_options.gpkg")
    else:
        full_routes = pd.read_csv(out_fp)
        option_routes = pd.read_csv(tmp_path / "routes_routing_options.csv")

    assert len(full_routes) == 1
    assert len(option_routes) == 2
    assert set(option_routes["routing_option"]) == {
        "overhead",
        "underground",
    }
    assert set(option_routes["route_id"]) == {"route_7"}
    assert np.all(option_routes["length_km"] > 0)
    full_route = full_routes.iloc[0]
    assert {
        "cost",
        "length_km",
        "total_transition_costs",
        "overhead_cost",
        "overhead_length_km",
        "underground_cost",
        "underground_length_km",
    } <= set(full_routes.columns)
    assert full_route["total_transition_costs"] == pytest.approx(3.5)
    assert full_route["cost"] == pytest.approx(
        full_route["overhead_cost"]
        + full_route["underground_cost"]
        + full_route["total_transition_costs"]
    )
    assert full_route["length_km"] == pytest.approx(
        full_route["overhead_length_km"] + full_route["underground_length_km"]
    )
    assert {
        "polarity_overhead",
        "voltage_overhead",
        "polarity_underground",
        "voltage_underground",
        "layer_1_overhead_cost",
        "layer_2_underground_cost",
        "layer_3_overhead_mean",
        "layer_3_underground_mean",
    } <= set(full_routes.columns)
    assert not {
        "total_transition_costs",
        "overhead_cost",
        "overhead_length_km",
        "underground_cost",
        "underground_length_km",
    } & set(option_routes.columns)
    assert not {
        "polarity_overhead",
        "voltage_overhead",
        "polarity_underground",
        "voltage_underground",
        "layer_1_overhead_cost",
        "layer_2_underground_cost",
        "layer_3_overhead_mean",
        "layer_3_underground_mean",
    } & set(option_routes.columns)
    by_option = option_routes.set_index("routing_option")
    assert by_option.loc["overhead", "cost"] == pytest.approx(
        full_route["overhead_cost"]
    )
    assert by_option.loc["underground", "cost"] == pytest.approx(
        full_route["underground_cost"]
    )
    assert by_option.loc["overhead", "polarity"] == "ac"
    assert by_option.loc["underground", "polarity"] == "dc"
    assert by_option.loc["overhead", "voltage"] == 500
    assert by_option.loc["underground", "voltage"] == 345
    assert by_option.loc["overhead", "layer_1_cost"] == pytest.approx(
        full_route["layer_1_overhead_cost"]
    )
    assert pd.isna(by_option.loc["underground", "layer_1_cost"])
    assert by_option.loc["underground", "layer_2_cost"] == pytest.approx(
        full_route["layer_2_underground_cost"]
    )
    assert pd.isna(by_option.loc["overhead", "layer_2_cost"])
    assert by_option.loc["overhead", "layer_3_mean"] == pytest.approx(
        full_route["layer_3_overhead_mean"]
    )
    assert by_option.loc["underground", "layer_3_mean"] == pytest.approx(
        full_route["layer_3_underground_mean"]
    )
    assert ("geometry" in option_routes.columns) is save_paths
    if save_paths:
        assert set(option_routes.geometry.geom_type) == {"MultiLineString"}


def test_routing_option_results_split_transition_segment_midpoint(
    sample_layered_data, tmp_path
):
    """Route option outputs split option changes at the segment midpoint"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "overhead": {"cost_layers": [{"layer_name": "layer_1"}]},
            "underground": {"cost_layers": [{"layer_name": "layer_2"}]},
        },
    )
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[],
    )
    result_writer = _RouteResultWriter(
        tmp_path / "test_routes.gpkg",
        True,
        route_computer.routing_layers.cost_crs,
        route_computer.routing_layers.transform,
    )

    try:
        indices = [
            (1, 1, "overhead"),
            (1, 2, "underground"),
            (1, 3, "underground"),
        ]
        route_result = RouteMetrics(
            route_computer.routing_layers,
            indices,
            optimized_objective=0,
            attrs={"route_id": "route_8"},
        ).compute()
        results = result_writer._routing_option_results(indices, route_result)
    finally:
        route_computer._reset_routing_layers()

    by_option = {result["routing_option"]: result for result in results}

    assert set(by_option) == {"overhead", "underground"}
    assert by_option["overhead"]["geometry"].geom_type == "MultiLineString"
    assert by_option["underground"]["geometry"].geom_type == "MultiLineString"
    assert np.allclose(
        np.asarray(by_option["overhead"]["geometry"].geoms[0].coords),
        np.asarray([(1.5, 5.5), (2.0, 5.5)]),
    )
    assert np.allclose(
        np.asarray(by_option["underground"]["geometry"].geoms[0].coords),
        np.asarray([(2.0, 5.5), (3.5, 5.5)]),
    )
    assert by_option["overhead"]["length_km"] == pytest.approx(0.0005)
    assert by_option["underground"]["length_km"] == pytest.approx(0.0015)
    assert by_option["overhead"]["route_id"] == "route_8"
    assert by_option["underground"]["route_id"] == "route_8"


def test_routing_scenario_rejects_invalid_algorithm(sample_layered_data):
    """RoutingScenario raises on unsupported algorithm names"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm="bellman_ford",
    )
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
        ],
    )
    with pytest.raises(ValueError, match="Unsupported routing algorithm"):
        list(route_computer._route_results())


def test_batch_route_processor_forwards_algorithm(
    sample_layered_data, monkeypatch
):
    """BatchRouteProcessor passes the selected algorithm to Rust"""

    captured = {}

    class FakeRouteFinder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(
        "revrt.routing.processing.RouteFinder", FakeRouteFinder
    )

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm="dijkstra",
    )
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
        ],
    )

    assert list(route_computer._route_results()) == []
    assert captured["algorithm"] == "dijkstra"


@pytest.mark.parametrize(
    "algorithm",
    [
        "astar",
        "dijkstra",
        "long-range-dijkstra",
        "bidirectional-long-range-dijkstra",
    ],
)
def test_basic_single_route_layered_file(
    sample_layered_data, tmp_path, algorithm
):
    """Test routing using a LayeredFile-generated cost surface"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
            ([(1, 2, "default")], [(2, 6, "default")]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 2
    first_route = output[
        (output["start_row"] == 1) & (output["start_col"] == 1)
    ].iloc[0]
    assert first_route["cost"] == pytest.approx(11.192389)
    assert first_route["length_km"] == pytest.approx(0.0090710678)
    assert np.isclose(first_route["cost"], first_route["optimized_objective"])

    second_route = output[
        (output["start_row"] == 1) & (output["start_col"] == 2)
    ].iloc[0]
    assert second_route["cost"] == pytest.approx(12.278174)
    assert second_route["length_km"] == pytest.approx(0.008656854)
    assert np.isclose(
        second_route["cost"], second_route["optimized_objective"]
    )


@pytest.mark.parametrize("single_rd", [True, False])
@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_multi_layer_route_layered_file(
    sample_layered_data, tmp_path, single_rd, algorithm
):
    """Test routing across multiple cost layers"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {"layer_name": "layer_1"},
                    {"layer_name": "layer_2"},
                ]
            }
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    if single_rd:
        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=[
                (
                    1,
                    [(1, 1, "default"), (1, 2, "default")],
                    [(2, 6, "default")],
                ),
            ],
            route_attrs={
                (1, (1, 2, "default")): {"route_type": "A"},
            },
        )
    else:
        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=[
                (1, [(1, 1, "default")], [(2, 6, "default")]),
                (2, [(1, 2, "default")], [(2, 6, "default")]),
            ],
            route_attrs={
                (2, (1, 2, "default")): {"route_type": "A"},
            },
        )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 2

    first_route = output[
        (output["start_row"] == 1) & (output["start_col"] == 1)
    ].iloc[0]
    assert first_route["cost"] == pytest.approx(
        27.606602,
        rel=1e-4,
    )
    assert first_route["length_km"] == pytest.approx(
        0.005414,
        rel=1e-4,
    )
    assert first_route["layer_1_default_cost"] == pytest.approx(
        17.571068,
        rel=1e-4,
    )
    assert first_route["layer_2_default_cost"] == pytest.approx(
        10.035534,
        rel=1e-4,
    )
    assert np.isclose(
        first_route["cost"], first_route["optimized_objective"], rtol=1e-6
    )
    assert np.isnan(first_route["route_type"])

    second_route = output[
        (output["start_row"] == 1) & (output["start_col"] == 2)
    ].iloc[0]
    assert second_route["cost"] == pytest.approx(
        25.106602,
        rel=1e-4,
    )
    assert second_route["length_km"] == pytest.approx(
        0.004414,
        rel=1e-4,
    )
    assert second_route["layer_1_default_cost"] == pytest.approx(
        16.071068,
        rel=1e-4,
    )
    assert second_route["layer_2_default_cost"] == pytest.approx(
        9.035534,
        rel=1e-4,
    )
    assert np.isclose(
        second_route["cost"], second_route["optimized_objective"], rtol=1e-6
    )
    assert second_route["route_type"] == "A"


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_save_paths_returns_expected_geometry(
    sample_layered_data, tmp_path, algorithm
):
    """Saving paths returns expected geometries for each route"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_gpkg = tmp_path / "routes.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
            ([(1, 2, "default")], [(2, 6, "default")]),
        ],
    )
    route_computer.process(out_fp=out_gpkg, save_paths=True)

    output = gpd.read_file(out_gpkg)
    assert len(output) == 2

    output = output.sort_values(by=["start_col"], ascending=True).reset_index(
        drop=True
    )

    route_geoms = [route["geometry"] for __, route in output.iterrows()]

    expected_geometries = [
        [
            (1.5, 5.5),
            (1.5, 4.5),
            (4.5, 1.5),
            (5.5, 2.5),
            (5.5, 3.5),
            (6.5, 4.5),
        ],
        [
            (2.5, 5.5),
            (2.5, 4.5),
            (3.5, 3.5),
            (3.5, 2.5),
            (4.5, 1.5),
            (5.5, 2.5),
            (5.5, 3.5),
            (6.5, 4.5),
        ],
    ]

    for geom, expected_coords in zip(
        route_geoms, expected_geometries, strict=True
    ):
        assert geom.geom_type == "LineString"
        assert np.allclose(
            np.asarray(geom.coords), np.asarray(expected_coords)
        )


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_empty_route_definitions_returns_empty_dataframe(
    sample_layered_data, tmp_path, algorithm
):
    """Empty route definitions return an empty dataframe"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)
    assert not out_csv.exists()


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_empty_route_definitions_returns_empty_geo_dataframe(
    sample_layered_data, tmp_path, algorithm
):
    """Empty route definitions return an empty dataframe"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_gpkg = tmp_path / "routes.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[],
    )
    route_computer.process(out_fp=out_gpkg, save_paths=True)
    assert not out_gpkg.exists()


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_multi_layer_route_with_multiplier(
    sample_layered_data, tmp_path, algorithm
):
    """Test routing with multiple layers and a scalar multiplier"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {"layer_name": "layer_1"},
                    {
                        "layer_name": "layer_2",
                        "multiplier_scalar": 0.5,
                    },
                ]
            }
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
            ([(1, 2, "default")], [(2, 6, "default")]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 2

    first_route = output[
        (output["start_row"] == 1) & (output["start_col"] == 1)
    ].iloc[0]
    assert first_route["cost"] == pytest.approx(
        22.588835,
        rel=1e-4,
    )
    assert first_route["length_km"] == pytest.approx(
        0.005414,
        rel=1e-4,
    )
    assert first_route["layer_1_default_cost"] == pytest.approx(
        17.571068,
        rel=1e-4,
    )
    assert first_route["layer_2_default_cost"] == pytest.approx(
        5.017767,
        rel=1e-4,
    )
    assert np.isclose(
        first_route["cost"],
        first_route["optimized_objective"],
        rtol=1e-4,
    )

    second_route = output[
        (output["start_row"] == 1) & (output["start_col"] == 2)
    ].iloc[0]
    assert second_route["cost"] == pytest.approx(
        20.588835,
        rel=1e-4,
    )
    assert second_route["length_km"] == pytest.approx(
        0.004414,
        rel=1e-4,
    )
    assert second_route["layer_1_default_cost"] == pytest.approx(
        16.071068,
        rel=1e-4,
    )
    assert second_route["layer_2_default_cost"] == pytest.approx(
        4.517767,
        rel=1e-4,
    )
    assert np.isclose(
        second_route["cost"],
        second_route["optimized_objective"],
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_multi_layer_route_with_scalar_and_layer_multipliers(
    sample_layered_data, tmp_path, algorithm
):
    """Test routing when combining scalar and layer multipliers"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {"layer_name": "layer_1"},
                    {"layer_name": "layer_2", "multiplier_scalar": 0.5},
                    {
                        "layer_name": "layer_3",
                        "multiplier_layer": ["layer_4"],
                    },
                    {
                        "layer_name": "layer_5",
                        "multiplier_scalar": 2,
                        "multiplier_layer": ["layer_4"],
                    },
                ]
            }
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(1, 2, "default")]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 1

    route = output.iloc[0]
    assert route["cost"] == pytest.approx(2.0, rel=1e-4)
    assert route["length_km"] == pytest.approx(0.001, rel=1e-4)
    assert route["layer_1_default_cost"] == pytest.approx(1.5, rel=1e-4)
    assert route["layer_2_default_cost"] == pytest.approx(0.5, rel=1e-4)
    assert route["layer_3_default_cost"] == pytest.approx(0.0, abs=1e-8)
    assert route["layer_5_default_cost"] == pytest.approx(0.0, abs=1e-8)
    assert route["layer_1_default_length_km"] == pytest.approx(0.001, rel=1e-4)
    assert route["layer_2_default_length_km"] == pytest.approx(0.001, rel=1e-4)
    assert route["layer_3_default_length_km"] == pytest.approx(0.0, abs=1e-8)
    assert route["layer_5_default_length_km"] == pytest.approx(0.0, abs=1e-8)
    assert np.isclose(route["cost"], route["optimized_objective"], rtol=1e-6)


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_routing_with_tracked_layers(sample_layered_data, tmp_path, algorithm):
    """Tracked layers report aggregated stats alongside routing results"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        tracked_layers=[
            {"layer_name": "layer_1", "agg_method": "mean"},
            {"layer_name": "layer_2", "agg_method": "max"},
            {"layer_name": "layer_3", "agg_method": "min"},
        ],
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(1, 2, "default")]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 1
    route = output.iloc[0]

    assert {
        "layer_1_default_mean",
        "layer_2_default_max",
        "layer_3_default_min",
    }.issubset(route.keys())

    assert route["layer_1_default_mean"] == pytest.approx(1.5)
    assert route["layer_2_default_max"] == pytest.approx(1.0)
    assert route["layer_3_default_min"] == pytest.approx(2.0)


@pytest.mark.parametrize("use_friction", [True, False])
@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_start_point_on_barrier_returns_no_route(
    sample_layered_data,
    assert_message_was_logged,
    use_friction,
    tmp_path,
    algorithm,
):
    """If the start point is on a barrier (cost <= 0) no route is returned"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_6"}]}
        },
        algorithm=algorithm,
    )
    if use_friction:
        scenario.routing_options["default"]["friction_layers"] = [
            {"multiplier_layer": ["layer_5"], "multiplier_scalar": -10}
        ]

    out_csv = tmp_path / "routes.csv"

    # (3, 1) in layer_6 is -1 -> treated as barrier
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(3, 1, "default")], [(2, 6, "default")]),
        ],
    )
    with pytest.warns(revrtWarning, match="invalid"):
        route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the start points have an invalid cost (must be > 0): "
        "{(3, 1, 'default')}",
        "WARNING",
    )
    assert_message_was_logged(
        "All start points are invalid for route with ID 0: "
        "[(3, 1, 'default')]",
        "WARNING",
    )
    assert not out_csv.exists()


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_invalid_start_point_logged(
    sample_layered_data, assert_message_was_logged, tmp_path, algorithm
):
    """Test that only the invalid starting point is logged"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"

    # (0, 3) in layer_1 is 0 -> treated as barrier
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            (
                [(1, 1, "default"), (0, 3, "default")],
                [(2, 6, "default")],
            ),
        ],
    )
    with pytest.warns(revrtWarning, match="invalid cost"):
        route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the start points have an invalid cost (must be > 0): "
        "{(0, 3, 'default')}",
        "WARNING",
    )

    output = pd.read_csv(out_csv)
    assert len(output) == 1

    route = output.iloc[0]
    assert route["cost"] == pytest.approx(11.192389)
    assert route["length_km"] == pytest.approx(0.0090710678)
    assert route["start_row"] == 1
    assert route["start_col"] == 1
    assert route["end_row"] == 2
    assert route["end_col"] == 6


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_invalid_start_point_explicitly_allowed(
    sample_layered_data, assert_message_was_logged, tmp_path, algorithm
):
    """Test out-of-bounds points logging"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        invalid_costs_block_routing=False,
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"

    # (0, 3) in layer_1 is 0 -> treated as barrier
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            (
                [
                    (1, 1, "default"),
                    (0, 3, "default"),
                    (10000, 10000, "default"),
                ],
                [(2, 6, "default"), (20000, 20000, "default")],
            ),
        ],
    )
    with pytest.warns(revrtWarning, match="out of bounds"):
        route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the start points are out of bounds for an array of "
        "shape (7, 8): [(10000, 10000, 'default')]",
        "WARNING",
    )
    assert_message_was_logged(
        "One or more of the end points are out of bounds for an array of "
        "shape (7, 8): [(20000, 20000, 'default')]",
        "WARNING",
    )

    output = pd.read_csv(out_csv)
    assert len(output) == 2

    first_route = output[
        (output["start_row"] == 1) & (output["start_col"] == 1)
    ].iloc[0]
    assert first_route["cost"] == pytest.approx(11.192389)
    assert first_route["length_km"] == pytest.approx(0.0090710678)
    assert first_route["start_row"] == 1
    assert first_route["start_col"] == 1
    assert first_route["end_row"] == 2
    assert first_route["end_col"] == 6

    second_route = output[
        (output["start_row"] == 0) & (output["start_col"] == 3)
    ].iloc[0]
    assert second_route["start_row"] == 0
    assert second_route["start_col"] == 3
    assert second_route["end_row"] == 2
    assert second_route["end_col"] == 6


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_some_endpoints_include_barriers_but_one_valid(
    sample_layered_data, tmp_path, algorithm
):
    """If some end points <=0 but at least one is valid, route is found"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"

    # include one barrier end (0,3) and one valid end (2,6)
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(0, 3, "default"), (2, 6, "default")]),
        ],
    )
    with pytest.warns(revrtWarning, match="invalid cost"):
        route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 1
    # At least one valid endpoint must be reached and cost must be positive.
    route = output.iloc[0]
    assert route["cost"] > 0

    end_row = int(route["end_row"])
    end_col = int(route["end_col"])
    assert (end_row, end_col) == (2, 6)


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_all_endpoints_are_barriers_returns_no_route(
    sample_layered_data, assert_message_was_logged, tmp_path, algorithm
):
    """If all end points are barriers, no route is returned"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(0, 3, "default"), (0, 7, "default")]),
        ],
    )
    with pytest.warns(revrtWarning, match="valid cost"):
        route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "None of the end points have a valid cost (must be > 0): "
        "[(0, 3, 'default'), (0, 7, 'default')]",
        "ERROR",
    )
    assert not out_csv.exists()


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_bad_start_index_returns_no_route(
    sample_layered_data, assert_message_was_logged, tmp_path, algorithm
):
    """If any points are out-of-bounds, no route is returned"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            (
                [(10000, 10000, "default")],
                [(0, 3, "default"), (0, 7, "default")],
            ),
        ],
    )
    with pytest.warns(revrtWarning, match=r"[(10000, 10000)]"):
        route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the start points are out of bounds for an array of "
        "shape (7, 8): [(10000, 10000, 'default')]",
        "WARNING",
    )
    assert not out_csv.exists()


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_bad_end_index_returns_no_route(
    sample_layered_data, assert_message_was_logged, tmp_path, algorithm
):
    """If any points are out-of-bounds, no route is returned"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(10000, 10000, "default")]),
        ],
    )
    with pytest.warns(revrtWarning, match="end points"):
        route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the end points are out of bounds for an array of "
        "shape (7, 8): [(10000, 10000, 'default')]",
        "WARNING",
    )
    assert_message_was_logged(
        "All end points are invalid for route with ID 0: "
        "[(10000, 10000, 'default')]",
        "WARNING",
    )
    assert not out_csv.exists()


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_bad_index_skipped(
    sample_layered_data, assert_message_was_logged, tmp_path, algorithm
):
    """Out-of-bounds points are skipped and routes are compute"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            (
                [(10000, 10000, "default"), (1, 1, "default")],
                [
                    (0, 3, "default"),
                    (2, 6, "default"),
                    (20000, 20000, "default"),
                ],
            ),
        ],
    )
    with pytest.warns(revrtWarning, match="Dropping these"):
        route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the start points are out of bounds for an array of "
        "shape (7, 8): [(10000, 10000, 'default')]",
        "WARNING",
    )
    assert_message_was_logged(
        "One or more of the end points are out of bounds for an array of "
        "shape (7, 8): [(20000, 20000, 'default')]",
        "WARNING",
    )
    output = pd.read_csv(out_csv)
    assert len(output) == 1

    route = output.iloc[0]
    assert route["cost"] == pytest.approx(11.192389)
    assert route["length_km"] == pytest.approx(0.0090710678)
    assert route["start_row"] == 1
    assert route["start_col"] == 1
    assert route["end_row"] == 2
    assert route["end_col"] == 6


def test_missing_cost_layer_raises_key_error(sample_layered_data, tmp_path):
    """Missing layers surface a revrtKeyError during build"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "not_there"}]}
        },
    )

    out_csv = tmp_path / "routes.csv"
    with pytest.raises(
        revrtKeyError, match="Did not find layer 'not_there' in cost file"
    ):
        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=[
                ([(1, 1, "default")], [(1, 2, "default")]),
            ],
        )
        route_computer.process(out_fp=out_csv, save_paths=False)


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_length_invariant_layers_sum_raw_values(
    sample_layered_data, tmp_path, algorithm
):
    """Length invariant layers sum raw cell values without distance scaling"""

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
        algorithm=algorithm,
    )

    out_gpkg = tmp_path / "routes.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
        ],
    )
    route_computer.process(out_fp=out_gpkg, save_paths=True)

    output = gpd.read_file(out_gpkg)
    assert len(output) == 1
    route = output.iloc[0]

    with xr.open_dataset(
        sample_layered_data,
        consolidated=False,
        engine="zarr",
    ) as ds:
        layer_two = ds["layer_2"].isel(band=0)
        mask = rasterio.features.geometry_mask(
            [route["geometry"]],
            out_shape=layer_two.shape,
            transform=ds.rio.transform(),
            invert=True,
        )
        rows, cols = np.where(mask)
        expected_invariant_cost = sum(
            layer_two.isel(y=row, x=col).item()
            for row, col in zip(rows, cols, strict=True)
        )

    assert route["layer_2_default_cost"] == pytest.approx(
        expected_invariant_cost,
        rel=1e-6,
    )
    assert route["cost"] == pytest.approx(
        route["layer_1_default_cost"] + expected_invariant_cost,
        rel=1e-6,
    )
    assert route["layer_2_default_length_km"] == pytest.approx(
        route["length_km"],
        rel=1e-6,
    )


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_length_invariant_hidden_and_friction_layers(
    sample_layered_data, tmp_path, algorithm
):
    """Combined layer settings preserve cost reporting expectations"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {"layer_name": "layer_1"},
                    {"layer_name": "layer_2", "is_invariant": True},
                    {
                        "layer_name": "layer_5",
                        "multiplier_scalar": 100,
                        "include_in_final_cost": False,
                        "include_in_report": True,
                    },
                ],
                "friction_layers": [
                    {
                        "multiplier_layer": ["layer_4"],
                        "multiplier_scalar": 0.5,
                    },
                ],
            }
        },
        algorithm=algorithm,
    )

    out_gpkg = tmp_path / "routes.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
        ],
    )
    route_computer.process(out_fp=out_gpkg, save_paths=True)

    output = gpd.read_file(out_gpkg)
    assert len(output) == 1
    route = output.iloc[0]

    assert route["length_km"] == pytest.approx(
        0.00682842712474619,
        rel=1e-6,
    )
    assert route["layer_2_default_length_km"] == pytest.approx(
        route["length_km"],
        rel=1e-6,
    )
    assert route["layer_1_default_cost"] == pytest.approx(26.156855, rel=1e-6)
    assert route["layer_2_default_cost"] == pytest.approx(19.0)
    assert route["cost"] == pytest.approx(
        route["layer_1_default_cost"] + route["layer_2_default_cost"],
        rel=1e-6,
    )
    assert route["cost"] == pytest.approx(
        45.15685424949238,
        rel=1e-6,
    )
    assert route["optimized_objective"] > route["cost"]

    assert route["layer_5_default_cost"] == pytest.approx(
        170.71068,
        rel=1e-6,
    )
    assert route["layer_5_default_length_km"] == pytest.approx(
        0.0017071,
        rel=1e-4,
    )
    assert list(route["geometry"].coords) == [
        (1.5, 5.5),
        (3.5, 3.5),
        (6.5, 3.5),
        (6.5, 4.5),
    ]


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_friction_layer_influences_objective_without_reporting(
    sample_layered_data, tmp_path, algorithm
):
    """Friction layers alter routing objective without affecting reports"""

    base_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    friction_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
                "friction_layers": [
                    {
                        "multiplier_layer": ["layer_4"],
                        "multiplier_scalar": 0.5,
                    }
                ],
            }
        },
        algorithm=algorithm,
    )

    base_csv = tmp_path / "base.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=base_scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
        ],
    )
    route_computer.process(out_fp=base_csv, save_paths=False)

    base_output = pd.read_csv(base_csv)
    assert len(base_output) == 1
    base_route = base_output.iloc[0]

    friction_csv = tmp_path / "friction.csv"

    route_computer = BatchRouteProcessor(
        routing_scenario=friction_scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
        ],
    )
    route_computer.process(out_fp=friction_csv, save_paths=False)

    friction_output = pd.read_csv(friction_csv)
    assert len(friction_output) == 1
    friction_route = friction_output.iloc[0]

    # Friction is unavoidable, so cost and path should be roughly the same
    assert np.allclose(base_route["cost"], friction_route["cost"])
    assert (
        friction_route["optimized_objective"]
        > base_route["optimized_objective"]
    )
    assert "layer_2_default_cost" not in friction_route
    assert "layer_2_default_length_km" not in friction_route


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_friction_layer_influences_objective(
    sample_layered_data, tmp_path, algorithm
):
    """Friction layers alter routing objective without affecting reports"""

    base_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    friction_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
                "friction_layers": [
                    {
                        "multiplier_layer": ["layer_4"],
                        "multiplier_scalar": 1000,
                    }
                ],
            }
        },
        algorithm=algorithm,
    )

    base_csv = tmp_path / "base.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=base_scenario,
        route_definitions=[
            ([(1, 1, "default")], [(3, 5, "default")]),
        ],
    )
    route_computer.process(out_fp=base_csv, save_paths=False)

    base_output = pd.read_csv(base_csv)
    assert len(base_output) == 1
    base_route = base_output.iloc[0]

    friction_csv = tmp_path / "friction.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=friction_scenario,
        route_definitions=[
            ([(1, 1, "default")], [(3, 5, "default")]),
        ],
    )
    route_computer.process(out_fp=friction_csv, save_paths=False)

    friction_output = pd.read_csv(friction_csv)
    assert len(friction_output) == 1
    friction_route = friction_output.iloc[0]

    # Friction path is shorter but more expensive
    assert friction_route["cost"] > base_route["cost"]
    assert friction_route["cost"] < 1000
    assert friction_route["optimized_objective"] > 1000
    assert friction_route["length_km"] < base_route["length_km"]
    assert (
        friction_route["optimized_objective"]
        > base_route["optimized_objective"]
    )

    assert "layer_5_default_cost" not in friction_route
    assert "layer_5_default_length_km" not in friction_route


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_negative_friction_layer_influences_objective(
    sample_layered_data, tmp_path, algorithm
):
    """Friction layers alter routing objective without affecting reports"""

    base_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    friction_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
                "friction_layers": [
                    {
                        "multiplier_layer": ["layer_5"],
                        "multiplier_scalar": -10,
                    }
                ],
            }
        },
        algorithm=algorithm,
    )

    base_gpkg = tmp_path / "base.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=base_scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
        ],
    )
    route_computer.process(out_fp=base_gpkg, save_paths=True)

    base_output = gpd.read_file(base_gpkg)
    assert len(base_output) == 1
    base_route = base_output.iloc[0]

    friction_gpkg = tmp_path / "friction.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=friction_scenario,
        route_definitions=[
            ([(1, 1, "default")], [(2, 6, "default")]),
        ],
    )
    route_computer.process(out_fp=friction_gpkg, save_paths=True)

    friction_output = gpd.read_file(friction_gpkg)
    assert len(friction_output) == 1
    friction_route = friction_output.iloc[0]

    # Friction path is shorter but more expensive
    assert friction_route["cost"] > base_route["cost"]
    assert friction_route["cost"] > 0
    assert friction_route["optimized_objective"] < 5
    assert friction_route["length_km"] < base_route["length_km"]
    assert (
        friction_route["optimized_objective"]
        < base_route["optimized_objective"]
    )

    assert "layer_5_default_cost" not in friction_route
    assert "layer_5_default_length_km" not in friction_route


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_negative_friction_layer_does_not_go_thru_barrier(
    sample_layered_data, tmp_path, algorithm
):
    """Friction layers alter routing objective without affecting reports"""

    base_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_6"}]}
        },
        algorithm=algorithm,
    )

    friction_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_6"}],
                "friction_layers": [
                    {
                        "multiplier_layer": ["layer_5"],
                        "multiplier_scalar": -10,
                    }
                ],
            },
        },
        algorithm=algorithm,
    )

    base_gpkg = tmp_path / "base.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=base_scenario,
        route_definitions=[
            ([(4, 0, "default")], [(2, 7, "default")]),
        ],
    )
    route_computer.process(out_fp=base_gpkg, save_paths=True)

    base_output = gpd.read_file(base_gpkg)
    assert len(base_output) == 1
    base_route = base_output.iloc[0]

    friction_gpkg = tmp_path / "friction.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=friction_scenario,
        route_definitions=[
            ([(4, 0, "default")], [(2, 7, "default")]),
        ],
    )
    route_computer.process(out_fp=friction_gpkg, save_paths=True)

    friction_output = gpd.read_file(friction_gpkg)
    assert len(friction_output) == 1
    friction_route = friction_output.iloc[0]

    # Friction path is shorter but more expensive
    assert friction_route["cost"] == pytest.approx(base_route["cost"])
    assert friction_route["cost"] > 0
    assert friction_route["length_km"] == pytest.approx(
        base_route["length_km"]
    )
    assert friction_route["geometry"].equals(base_route["geometry"])
    assert (
        friction_route["optimized_objective"]
        < base_route["optimized_objective"]
    )

    assert "layer_5_default_cost" not in friction_route
    assert "layer_5_default_length_km" not in friction_route


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_include_in_final_cost_false_behaves_like_friction(
    sample_layered_data, tmp_path, algorithm
):
    """Non-final cost layers steer routing but stay out of reports"""

    base_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm=algorithm,
    )

    out_gpkg = tmp_path / "base.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=base_scenario,
        route_definitions=[
            ([(1, 1, "default")], [(3, 5, "default")]),
        ],
    )
    route_computer.process(out_fp=out_gpkg, save_paths=True)

    base_output = gpd.read_file(out_gpkg)
    assert len(base_output) == 1
    base_route = base_output.iloc[0]

    penalized_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {"layer_name": "layer_1"},
                    {
                        "layer_name": "layer_5",
                        "multiplier_scalar": 1000,
                        "include_in_final_cost": False,
                        "include_in_report": False,
                    },
                ]
            }
        },
    )

    penalized_gpkg = tmp_path / "penalized.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=penalized_scenario,
        route_definitions=[
            ([(1, 1, "default")], [(3, 5, "default")]),
        ],
    )
    route_computer.process(out_fp=penalized_gpkg, save_paths=True)

    penalized_output = gpd.read_file(penalized_gpkg)
    assert len(penalized_output) == 1
    penalized_route = penalized_output.iloc[0]

    assert not base_route["geometry"].equals(penalized_route["geometry"])
    assert (
        penalized_route["optimized_objective"]
        > base_route["optimized_objective"]
    )
    assert penalized_route["optimized_objective"] > penalized_route["cost"]
    assert penalized_route["cost"] < 1000
    assert penalized_route["cost"] == pytest.approx(
        penalized_route["layer_1_default_cost"],
        rel=1e-6,
    )
    assert "layer_5_default_cost" not in penalized_route


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_negative_cost_path_returns_no_route(
    sample_layered_data, tmp_path, algorithm
):
    """If all points between start and end are negative, return no route"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_7"}]}
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(4, 0, "default")], [(4, 5, "default")]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)
    assert not out_csv.exists()


def test_explicit_barrier_blocks_route_even_with_soft_invalid_costs(
    sample_layered_data, tmp_path
):
    """Explicit barriers block routes regardless of invalid cost setting"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_2"}],
                "barrier_layers": [{"layer_name": "layer_4", "where": "==1"}],
            }
        },
        invalid_costs_block_routing=False,
        algorithm="dijkstra",
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(1, 5, "default")]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    assert not out_csv.exists()


def test_soft_barrier_points_remain_valid_for_retry(sample_layered_data):
    """Soft barriers do not invalidate Python-side route endpoints"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_2"}],
                "barrier_layers": [
                    {
                        "layer_name": "layer_4",
                        "where": "==1",
                        "barrier_importance": 1,
                    },
                    {
                        "layer_name": "layer_5",
                        "where": "==1",
                        "barrier_importance": 1,
                    },
                ],
            }
        },
        invalid_costs_block_routing=False,
    )

    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(1, 5, "default")]),
        ],
    )
    route_formatter = _RouteDefinitionFormatter(
        route_computer.route_definitions,
        route_computer.routing_layers,
        route_computer.routing_scenario,
    )
    try:
        assert route_formatter._validate_start_points([(0, 3, "default")]) == [
            (0, 3, "default")
        ]
        assert route_formatter._validate_end_points([(0, 3, "default")]) == [
            (0, 3, "default")
        ]
    finally:
        route_computer._reset_routing_layers()


def test_soft_barrier_retry_returns_route_with_metadata(
    sample_layered_data, tmp_path
):
    """Soft barrier retries drop ranked barriers and record metadata"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_2"}],
                "barrier_layers": [
                    {
                        "layer_name": "layer_4",
                        "where": "==1",
                        "barrier_importance": 1,
                    }
                ],
            }
        },
        invalid_costs_block_routing=False,
        algorithm="dijkstra",
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1, "default")], [(1, 5, "default")]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 1
    route = output.iloc[0]
    assert route["dropped_barrier_layers"] == '["layer_4"]'


def test_soft_barrier_start_point_retries_and_records_metadata(
    sample_layered_data, tmp_path
):
    """Routes starting on a soft barrier succeed after retry"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_2"}],
                "barrier_layers": [
                    {
                        "layer_name": "layer_4",
                        "where": "==1",
                        "barrier_importance": 1,
                    }
                ],
            }
        },
        invalid_costs_block_routing=False,
        algorithm="dijkstra",
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 3, "default")], [(1, 5, "default")]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 1
    route = output.iloc[0]
    assert route["start_row"] == 1
    assert route["start_col"] == 3
    assert route["end_row"] == 1
    assert route["end_col"] == 5
    assert route["dropped_barrier_layers"] == '["layer_4"]'


def test_soft_barrier_retry_exhaustion_returns_no_route(
    sample_layered_data, assert_message_was_logged, tmp_path
):
    """Routing reports no solution after exhausting soft barrier retries"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_7"}],
                "barrier_layers": [
                    {
                        "layer_name": "layer_4",
                        "where": "==1",
                        "barrier_importance": 1,
                    }
                ],
            }
        },
        invalid_costs_block_routing=True,
        algorithm="dijkstra",
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(4, 0, "default")], [(4, 5, "default")]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "Unable to find route from [(4, 0, 'default')] to any of "
        "[(4, 5, 'default')]",
        "ERROR",
    )
    assert not out_csv.exists()


def test_skip_failed_routes_preserves_per_solution_retry_metadata(
    sample_layered_data,
):
    """Batch routing applies dropped barrier metadata per solution"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_1"}]}
        },
        algorithm="dijkstra",
    )
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            (
                13,
                [(1, 1, "default"), (1, 2, "default")],
                [(2, 6, "default")],
            ),
        ],
        route_attrs={
            (13, (1, 2, "default")): {"route_type": "secondary"},
        },
    )

    routed = list(
        route_computer._skip_failed_routes(
            [
                (
                    13,
                    [
                        (
                            [
                                (1, 1, "default"),
                                (2, 2, "default"),
                                (2, 6, "default"),
                            ],
                            10.0,
                            [],
                        ),
                        (
                            [
                                (1, 2, "default"),
                                (2, 3, "default"),
                                (2, 6, "default"),
                            ],
                            12.0,
                            ["layer_4"],
                        ),
                    ],
                )
            ]
        )
    )

    assert len(routed) == 2

    first_indices, first_objective, first_attrs = routed[0]
    assert first_indices[0] == (1, 1, "default")
    assert first_objective == pytest.approx(10.0)
    assert first_attrs["dropped_barrier_layers"] == "[]"

    second_indices, second_objective, second_attrs = routed[1]
    assert second_indices[0] == (1, 2, "default")
    assert second_objective == pytest.approx(12.0)
    assert second_attrs["route_type"] == "secondary"
    assert second_attrs["dropped_barrier_layers"] == '["layer_4"]'


@pytest.mark.parametrize("invalid_costs_block_routing", [True, False])
@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_soft_barrier(
    sample_layered_data, invalid_costs_block_routing, tmp_path, algorithm
):
    """Test that soft barriers work as expected in point-to-many routing"""
    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {"cost_layers": [{"layer_name": "layer_7"}]}
        },
        invalid_costs_block_routing=invalid_costs_block_routing,
        algorithm=algorithm,
    )

    out_gpkg = tmp_path / "routes.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(4, 0, "default")], [(4, 5, "default")]),
        ],
    )
    route_computer.process(out_fp=out_gpkg, save_paths=True)

    if invalid_costs_block_routing:
        assert not out_gpkg.exists()
    else:
        output = gpd.read_file(out_gpkg)
        assert len(output) == 1
        route = output.iloc[0]
        assert route["cost"] == pytest.approx(6)
        assert route["length_km"] == pytest.approx(0.005)
        x, y = route["geometry"].xy
        assert np.allclose(x, [0.5, 5.5])
        assert np.allclose(y, 2.5)


@pytest.mark.parametrize("single_rd", [True, False])
@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_route_many_attrs(sample_layered_data, tmp_path, single_rd, algorithm):
    """Test routing with multiple layers and a scalar multiplier"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        routing_options={
            "default": {
                "cost_layers": [
                    {"layer_name": "layer_1"},
                ]
            }
        },
        algorithm=algorithm,
    )

    out_csv = tmp_path / "routes.csv"
    if single_rd:
        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=[
                (
                    1,
                    [
                        (1, 1, "default"),
                        (1, 2, "default"),
                        (1, 3, "default"),
                        (1, 4, "default"),
                    ],
                    [(2, 6, "default")],
                ),
            ],
            route_attrs={
                (1, (1, 2, "default")): {"route_type": "A"},
                (1, (1, 4, "default")): {"my_attr": "B"},
                (1, (1, 3, "default")): {
                    "route_type": "C",
                    "my_attr": "D",
                    "final": True,
                },
            },
        )
    else:
        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=[
                (6, [(1, 1, "default")], [(2, 6, "default")]),
                (7, [(1, 2, "default")], [(2, 6, "default")]),
                (8, [(1, 3, "default")], [(2, 6, "default")]),
                (9, [(1, 4, "default")], [(2, 6, "default")]),
            ],
            route_attrs={
                (7, (1, 2, "default")): {"route_type": "A"},
                (9, (1, 4, "default")): {"my_attr": "B"},
                (8, (1, 3, "default")): {
                    "route_type": "C",
                    "my_attr": "D",
                    "final": True,
                },
            },
        )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 4

    first_route = output[
        (output["start_row"] == 1) & (output["start_col"] == 1)
    ].iloc[0]
    assert np.isnan(first_route["route_type"])
    assert np.isnan(first_route["my_attr"])
    assert np.isnan(first_route["final"])

    second_route = output[
        (output["start_row"] == 1) & (output["start_col"] == 2)
    ].iloc[0]
    assert second_route["route_type"] == "A"
    assert np.isnan(second_route["my_attr"])
    assert np.isnan(second_route["final"])

    third_route = output[
        (output["start_row"] == 1) & (output["start_col"] == 3)
    ].iloc[0]
    assert third_route["route_type"] == "C"
    assert third_route["my_attr"] == "D"
    assert third_route["final"]

    fourth_route = output[
        (output["start_row"] == 1) & (output["start_col"] == 4)
    ].iloc[0]
    assert np.isnan(fourth_route["route_type"])
    assert fourth_route["my_attr"] == "B"
    assert np.isnan(fourth_route["final"])


def test_validate_out_fp_ok(caplog):
    """Valid output file paths pass through unchanged without warnings"""
    assert _validate_out_fp("test/out.csv", save_paths=False) == Path(
        "test/out.csv"
    )
    assert _validate_out_fp("test/out.gpkg", save_paths=True) == Path(
        "test/out.gpkg"
    )

    for record in caplog.records:
        assert "the output file should have a" not in record.message


def test_validate_out_fp_bad_gpkg(assert_message_was_logged):
    """Invalid output file paths are corrected with warnings"""
    with pytest.warns(
        revrtWarning,
        match="When saving paths, the output file should have a '.gpkg'",
    ):
        out_fp = _validate_out_fp("test/out.csv", save_paths=True)

    assert out_fp == Path("test/out.gpkg")
    assert_message_was_logged(
        "When saving paths, the output file should have a '.gpkg'", "WARNING"
    )


def test_validate_out_fp_bad_csv(assert_message_was_logged):
    """Invalid output file paths are corrected with warnings"""
    with pytest.warns(
        revrtWarning,
        match="When not saving paths, the output file should have a '.csv'",
    ):
        out_fp = _validate_out_fp("test/out.gpkg", save_paths=False)

    assert out_fp == Path("test/out.csv")
    assert_message_was_logged(
        "When not saving paths, the output file should have a '.csv'",
        "WARNING",
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
