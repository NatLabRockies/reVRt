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
from revrt.routing.base import (
    BatchRouteProcessor,
    RouteMetrics,
    RoutingLayerManager,
    RoutingScenario,
    _validate_out_fp,
)
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
                [ 0, -1, -1, -1, -1, -1, -1,  0],  # noqa: E201, E241
                [ 0, -1, -1, -1, -1, -1, -1,  3],  # noqa: E201, E241
                [ 1, -1, -1, -1, -1, -1, -1,  1],  # noqa: E201, E241
                [ 1, -1, -1, -1, -1, -1, -1,  1],  # noqa: E201, E241
                [ 1, -1, -1, -1, -1, -1, -1,  1],  # noqa: E201, E241
                [ 0,  1,  1,  1,  1,  1,  1,  1],  # noqa: E201, E241
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


def test_basic_single_route_layered_file_short_path(
    sample_layered_data, tmp_path
):
    """Test routing using a LayeredFile-generated cost surface"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(1, 2)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 1
    route = output.iloc[0]
    assert route["cost"] == pytest.approx((1 + 2) / 2)
    assert route["length_km"] == pytest.approx(1 / 1000)
    assert route["cost"] == route["optimized_objective"]


def test_routing_scenario_normalizes_algorithm(sample_layered_data):
    """RoutingScenario normalizes supported algorithm aliases"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        algorithm="long-range",
    )

    # assert scenario.algorithm is RoutingAlgorithm.LONG_RANGE
    assert "algorithm: long-range" in repr(scenario)


def test_routing_scenario_rejects_invalid_algorithm(sample_layered_data):
    """RoutingScenario raises on unsupported algorithm names"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        algorithm="bellman_ford",
    )
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(2, 6)]),
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

    monkeypatch.setattr("revrt.routing.base.RouteFinder", FakeRouteFinder)

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        algorithm="dijkstra",
    )
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(2, 6)]),
        ],
    )

    assert list(route_computer._route_results()) == []
    assert captured["algorithm"] == "dijkstra"


def test_basic_single_route_layered_file(sample_layered_data, tmp_path):
    """Test routing using a LayeredFile-generated cost surface"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(2, 6)]),
            ([(1, 2)], [(2, 6)]),
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
def test_multi_layer_route_layered_file(
    sample_layered_data, tmp_path, single_rd
):
    """Test routing across multiple cost layers"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1"},
            {"layer_name": "layer_2"},
        ],
    )

    out_csv = tmp_path / "routes.csv"
    if single_rd:
        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=[
                (1, [(1, 1), (1, 2)], [(2, 6)]),
            ],
            route_attrs={
                (1, (1, 2)): {"route_type": "A"},
            },
        )
    else:
        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=[
                (1, [(1, 1)], [(2, 6)]),
                (2, [(1, 2)], [(2, 6)]),
            ],
            route_attrs={
                (2, (1, 2)): {"route_type": "A"},
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
    assert first_route["layer_1_cost"] == pytest.approx(
        17.571068,
        rel=1e-4,
    )
    assert first_route["layer_2_cost"] == pytest.approx(
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
    assert second_route["layer_1_cost"] == pytest.approx(
        16.071068,
        rel=1e-4,
    )
    assert second_route["layer_2_cost"] == pytest.approx(
        9.035534,
        rel=1e-4,
    )
    assert np.isclose(
        second_route["cost"], second_route["optimized_objective"], rtol=1e-6
    )
    assert second_route["route_type"] == "A"


def test_save_paths_returns_expected_geometry(sample_layered_data, tmp_path):
    """Saving paths returns expected geometries for each route"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_gpkg = tmp_path / "routes.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(2, 6)]),
            ([(1, 2)], [(2, 6)]),
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


def test_empty_route_definitions_returns_empty_dataframe(
    sample_layered_data, tmp_path
):
    """Empty route definitions return an empty dataframe"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)
    assert not out_csv.exists()


def test_empty_route_definitions_returns_empty_geo_dataframe(
    sample_layered_data, tmp_path
):
    """Empty route definitions return an empty dataframe"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_gpkg = tmp_path / "routes.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[],
    )
    route_computer.process(out_fp=out_gpkg, save_paths=True)
    assert not out_gpkg.exists()


def test_multi_layer_route_with_multiplier(sample_layered_data, tmp_path):
    """Test routing with multiple layers and a scalar multiplier"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1"},
            {
                "layer_name": "layer_2",
                "multiplier_scalar": 0.5,
            },
        ],
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(2, 6)]),
            ([(1, 2)], [(2, 6)]),
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
    assert first_route["layer_1_cost"] == pytest.approx(
        17.571068,
        rel=1e-4,
    )
    assert first_route["layer_2_cost"] == pytest.approx(
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
    assert second_route["layer_1_cost"] == pytest.approx(
        16.071068,
        rel=1e-4,
    )
    assert second_route["layer_2_cost"] == pytest.approx(
        4.517767,
        rel=1e-4,
    )
    assert np.isclose(
        second_route["cost"],
        second_route["optimized_objective"],
        rtol=1e-4,
    )


def test_multi_layer_route_with_scalar_and_layer_multipliers(
    sample_layered_data, tmp_path
):
    """Test routing when combining scalar and layer multipliers"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1"},
            {"layer_name": "layer_2", "multiplier_scalar": 0.5},
            {
                "layer_name": "layer_3",
                "multiplier_layer": "layer_4",
            },
            {
                "layer_name": "layer_5",
                "multiplier_scalar": 2,
                "multiplier_layer": "layer_4",
            },
        ],
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(1, 2)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 1

    route = output.iloc[0]
    assert route["cost"] == pytest.approx(2.0, rel=1e-4)
    assert route["length_km"] == pytest.approx(0.001, rel=1e-4)
    assert route["layer_1_cost"] == pytest.approx(1.5, rel=1e-4)
    assert route["layer_2_cost"] == pytest.approx(0.5, rel=1e-4)
    assert route["layer_3_cost"] == pytest.approx(0.0, abs=1e-8)
    assert route["layer_5_cost"] == pytest.approx(0.0, abs=1e-8)
    assert route["layer_1_length_km"] == pytest.approx(0.001, rel=1e-4)
    assert route["layer_2_length_km"] == pytest.approx(0.001, rel=1e-4)
    assert route["layer_3_length_km"] == pytest.approx(0.0, abs=1e-8)
    assert route["layer_5_length_km"] == pytest.approx(0.0, abs=1e-8)
    assert np.isclose(route["cost"], route["optimized_objective"], rtol=1e-6)


def test_routing_with_tracked_layers(sample_layered_data, tmp_path):
    """Tracked layers report aggregated stats alongside routing results"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        tracked_layers={
            "layer_1": "mean",
            "layer_2": "max",
            "layer_3": "min",
        },
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(1, 2)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 1
    route = output.iloc[0]

    assert {
        "layer_1_mean",
        "layer_2_max",
        "layer_3_min",
    }.issubset(route.keys())

    assert route["layer_1_mean"] == pytest.approx(1.5)
    assert route["layer_2_max"] == pytest.approx(1.0)
    assert route["layer_3_min"] == pytest.approx(2.0)


@pytest.mark.parametrize("use_friction", [True, False])
def test_start_point_on_barrier_returns_no_route(
    sample_layered_data, assert_message_was_logged, use_friction, tmp_path
):
    """If the start point is on a barrier (cost <= 0) no route is returned"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_6"}],
    )
    if use_friction:
        scenario.friction_layers = [
            {"mask": "layer_5", "multiplier_scalar": -10}
        ]

    out_csv = tmp_path / "routes.csv"

    # (3, 1) in layer_6 is -1 -> treated as barrier
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(3, 1)], [(2, 6)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the start points have an invalid cost (must be > 0): "
        "{(3, 1)}",
        "WARNING",
    )
    assert_message_was_logged(
        "All start points are invalid for route with ID 0: [(3, 1)]",
        "WARNING",
    )
    assert not out_csv.exists()


def test_invalid_start_point_logged(
    sample_layered_data, assert_message_was_logged, tmp_path
):
    """Test that only the invalid starting point is logged"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_csv = tmp_path / "routes.csv"

    # (0, 3) in layer_1 is 0 -> treated as barrier
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1), (0, 3)], [(2, 6)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the start points have an invalid cost (must be > 0): "
        "{(0, 3)}",
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


def test_invalid_start_point_explicitly_allowed(
    sample_layered_data, assert_message_was_logged, tmp_path
):
    """Test out-of-bounds points logging when ignore_invalid_costs is False"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        ignore_invalid_costs=False,
    )

    out_csv = tmp_path / "routes.csv"

    # (0, 3) in layer_1 is 0 -> treated as barrier
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1), (0, 3), (10000, 10000)], [(2, 6), (20000, 20000)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the start points are out of bounds for an array of "
        "shape (7, 8): [(10000, 10000)]",
        "WARNING",
    )
    assert_message_was_logged(
        "One or more of the end points are out of bounds for an array of "
        "shape (7, 8): [(20000, 20000)]",
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


def test_some_endpoints_include_barriers_but_one_valid(
    sample_layered_data, tmp_path
):
    """If some end points <=0 but at least one is valid, route is found"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_csv = tmp_path / "routes.csv"

    # include one barrier end (0,3) and one valid end (2,6)
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(0, 3), (2, 6)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    output = pd.read_csv(out_csv)
    assert len(output) == 1
    # At least one valid endpoint must be reached and cost must be positive.
    route = output.iloc[0]
    assert route["cost"] > 0

    end_row = int(route["end_row"])
    end_col = int(route["end_col"])
    assert (end_row, end_col) == (2, 6)


def test_all_endpoints_are_barriers_returns_no_route(
    sample_layered_data, assert_message_was_logged, tmp_path
):
    """If all end points are barriers, no route is returned"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(0, 3), (0, 7)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "None of the end points have a valid cost (must be > 0): "
        "[(0, 3), (0, 7)]",
        "ERROR",
    )
    assert not out_csv.exists()


def test_bad_start_index_returns_no_route(
    sample_layered_data, assert_message_was_logged, tmp_path
):
    """If any points are out-of-bounds, no route is returned"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(10000, 10000)], [(0, 3), (0, 7)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the start points are out of bounds for an array of "
        "shape (7, 8): [(10000, 10000)]",
        "WARNING",
    )
    assert not out_csv.exists()


def test_bad_end_index_returns_no_route(
    sample_layered_data, assert_message_was_logged, tmp_path
):
    """If any points are out-of-bounds, no route is returned"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(10000, 10000)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the end points are out of bounds for an array of "
        "shape (7, 8): [(10000, 10000)]",
        "WARNING",
    )
    assert_message_was_logged(
        "All end points are invalid for route with ID 0: [(10000, 10000)]",
        "WARNING",
    )
    assert not out_csv.exists()


def test_bad_index_skipped(
    sample_layered_data, assert_message_was_logged, tmp_path
):
    """Out-of-bounds points are skipped and routes are compute"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(10000, 10000), (1, 1)], [(0, 3), (2, 6), (20000, 20000)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)

    assert_message_was_logged(
        "One or more of the start points are out of bounds for an array of "
        "shape (7, 8): [(10000, 10000)]",
        "WARNING",
    )
    assert_message_was_logged(
        "One or more of the end points are out of bounds for an array of "
        "shape (7, 8): [(20000, 20000)]",
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


def test_routing_scenario_repr_contains_fields(sample_layered_data):
    """RoutingScenario repr surfaces configured layer metadata"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        friction_layers=[{"mask": "layer_2"}],
        cost_multiplier_layer="layer_3",
        cost_multiplier_scalar=1.5,
    )

    representation = repr(scenario)

    assert "layer_1" in representation
    assert "layer_2" in representation
    assert "cost_multiplier_scalar: 1.5" in representation


def test_missing_cost_layer_raises_key_error(sample_layered_data, tmp_path):
    """Missing layers surface a revrtKeyError during build"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "not_there"}],
    )

    out_csv = tmp_path / "routes.csv"
    with pytest.raises(
        revrtKeyError, match="Did not find layer 'not_there' in cost file"
    ):
        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=[
                ([(1, 1)], [(1, 2)]),
            ],
        )
        route_computer.process(out_fp=out_csv, save_paths=False)


def test_cost_multiplier_layer_and_scalar_applied(sample_layered_data):
    """Cost multipliers scale base costs before routing aggregation"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        cost_multiplier_layer="layer_3",
        cost_multiplier_scalar=2.0,
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        cost_val = routing_layers.cost.isel(y=1, x=1).compute().item()
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
        expected = layer_one * layer_three * scenario.cost_multiplier_scalar

        assert cost_val == pytest.approx(expected)
    finally:
        routing_layers.close()


def test_length_invariant_layer_costs_ignore_path_length(
    sample_layered_data,
):
    """Length invariant cost layers ignore per-cell distances"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1"},
            {"layer_name": "layer_2", "is_invariant": True},
        ],
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        route = [(1, 1), (1, 2)]
        result = RouteMetrics(
            routing_layers,
            route,
            optimized_objective=0.0,
        ).compute()

        layer_two = (
            routing_layers._layer_fh["layer_2"]
            .isel(band=0)
            .compute()
            .to_numpy()
        )
        expected = sum(layer_two[row, col] for row, col in route[1:])

        assert result["layer_2_cost"] == pytest.approx(expected)
    finally:
        routing_layers.close()


def test_length_invariant_layers_sum_raw_values(sample_layered_data, tmp_path):
    """Length invariant layers sum raw cell values without distance scaling"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1"},
            {"layer_name": "layer_2", "is_invariant": True},
        ],
    )

    out_gpkg = tmp_path / "routes.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(2, 6)]),
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
        start_row, start_col = rasterio.transform.rowcol(
            ds.rio.transform(), *route["geometry"].coords[0]
        )
        rows, cols = np.where(mask)
        expected_invariant_cost = sum(
            layer_two.isel(y=row, x=col).item()
            for row, col in zip(rows, cols, strict=True)
            if (row, col) != (start_row, start_col)
        )

    assert route["layer_2_cost"] == pytest.approx(
        expected_invariant_cost,
        rel=1e-6,
    )
    assert route["cost"] == pytest.approx(
        route["layer_1_cost"] + expected_invariant_cost,
        rel=1e-6,
    )
    assert route["layer_2_length_km"] == pytest.approx(
        route["length_km"],
        rel=1e-6,
    )
    assert route["cost"] == pytest.approx(
        route["optimized_objective"],
        rel=1e-5,
    )


def test_length_invariant_hidden_and_friction_layers(
    sample_layered_data, tmp_path
):
    """Combined layer settings preserve cost reporting expectations"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1"},
            {"layer_name": "layer_2", "is_invariant": True},
            {
                "layer_name": "layer_5",
                "multiplier_scalar": 100,
                "include_in_final_cost": False,
                "include_in_report": True,
            },
        ],
        friction_layers=[
            {
                "mask": "layer_4",
                "multiplier_scalar": 0.5,
            },
        ],
    )

    out_gpkg = tmp_path / "routes.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(1, 1)], [(2, 6)]),
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
    assert route["layer_2_length_km"] == pytest.approx(
        route["length_km"],
        rel=1e-6,
    )
    assert route["layer_1_cost"] == pytest.approx(26.156855, rel=1e-6)
    assert route["layer_2_cost"] == pytest.approx(18.0)
    assert route["cost"] == pytest.approx(
        route["layer_1_cost"] + route["layer_2_cost"],
        rel=1e-6,
    )
    assert route["cost"] == pytest.approx(
        44.15685424949238,
        rel=1e-6,
    )
    assert route["optimized_objective"] == pytest.approx(
        276.3262939453125,
        rel=1e-6,
    )
    assert route["optimized_objective"] > route["cost"]

    assert route["layer_5_cost"] == pytest.approx(
        170.71068,
        rel=1e-6,
    )
    assert route["layer_5_length_km"] == pytest.approx(
        0.0017071,
        rel=1e-4,
    )
    assert list(route["geometry"].coords) == [
        (1.5, 5.5),
        (3.5, 3.5),
        (6.5, 3.5),
        (6.5, 4.5),
    ]


def test_tracked_layers_invalid_configs_warn(
    sample_layered_data, assert_message_was_logged
):
    """Tracked layer config issues emit revrtWarning messages"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        tracked_layers={
            "layer_1": "does_not_exist",
            "missing_layer": "mean",
        },
    )

    with pytest.warns(revrtWarning) as warning_record:
        routing_layers = RoutingLayerManager(scenario).build()

    assert_message_was_logged("Did not find layer", "WARNING")
    assert_message_was_logged("Did not find method", "WARNING")

    try:
        assert len(warning_record) == 2
    finally:
        routing_layers.close()


def test_friction_layers_and_lcp_agg_costs(sample_layered_data):
    """Friction layers may include cost stack and tracked layer toggles"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1", "include_in_report": False},
            {
                "layer_name": "layer_2",
                "multiplier_scalar": 0.5,
                "include_in_report": True,
                "include_in_final_cost": False,
            },
        ],
        friction_layers=[
            {
                "mask": "layer_3",
                "multiplier_scalar": 0.1,
            },
        ],
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        tracked_names = {layer.name for layer in routing_layers.tracked_layers}
        assert "layer_1" not in tracked_names
        assert "layer_2" in tracked_names

        base_value = routing_layers.cost.isel(y=1, x=1).compute().item()
        final_value = (
            routing_layers.final_routing_layer.isel(y=1, x=1).compute().item()
        )

        assert final_value > base_value
    finally:
        routing_layers.close()


def test_friction_layer_include_in_report_adds_tracker(sample_layered_data):
    """Friction layers flagged for reports extend tracked layers"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        friction_layers=[
            {
                "mask": "layer_4",
                "multiplier_scalar": 0.5,
                "include_in_report": True,
            }
        ],
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        tracked_names = {layer.name for layer in routing_layers.tracked_layers}
        assert "layer_4" in tracked_names
    finally:
        routing_layers.close()


def test_friction_layer_influences_objective_without_reporting(
    sample_layered_data, tmp_path
):
    """Friction layers alter routing objective without affecting reports"""

    base_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    friction_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        friction_layers=[
            {
                "mask": "layer_4",
                "multiplier_scalar": 0.5,
            }
        ],
    )

    base_csv = tmp_path / "base.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=base_scenario,
        route_definitions=[
            ([(1, 1)], [(2, 6)]),
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
            ([(1, 1)], [(2, 6)]),
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
    assert "layer_2_cost" not in friction_route
    assert "layer_2_length_km" not in friction_route


def test_friction_layer_influences_objective(sample_layered_data, tmp_path):
    """Friction layers alter routing objective without affecting reports"""

    base_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    friction_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        friction_layers=[
            {
                "mask": "layer_5",
                "multiplier_scalar": 1000,
            }
        ],
    )

    base_csv = tmp_path / "base.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=base_scenario,
        route_definitions=[
            ([(1, 1)], [(3, 5)]),
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
            ([(1, 1)], [(3, 5)]),
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

    assert "layer_5_cost" not in friction_route
    assert "layer_5_length_km" not in friction_route


def test_negative_friction_layer_influences_objective(
    sample_layered_data, tmp_path
):
    """Friction layers alter routing objective without affecting reports"""

    base_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    friction_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        friction_layers=[
            {
                "mask": "layer_5",
                "multiplier_scalar": -10,
            }
        ],
    )

    base_gpkg = tmp_path / "base.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=base_scenario,
        route_definitions=[
            ([(1, 1)], [(2, 6)]),
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
            ([(1, 1)], [(2, 6)]),
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

    assert "layer_5_cost" not in friction_route
    assert "layer_5_length_km" not in friction_route


def test_negative_friction_layer_does_not_go_thru_barrier(
    sample_layered_data, tmp_path
):
    """Friction layers alter routing objective without affecting reports"""

    base_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_6"}],
    )

    friction_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_6"}],
        friction_layers=[
            {
                "mask": "layer_5",
                "multiplier_scalar": -10,
            }
        ],
    )

    base_gpkg = tmp_path / "base.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=base_scenario,
        route_definitions=[
            ([(4, 0)], [(2, 7)]),
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
            ([(4, 0)], [(2, 7)]),
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

    assert "layer_5_cost" not in friction_route
    assert "layer_5_length_km" not in friction_route


def test_include_in_final_cost_false_behaves_like_friction(
    sample_layered_data, tmp_path
):
    """Non-final cost layers steer routing but stay out of reports"""

    base_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    out_gpkg = tmp_path / "base.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=base_scenario,
        route_definitions=[
            ([(1, 1)], [(3, 5)]),
        ],
    )
    route_computer.process(out_fp=out_gpkg, save_paths=True)

    base_output = gpd.read_file(out_gpkg)
    assert len(base_output) == 1
    base_route = base_output.iloc[0]

    penalized_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1"},
            {
                "layer_name": "layer_5",
                "multiplier_scalar": 1000,
                "include_in_final_cost": False,
                "include_in_report": False,
            },
        ],
    )

    penalized_gpkg = tmp_path / "penalized.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=penalized_scenario,
        route_definitions=[
            ([(1, 1)], [(3, 5)]),
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
        penalized_route["layer_1_cost"],
        rel=1e-6,
    )
    assert "layer_5_cost" not in penalized_route


def test_route_result_geom_returns_point_for_single_cell(sample_layered_data):
    """RouteMetrics.geom returns a Point geometry for single-cell routes"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        route = [(1, 1)]
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
        cost_layers=[{"layer_name": "layer_1"}],
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        layer = next(
            tracked
            for tracked in routing_layers.tracked_layers
            if tracked.name == "layer_1"
        )
        route = [(1, 1), (1, 2), (2, 3)]
        metrics = layer.compute(route, abs(routing_layers.transform.a))

        assert metrics["layer_1_length_km"] >= 0
    finally:
        routing_layers.close()


def test_route_result_cached_properties_reuse_computed_values(
    sample_layered_data,
):
    """RouteMetrics caches per-route lengths after first computation"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        route = [(1, 1), (1, 2), (2, 3)]
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
        cost_layers=[{"layer_name": "layer_1"}],
    )

    routing_layers = RoutingLayerManager(scenario).build()
    try:
        route = [(1, 1), (1, 2), (2, 3)]
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
        cost_layers=[{"layer_name": "layer_1"}],
    )

    routing_layers = RoutingLayerManager(scenario, chunks=None).build()
    try:
        layer = next(
            tracked
            for tracked in routing_layers.tracked_layers
            if tracked.name == "layer_1"
        )
        route = [(1, 1), (1, 2), (2, 3)]
        metrics = layer.compute(route, abs(routing_layers.transform.a))

        assert metrics["layer_1_cost"] > 0
        assert metrics["layer_1_length_km"] >= 0
    finally:
        routing_layers.close()


def test_negative_cost_path_returns_no_route(sample_layered_data, tmp_path):
    """If all points between start and end are negative, return no route"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_6"},
            {"layer_name": "layer_4", "multiplier_scalar": -3},
        ],
        friction_layers=[{"mask": "layer_5", "multiplier_scalar": -10}],
    )

    out_csv = tmp_path / "routes.csv"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(4, 0)], [(2, 7)]),
        ],
    )
    route_computer.process(out_fp=out_csv, save_paths=False)
    assert not out_csv.exists()


def test_friction_layer_with_layer_name_warns(sample_layered_data):
    """Layer name on friction layer drops with deprecation warning"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        friction_layers=[
            {
                "layer_name": "legacy_friction",
                "mask": "layer_4",
            }
        ],
    )

    with pytest.warns(revrtDeprecationWarning) as warning_record:
        layers_for_rust = list(scenario._friction_layers_for_rust())

    assert len(warning_record) == 1
    friction_payload = layers_for_rust[-1]
    assert "layer_name" not in friction_payload
    assert friction_payload["multiplier_layer"] == "layer_4"


def test_friction_layer_with_multiplier_layer_only(sample_layered_data):
    """Friction layers support multiplier layer without mask"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        friction_layers=[
            {
                "multiplier_layer": "layer_4",
                "multiplier_scalar": 0.25,
            }
        ],
    )

    layers_for_rust = list(scenario._friction_layers_for_rust())
    friction_payload = layers_for_rust[-1]
    assert friction_payload["multiplier_layer"] == "layer_4"
    assert "mask" not in friction_payload


def test_friction_layer_requires_mask(sample_layered_data):
    """Friction layer build enforces presence of mask metadata"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        friction_layers=[{"multiplier_scalar": 5}],
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


def test_soft_barrier_setting_controls_barrier_value(sample_layered_data):
    """Soft barriers convert impassable cells to large positive costs"""

    hard_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        ignore_invalid_costs=True,
    )
    hard_layers = RoutingLayerManager(hard_scenario).build()
    try:
        hard_value = (
            hard_layers.final_routing_layer.isel(y=0, x=3).compute().item()
        )
    finally:
        hard_layers.close()

    soft_scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        ignore_invalid_costs=False,
    )
    soft_layers = RoutingLayerManager(soft_scenario).build()
    try:
        soft_value = (
            soft_layers.final_routing_layer.isel(y=0, x=3).compute().item()
        )
        assert hard_value == -1
        assert soft_value > 0
        assert soft_value > abs(hard_value)
    finally:
        soft_layers.close()


@pytest.mark.parametrize("ignore_invalid_costs", [True, False])
def test_soft_barrier(sample_layered_data, ignore_invalid_costs, tmp_path):
    """Test that soft barriers work as expected in point-to-many routing"""
    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_7"}],
        ignore_invalid_costs=ignore_invalid_costs,
    )

    out_gpkg = tmp_path / "routes.gpkg"
    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[
            ([(4, 0)], [(4, 5)]),
        ],
    )
    route_computer.process(out_fp=out_gpkg, save_paths=True)

    if ignore_invalid_costs:
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
def test_route_many_attrs(sample_layered_data, tmp_path, single_rd):
    """Test routing with multiple layers and a scalar multiplier"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1"},
        ],
    )

    out_csv = tmp_path / "routes.csv"
    if single_rd:
        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=[
                (1, [(1, 1), (1, 2), (1, 3), (1, 4)], [(2, 6)]),
            ],
            route_attrs={
                (1, (1, 2)): {"route_type": "A"},
                (1, (1, 4)): {"my_attr": "B"},
                (1, (1, 3)): {
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
                (6, [(1, 1)], [(2, 6)]),
                (7, [(1, 2)], [(2, 6)]),
                (8, [(1, 3)], [(2, 6)]),
                (9, [(1, 4)], [(2, 6)]),
            ],
            route_attrs={
                (7, (1, 2)): {"route_type": "A"},
                (9, (1, 4)): {"my_attr": "B"},
                (8, (1, 3)): {
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
