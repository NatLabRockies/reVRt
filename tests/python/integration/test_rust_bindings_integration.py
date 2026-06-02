"""revrt rust binding tests"""

import json
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
from skimage.graph import MCP_Geometric

from revrt import find_paths, RouteFinder
from revrt.routing.base import RoutingLayerManager, RoutingScenario

from revrt.utilities import LayeredFile


def _write_layers_to_layered_file(tmp_path, layers):
    """Write in-memory arrays to a layered-file test store"""

    first_layer = next(iter(layers.values()))
    height, width = first_layer.shape[1:]
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    layered_fp = tmp_path / "test_layered.zarr"
    layer_file = LayeredFile(layered_fp)
    for layer_name, layer_values in layers.items():
        da = xr.DataArray(
            layer_values,
            dims=("band", "y", "x"),
            coords={"y": y_coords, "x": x_coords},
        )
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.write_transform(transform)

        geotiff_fp = tmp_path / f"{layer_name}.tif"
        da.rio.to_raster(geotiff_fp, driver="GTiff")
        layer_file.write_geotiff_to_file(
            geotiff_fp,
            layer_name,
            overwrite=True,
        )

    return layered_fp


def test_find_paths_basic_single_route_layered_file(tmp_path):
    """Test routing using a LayeredFile-generated cost surface"""

    cost_values = np.array(
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

    height, width = cost_values.shape[1:]
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    da = xr.DataArray(
        cost_values,
        dims=("band", "y", "x"),
        coords={"y": y_coords, "x": x_coords},
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_transform(transform)

    geotiff_fp = tmp_path / "costs.tif"
    da.rio.to_raster(geotiff_fp, driver="GTiff")

    layered_fp = tmp_path / "test_layered.zarr"
    layer_file = LayeredFile(layered_fp)
    layer_file.write_geotiff_to_file(
        geotiff_fp,
        "test_costs",
        overwrite=True,
    )
    cost_definition = {
        "routing_options": {
            "default": {"cost_layers": [{"layer_name": "test_costs"}]}
        },
        "ignore_invalid_costs": True,
    }
    results = find_paths(
        zarr_fp=layered_fp,
        cost_function=json.dumps(cost_definition),
        start=[(1, 1)],
        end=[(2, 6)],
    )

    assert len(results) == 1
    test_path, test_cost, dropped_barrier_layers = results[0]

    mcp = MCP_Geometric(cost_values[0])
    costs, __ = mcp.find_costs(starts=[(1, 1)], ends=[(2, 6)])

    assert test_path == mcp.traceback((2, 6))
    assert np.isclose(test_cost, costs[(2, 6)])
    assert not dropped_barrier_layers


def test_find_paths_respects_hard_barrier_layered_file(tmp_path):
    """Test that hard barriers block routing for layered-file inputs"""

    layered_fp = _write_layers_to_layered_file(
        tmp_path,
        {
            "test_costs": np.ones((1, 3, 3), dtype=np.float32),
            "test_barrier": np.array(
                [
                    [
                        [1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ]
                ],
                dtype=np.float32,
            ),
        },
    )

    cost_definition = {
        "routing_options": {
            "default": {
                "cost_layers": [{"layer_name": "test_costs"}],
                "barrier_layers": [
                    {
                        "layer_name": "test_barrier",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1,
                    }
                ],
            }
        },
        "ignore_invalid_costs": False,
    }
    results = find_paths(
        zarr_fp=layered_fp,
        cost_function=json.dumps(cost_definition),
        start=[(1, 1)],
        end=[(0, 0)],
    )

    assert results == []


def test_find_paths_respects_not_equal_barrier_layered_file(tmp_path):
    """Test that not-equal barriers block routing for layered-file inputs"""

    layered_fp = _write_layers_to_layered_file(
        tmp_path,
        {
            "test_costs": np.ones((1, 3, 3), dtype=np.float32),
            "test_barrier": np.array(
                [
                    [
                        [1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ]
                ],
                dtype=np.float32,
            ),
        },
    )

    cost_definition = {
        "routing_options": {
            "default": {
                "cost_layers": [{"layer_name": "test_costs"}],
                "barrier_layers": [
                    {
                        "layer_name": "test_barrier",
                        "barrier_operator": "ne",
                        "barrier_threshold": 0,
                    }
                ],
            }
        },
        "ignore_invalid_costs": False,
    }
    results = find_paths(
        zarr_fp=layered_fp,
        cost_function=json.dumps(cost_definition),
        start=[(1, 1)],
        end=[(0, 0)],
    )

    assert results == []


@pytest.mark.parametrize(
    "algorithm",
    [
        "astar",
        "dijkstra",
        "long-range-astar",
        "long-range-dijkstra",
        "bidirectional-long-range-dijkstra",
    ],
)
def test_route_finder_basic_single_route_layered_file(tmp_path, algorithm):
    """Test routing using a LayeredFile-generated cost surface"""

    cost_values = np.array(
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

    height, width = cost_values.shape[1:]
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    da = xr.DataArray(
        cost_values,
        dims=("band", "y", "x"),
        coords={"y": y_coords, "x": x_coords},
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_transform(transform)

    geotiff_fp = tmp_path / "costs.tif"
    da.rio.to_raster(geotiff_fp, driver="GTiff")

    layered_fp = tmp_path / "test_layered.zarr"
    layer_file = LayeredFile(layered_fp)
    layer_file.write_geotiff_to_file(
        geotiff_fp,
        "test_costs",
        overwrite=True,
    )
    cost_definition = {
        "routing_options": {
            "default": {"cost_layers": [{"layer_name": "test_costs"}]}
        }
    }
    routing_results = RouteFinder(
        zarr_fp=layered_fp,
        cost_function=json.dumps(cost_definition),
        route_definitions=[
            (2, [(1, 1)], [(2, 6)]),
            (4, [(1, 2)], [(1000, 1000)]),
        ],
        algorithm=algorithm,
    )

    for route_id, solutions in routing_results:
        if route_id == 4:
            assert len(solutions) == 0
        else:
            assert route_id == 2
            assert len(solutions) == 1
            test_path, test_cost, dropped_barrier_layers = solutions[0][:3]
            assert dropped_barrier_layers == []
            if len(solutions[0]) > 3:
                assert solutions[0][3] == []

    mcp = MCP_Geometric(cost_values[0])
    costs, __ = mcp.find_costs(starts=[(1, 1)], ends=[(2, 6)])

    assert test_path == mcp.traceback((2, 6))
    assert np.isclose(test_cost, costs[(2, 6)])


@pytest.mark.parametrize(
    "algorithm",
    [
        "astar",
        "dijkstra",
        "long-range-astar",
        "long-range-dijkstra",
        "bidirectional-long-range-dijkstra",
    ],
)
def test_route_finder_retries_soft_barriers_layered_file(tmp_path, algorithm):
    """Test mixed hard and soft barriers for layered-file inputs"""

    layered_fp = _write_layers_to_layered_file(
        tmp_path,
        {
            "test_costs": np.ones((1, 3, 5), dtype=np.float32),
            "hard_barrier": np.array(
                [
                    [
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0],
                    ]
                ],
                dtype=np.float32,
            ),
            "soft_barrier": np.array(
                [
                    [
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ],
                dtype=np.float32,
            ),
        },
    )

    cost_definition = {
        "routing_options": {
            "default": {
                "cost_layers": [{"layer_name": "test_costs"}],
                "barrier_layers": [
                    {
                        "layer_name": "hard_barrier",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1,
                    },
                    {
                        "layer_name": "soft_barrier",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1,
                        "barrier_importance": 1,
                    },
                ],
            }
        },
        "ignore_invalid_costs": False,
    }
    results = list(
        RouteFinder(
            zarr_fp=layered_fp,
            cost_function=json.dumps(cost_definition),
            route_definitions=[(7, [(0, 0), (2, 0)], [(0, 4), (2, 4)])],
            algorithm=algorithm,
        )
    )

    assert len(results) == 1
    route_id, solutions = results[0]
    assert route_id == 7
    assert len(solutions) == 2

    solutions_by_start = {
        tuple(solution[0][0]): solution for solution in solutions
    }

    top_solution = solutions_by_start[(0, 0)]
    assert top_solution[0][-1] == (0, 4)
    assert top_solution[1] > 0
    assert top_solution[2] == ["soft_barrier"]
    if len(top_solution) > 3:
        assert top_solution[3] == [1]

    bottom_solution = solutions_by_start[(2, 0)]
    assert bottom_solution[0][-1] == (2, 4)
    assert bottom_solution[1] > 0
    assert bottom_solution[2] == []
    if len(bottom_solution) > 3:
        assert bottom_solution[3] == []


@pytest.mark.parametrize(
    "algorithm",
    [
        "astar",
        "dijkstra",
        "long-range-astar",
        "long-range-dijkstra",
        "bidirectional-long-range-dijkstra",
    ],
)
def test_route_finder_drops_multiple_soft_barrier_groups_layered_file(
    tmp_path, algorithm
):
    """Layered-file routing drops soft barriers in importance order"""

    layered_fp = _write_layers_to_layered_file(
        tmp_path,
        {
            "test_costs": np.ones((1, 3, 5), dtype=np.float32),
            "soft_barrier_low": np.array(
                [
                    [
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                    ]
                ],
                dtype=np.float32,
            ),
            "soft_barrier_high": np.array(
                [
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0],
                    ]
                ],
                dtype=np.float32,
            ),
        },
    )

    cost_definition = {
        "routing_options": {
            "default": {
                "cost_layers": [{"layer_name": "test_costs"}],
                "barrier_layers": [
                    {
                        "layer_name": "soft_barrier_low",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1,
                        "barrier_importance": 1,
                    },
                    {
                        "layer_name": "soft_barrier_high",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1,
                        "barrier_importance": 2,
                    },
                ],
            }
        },
        "ignore_invalid_costs": False,
    }
    results = list(
        RouteFinder(
            zarr_fp=layered_fp,
            cost_function=json.dumps(cost_definition),
            route_definitions=[(9, [(1, 0)], [(1, 4)])],
            algorithm=algorithm,
        )
    )

    assert len(results) == 1
    route_id, solutions = results[0]
    assert route_id == 9
    assert len(solutions) == 1

    solution = solutions[0]
    assert solution[0][0] == (1, 0)
    assert solution[0][-1] == (1, 4)
    assert solution[1] > 0
    assert solution[2] == ["soft_barrier_low", "soft_barrier_high"]
    if len(solution) > 3:
        assert solution[3] == [1, 2]


@pytest.mark.parametrize(
    "algorithm",
    [
        "astar",
        "dijkstra",
        "long-range-astar",
        "long-range-dijkstra",
        "bidirectional-long-range-dijkstra",
    ],
)
def test_route_finder_writes_routing_layer_to_expected_path_layered_file(
    tmp_path, algorithm
):
    """Test RouteFinder routing layer output for LayeredFile inputs"""

    cost_values = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        ],
        dtype=np.float32,
    )

    height, width = cost_values.shape[1:]
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    da = xr.DataArray(
        cost_values,
        dims=("band", "y", "x"),
        coords={"y": y_coords, "x": x_coords},
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_transform(transform)

    geotiff_fp = tmp_path / "costs.tif"
    da.rio.to_raster(geotiff_fp, driver="GTiff")

    layered_fp = tmp_path / "test_layered.zarr"
    layer_file = LayeredFile(layered_fp)
    layer_file.write_geotiff_to_file(
        geotiff_fp,
        "test_costs",
        overwrite=True,
    )

    cost_definition = {
        "routing_options": {
            "default": {"cost_layers": [{"layer_name": "test_costs"}]}
        },
        "ignore_invalid_costs": True,
    }
    routing_layer_out_fp = tmp_path / "routing_layer.zarr"

    routing_results = RouteFinder(
        zarr_fp=layered_fp,
        cost_function=json.dumps(cost_definition),
        route_definitions=[
            (11, [(0, 0)], [(2, 2)]),
        ],
        routing_layer_out_fp=routing_layer_out_fp,
        algorithm=algorithm,
    )
    list(routing_results)

    assert routing_layer_out_fp.exists()

    with xr.open_dataset(
        routing_layer_out_fp, engine="zarr", consolidated=False
    ) as routing_ds:
        assert "cost" in routing_ds
        written_costs = routing_ds["cost"].astype(np.float32).values
        assert written_costs.shape == (1, 3, 3)

    scenario = RoutingScenario(
        cost_fpath=layered_fp,
        cost_layers=[{"layer_name": "test_costs"}],
        ignore_invalid_costs=True,
    )
    routing_layers = RoutingLayerManager(scenario).build()
    try:
        expected_costs = (
            routing_layers.final_routing_layer.astype(np.float32)
            .compute()
            .values
        )
    finally:
        routing_layers.close()

    np.testing.assert_allclose(
        written_costs[0],
        expected_costs,
        rtol=1e-6,
        atol=1e-6,
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
