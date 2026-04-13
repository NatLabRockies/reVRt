"""revrt rust binding tests"""

import json
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
from skimage.graph import MCP_Geometric

from revrt import RouteFinder, find_paths, simplify_using_slopes
from revrt.routing.base import RoutingLayerManager, RoutingScenario


def test_find_paths_basic_single_route(tmp_path):
    """Test a basic routing invocation"""

    da = xr.DataArray(
        np.array(
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
        ),
        dims=("band", "y", "x"),
    )

    test_cost_fp = tmp_path / "test.zarr"
    ds = xr.Dataset({"test_costs": da})
    ds["test_costs"].encoding = {"fill_value": 1_000.0, "_FillValue": 1_000.0}
    ds.chunk({"x": 4, "y": 3}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    cost_definition = {
        "cost_layers": [{"layer_name": "test_costs"}],
        "ignore_invalid_costs": True,
    }
    results = find_paths(
        zarr_fp=test_cost_fp,
        cost_function=json.dumps(cost_definition),
        start=[(1, 1)],
        end=[(2, 6)],
    )

    assert len(results) == 1
    test_path, test_cost = results[0]

    mcp = MCP_Geometric(da.values[0])
    costs, __ = mcp.find_costs(starts=[(1, 1)], ends=[(2, 6)])

    assert test_path == mcp.traceback((2, 6))
    assert np.isclose(test_cost, costs[(2, 6)])


@pytest.mark.parametrize(
    "algorithm",
    [
        "astar",
        "dijkstra",
        "long-range-dijkstra",
        "bidirectional-long-range-dijkstra",
    ],
)
def test_route_finder_basic_single_route(tmp_path, algorithm):
    """Test a basic routing invocation"""

    da = xr.DataArray(
        np.array(
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
        ),
        dims=("band", "y", "x"),
    )

    test_cost_fp = tmp_path / "test.zarr"
    ds = xr.Dataset({"test_costs": da})
    ds["test_costs"].encoding = {"fill_value": 1_000.0, "_FillValue": 1_000.0}
    ds.chunk({"x": 4, "y": 3}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    cost_definition = {"cost_layers": [{"layer_name": "test_costs"}]}
    routing_results = RouteFinder(
        zarr_fp=test_cost_fp,
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
            test_path, test_cost = solutions[0]

    mcp = MCP_Geometric(da.values[0])
    costs, __ = mcp.find_costs(starts=[(1, 1)], ends=[(2, 6)])

    assert test_path == mcp.traceback((2, 6))
    assert np.isclose(test_cost, costs[(2, 6)])


@pytest.mark.parametrize(
    "algorithm",
    [
        "astar",
        "dijkstra",
        "long-range-dijkstra",
        "bidirectional-long-range-dijkstra",
    ],
)
def test_route_finder_writes_routing_layer_to_expected_path(
    tmp_path, algorithm
):
    """Test RouteFinder writes readable routing layer with expected costs"""
    da = xr.DataArray(
        np.array(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ],
            dtype=np.float32,
        ),
        dims=("band", "y", "x"),
        coords={
            "x": np.array([0.5, 1.5, 2.5], dtype=np.float32),
            "y": np.array([2.5, 1.5, 0.5], dtype=np.float32),
        },
    )
    test_cost_fp = tmp_path / "test.zarr"
    ds = xr.Dataset({"test_costs": da})
    ds["test_costs"].encoding = {
        "fill_value": 1_000.0,
        "_FillValue": 1_000.0,
    }
    ds.chunk({"x": 3, "y": 3}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    routing_layer_out_fp = tmp_path / "routing_layer.zarr"

    cost_definition = {
        "cost_layers": [{"layer_name": "test_costs"}],
        "ignore_invalid_costs": True,
    }
    routing_results = RouteFinder(
        zarr_fp=test_cost_fp,
        cost_function=json.dumps(cost_definition),
        route_definitions=[
            (11, [(0, 0)], [(2, 2)]),
        ],
        routing_layer_out_fp=routing_layer_out_fp,
        algorithm=algorithm,
    )
    list(routing_results)  # force routing to run

    assert routing_layer_out_fp.exists()

    with xr.open_dataset(
        routing_layer_out_fp, engine="zarr", consolidated=False
    ) as routing_ds:
        assert "cost" in routing_ds
        written_costs = routing_ds["cost"].astype(np.float32).values
        assert written_costs.shape == (1, 3, 3)

    scenario = RoutingScenario(
        cost_fpath=test_cost_fp,
        cost_layers=[{"layer_name": "test_costs"}],
        ignore_invalid_costs=True,
        algorithm=algorithm,
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


@pytest.mark.parametrize(
    "algorithm",
    [
        "astar",
        "dijkstra",
        "long-range-dijkstra",
        "bidirectional-long-range-dijkstra",
    ],
)
def test_find_paths_supports_explicit_algorithm(tmp_path, algorithm):
    """find_paths accepts explicit routing algorithm selection"""

    da = xr.DataArray(
        np.array(
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
        ),
        dims=("band", "y", "x"),
    )

    test_cost_fp = tmp_path / "test.zarr"
    ds = xr.Dataset({"test_costs": da})
    ds["test_costs"].encoding = {
        "fill_value": 1_000.0,
        "_FillValue": 1_000.0,
    }
    ds.chunk({"x": 4, "y": 3}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    cost_definition = {
        "cost_layers": [{"layer_name": "test_costs"}],
        "ignore_invalid_costs": True,
    }
    results = find_paths(
        zarr_fp=test_cost_fp,
        cost_function=json.dumps(cost_definition),
        start=[(1, 1)],
        end=[(2, 6)],
        algorithm=algorithm,
    )

    assert len(results) == 1
    path, cost = results[0]
    assert path[0] == (1, 1)
    assert path[-1] == (2, 6)
    assert cost > 0


@pytest.mark.parametrize(
    "algorithm",
    ["dijkstra", "long-range-dijkstra", "bidirectional-long-range-dijkstra"],
)
def test_route_finder_supports_explicit_algorithm(tmp_path, algorithm):
    """RouteFinder accepts explicit routing algorithm selection"""

    da = xr.DataArray(
        np.array(
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
        ),
        dims=("band", "y", "x"),
    )

    test_cost_fp = tmp_path / "test.zarr"
    ds = xr.Dataset({"test_costs": da})
    ds["test_costs"].encoding = {
        "fill_value": 1_000.0,
        "_FillValue": 1_000.0,
    }
    ds.chunk({"x": 4, "y": 3}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    cost_definition = {"cost_layers": [{"layer_name": "test_costs"}]}
    results = list(
        RouteFinder(
            zarr_fp=test_cost_fp,
            cost_function=json.dumps(cost_definition),
            route_definitions=[(2, [(1, 1)], [(2, 6)])],
            algorithm=algorithm,
        )
    )

    assert len(results) == 1
    route_id, solutions = results[0]
    assert route_id == 2
    assert len(solutions) == 1
    path, cost = solutions[0]
    assert path[0] == (1, 1)
    assert path[-1] == (2, 6)
    assert cost > 0


def test_find_paths_supports_a_star_alias(tmp_path):
    """find_paths accepts the hyphenated A* alias"""

    da = xr.DataArray(
        np.array([[[1, 1], [1, 1]]], dtype=np.float32),
        dims=("band", "y", "x"),
    )

    test_cost_fp = tmp_path / "test.zarr"
    ds = xr.Dataset({"test_costs": da})
    ds["test_costs"].encoding = {
        "fill_value": 1_000.0,
        "_FillValue": 1_000.0,
    }
    ds.chunk({"x": 2, "y": 2}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    results = find_paths(
        zarr_fp=test_cost_fp,
        cost_function=json.dumps(
            {"cost_layers": [{"layer_name": "test_costs"}]}
        ),
        start=[(0, 0)],
        end=[(1, 1)],
        algorithm="a-star",
    )

    assert len(results) == 1


def test_find_paths_rejects_invalid_algorithm(tmp_path):
    """find_paths rejects unsupported routing algorithms"""

    da = xr.DataArray(
        np.array([[[1, 1], [1, 1]]], dtype=np.float32),
        dims=("band", "y", "x"),
    )

    test_cost_fp = tmp_path / "test.zarr"
    ds = xr.Dataset({"test_costs": da})
    ds["test_costs"].encoding = {
        "fill_value": 1_000.0,
        "_FillValue": 1_000.0,
    }
    ds.chunk({"x": 2, "y": 2}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    with pytest.raises(
        ValueError,
        match=r'Unsupported routing algorithm "DNE". Supported values: ',
    ):
        find_paths(
            zarr_fp=test_cost_fp,
            cost_function=json.dumps(
                {"cost_layers": [{"layer_name": "test_costs"}]}
            ),
            start=[(0, 0)],
            end=[(1, 1)],
            algorithm="DNE",
        )


@pytest.mark.parametrize(
    "in_path, out_path",
    [
        (
            [
                (0.0, 0.0),
                (1.0, 1.0),
                (2.0, 2.0),
                (3.0, 3.0),
                (4.0, 4.0),
                (5.0, 5.0),
                (6.0, 5.0),
                (7.0, 5.0),
                (8.0, 5.0),
                (9.0, 6.0),
                (10.0, 7.0),
            ],
            [
                (0.0, 0.0),
                (5.0, 5.0),
                (8.0, 5.0),
                (10.0, 7.0),
            ],
        ),
        (
            [
                (1.5, 5.5),
                (1.5, 4.5),
                (2.5, 3.5),
                (3.5, 2.5),
                (4.5, 1.5),
                (5.5, 2.5),
                (5.5, 3.5),
                (6.5, 4.5),
            ],
            [
                (1.5, 5.5),
                (1.5, 4.5),
                (4.5, 1.5),
                (5.5, 2.5),
                (5.5, 3.5),
                (6.5, 4.5),
            ],
        ),
        (
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
        ),
        (
            [
                (1.5, 5.5),
                (2.5, 4.5),
                (3.5, 3.5),
                (4.5, 3.5),
                (5.5, 3.5),
                (6.5, 3.5),
                (6.5, 4.5),
            ],
            [
                (1.5, 5.5),
                (3.5, 3.5),
                (6.5, 3.5),
                (6.5, 4.5),
            ],
        ),
        (
            [
                (1, 5),
                (2, 4),
                (3, 3),
                (4, 3),
                (5, 3),
                (6, 3),
                (6, 4),
            ],
            [
                (1, 5),
                (3, 3),
                (6, 3),
                (6, 4),
            ],
        ),
    ],
)
@pytest.mark.parametrize("use_default_tol", [True, False])
def test_simplify_using_slopes_basic(in_path, out_path, use_default_tol):
    """Test basic slope simplification"""

    if use_default_tol:
        simplified_path = simplify_using_slopes(in_path)
    else:
        simplified_path = simplify_using_slopes(in_path, slope_tolerance=1)
    assert simplified_path == out_path


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
