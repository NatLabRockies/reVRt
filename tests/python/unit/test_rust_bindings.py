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
    test_path, test_cost, dropped_barrier_layers = results[0]

    mcp = MCP_Geometric(da.values[0])
    costs, __ = mcp.find_costs(starts=[(1, 1)], ends=[(2, 6)])

    assert test_path == mcp.traceback((2, 6))
    assert np.isclose(test_cost, costs[(2, 6)])
    assert not dropped_barrier_layers


def test_find_paths_respects_explicit_barrier_layers(tmp_path):
    """find_paths treats explicit barrier layers as impassable"""

    cost_layer = xr.DataArray(
        np.ones((1, 3, 3), dtype=np.float32),
        dims=("band", "y", "x"),
    )
    barrier_layer = xr.DataArray(
        np.array(
            [
                [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ]
            ],
            dtype=np.float32,
        ),
        dims=("band", "y", "x"),
    )

    test_cost_fp = tmp_path / "test_barrier.zarr"
    ds = xr.Dataset({"test_costs": cost_layer, "test_barrier": barrier_layer})
    for layer_name in ds.data_vars:
        ds[layer_name].encoding = {
            "fill_value": 1_000.0,
            "_FillValue": 1_000.0,
        }

    ds.chunk({"x": 3, "y": 3}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    cost_definition = {
        "cost_layers": [{"layer_name": "test_costs"}],
        "barrier_layers": [
            {
                "layer_name": "test_barrier",
                "barrier_values": "== 1",
            }
        ],
        "ignore_invalid_costs": False,
    }
    results = find_paths(
        zarr_fp=test_cost_fp,
        cost_function=json.dumps(cost_definition),
        start=[(1, 1)],
        end=[(0, 0)],
    )

    assert results == []


def test_find_paths_rejects_invalid_barrier_values(tmp_path):
    """find_paths returns a validation error for malformed barriers"""

    da = xr.DataArray(
        np.ones((1, 2, 2), dtype=np.float32),
        dims=("band", "y", "x"),
    )

    test_cost_fp = tmp_path / "test_invalid_barrier.zarr"
    ds = xr.Dataset({"test_costs": da, "test_barrier": da})
    for layer_name in ds.data_vars:
        ds[layer_name].encoding = {
            "fill_value": 1_000.0,
            "_FillValue": 1_000.0,
        }

    ds.chunk({"x": 2, "y": 2}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    cost_definition = {
        "cost_layers": [{"layer_name": "test_costs"}],
        "barrier_layers": [
            {
                "layer_name": "test_barrier",
                "barrier_values": "~1",
            }
        ],
    }

    with pytest.raises(ValueError, match="Barrier values must use"):
        find_paths(
            zarr_fp=test_cost_fp,
            cost_function=json.dumps(cost_definition),
            start=[(0, 0)],
            end=[(1, 1)],
        )


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
            (test_path, test_cost, dropped_barrier_layers) = solutions[0]
            assert dropped_barrier_layers == []

    mcp = MCP_Geometric(da.values[0])
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
        "long-range-astar",
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
    path, cost, dropped_barrier_layers = results[0]
    assert path[0] == (1, 1)
    assert path[-1] == (2, 6)
    assert cost > 0
    assert not dropped_barrier_layers


@pytest.mark.parametrize(
    "algorithm",
    [
        "dijkstra",
        "long-range-astar",
        "long-range-dijkstra",
        "bidirectional-long-range-dijkstra",
    ],
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
    path, cost, dropped_barrier_layers = solutions[0]
    assert dropped_barrier_layers == []
    assert path[0] == (1, 1)
    assert path[-1] == (2, 6)
    assert cost > 0


def test_route_finder_tracks_dropped_barriers_per_start_point(tmp_path):
    """RouteFinder returns per-solution retry metadata for mixed starts"""

    cost_layer = xr.DataArray(
        np.ones((1, 3, 5), dtype=np.float32),
        dims=("band", "y", "x"),
    )
    hard_barrier = xr.DataArray(
        np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                ]
            ],
            dtype=np.float32,
        ),
        dims=("band", "y", "x"),
    )
    soft_barrier = xr.DataArray(
        np.array(
            [
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ],
            dtype=np.float32,
        ),
        dims=("band", "y", "x"),
    )

    test_cost_fp = tmp_path / "mixed_retry.zarr"
    ds = xr.Dataset(
        {
            "test_costs": cost_layer,
            "hard_barrier": hard_barrier,
            "soft_barrier": soft_barrier,
        }
    )
    for layer_name in ds.data_vars:
        ds[layer_name].encoding = {
            "fill_value": 1_000.0,
            "_FillValue": 1_000.0,
        }

    ds.chunk({"x": 5, "y": 3}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    cost_definition = {
        "cost_layers": [{"layer_name": "test_costs"}],
        "barrier_layers": [
            {
                "layer_name": "hard_barrier",
                "barrier_operator": "eq",
                "barrier_threshold": 1.0,
            },
            {
                "layer_name": "soft_barrier",
                "barrier_operator": "eq",
                "barrier_threshold": 1.0,
                "barrier_importance": 1,
            },
        ],
        "ignore_invalid_costs": False,
    }
    results = list(
        RouteFinder(
            zarr_fp=test_cost_fp,
            cost_function=json.dumps(cost_definition),
            route_definitions=[(7, [(0, 0), (2, 0)], [(0, 4), (2, 4)])],
            algorithm="dijkstra",
        )
    )

    assert len(results) == 1
    route_id, solutions = results[0]
    assert route_id == 7
    assert len(solutions) == 2

    solutions_by_start = {
        tuple(path[0]): (path, cost, dropped_layers)
        for path, cost, dropped_layers in solutions
    }

    top_path, top_cost, top_layers = solutions_by_start[(0, 0)]
    assert top_path[-1] == (0, 4)
    assert top_cost > 0
    assert top_layers == ["soft_barrier"]

    bottom_path, bottom_cost, bottom_layers = solutions_by_start[(2, 0)]
    assert bottom_path[-1] == (2, 4)
    assert bottom_cost > 0
    assert bottom_layers == []


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
def test_route_finder_drops_soft_barriers_by_importance(tmp_path, algorithm):
    """RouteFinder drops soft barrier groups in ascending importance order"""

    cost_layer = xr.DataArray(
        np.ones((1, 3, 5), dtype=np.float32),
        dims=("band", "y", "x"),
    )
    soft_barrier_low = xr.DataArray(
        np.array(
            [
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                ]
            ],
            dtype=np.float32,
        ),
        dims=("band", "y", "x"),
    )
    soft_barrier_high = xr.DataArray(
        np.array(
            [
                [
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                ]
            ],
            dtype=np.float32,
        ),
        dims=("band", "y", "x"),
    )

    test_cost_fp = tmp_path / "ordered_retry.zarr"
    ds = xr.Dataset(
        {
            "test_costs": cost_layer,
            "soft_barrier_low": soft_barrier_low,
            "soft_barrier_high": soft_barrier_high,
        }
    )
    for layer_name in ds.data_vars:
        ds[layer_name].encoding = {
            "fill_value": 1_000.0,
            "_FillValue": 1_000.0,
        }

    ds.chunk({"x": 5, "y": 3}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    cost_definition = {
        "cost_layers": [{"layer_name": "test_costs"}],
        "barrier_layers": [
            {
                "layer_name": "soft_barrier_low",
                "barrier_operator": "eq",
                "barrier_threshold": 1.0,
                "barrier_importance": 1,
            },
            {
                "layer_name": "soft_barrier_high",
                "barrier_operator": "eq",
                "barrier_threshold": 1.0,
                "barrier_importance": 2,
            },
        ],
        "ignore_invalid_costs": False,
    }
    results = list(
        RouteFinder(
            zarr_fp=test_cost_fp,
            cost_function=json.dumps(cost_definition),
            route_definitions=[(9, [(1, 0)], [(1, 4)])],
            algorithm=algorithm,
        )
    )

    assert len(results) == 1
    route_id, solutions = results[0]
    assert route_id == 9
    assert len(solutions) == 1

    path, cost, dropped_layers = solutions[0][:3]
    assert path[0] == (1, 0)
    assert path[-1] == (1, 4)
    assert cost > 0
    assert dropped_layers == ["soft_barrier_low", "soft_barrier_high"]
    if len(solutions[0]) > 3:
        assert solutions[0][3] == [1, 2]


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
