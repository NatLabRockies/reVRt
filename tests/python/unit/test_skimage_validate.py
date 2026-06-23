"""Validate reVRt against scikit-image

Compare the solution given by scikit-image with reVRt, for the
same conditions.
"""

import json

import hypothesis
import numpy as np
import xarray as xr
from shapely.geometry import LineString
from skimage.graph import MCP_Geometric
from hypothesis.extra.numpy import arrays, array_shapes

from revrt import RouteFinder, find_paths, simplify_using_slopes

# Maximum value for input features used to calculate cost
# The test never ends for large values, such as 1e10.
MAX_COST = 1e6
ALGORITHMS = [
    "dijkstra",
    "long-range-dijkstra",
    "bidirectional-long-range-dijkstra",
]


def validate_find_paths_single_var(data, start, end, tmp_path, algorithm):
    """Validate reVRt against skimage for a given feature array

    Currently only for a single variable
    """
    da = xr.DataArray(data[None], dims=("band", "y", "x"))

    test_cost_fp = tmp_path / "test.zarr"
    ds = xr.Dataset({"test_costs": da})
    ds["test_costs"].encoding = {"fill_value": 1_000.0, "_FillValue": 1_000.0}
    ds.chunk({"x": 4, "y": 3}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    cost_definition = {
        "routing_options": {
            "default": {"cost_layers": [{"layer_name": "test_costs"}]}
        },
        "ignore_invalid_costs": True,
    }
    results = find_paths(
        zarr_fp=test_cost_fp,
        cost_function=json.dumps(cost_definition),
        start=[(*start, "default")],
        end=[(*end, "default")],
        algorithm=algorithm,
    )

    assert len(results) == 1
    revrt_route, revrt_cost, dropped_barrier_layers = results[0]
    revrt_route = [p[:2] for p in revrt_route]

    cost = da.values[0]
    mcp = MCP_Geometric(cost)
    costs, __ = mcp.find_costs(starts=[start], ends=[end])
    skimage_route = mcp.traceback(end)

    # compare route
    assert np.array_equal(skimage_route, revrt_route)
    # compare final cost
    assert np.isclose(revrt_cost, costs[end])
    assert not dropped_barrier_layers

    # make sure path simplification is equivalent
    if len(revrt_route) > 1:
        assert LineString(skimage_route).equals(
            LineString(simplify_using_slopes(revrt_route))
        )


def validate_route_finder_single_var(data, start, end, tmp_path, algorithm):
    """Validate reVRt against skimage for a given feature array

    Currently only for a single variable
    """
    da = xr.DataArray(data[None], dims=("band", "y", "x"))

    test_cost_fp = tmp_path / "test.zarr"
    ds = xr.Dataset({"test_costs": da})
    ds["test_costs"].encoding = {"fill_value": 1_000.0, "_FillValue": 1_000.0}
    ds.chunk({"x": 4, "y": 3}).to_zarr(
        test_cost_fp, mode="w", zarr_format=3, consolidated=False
    )

    cost_definition = {
        "routing_options": {
            "default": {"cost_layers": [{"layer_name": "test_costs"}]}
        }
    }
    routing_results = RouteFinder(
        zarr_fp=test_cost_fp,
        cost_function=json.dumps(cost_definition),
        route_definitions=[(0, [(*start, "default")], [(*end, "default")])],
        algorithm=algorithm,
    )

    results = list(routing_results)

    assert len(results) == 1
    route_id, solutions = results[0]
    assert route_id == 0
    assert len(solutions) == 1
    revrt_route, revrt_cost, dropped_barrier_layers = solutions[0]
    revrt_route = [p[:2] for p in revrt_route]
    assert dropped_barrier_layers == []

    cost = da.values[0]
    mcp = MCP_Geometric(cost)
    costs, __ = mcp.find_costs(starts=[start], ends=[end])
    skimage_route = mcp.traceback(end)

    # compare route
    assert np.array_equal(skimage_route, revrt_route)
    # compare final cost
    assert np.isclose(revrt_cost, costs[end])

    # make sure path simplification is equivalent
    if len(revrt_route) > 1:
        assert LineString(skimage_route).equals(
            LineString(simplify_using_slopes(revrt_route))
        )


@hypothesis.given(
    arrays(
        np.float32,
        array_shapes(min_dims=2, max_dims=2, min_side=7, max_side=32),
        elements=hypothesis.strategies.integers(
            min_value=1, max_value=MAX_COST
        ),
        unique=True,
    ),
    hypothesis.strategies.tuples(
        hypothesis.strategies.floats(0, 1), hypothesis.strategies.floats(0, 1)
    ),
    hypothesis.strategies.tuples(
        hypothesis.strategies.floats(0, 1), hypothesis.strategies.floats(0, 1)
    ),
    hypothesis.strategies.sampled_from(ALGORITHMS),
)
@hypothesis.settings(deadline=10_000, max_examples=100)
def test_basic_find_paths(tmp_path_factory, data, start, end, algorithm):
    """Validate single f32 variable"""
    start = (
        round(start[0] * max(0, data.shape[0] - 1)),
        round(start[1] * max(0, data.shape[1] - 1)),
    )
    end = (
        round(end[0] * max(0, data.shape[0] - 1)),
        round(end[1] * max(0, data.shape[1] - 1)),
    )

    tmpdir = tmp_path_factory.mktemp("skimage_test")
    validate_find_paths_single_var(data, start, end, tmpdir, algorithm)


@hypothesis.given(
    arrays(
        np.float32,
        array_shapes(min_dims=2, max_dims=2, min_side=7, max_side=32),
        elements=hypothesis.strategies.integers(
            min_value=1, max_value=MAX_COST
        ),
        unique=True,
    ),
    hypothesis.strategies.tuples(
        hypothesis.strategies.floats(0, 1), hypothesis.strategies.floats(0, 1)
    ),
    hypothesis.strategies.tuples(
        hypothesis.strategies.floats(0, 1), hypothesis.strategies.floats(0, 1)
    ),
    hypothesis.strategies.sampled_from(ALGORITHMS),
)
@hypothesis.settings(deadline=10_000, max_examples=100)
def test_basic_route_finder(tmp_path_factory, data, start, end, algorithm):
    """Validate single f32 variable"""
    start = (
        round(start[0] * max(0, data.shape[0] - 1)),
        round(start[1] * max(0, data.shape[1] - 1)),
    )
    end = (
        round(end[0] * max(0, data.shape[0] - 1)),
        round(end[1] * max(0, data.shape[1] - 1)),
    )

    tmpdir = tmp_path_factory.mktemp("skimage_test")
    validate_route_finder_single_var(data, start, end, tmpdir, algorithm)
