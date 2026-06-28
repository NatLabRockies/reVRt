"""reVrt tests ported from reVX"""

import os
import json
import random
import shutil
import platform
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from revrt.utilities import LayeredFile, features_to_route_table
from revrt.costs.config import TransmissionConfig, parse_cap_class
from revrt.routing.base import RoutingScenario
from revrt.routing.processing import BatchRouteProcessor
from revrt.routing.cli.base import _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL
from revrt.routing.cli.point_to_point import (
    PointToPointRouteDefinitionConverter,
)
from revrt.routing.cli.build_route_table import point_to_feature_route_table
from revrt.routing.cli.point_to_feature import compute_lcp_routes
from revrt.routing.utilities import map_to_costs

DEFAULT_CONFIG = TransmissionConfig()
DEFAULT_BARRIER_CONFIG = {
    "multiplier_layer": "transmission_barrier",
    "multiplier_scalar": 100,
}
CHECK_COLS = ("start_index", "length_km", "cost", "index")


def _set_default_route_values(route_table, voltage, polarity):
    """Populate explicit default-option route value columns"""
    route_table["voltage_default"] = voltage
    route_table["polarity_default"] = polarity


def _cap_class_to_cap(capacity):
    """Get capacity for a capacity class"""
    capacity_class = parse_cap_class(capacity)
    return DEFAULT_CONFIG["power_classes"][capacity_class]


def check(truth, test, check_cols=CHECK_COLS):
    """Compare values in truth and test for given columns"""
    if check_cols is None:
        check_cols = truth.columns

    truth = truth.sort_values(["start_index", "index"])
    test = test.sort_values(["start_index", "index"])

    for c in check_cols:
        msg = f"values for {c} do not match!"
        c_truth = truth[c]
        c_test = test[c]
        assert np.allclose(c_truth, c_test, equal_nan=True), msg


def _run_lcp(tmp_path, routing_scenario, route_table):
    """Run least cost path computation and return resulting dataframe"""
    out_fp = tmp_path / "least_cost_paths_test.csv"
    routes_generator = PointToPointRouteDefinitionConverter(
        cost_fpath=routing_scenario.cost_fpath,
        route_points=route_table,
        out_fp=out_fp,
        routing_options=routing_scenario.routing_options,
        drivers=routing_scenario.drivers,
        transition_costs=routing_scenario.transition_costs,
    )
    # routes_generator._rp_with_expected_cols()

    route_definitions, route_attrs = (
        routes_generator._convert_to_route_definitions(
            routes_generator.route_points
        )
    )
    route_computer = BatchRouteProcessor(
        routing_scenario=routing_scenario,
        route_definitions=route_definitions,
        route_attrs=route_attrs,
    )
    route_computer.process(out_fp=out_fp, save_paths=False)

    return pd.read_csv(out_fp)


def _run_cli(
    tmp_path,
    routing_data_dir,
    revx_transmission_layers,
    row_config,
    polarity_config,
    route_table,
    cost_layer_config,
    cli_command_run_func,
    save_paths=False,
):
    """Run reVRt CLI with given configs and return test and truth dataframes"""
    capacity = random.choice([100, 200, 400, 1000, 3000])  # noqa: S311
    cost_layer = f"tie_line_costs_{_cap_class_to_cap(capacity)}MW"
    cost_layer_config["layer_name"] = cost_layer
    truth = routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    truth = pd.read_csv(truth)

    row_config_path = tmp_path / "config_row.json"
    row_config_path.write_text(json.dumps(row_config))

    polarity_config_path = tmp_path / "config_polarity.json"
    polarity_config_path.write_text(json.dumps(polarity_config))

    routes_fp = tmp_path / "routes.csv"
    route_table.to_csv(routes_fp, index=False)

    config = {
        "transmission_config": {
            "row_width": str(row_config_path),
            "voltage_polarity_mult": str(polarity_config_path),
        },
        "cost_fpath": str(revx_transmission_layers),
        "route_table_fpath": str(routes_fp),
        "save_paths": save_paths,
        "routing_options": {
            "default": {
                "cost_layers": [cost_layer_config],
                "friction_layers": [DEFAULT_BARRIER_CONFIG],
            }
        },
    }

    out_fp = cli_command_run_func("route-points", config, tmp_path)

    if save_paths:
        test = gpd.read_file(out_fp)
        assert test.geometry is not None
    else:
        test = pd.read_csv(out_fp)
    return test, truth


@pytest.fixture(scope="module")
def route_table(revx_transmission_layers, test_routing_data_dir):
    """Generate test BA regions and network nodes from ISO shapes"""

    with xr.open_dataset(
        revx_transmission_layers, consolidated=False, engine="zarr"
    ) as f:
        cost_crs = f.rio.crs
        features = test_routing_data_dir / "ri_county_centroids.gpkg"
        route_feats = gpd.read_file(features).to_crs(cost_crs)
        route_points = features_to_route_table(route_feats)
        return map_to_costs(
            route_points,
            crs=f.rio.crs,
            transform=f.rio.transform(),
            shape=f.rio.shape,
        )


@pytest.mark.parametrize("capacity", [100, 200, 400, 1000, 3000])
def test_capacity_class(
    revx_transmission_layers,
    capacity,
    route_table,
    tmp_path,
    test_routing_data_dir,
):
    """Test reVX capacity class routing against known outputs"""
    cap = _cap_class_to_cap(capacity)
    routing_scenario = RoutingScenario(
        cost_fpath=revx_transmission_layers,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": f"tie_line_costs_{cap}MW"}],
                "friction_layers": [DEFAULT_BARRIER_CONFIG],
            }
        },
    )

    test = _run_lcp(tmp_path, routing_scenario, route_table)

    truth = test_routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    truth = pd.read_csv(truth)

    check(truth, test)


def test_invariant_costs(
    revx_transmission_layers, route_table, tmp_path, test_routing_data_dir
):
    """Test reVX invariant cost layer routing against known outputs"""

    capacity = random.choice([100, 200, 400, 1000, 3000])  # noqa: S311
    cap = _cap_class_to_cap(capacity)
    base_costs = {
        "layer_name": f"tie_line_costs_{cap}MW",
        "multiplier_scalar": 1e-9,
    }
    invariant_cost_layer = {
        "layer_name": f"tie_line_costs_{cap}MW",
        "is_invariant": True,
    }
    routing_scenario = RoutingScenario(
        cost_fpath=revx_transmission_layers,
        routing_options={
            "default": {
                "cost_layers": [base_costs, invariant_cost_layer],
                "friction_layers": [DEFAULT_BARRIER_CONFIG],
            }
        },
        invalid_costs_block_routing=False,
    )

    test = _run_lcp(tmp_path, routing_scenario, route_table)

    truth = test_routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    truth = pd.read_csv(truth)

    truth = truth.sort_values(["start_index", "index"])
    test = test.sort_values(["start_index", "index"])

    assert (test["cost"].to_numpy() < truth["cost"].to_numpy()).all()


def test_cost_multiplier_layer(
    revx_transmission_layers, route_table, tmp_path, test_routing_data_dir
):
    """Test routing with a per-option cost_multiplier_layer"""

    capacity = random.choice([100, 200, 400, 1000, 3000])  # noqa: S311
    cap = _cap_class_to_cap(capacity)
    cost_layer = {"layer_name": f"tie_line_costs_{cap}MW"}

    temp_layer_file = tmp_path / "temp_multiplier_layer.zarr"
    shutil.copytree(revx_transmission_layers, temp_layer_file)

    lf = LayeredFile(temp_layer_file)
    lf.write_layer(
        np.ones(lf.shape, dtype=np.float32) * 7, "test_layer", overwrite=True
    )

    routing_scenario = RoutingScenario(
        cost_fpath=temp_layer_file,
        routing_options={
            "default": {
                "cost_layers": [cost_layer],
                "friction_layers": [DEFAULT_BARRIER_CONFIG],
                "cost_multiplier_layer": "test_layer",
            }
        },
    )
    test = _run_lcp(tmp_path, routing_scenario, route_table)

    truth = test_routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    truth = pd.read_csv(truth)

    truth = truth.sort_values(["start_index", "index"])
    test = test.sort_values(["start_index", "index"])

    assert np.allclose(test["length_km"], truth["length_km"])
    assert np.allclose(test["cost"].to_numpy(), truth["cost"].to_numpy() * 7)


def test_cost_multiplier_scalar(
    revx_transmission_layers, route_table, tmp_path, test_routing_data_dir
):
    """Test routing with a cost_multiplier_scalar"""

    capacity = random.choice([100, 200, 400, 1000, 3000])  # noqa: S311
    cap = _cap_class_to_cap(capacity)
    cost_layer = {"layer_name": f"tie_line_costs_{cap}MW"}

    routing_scenario = RoutingScenario(
        cost_fpath=revx_transmission_layers,
        routing_options={
            "default": {
                "cost_layers": [cost_layer],
                "friction_layers": [DEFAULT_BARRIER_CONFIG],
                "cost_multiplier_scalar": 5,
            }
        },
    )
    test = _run_lcp(tmp_path, routing_scenario, route_table)

    truth = test_routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    truth = pd.read_csv(truth)

    truth = truth.sort_values(["start_index", "index"])
    test = test.sort_values(["start_index", "index"])

    assert np.allclose(test["length_km"], truth["length_km"])
    assert np.allclose(test["cost"].to_numpy(), truth["cost"].to_numpy() * 5)


def test_not_hard_barrier(revx_transmission_layers, tmp_path):
    """Test routing to cut off points"""

    temp_layer_file = tmp_path / "temp_multiplier_layer.zarr"
    shutil.copytree(revx_transmission_layers, temp_layer_file)
    out_fp = tmp_path / "least_cost_paths_barrier.csv"

    lf = LayeredFile(temp_layer_file)
    costs = np.ones(shape=lf.shape)
    costs[0, 3] = costs[1, 3] = costs[2, 3] = costs[3, 3] = -1
    costs[3, 0] = costs[3, 1] = costs[3, 2] = -1
    lf.write_layer(costs, "test_layer", overwrite=True)

    route_feats = gpd.GeoDataFrame(
        data={"index": [0, 1]},
        geometry=[Point(-70.868065, 40.85588), Point(-71.9096, 42.016506)],
        crs="EPSG:4326",
    ).to_crs(lf.profile["crs"])
    route_points = features_to_route_table(route_feats)
    route_table = map_to_costs(
        route_points,
        crs=lf.profile["crs"],
        transform=lf.profile["transform"],
        shape=lf.shape,
    )
    routes_generator = PointToPointRouteDefinitionConverter(
        cost_fpath=temp_layer_file,
        route_points=route_table,
        out_fp=out_fp,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "test_layer"}],
                "friction_layers": [DEFAULT_BARRIER_CONFIG],
            }
        },
    )

    route_definitions, route_attrs = (
        routes_generator._convert_to_route_definitions(
            routes_generator.route_points
        )
    )

    routing_scenario = RoutingScenario(
        cost_fpath=temp_layer_file,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "test_layer"}],
                "friction_layers": [DEFAULT_BARRIER_CONFIG],
                "cost_multiplier_layer": "test_layer",
            }
        },
        invalid_costs_block_routing=True,
    )

    assert not out_fp.exists()
    route_computer = BatchRouteProcessor(
        routing_scenario=routing_scenario,
        route_definitions=route_definitions,
        route_attrs=route_attrs,
    )
    route_computer.process(out_fp=out_fp, save_paths=False)
    assert not out_fp.exists()

    routing_scenario = RoutingScenario(
        cost_fpath=temp_layer_file,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "test_layer"}],
                "friction_layers": [DEFAULT_BARRIER_CONFIG],
                "cost_multiplier_layer": "test_layer",
            }
        },
        invalid_costs_block_routing=False,
    )

    out_fp = tmp_path / "least_cost_paths_barrier.csv"
    assert not out_fp.exists()
    route_computer = BatchRouteProcessor(
        routing_scenario=routing_scenario,
        route_definitions=route_definitions,
        route_attrs=route_attrs,
    )
    route_computer.process(out_fp=out_fp, save_paths=False)
    assert out_fp.exists()

    test = pd.read_csv(out_fp)
    assert (test["length_km"] > 193).all()


@pytest.mark.parametrize("save_paths", [False, True])
@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli(
    revx_transmission_layers,
    route_table,
    tmp_path,
    test_routing_data_dir,
    save_paths,
    run_gaps_cli_with_expected_file,
):
    """Test reVX invariant cost layer routing against known outputs"""

    row_config = polarity_config = cost_layer_config = {}
    test, truth = _run_cli(
        tmp_path,
        test_routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        run_gaps_cli_with_expected_file,
        save_paths=save_paths,
    )

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_config_given_but_no_mult_in_layers(
    revx_transmission_layers,
    route_table,
    tmp_path,
    test_routing_data_dir,
    run_gaps_cli_with_expected_file,
):
    """Test Least cost path with transmission config but no volt in points"""

    row_config = {"138": 2}
    polarity_config = {"138": {"ac": 2, "dc": 3}}
    cost_layer_config = {}
    test, truth = _run_cli(
        tmp_path,
        test_routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        run_gaps_cli_with_expected_file,
    )

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_apply_row_mult(
    revx_transmission_layers,
    route_table,
    tmp_path,
    test_routing_data_dir,
    run_gaps_cli_with_expected_file,
):
    """Test applying row multiplier"""

    _set_default_route_values(route_table, 138, "dc")

    row_config = {"138": 2}
    polarity_config = {"138": {"ac": 2, "dc": 3}}
    cost_layer_config = {"apply_row_mult": True}
    test, truth = _run_cli(
        tmp_path,
        test_routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        run_gaps_cli_with_expected_file,
    )
    test["cost"] /= 2

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_apply_polarity_mult(
    revx_transmission_layers,
    route_table,
    tmp_path,
    test_routing_data_dir,
    run_gaps_cli_with_expected_file,
):
    """Test applying polarity multiplier"""

    _set_default_route_values(route_table, 138, "dc")

    row_config = {"138": 2}
    polarity_config = {"138": {"ac": 2, "dc": 3}}
    cost_layer_config = {"apply_polarity_mult": True}
    test, truth = _run_cli(
        tmp_path,
        test_routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        run_gaps_cli_with_expected_file,
    )
    test["cost"] /= 3 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_apply_row_and_polarity_mult(
    revx_transmission_layers,
    route_table,
    tmp_path,
    test_routing_data_dir,
    run_gaps_cli_with_expected_file,
):
    """Test applying row multiplier"""

    _set_default_route_values(route_table, 138, "dc")

    row_config = {"138": 2}
    polarity_config = {"138": {"ac": 2, "dc": 3}}
    cost_layer_config = {"apply_row_mult": True, "apply_polarity_mult": True}
    test, truth = _run_cli(
        tmp_path,
        test_routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        run_gaps_cli_with_expected_file,
    )
    test["cost"] /= 6 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_apply_row_and_polarity_with_existing_mult(
    revx_transmission_layers,
    route_table,
    tmp_path,
    test_routing_data_dir,
    run_gaps_cli_with_expected_file,
):
    """Test applying both row and polarity multiplier when mult exists"""

    _set_default_route_values(route_table, 138, "dc")

    row_config = {"138": 2}
    polarity_config = {"138": {"ac": 2, "dc": 3}}
    cost_layer_config = {
        "multiplier_scalar": 5,
        "apply_row_mult": True,
        "apply_polarity_mult": True,
    }
    test, truth = _run_cli(
        tmp_path,
        test_routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        run_gaps_cli_with_expected_file,
    )
    test["cost"] /= 30 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_apply_multipliers_by_route(
    revx_transmission_layers,
    route_table,
    tmp_path,
    test_routing_data_dir,
    run_gaps_cli_with_expected_file,
):
    """Test applying unique multipliers per route"""

    idx_to_volt = {0: 138, 1: 69, 2: 345, 3: 500}
    idx_to_polarity = {0: "ac", 1: "dc", 2: "ac", 3: "dc", 4: "dc"}
    for idx, volt in idx_to_volt.items():
        mask = route_table["start_index"] == idx
        route_table.loc[mask, "voltage_default"] = volt
    for idx, polarity in idx_to_polarity.items():
        mask = route_table["start_index"] == idx
        route_table.loc[mask, "polarity_default"] = polarity

    row_config = {"138": 2, "69": 2.5, "345": 3, "500": 3.5}
    polarity_config = {
        "138": {"ac": 4, "dc": 4.5},
        "69": {"ac": 5, "dc": 5.5},
        "345": {"ac": 6, "dc": 6.5},
        "500": {"ac": 7, "dc": 7.5},
    }
    cost_layer_config = {
        "multiplier_scalar": 1.2,
        "apply_row_mult": True,
        "apply_polarity_mult": True,
    }
    test, truth = _run_cli(
        tmp_path,
        test_routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        run_gaps_cli_with_expected_file,
    )
    divisors = []
    for __, row in test.iterrows():
        voltage = str(int(row["voltage_default"]))
        polarity = row["polarity_default"]
        divisors.append(
            1.2
            * row_config[voltage]
            * polarity_config[voltage][polarity]
            * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL
        )
    test["cost"] /= divisors

    check(truth, test)


def test_tracked_layers(
    tmp_path, revx_transmission_layers, test_routing_data_dir
):
    """Test tracked layers functionality"""

    route_table_path = tmp_path / "route_table.csv"
    mapped_features_path = tmp_path / "mapped_features.gpkg"
    assert not route_table_path.exists()
    assert not mapped_features_path.exists()

    point_to_feature_route_table(
        revx_transmission_layers,
        test_routing_data_dir / "ri_transmission_features.gpkg",
        out_dir=tmp_path,
        regions_fpath=test_routing_data_dir / "ri_regions.gpkg",
        resolution=64,
        radius=10_000,
        points_fpath=test_routing_data_dir / "sample_ri_points.csv",
        expand_radius=False,
    )

    assert route_table_path.exists()
    assert mapped_features_path.exists()

    route_table = pd.read_csv(route_table_path)
    mapped_features = gpd.read_file(mapped_features_path)

    assert len(mapped_features) == 89
    assert len(route_table) == 36

    assert {"start_row", "start_col", "end_feat_id"}.issubset(
        route_table.columns
    )
    assert route_table["start_row"].between(480, 1248).all()
    assert route_table["start_col"].between(32, 416).all()
    assert route_table["end_feat_id"].notna().all()

    temp_layer_file = tmp_path / "temp_multiplier_layer.zarr"
    shutil.copytree(revx_transmission_layers, temp_layer_file)

    lf = LayeredFile(temp_layer_file)
    lf.write_layer(np.ones(shape=lf.shape) * 1, "layer1", overwrite=True)
    lf.write_layer(np.ones(shape=lf.shape) * 2, "layer2", overwrite=True)
    lf.write_layer(np.ones(shape=lf.shape) * 4, "layer4", overwrite=True)
    lf.write_layer(np.ones(shape=lf.shape) * 1, "layer5", overwrite=True)

    out_fp = compute_lcp_routes(
        cost_fpath=temp_layer_file,
        route_table_fpath=route_table_path,
        features_fpath=mapped_features_path,
        routing_options={
            "default": {
                "cost_layers": [
                    {"layer_name": "tie_line_costs_102MW"},
                ]
            }
        },
        out_dir=tmp_path,
        job_name="test_route_to_features",
        tracked_layers=[
            {
                "layer_name": "layer1",
                "multiplier_scalar": 2,
                "agg_method": "sum",
            },
            {
                "layer_name": "layer2",
                "multiplier_layer": "layer4",
                "agg_method": "max",
            },
            {"layer_name": "layer3", "agg_method": "min"},
            {"layer_name": "layer5", "agg_method": "dne"},
        ],
        save_paths=True,
    )

    assert Path(out_fp)

    test = gpd.read_file(out_fp)
    assert len(test) == 36

    assert "layer1_default_sum" in test
    assert "layer2_default_max" in test
    assert "layer3_default_min" not in test
    assert "layer5_default_dne" not in test

    assert (
        test["layer1_default_sum"] <= 2 * (test["length_km"] / 90 * 1000 + 1)
    ).all()
    assert np.allclose(test["layer2_default_max"], 8)


@pytest.mark.parametrize("save_paths", [False, True])
@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_regional_end_to_end_cli(
    save_paths,
    run_gaps_cli_with_expected_file,
    tmp_path,
    test_routing_data_dir,
    revx_transmission_layers,
):
    """Test Regional cost routines and CLI"""

    ri_feats_path = test_routing_data_dir / "ri_transmission_features.gpkg"
    ri_ba_path = test_routing_data_dir / "ri_regions.gpkg"

    # -- Build features (substations) to route to --

    config = {
        "features_fpath": str(ri_feats_path),
        "regions_fpath": str(ri_ba_path),
        "region_identifier_column": "feature_id",
    }
    subs_fp = run_gaps_cli_with_expected_file("map-ss-to-rr", config, tmp_path)

    assert len(gpd.read_file(subs_fp)) == 380

    # -- Build Route Table --

    config = {
        "cost_fpath": str(revx_transmission_layers),
        "features_fpath": str(subs_fp),
        "out_dir": str(tmp_path),
        "regions_fpath": str(ri_ba_path),
        "resolution": 64,
        "expand_radius": True,
        "clip_points_to_regions": False,
        "feature_out_fp": "config_features.gpkg",
        "route_table_out_fp": "config_routes.csv",
        "region_identifier_column": "feature_id",
        "connection_identifier_column": "end_feat_id",
    }

    mapped_features_path = tmp_path / "config_features.gpkg"
    assert not mapped_features_path.exists()

    route_table_path = run_gaps_cli_with_expected_file(
        "build-feature-route-table",
        config,
        tmp_path,
        glob_pattern="config_routes.csv",
    )

    assert mapped_features_path.exists()
    assert len(pd.read_csv(route_table_path)) == 270
    assert len(gpd.read_file(mapped_features_path)) == 4045

    # -- RUN LCP --

    config = {
        "cost_fpath": str(revx_transmission_layers),
        "route_table_fpath": str(route_table_path),
        "features_fpath": str(mapped_features_path),
        "routing_options": {
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_1500MW"}],
                "friction_layers": [DEFAULT_BARRIER_CONFIG],
            }
        },
        "save_paths": save_paths,
    }

    out_fp = run_gaps_cli_with_expected_file(
        "route-features", config, tmp_path
    )

    if save_paths:
        test_routes = gpd.read_file(out_fp)
        assert test_routes.geometry is not None
    else:
        test_routes = pd.read_csv(out_fp)

    assert len(test_routes) == 270

    # -- RUN Post-Processing --

    config = {
        "collect_pattern": str(out_fp),
        "chunk_size": 200,
        "purge_chunks": False,
        "cost_fpath": str(revx_transmission_layers),
        "features_fpath": str(ri_feats_path),
        "transmission_feature_id_col": "trans_gid",
    }

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes", config, tmp_path
    )

    if save_paths:
        test = gpd.read_file(merged_fp)
        assert test.geometry is not None
    else:
        test = pd.read_csv(merged_fp)

    assert "poi_lat" in test
    assert "poi_lon" in test
    assert "feature_id" in test
    assert "trans_gid" in test
    assert "ac_cap" in test
    assert "category" in test
    assert "voltage" in test
    assert "trans_gids" in test

    assert "feature_id_transmission_feature" not in test
    assert "min_volts" not in test
    assert "max_volts" not in test
    assert "end_feat_id_transmission_feature" not in test

    assert len(test) == 806
    assert len(test) >= len(test_routes)
    assert test["trans_gid"].notna().all()
    assert test["cost"].notna().all()
    assert test["length_km"].notna().all()

    assert (test["length_km"] >= 0.45).all()
    assert (test["length_km"] <= 168).all()
    assert (test["cost"] >= 1.46e6).all()
    assert (test["cost"] <= 7.05e8).all()
    assert set(test["feature_id"].unique()) == {1, 2, 3, 4}

    assert len(set(test["trans_gid"].unique())) == 138
    assert len(test.groupby(["poi_lat", "poi_lon"])) == 69

    mask = test["trans_gid"] == 69130
    assert len(test[mask]) == 6
    assert set(test[mask]["feature_id"].unique()) == {4}

    assert np.allclose(
        test["tie_line_costs_1500MW_default_cost"].astype(float),
        test["cost"].astype(float),
    )
    assert np.allclose(
        test["tie_line_costs_1500MW_default_length_km"].astype(float),
        test["length_km"].astype(float),
    )

    # -- Test other linking configurations --
    merged_fp.unlink()
    shutil.move(tmp_path / "chunk_files" / out_fp.name, tmp_path / out_fp.name)
    shutil.rmtree(tmp_path / ".gaps", ignore_errors=True)
    config = {
        "collect_pattern": str(out_fp),
        "chunk_size": 200,
        "purge_chunks": False,
        "cost_fpath": str(revx_transmission_layers),
        "features_fpath": str(mapped_features_path),
        "transmission_feature_id_col": "trans_gid",
    }

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes", config, tmp_path
    )

    if save_paths:
        test = gpd.read_file(merged_fp)
        assert test.geometry is not None
    else:
        test = pd.read_csv(merged_fp)

    assert "poi_lat" in test
    assert "poi_lon" in test
    assert "feature_id" in test
    assert "ac_cap" in test
    assert "category" in test
    assert "trans_gids" in test
    assert "feature_id_transmission_feature" in test
    assert "min_volts" in test
    assert "max_volts" in test
    assert "end_feat_id_transmission_feature" in test

    assert len(test) == 329  # ensures features are de-duplicated
    assert len(test) >= len(test_routes)

    assert (test["length_km"] >= 0.45).all()
    assert (test["length_km"] <= 168).all()
    assert (test["cost"] >= 1.46e6).all()
    assert (test["cost"] <= 7.05e8).all()
    assert set(test["feature_id"].unique()) == {1, 2, 3, 4}

    # -- Test final linking configurations --

    merged_fp.unlink()
    shutil.move(tmp_path / "chunk_files" / out_fp.name, tmp_path / out_fp.name)
    shutil.rmtree(tmp_path / ".gaps", ignore_errors=True)
    config = {
        "collect_pattern": str(out_fp),
        "chunk_size": 200,
        "purge_chunks": True,
        "cost_fpath": str(revx_transmission_layers),
        "features_fpath": str(subs_fp),
        "transmission_feature_id_col": "trans_gid",
    }

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes", config, tmp_path
    )

    if save_paths:
        test = gpd.read_file(merged_fp)
        assert test.geometry is not None
    else:
        test = pd.read_csv(merged_fp)

    assert "poi_lat" in test
    assert "poi_lon" in test
    assert "feature_id" in test
    assert "ac_cap" in test
    assert "category" in test
    assert "trans_gids" in test
    assert "feature_id_transmission_feature" in test
    assert "min_volts" in test
    assert "max_volts" in test
    assert "end_feat_id_transmission_feature" not in test

    assert len(test) == 329
    assert len(test) >= len(test_routes)

    assert (test["length_km"] >= 0.45).all()
    assert (test["length_km"] <= 168).all()
    assert (test["cost"] >= 1.46e6).all()
    assert (test["cost"] <= 7.05e8).all()
    assert set(test["feature_id"].unique()) == {1, 2, 3, 4}

    assert not out_fp.exists()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
