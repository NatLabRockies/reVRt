"""reVRt point-to-point CLI command unit tests"""

import os
import json
import shutil
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import from_origin

from revrt._cli import main
from revrt.utilities import LayeredFile
from revrt.utilities.handlers import ZARR_COMPRESSORS
from revrt.routing.utilities import map_to_costs

from revrt.routing.cli.point_to_point import (
    compute_lcp_routes,
    PointToPointRouteDefinitionConverter,
)


@pytest.fixture(scope="module")
def sample_layered_data(tmp_path_factory):
    """Create layered routing data mimicking point_to_point tests"""

    data_dir = tmp_path_factory.mktemp("routing_cli_data")

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

    layer_values = [
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
        np.array(
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
        ),
        np.array(
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
        ),
    ]

    for ind, routing_layer in enumerate(layer_values, start=1):
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


def _build_route_table(layered_fp, rows_cols):
    """Helper to construct route tables with CRS-aligned coordinates"""

    with xr.open_dataset(layered_fp, consolidated=False, engine="zarr") as ds:
        latitudes = ds["latitude"].to_numpy()
        longitudes = ds["longitude"].to_numpy()

    records = []
    for idx, (start, end) in enumerate(rows_cols):
        s_row, s_col = start
        e_row, e_col = end
        records.append(
            {
                "route_id": f"route_{idx}",
                "start_lat": float(latitudes[s_row, s_col]),
                "start_lon": float(longitudes[s_row, s_col]),
                "end_lat": float(latitudes[e_row, e_col]),
                "end_lon": float(longitudes[e_row, e_col]),
                "voltage": 138,
                "polarity": "ac",
            }
        )

    return pd.DataFrame.from_records(records)


def _assert_shared_compressor(codec):
    """Validate shared Blosc compressor settings"""
    assert isinstance(codec, type(ZARR_COMPRESSORS))
    assert codec.cname == ZARR_COMPRESSORS.cname
    assert codec.clevel == ZARR_COMPRESSORS.clevel
    assert codec.shuffle == ZARR_COMPRESSORS.shuffle


def test_compute_lcp_routes_generates_csv(sample_layered_data, tmp_path):
    """compute_lcp_routes should map points and persist CSV outputs"""

    routes = _build_route_table(
        sample_layered_data,
        rows_cols=[((1, 1), (2, 3)), ((0, 0), (3, 4))],
    )
    route_table_fp = tmp_path / "route_table.csv"
    routes.to_csv(route_table_fp, index=False)

    out_dir = tmp_path / "routing_outputs"
    cost_layers = [
        {
            "layer_name": "layer_1",
            "apply_row_mult": True,
            "multiplier_scalar": 1,
        },
        {
            "layer_name": "layer_2",
            "apply_polarity_mult": True,
            "multiplier_scalar": 1,
        },
    ]

    transmission_config = {
        "row_width": {"138": 1.5},
        "voltage_polarity_mult": {"138": {"ac": 2e-5}},
    }

    result_fp = compute_lcp_routes(
        cost_fpath=sample_layered_data,
        route_table_fpath=route_table_fp,
        cost_layers=cost_layers,
        out_dir=out_dir,
        job_name="run",
        transmission_config=transmission_config,
        save_paths=False,
        _split_params=(0, 1),
    )

    output_path = Path(result_fp)
    assert output_path.exists()

    df = pd.read_csv(output_path)
    df = df[df["route_id"] != "route_id"].reset_index(drop=True)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    numeric_cols = [
        "cost",
        "length_km",
        "optimized_objective",
        "layer_1_cost",
        "layer_1_length_km",
        "layer_2_cost",
        "layer_2_length_km",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    assert len(df) == len(routes)
    assert set(df["route_id"]) == set(routes["route_id"])

    with xr.open_dataset(
        sample_layered_data, consolidated=False, engine="zarr"
    ) as ds:
        mapped_routes = map_to_costs(
            routes.copy(), ds.rio.crs, ds.rio.transform(), ds.rio.shape
        )

    merged = df.merge(
        mapped_routes[
            [
                "route_id",
                "start_row",
                "start_col",
                "end_row",
                "end_col",
            ]
        ],
        on="route_id",
        how="left",
        suffixes=("", "_expected"),
    )

    for col in ["start_row", "start_col", "end_row", "end_col"]:
        assert (
            merged[col].astype(int) == merged[f"{col}_expected"].astype(int)
        ).all()

    assert np.allclose(
        merged["cost"], merged["layer_1_cost"] + merged["layer_2_cost"]
    )
    assert np.all(merged["length_km"] > 0)
    assert np.allclose(
        merged["cost"], merged["optimized_objective"], rtol=1e-5
    )


def test_compute_lcp_routes_saves_routing_layer(sample_layered_data, tmp_path):
    """compute_lcp_routes should persist routing layers when requested"""

    routes = _build_route_table(
        sample_layered_data,
        rows_cols=[((1, 1), (2, 3))],
    )
    route_table_fp = tmp_path / "route_table.csv"
    routes.to_csv(route_table_fp, index=False)

    out_dir = tmp_path / "routing_outputs"
    result_fp = compute_lcp_routes(
        cost_fpath=sample_layered_data,
        route_table_fpath=route_table_fp,
        cost_layers=[{"layer_name": "layer_1"}],
        out_dir=out_dir,
        job_name="save_layer",
        save_routing_layer=True,
        _split_params=(0, 1),
    )

    assert Path(result_fp).exists()

    extra_outputs = out_dir / "extra_outputs"
    saved_layers = sorted(extra_outputs.glob("*.zarr"))
    assert saved_layers

    with xr.open_dataset(
        saved_layers[0], engine="zarr", consolidated=False
    ) as ds:
        assert "cost" in ds
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        _assert_shared_compressor(ds["cost"].encoding["compressors"][0])
        _assert_shared_compressor(ds["latitude"].encoding["compressors"][0])
        _assert_shared_compressor(ds["longitude"].encoding["compressors"][0])


def test_compute_lcp_routes_returns_none_on_empty_indices(
    sample_layered_data, tmp_path
):
    """Ensure compute_lcp_routes short-circuits when route points are empty"""

    route_table_fp = tmp_path / "routes.csv"
    _build_route_table(sample_layered_data, [((1, 1), (2, 2))]).to_csv(
        route_table_fp, index=False
    )

    result = compute_lcp_routes(
        cost_fpath=sample_layered_data,
        route_table_fpath=route_table_fp,
        cost_layers=[{"layer_name": "layer_1"}],
        out_dir=tmp_path,
        job_name="no_routes",
        _split_params=(1000, 1),
    )

    assert result is None
    assert not (tmp_path / "no_routes.csv").exists()


def test_route_generator_existing_routes_csv(tmp_path):
    """route_generator.existing_routes should read CSV outputs"""

    data = pd.DataFrame(
        [
            {
                "start_row": 0,
                "start_col": 1,
                "end_row": 2,
                "end_col": 3,
                "polarity": "ac",
                "voltage": "230",
            }
        ]
    )
    csv_fp = tmp_path / "routes.csv"
    data.to_csv(csv_fp, index=False)

    route_generator = PointToPointRouteDefinitionConverter(
        None, None, csv_fp, None, None, None
    )
    result = route_generator.existing_routes
    assert result == {(0, 1, 2, 3, "ac", "230")}


def test_route_generator_existing_routes_gpkg(tmp_path):
    """route_generator.existing_routes should support GeoPackage outputs"""

    gpkg_fp = tmp_path / "routes.gpkg"
    gdf = gpd.GeoDataFrame(
        {
            "start_row": [1],
            "start_col": [2],
            "end_row": [3],
            "end_col": [4],
            "polarity": ["unknown"],
            "voltage": ["unknown"],
            "geometry": [Point(0, 0)],
        },
        crs="EPSG:4326",
    )
    gdf.to_file(gpkg_fp, driver="GPKG")

    route_generator = PointToPointRouteDefinitionConverter(
        None, None, gpkg_fp, None, None, None
    )
    result = route_generator.existing_routes
    assert result == {(1, 2, 3, 4, "unknown", "unknown")}


def test_route_generator_existing_routes_when_missing(tmp_path):
    """Missing outputs should result in an empty existing route set"""

    route_generator = PointToPointRouteDefinitionConverter(
        None, None, tmp_path / "missing.csv", None, None, None
    )
    assert route_generator.existing_routes == set()
    route_generator = PointToPointRouteDefinitionConverter(
        None, None, tmp_path / "missing.csv", None, None, None
    )
    assert route_generator.existing_routes == set()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_route_points_skips_precomputed_routes(
    cli_runner, sample_layered_data, tmp_path
):
    """route-points CLI should append only new routes"""

    routes = _build_route_table(
        sample_layered_data,
        rows_cols=[((1, 1), (2, 3)), ((0, 0), (3, 4))],
    )

    route_table_fp = tmp_path / "route_points.csv"
    routes.iloc[:1].to_csv(route_table_fp, index=False)

    config = {
        "cost_fpath": str(sample_layered_data),
        "route_table_fpath": str(route_table_fp),
        "cost_layers": [{"layer_name": "layer_1"}],
    }
    config_fp = tmp_path / "route_points_config.json"
    config_fp.write_text(json.dumps(config))

    first_result = cli_runner.invoke(
        main, ["route-points", "-c", str(config_fp)]
    )
    assert first_result.exit_code == 0, first_result.output

    out_fp = list(tmp_path.glob("*test*.csv"))
    assert len(out_fp) == 1
    out_fp = out_fp[0]
    assert out_fp.exists()

    first_run = pd.read_csv(out_fp)
    assert len(first_run) == 1

    base_route_id = routes.iloc[0]["route_id"]
    assert first_run["route_id"].tolist() == [base_route_id]
    first_row = first_run.iloc[0]

    sentinel_length = -4321.0
    raw_first = pd.read_csv(out_fp)
    assert "length_km" in raw_first.columns
    raw_first.loc[
        raw_first["route_id"].astype(str) == str(base_route_id),
        "length_km",
    ] = sentinel_length
    raw_first.to_csv(out_fp, index=False)

    routes.to_csv(route_table_fp, index=False)

    shutil.rmtree(tmp_path / ".gaps")
    second_result = cli_runner.invoke(
        main, ["route-points", "-c", str(config_fp)]
    )
    assert second_result.exit_code == 0, second_result.output

    assert len(list(tmp_path.glob("*test*.csv"))) == 1
    all_routes = pd.read_csv(out_fp)
    assert len(all_routes) == 2

    counts = all_routes["route_id"].value_counts()
    for route_id in routes["route_id"]:
        assert counts[route_id] == 1

    first_after = all_routes.loc[all_routes["route_id"] == base_route_id].iloc[
        0
    ]

    for col in ["cost", "optimized_objective"]:
        assert first_after[col] == pytest.approx(first_row[col])

    if "length_km" in all_routes.columns:
        assert first_after["length_km"] == pytest.approx(sentinel_length)

    with xr.open_dataset(
        sample_layered_data, consolidated=False, engine="zarr"
    ) as ds:
        mapped = map_to_costs(
            routes.copy(), ds.rio.crs, ds.rio.transform(), ds.rio.shape
        )

    merged = all_routes.merge(
        mapped[
            [
                "route_id",
                "start_row",
                "start_col",
                "end_row",
                "end_col",
            ]
        ],
        on="route_id",
        how="left",
        suffixes=("", "_expected"),
    )

    for col in ["start_row", "start_col", "end_row", "end_col"]:
        assert (
            merged[col].astype(int) == merged[f"{col}_expected"].astype(int)
        ).all()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_route_points_strips_required_path_whitespace(
    run_gaps_cli_with_expected_file, sample_layered_data, tmp_path
):
    """route-points CLI strips whitespace on required path inputs"""

    routes = _build_route_table(
        sample_layered_data, rows_cols=[((1, 1), (2, 3))]
    )
    route_table_fp = tmp_path / "route_points_trimmed.csv"
    routes.to_csv(route_table_fp, index=False)

    config = {
        "cost_fpath": f"  {sample_layered_data}  ",
        "route_table_fpath": f"  {route_table_fp}  ",
        "cost_layers": [{"layer_name": "layer_1"}],
    }
    out_fp = run_gaps_cli_with_expected_file(
        "route-points", config, tmp_path, glob_pattern="*test*.csv"
    )

    assert Path(out_fp).exists()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_route_points_skips_precomputed_routes_gpkg(
    cli_runner, sample_layered_data, tmp_path
):
    """route-points CLI should append only new routes (GPKG)"""

    routes = _build_route_table(
        sample_layered_data,
        rows_cols=[((1, 1), (2, 3)), ((0, 0), (3, 4))],
    )

    route_table_fp = tmp_path / "route_points.csv"
    routes.iloc[:1].to_csv(route_table_fp, index=False)

    config = {
        "cost_fpath": str(sample_layered_data),
        "route_table_fpath": str(route_table_fp),
        "cost_layers": [{"layer_name": "layer_1"}],
        "save_paths": True,
    }
    config_fp = tmp_path / "route_points_config.json"
    config_fp.write_text(json.dumps(config))

    first_result = cli_runner.invoke(
        main, ["route-points", "-c", str(config_fp)]
    )
    assert first_result.exit_code == 0, first_result.output

    out_fp = list(tmp_path.glob("*test*.gpkg"))
    assert len(out_fp) == 1
    out_fp = out_fp[0]
    assert out_fp.exists()

    first_run = gpd.read_file(out_fp)
    assert len(first_run) == 1

    base_route_id = routes.iloc[0]["route_id"]
    assert first_run["route_id"].tolist() == [base_route_id]
    first_row = first_run.iloc[0]

    sentinel_length = -4321.0
    raw_first = gpd.read_file(out_fp)
    assert "length_km" in raw_first.columns
    raw_first.loc[
        raw_first["route_id"].astype(str) == str(base_route_id),
        "length_km",
    ] = sentinel_length
    raw_first.to_file(out_fp, index=False, driver="GPKG")

    routes.to_csv(route_table_fp, index=False)

    shutil.rmtree(tmp_path / ".gaps")
    second_result = cli_runner.invoke(
        main, ["route-points", "-c", str(config_fp)]
    )
    assert second_result.exit_code == 0, second_result.output

    assert len(list(tmp_path.glob("*test*.gpkg"))) == 1
    all_routes = gpd.read_file(out_fp)
    assert len(all_routes) == 2

    counts = all_routes["route_id"].value_counts()
    for route_id in routes["route_id"]:
        assert counts[route_id] == 1

    first_after = all_routes.loc[all_routes["route_id"] == base_route_id].iloc[
        0
    ]

    for col in ["cost", "optimized_objective"]:
        assert first_after[col] == pytest.approx(first_row[col])

    if "length_km" in all_routes.columns:
        assert first_after["length_km"] == pytest.approx(sentinel_length)

    with xr.open_dataset(
        sample_layered_data, consolidated=False, engine="zarr"
    ) as ds:
        mapped = map_to_costs(
            routes.copy(), ds.rio.crs, ds.rio.transform(), ds.rio.shape
        )

    merged = all_routes.merge(
        mapped[
            [
                "route_id",
                "start_row",
                "start_col",
                "end_row",
                "end_col",
            ]
        ],
        on="route_id",
        how="left",
        suffixes=("", "_expected"),
    )

    for col in ["start_row", "start_col", "end_row", "end_col"]:
        assert (
            merged[col].astype(int) == merged[f"{col}_expected"].astype(int)
        ).all()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_route_points_flip_start_end(
    run_gaps_cli_with_expected_file, sample_layered_data, tmp_path
):
    """route-points CLI that internally flips start and end points"""

    to_test_routes = [
        # More unique endpoints than start points,
        # so computation will flip them
        ((1, 1), (2, 3)),
        ((1, 1), (3, 4)),
        ((1, 1), (4, 5)),
        ((1, 2), (5, 6)),
        ((1, 3), (5, 6)),
    ]

    routes = _build_route_table(
        sample_layered_data,
        rows_cols=to_test_routes,
    )
    route_table_fp = tmp_path / "route_points.csv"
    routes.to_csv(route_table_fp, index=False)

    config = {
        "cost_fpath": str(sample_layered_data),
        "route_table_fpath": str(route_table_fp),
        "cost_layers": [{"layer_name": "layer_1"}],
    }
    out_fp = run_gaps_cli_with_expected_file(
        "route-points", config, tmp_path, glob_pattern="*test*.csv"
    )

    output = pd.read_csv(out_fp)
    assert len(output) == len(routes)
    for route_id, route in enumerate(to_test_routes):
        (start_row, start_col), (end_row, end_col) = route

        output_route = output.loc[
            output["route_id"] == f"route_{route_id}"
        ].iloc[0]

        assert output_route["start_row"] == start_row
        assert output_route["start_col"] == start_col
        assert output_route["end_row"] == end_row
        assert output_route["end_col"] == end_col


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
