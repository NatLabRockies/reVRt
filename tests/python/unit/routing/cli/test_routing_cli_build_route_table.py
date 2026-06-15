"""reVrt tests for routing one point to many endpoints"""

import json
import os
import platform
import warnings
from pathlib import Path

import pytest
import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import LineString, Point, Polygon

from revrt._cli import main
from revrt.exceptions import revrtValueError
from revrt.warn import revrtWarning
from revrt.routing.cli.build_route_table import (
    _check_output_filepaths,
    _make_points,
    point_to_feature_route_table,
)
from revrt.routing.utilities import (
    PointToFeatureMapper,
    make_rev_sc_points,
    map_to_costs,
)


@pytest.fixture(scope="module")
def cost_metadata(revx_transmission_layers):
    """Return metadata describing the test routing cost grid"""

    with xr.open_dataset(
        revx_transmission_layers, consolidated=False, engine="zarr"
    ) as ds:
        return {
            "crs": ds.rio.crs,
            "transform": ds.rio.transform(),
            "shape": (ds.rio.height, ds.rio.width),
        }


def test_make_points_requires_input(cost_metadata):
    """Verify the helper enforces mutually exclusive inputs"""

    with pytest.raises(
        revrtValueError,
        match=(
            "Either `points_fpath` or `resolution` must be provided to "
            "create route table!"
        ),
    ):
        _make_points(
            cost_metadata["crs"],
            cost_metadata["transform"],
            cost_metadata["shape"],
        )


def test_make_points_from_csv(tmp_path, cost_metadata):
    """Build supply-curve points from a CSV input"""

    height, width = cost_metadata["shape"]
    resolution = 64
    sc_points = make_rev_sc_points(
        height,
        width,
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=resolution,
    ).head(5)
    csv_path = tmp_path / "points.csv"
    sc_points.drop(columns="geometry").to_csv(csv_path, index=False)

    points = _make_points(
        cost_metadata["crs"],
        cost_metadata["transform"],
        cost_metadata["shape"],
        points_fpath=csv_path,
    )

    assert {"start_row", "start_col"}.issubset(points.columns)


def test_map_to_costs_filters_out_of_bounds(cost_metadata):
    """map_to_costs converts coordinates and drops routes out of domain"""

    base_points = make_rev_sc_points(
        cost_metadata["shape"][0],
        cost_metadata["shape"][1],
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=64,
    )

    start = base_points.iloc[0]
    end = base_points.iloc[len(base_points) // 2]
    routes = pd.DataFrame(
        {
            "start_lat": [start.latitude, 90.0],
            "start_lon": [start.longitude, 0.0],
            "end_lat": [end.latitude, end.latitude],
            "end_lon": [end.longitude, end.longitude],
        }
    )

    with pytest.warns(revrtWarning, match="outside of the cost domain"):
        mapped = map_to_costs(
            routes,
            cost_metadata["crs"],
            cost_metadata["transform"],
            cost_metadata["shape"],
        )

    assert {"start_row", "start_col", "end_row", "end_col"}.issubset(
        mapped.columns
    )
    assert len(mapped) == 1


def test_point_to_feature_mapper_clips_features_to_region_boundary(tmp_path):
    """Features are trimmed to region boundary when regions provided"""

    crs = "EPSG:3857"
    region_geom = Polygon([(0, 0), (0, 1_000), (1_000, 1_000), (1_000, 0)])
    regions = gpd.GeoDataFrame({"rid": [7]}, geometry=[region_geom], crs=crs)
    regions_fp = tmp_path / "clip_regions.gpkg"
    regions.to_file(regions_fp, driver="GPKG")

    original = LineString([(-250, 500), (1_250, 500)])
    features = gpd.GeoDataFrame(
        {"category": ["keep"], "gid": [101]}, geometry=[original], crs=crs
    )
    features_fp = tmp_path / "clip_features.gpkg"
    features.to_file(features_fp, driver="GPKG")

    mapper = PointToFeatureMapper(
        crs,
        features_fp,
        regions=regions_fp,
        region_identifier_column="rid",
        batch_size=1,
    )

    points = gpd.GeoDataFrame(
        {"start_row": [0], "start_col": [0]},
        geometry=[Point(500, 500)],
        crs=crs,
    )

    out_fp = tmp_path / "clipped_outputs.gpkg"
    mapper.map_points(points, out_fp, radius=800, expand_radius=False)

    clipped = gpd.read_file(out_fp)
    assert len(clipped) == 1
    clipped_geom = clipped.geometry.iloc[0]
    assert clipped_geom.geom_type == "LineString"
    assert clipped_geom.length < original.length
    assert clipped["rid"].tolist() == [7]

    expected = LineString([(0, 500), (1_000, 500)])
    assert clipped_geom.equals_exact(expected, tolerance=1e-6)

    region_difference = clipped_geom.difference(region_geom)
    assert region_difference.is_empty


def test_point_to_feature_mapper_clips_features_to_radius(tmp_path):
    """Features are trimmed to circular radius when radius provided"""

    crs = "EPSG:3857"
    radius = 400.0
    point_geom = Point(0, 0)
    original = LineString([(-1_000, 0), (1_000, 0)])
    features = gpd.GeoDataFrame({"gid": [5]}, geometry=[original], crs=crs)
    features_fp = tmp_path / "radius_features.gpkg"
    features.to_file(features_fp, driver="GPKG")

    mapper = PointToFeatureMapper(crs, features_fp, batch_size=1)
    points = gpd.GeoDataFrame(
        {"start_row": [0], "start_col": [0]}, geometry=[point_geom], crs=crs
    )

    out_fp = tmp_path / "radius_outputs.gpkg"
    mapper.map_points(points, out_fp, radius=radius, expand_radius=False)

    clipped = gpd.read_file(out_fp)
    assert len(clipped) == 1
    clipped_geom = clipped.geometry.iloc[0]
    expected = original.intersection(point_geom.buffer(radius))
    assert clipped_geom.equals_exact(expected, tolerance=1e-6)
    difference = clipped_geom.difference(point_geom.buffer(radius))
    assert difference.is_empty
    assert clipped_geom.length < original.length


def test_point_to_feature_mapper_clips_features_to_region_and_radius(
    tmp_path,
):
    """Features respect both region and radius constraints when provided"""

    crs = "EPSG:3857"
    radius = 400.0
    point_geom = Point(800, 500)
    region_geom = Polygon([(0, 0), (0, 1_000), (1_000, 1_000), (1_000, 0)])
    regions = gpd.GeoDataFrame({"rid": [1]}, geometry=[region_geom], crs=crs)
    regions_fp = tmp_path / "region_and_radius_regions.gpkg"
    regions.to_file(regions_fp, driver="GPKG")

    original = LineString([(-500, 500), (1_500, 500)])
    features = gpd.GeoDataFrame({"gid": [1]}, geometry=[original], crs=crs)
    features_fp = tmp_path / "region_and_radius_features.gpkg"
    features.to_file(features_fp, driver="GPKG")

    mapper = PointToFeatureMapper(
        crs,
        features_fp,
        regions=regions_fp,
        region_identifier_column="rid",
        batch_size=1,
    )

    points = gpd.GeoDataFrame(
        {"start_row": [0], "start_col": [0]},
        geometry=[point_geom],
        crs=crs,
    )

    out_fp = tmp_path / "region_and_radius_outputs.gpkg"
    mapper.map_points(points, out_fp, radius=radius, expand_radius=False)

    clipped = gpd.read_file(out_fp)
    assert len(clipped) == 1
    clipped_geom = clipped.geometry.iloc[0]

    region_only = original.intersection(region_geom)
    radius_only = original.intersection(point_geom.buffer(radius))
    expected = region_only.intersection(point_geom.buffer(radius))

    assert clipped_geom.equals_exact(expected, tolerance=1e-6)
    assert clipped_geom.length < region_only.length
    assert clipped_geom.length < radius_only.length
    assert clipped["rid"].tolist() == [1]


def test_check_output_filepaths_preserves_valid_extensions(tmp_path):
    """Ensure helper respects valid output extensions without warnings"""

    feature_path = tmp_path / "features.gpkg"
    route_path = tmp_path / "routes.csv"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        checked_feature, checked_route = _check_output_filepaths(
            tmp_path, feature_path.name, route_path.name
        )

    assert checked_feature == feature_path
    assert checked_route == route_path
    assert not caught


def test_point_to_feature_route_table_builds_outputs(
    tmp_path, revx_transmission_layers
):
    """Route table CLI creates outputs and normalizes extensions"""

    cost_fp = tmp_path / "cost_surface.zarr"
    with xr.open_dataset(
        revx_transmission_layers, consolidated=False, engine="zarr"
    ) as ds:
        subset = ds.isel(y=slice(0, 6), x=slice(0, 6))
        subset.to_zarr(cost_fp, mode="w", zarr_format=3, consolidated=False)
        crs = subset.rio.crs
        transform = subset.rio.transform()
        shape = (subset.rio.height, subset.rio.width)

    resolution = shape[0] + 2
    cell_size = max(abs(transform.a), abs(transform.e))

    sc_points = make_rev_sc_points(
        shape[0], shape[1], crs, transform, resolution=resolution
    )
    point_geom = sc_points.geometry.iloc[0]

    line_geom = LineString(
        [
            (point_geom.x - cell_size, point_geom.y),
            (point_geom.x + cell_size, point_geom.y),
        ]
    )
    features = gpd.GeoDataFrame(
        {"gid": [1], "category": ["test"]},
        geometry=[line_geom],
        crs=crs,
    )
    features_fp = tmp_path / "features_source.gpkg"
    features.to_file(features_fp, driver="GPKG")

    region_geom = Polygon(
        [
            (point_geom.x - 4 * cell_size, point_geom.y - 4 * cell_size),
            (point_geom.x - 4 * cell_size, point_geom.y + 4 * cell_size),
            (point_geom.x + 4 * cell_size, point_geom.y + 4 * cell_size),
            (point_geom.x + 4 * cell_size, point_geom.y - 4 * cell_size),
        ]
    )
    regions = gpd.GeoDataFrame({"rid": [3]}, geometry=[region_geom], crs=crs)
    regions_fp = tmp_path / "regions.gpkg"
    regions.to_file(regions_fp, driver="GPKG")

    out_dir = tmp_path / "outputs"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outputs = point_to_feature_route_table(
            cost_fpath=cost_fp,
            features_fpath=features_fp,
            out_dir=out_dir,
            regions_fpath=regions_fp,
            resolution=resolution,
            radius=3 * cell_size,
            feature_out_fp="clipped_features",
            route_table_out_fp="route_listing",
            expand_radius=False,
            batch_size=1,
        )

    expected_feature = out_dir / "clipped_features.gpkg"
    expected_route_table = out_dir / "route_listing.csv"

    assert set(outputs) == {
        str(expected_route_table),
        str(expected_feature),
    }
    assert expected_feature.exists()
    assert expected_route_table.exists()

    route_table = pd.read_csv(expected_route_table)
    assert len(route_table) == 1
    assert route_table["end_feat_id"].tolist() == [0]
    assert route_table["start_row"].notna().all()
    assert route_table["start_col"].notna().all()

    clipped = gpd.read_file(expected_feature)
    assert len(clipped) == 1
    assert clipped["rid"].tolist() == [3]

    warning_messages = [str(w.message) for w in caught]
    assert any(".gpkg" in msg for msg in warning_messages)
    assert any(".csv" in msg for msg in warning_messages)


def test_point_to_feature_route_table_radius_only(
    tmp_path, revx_transmission_layers
):
    """Routing CLI executes when only a radius constraint is provided"""

    cost_fp = tmp_path / "cost_surface.zarr"
    with xr.open_dataset(
        revx_transmission_layers, consolidated=False, engine="zarr"
    ) as ds:
        subset = ds.isel(y=slice(0, 6), x=slice(0, 6))
        subset.to_zarr(cost_fp, mode="w", zarr_format=3, consolidated=False)
        crs = subset.rio.crs
        transform = subset.rio.transform()
        shape = (subset.rio.height, subset.rio.width)

    resolution = shape[0] + 2
    cell_size = max(abs(transform.a), abs(transform.e))

    sc_points = make_rev_sc_points(
        shape[0], shape[1], crs, transform, resolution=resolution
    )
    point_geom = sc_points.geometry.iloc[0]

    line_geom = LineString(
        [
            (point_geom.x - cell_size, point_geom.y),
            (point_geom.x + cell_size, point_geom.y),
        ]
    )
    features = gpd.GeoDataFrame(
        {"gid": [2]},
        geometry=[line_geom],
        crs=crs,
    )
    features_fp = tmp_path / "features_radius.gpkg"
    features.to_file(features_fp, driver="GPKG")

    out_dir = tmp_path / "outputs"
    outputs = point_to_feature_route_table(
        cost_fpath=cost_fp,
        features_fpath=features_fp,
        out_dir=out_dir,
        resolution=resolution,
        radius=3 * cell_size,
        expand_radius=False,
        feature_out_fp="clipped_radius.gpkg",
        route_table_out_fp="routes_radius.csv",
    )

    expected_feature = out_dir / "clipped_radius.gpkg"
    expected_route_table = out_dir / "routes_radius.csv"

    assert outputs == [str(expected_route_table), str(expected_feature)]
    assert expected_feature.exists()
    assert expected_route_table.exists()

    route_table = pd.read_csv(expected_route_table)
    assert len(route_table) == 1
    assert route_table["end_feat_id"].tolist() == [0]

    clipped = gpd.read_file(expected_feature)
    assert len(clipped) == 1
    assert "end_feat_id" in clipped.columns


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_build_feature_route_table_cli_strips_required_path_whitespace(
    cli_runner, tmp_path, revx_transmission_layers
):
    """build-feature-route-table CLI strips required path whitespace"""

    cost_fp = tmp_path / "cost_surface_cli.zarr"
    with xr.open_dataset(
        revx_transmission_layers, consolidated=False, engine="zarr"
    ) as ds:
        subset = ds.isel(y=slice(0, 6), x=slice(0, 6))
        subset.to_zarr(cost_fp, mode="w", zarr_format=3, consolidated=False)
        crs = subset.rio.crs
        transform = subset.rio.transform()
        shape = (subset.rio.height, subset.rio.width)

    resolution = shape[0] + 2
    cell_size = max(abs(transform.a), abs(transform.e))
    sc_points = make_rev_sc_points(
        shape[0], shape[1], crs, transform, resolution=resolution
    )
    point_geom = sc_points.geometry.iloc[0]

    features = gpd.GeoDataFrame(
        {"gid": [1]},
        geometry=[
            LineString(
                [
                    (point_geom.x - cell_size, point_geom.y),
                    (point_geom.x + cell_size, point_geom.y),
                ]
            )
        ],
        crs=crs,
    )
    features_fp = tmp_path / "features_cli.gpkg"
    features.to_file(features_fp, driver="GPKG")

    config = {
        "cost_fpath": f"  {cost_fp}  ",
        "features_fpath": f"  {features_fp}  ",
        "resolution": resolution,
        "radius": 3 * cell_size,
        "expand_radius": False,
    }

    config_fp = tmp_path / "build_route_table_whitespace.json"
    config_fp.write_text(json.dumps(config))

    result = cli_runner.invoke(
        main, ["build-feature-route-table", "-c", str(config_fp)]
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "mapped_features.gpkg").exists()
    assert (tmp_path / "route_table.csv").exists()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
