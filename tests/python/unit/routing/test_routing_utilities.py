"""reVrt tests for routing utilities"""

import math
from pathlib import Path
import warnings

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin, xy
from shapely.geometry import LineString, Point, Polygon

from revrt.routing.utilities import (
    PointToFeatureMapper,
    _add_voltage_polarity,
    _filter_transmission_features,
    _init_streaming_writer,
    _transform_lat_lon_to_row_col,
    compute_lens,
    points_csv_to_geo_dataframe,
    convert_lat_lon_to_row_col,
    filter_points_outside_cost_domain,
    make_rev_sc_points,
    map_to_costs,
)
from revrt.exceptions import revrtValueError
from revrt.warn import revrtWarning


@pytest.fixture
def cost_grid():
    """Simple grid metadata for routing tests"""

    height, width = (4, 5)
    cell_size = 1.0
    transform = from_origin(0.0, float(height), cell_size, cell_size)
    return "EPSG:4326", transform, (height, width)


@pytest.fixture
def transmission_features(tmp_path):
    """Minimal transmission features written to GeoPackage"""
    features = gpd.GeoDataFrame(
        {
            "gid": [101, 102],
            "bgid": [11, 12],
            "egid": [21, 22],
            "cap_left": [1.0, 2.0],
            "category": ["keep", "keep"],
        },
        geometry=[
            LineString([(0.0, 0.0), (0.0, 0.8)]),
            LineString([(10.08, 10.0), (10.08, 10.2)]),
        ],
        crs="EPSG:4326",
    )
    features_fp = tmp_path / "transmission_features.gpkg"
    features.to_file(features_fp, driver="GPKG")
    return features_fp, features


@pytest.fixture
def transmission_regions():
    """Two regions spanning the synthetic transmission features"""
    return gpd.GeoDataFrame(
        geometry=[
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]),
            Polygon([(10.0, 10.0), (10.0, 10.5), (10.5, 10.5), (10.5, 10.0)]),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def candidate_points():
    """Candidate points to map onto transmission features"""
    return gpd.GeoDataFrame(
        {
            "search_radius": [0.3, 0.02],
            "geometry": [Point(0.2, 0.2), Point(10.05, 10.05)],
        },
        crs="EPSG:4326",
    )


def test_transform_lat_lon_to_row_col_expected_indices(cost_grid):
    """Map lat/lon pairs to expected raster indices"""

    crs, transform, _ = cost_grid
    lon_a, lat_a = xy(transform, 0, 0, offset="center")
    lon_b, lat_b = xy(transform, 3, 4, offset="center")

    row, col = _transform_lat_lon_to_row_col(
        transform, crs, np.array([lat_a, lat_b]), np.array([lon_a, lon_b])
    )

    assert isinstance(row, np.ndarray)
    assert isinstance(col, np.ndarray)
    np.testing.assert_array_equal(row, np.array([0, 3]))
    np.testing.assert_array_equal(col, np.array([0, 4]))


def test_points_csv_to_geo_dataframe_from_dataframe(cost_grid):
    """DataFrame input adds grid indices and preserves CRS"""

    crs, transform, _ = cost_grid
    lon_a, lat_a = xy(transform, 0, 0, offset="center")
    lon_b, lat_b = xy(transform, 1, 2, offset="center")

    points = pd.DataFrame(
        {
            "latitude": [str(lat_a), str(lat_b)],
            "longitude": [str(lon_a), str(lon_b)],
            "label": ["first", "second"],
        }
    )

    geo_df = points_csv_to_geo_dataframe(points, crs, transform)

    assert isinstance(geo_df, gpd.GeoDataFrame)
    assert geo_df.crs.to_string() == crs
    np.testing.assert_array_equal(geo_df["start_row"], np.array([0, 1]))
    np.testing.assert_array_equal(geo_df["start_col"], np.array([0, 2]))
    np.testing.assert_allclose(
        geo_df.geometry.x.to_numpy(), np.array([lon_a, lon_b])
    )
    np.testing.assert_allclose(
        geo_df.geometry.y.to_numpy(), np.array([lat_a, lat_b])
    )


def test_points_csv_to_geo_dataframe_from_csv_path(cost_grid, tmp_path):
    """CSV path input is read and re-projected to target CRS"""

    crs, transform, _ = cost_grid
    lon, lat = xy(transform, 2, 3, offset="center")
    csv_points = pd.DataFrame(
        {
            "latitude": [lat],
            "longitude": [lon],
            "category": ["test"],
        }
    )
    csv_fp = tmp_path / "route_points.csv"
    csv_points.to_csv(csv_fp, index=False)

    geo_df = points_csv_to_geo_dataframe(csv_fp, crs, transform)

    assert isinstance(geo_df, gpd.GeoDataFrame)
    assert geo_df.crs.to_string() == crs
    assert geo_df["category"].tolist() == ["test"]
    np.testing.assert_array_equal(geo_df["start_row"], np.array([2]))
    np.testing.assert_array_equal(geo_df["start_col"], np.array([3]))
    np.testing.assert_allclose(geo_df.geometry.iloc[0].x, lon)
    np.testing.assert_allclose(geo_df.geometry.iloc[0].y, lat)


def test_map_to_costs_adds_expected_columns(cost_grid):
    """Populate start/end index columns from coordinates"""

    crs, transform, shape = cost_grid
    lon_start_a, lat_start_a = xy(transform, 0, 0, offset="center")
    lon_start_b, lat_start_b = xy(transform, 1, 2, offset="center")
    lon_end_a, lat_end_a = xy(transform, 2, 3, offset="center")
    lon_end_b, lat_end_b = xy(transform, 3, 4, offset="center")

    route_points = pd.DataFrame(
        {
            "start_lat": [str(lat_start_a), str(lat_start_b)],
            "start_lon": [str(lon_start_a), str(lon_start_b)],
            "end_lat": [str(lat_end_a), str(lat_end_b)],
            "end_lon": [str(lon_end_a), str(lon_end_b)],
        }
    )

    updated = map_to_costs(route_points, crs, transform, shape)

    assert updated is route_points
    np.testing.assert_array_equal(
        route_points["start_row"].to_numpy(), np.array([0, 1])
    )
    np.testing.assert_array_equal(
        route_points["start_col"].to_numpy(), np.array([0, 2])
    )
    np.testing.assert_array_equal(
        route_points["end_row"].to_numpy(), np.array([2, 3])
    )
    np.testing.assert_array_equal(
        route_points["end_col"].to_numpy(), np.array([3, 4])
    )


def test_filter_points_outside_cost_domain_no_warning(cost_grid):
    """Do not warn when all routes remain in bounds"""

    _, _, shape = cost_grid
    route_points = pd.DataFrame(
        {
            "start_row": [0, 1],
            "start_col": [0, 2],
            "end_row": [2, 3],
            "end_col": [3, 4],
        }
    )
    expected = route_points.copy(deep=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filtered = filter_points_outside_cost_domain(route_points, shape)

    assert not caught
    pd.testing.assert_frame_equal(filtered, expected)


def test_filter_points_outside_cost_domain_warns_and_drops(cost_grid):
    """Warn and drop routes that fall outside bounds"""

    _, _, shape = cost_grid
    route_points = pd.DataFrame(
        {
            "start_row": [0, -1],
            "start_col": [0, 1],
            "end_row": [1, 5],
            "end_col": [1, 6],
            "label": ["valid", "invalid"],
        }
    )

    with pytest.warns(revrtWarning) as record:
        filtered = filter_points_outside_cost_domain(route_points, shape)

    assert len(record) == 1
    assert (
        "The following 1 feature(s) are outside of the cost "
        "domain and will be dropped"
    ) in str(record[0].message)
    assert filtered.index.tolist() == [0]
    assert filtered["label"].tolist() == ["valid"]


def test_map_to_costs_maps_and_preserves_valid_routes(cost_grid):
    """End-to-end mapping keeps valid route intact"""

    crs, transform, shape = cost_grid
    lon_start, lat_start = xy(transform, 0, 0, offset="center")
    lon_end, lat_end = xy(transform, 2, 3, offset="center")

    route_points = pd.DataFrame(
        {
            "start_lat": [lat_start],
            "start_lon": [lon_start],
            "end_lat": [lat_end],
            "end_lon": [lon_end],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mapped = map_to_costs(
            route_points.copy(deep=True), crs, transform, shape
        )

    revrt_warnings = [w for w in caught if isinstance(w.message, revrtWarning)]
    assert not revrt_warnings

    np.testing.assert_array_equal(
        mapped["start_row"].to_numpy(), np.array([0])
    )
    np.testing.assert_array_equal(
        mapped["start_col"].to_numpy(), np.array([0])
    )
    np.testing.assert_array_equal(
        mapped["end_row"].to_numpy(),
        np.array([2]),
    )
    np.testing.assert_array_equal(
        mapped["end_col"].to_numpy(),
        np.array([3]),
    )


def test_map_to_costs_filters_routes_outside_cost_domain(cost_grid):
    """Remove routes that leave the cost domain"""

    crs, transform, shape = cost_grid
    lon_valid_start, lat_valid_start = xy(transform, 0, 0, offset="center")
    lon_valid_end, lat_valid_end = xy(transform, 1, 1, offset="center")

    # Column beyond grid width keeps latitude inside range but forces filtering
    lon_outside = lon_valid_start + shape[1]
    lat_outside = lat_valid_start

    route_points = pd.DataFrame(
        {
            "start_lat": [lat_valid_start, lat_outside],
            "start_lon": [lon_valid_start, lon_outside],
            "end_lat": [lat_valid_end, lat_outside],
            "end_lon": [lon_valid_end, lon_outside],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mapped = map_to_costs(route_points, crs, transform, shape)

    revrt_warnings = [w for w in caught if isinstance(w.message, revrtWarning)]
    assert len(revrt_warnings) == 1
    np.testing.assert_array_equal(
        mapped["start_row"].to_numpy(), np.array([0])
    )
    np.testing.assert_array_equal(
        mapped["start_col"].to_numpy(), np.array([0])
    )
    np.testing.assert_array_equal(
        mapped["end_row"].to_numpy(),
        np.array([1]),
    )
    np.testing.assert_array_equal(
        mapped["end_col"].to_numpy(),
        np.array([1]),
    )
    assert mapped.index.tolist() == [0]


def test_filter_transmission_features_drops_empty_categories(
    test_routing_data_dir,
):
    """_filter_transmission_features removes empty category records"""

    features_src = test_routing_data_dir / "ri_transmission_features.gpkg"
    features = gpd.read_file(features_src, rows=2)
    features["bgid"] = [1, 2]
    features["egid"] = [3, 4]
    features["cap_left"] = [0.0, 0.0]
    features["gid"] = [11, 12]
    features.loc[0, "category"] = math.nan
    features.loc[1, "category"] = "keep"

    with pytest.warns(revrtWarning, match="Dropping 1 feature"):
        cleaned = _filter_transmission_features(features)

    assert not any(c in cleaned.columns for c in ["bgid", "egid", "cap_left"])
    assert "trans_gid" in cleaned.columns
    assert cleaned["category"].tolist() == ["keep"]


def test_filter_transmission_features_without_category_column(
    test_routing_data_dir,
):
    """_filter_transmission_features tolerates missing category column"""

    features_src = test_routing_data_dir / "ri_transmission_features.gpkg"
    features = gpd.read_file(features_src, rows=1)
    features = features.drop(columns="category", errors="ignore")

    cleaned = _filter_transmission_features(features)

    assert "category" not in cleaned.columns


def test_point_to_feature_mapper_requires_constraints(
    transmission_features, candidate_points, tmp_path
):
    """PointToFeatureMapper raises if neither regions nor radius provided"""

    features_fp, _ = transmission_features
    mapper = PointToFeatureMapper("EPSG:4326", features_fp)

    with pytest.raises(
        revrtValueError,
        match=(
            "Must provide either `regions` or a radius to map points "
            "to features!"
        ),
    ):
        mapper.map_points(
            candidate_points, tmp_path / "missing_constraints.gpkg"
        )


def test_point_to_feature_mapper_maps_points_and_writes_features(
    transmission_features, transmission_regions, candidate_points, tmp_path
):
    """Map points onto features with region clipping and radius expansion"""

    features_fp, _ = transmission_features
    regions_fp = tmp_path / "regions.gpkg"
    transmission_regions.to_file(regions_fp, driver="GPKG")
    mapper = PointToFeatureMapper(
        "EPSG:4326", features_fp, regions=regions_fp, batch_size=5
    )
    feature_out = tmp_path / "clipped_features"

    with pytest.warns(
        revrtWarning,
        match="Output feature file should have a '.gpkg' extension",
    ):
        mapped = mapper.map_points(
            candidate_points.copy(deep=True),
            feature_out,
            radius="search_radius",
        )

    clipped_fp = feature_out.with_suffix(".gpkg")
    assert clipped_fp.exists()
    assert mapped["end_feat_id"].tolist() == [0, 1]
    assert mapped["rid"].tolist() == [0, 1]
    clipped = gpd.read_file(clipped_fp)
    assert set(clipped["end_feat_id"]) == {0, 1}


def test_point_to_feature_mapper_region_only(
    transmission_features, transmission_regions, tmp_path
):
    """Mapper clips solely by regions when no radius is provided"""

    features_fp, _ = transmission_features
    mapper = PointToFeatureMapper(
        "EPSG:4326", features_fp, regions=transmission_regions
    )
    points = gpd.GeoDataFrame({"geometry": [Point(0.2, 0.2)]}, crs="EPSG:4326")
    mapped = mapper.map_points(points, tmp_path / "region_only.gpkg")
    region_feats = gpd.read_file(tmp_path / "region_only.gpkg")

    assert mapped["end_feat_id"].tolist() == [0]
    assert region_feats["end_feat_id"].tolist() == [0]


def test_point_to_feature_mapper_radius_only(transmission_features, tmp_path):
    """Mapper clips by radius when no regions supplied"""

    features_fp, _ = transmission_features
    mapper = PointToFeatureMapper("EPSG:4326", features_fp, batch_size=1)
    points = gpd.GeoDataFrame(
        {"geometry": [Point(0.1, 0.1)]},
        crs="EPSG:4326",
    )

    mapped = mapper.map_points(
        points.copy(deep=True),
        tmp_path / "radius_only.gpkg",
        radius=0.25,
        expand_radius=False,
    )

    assert mapped["end_feat_id"].tolist() == [0]


def test_point_to_feature_mapper_parallel_batch(
    transmission_features, tmp_path, monkeypatch
):
    """Mapper can compute point batches via Dask and preserve IDs"""

    features_fp, _ = transmission_features
    mapper = PointToFeatureMapper(
        "EPSG:4326", features_fp, batch_size=2, max_workers=2
    )
    points = gpd.GeoDataFrame(
        {"geometry": [Point(0.1, 0.1), Point(10.1, 10.1)]},
        crs="EPSG:4326",
    )
    fake_features = gpd.GeoDataFrame(
        {"trans_gid": [1]},
        geometry=[LineString([(0.0, 0.0), (0.0, 0.5)])],
        crs="EPSG:4326",
    )

    def _fake_clip_to_point(self, point, radius, expand_radius):
        return fake_features.copy()

    compute_calls = []
    original_compute = dask.compute

    def _tracking_compute(*args, **kwargs):
        compute_calls.append(kwargs)
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(
        PointToFeatureMapper, "_clip_to_point", _fake_clip_to_point
    )
    monkeypatch.setattr(
        "revrt.routing.utilities.dask.compute", _tracking_compute
    )

    mapped = mapper.map_points(
        points.copy(deep=True),
        tmp_path / "parallel_batch.gpkg",
        radius=0.25,
        expand_radius=False,
    )

    assert mapped["end_feat_id"].tolist() == [0, 1]
    assert compute_calls
    assert compute_calls[0]["scheduler"] == "processes"
    assert compute_calls[0]["num_workers"] == 2


def test_point_to_feature_mapper_drops_unpaired_points(
    transmission_features, candidate_points, tmp_path
):
    """Mapper drops points that never find nearby features"""

    features_fp, _ = transmission_features
    mapper = PointToFeatureMapper("EPSG:4326", features_fp, batch_size=1)
    distant = candidate_points.iloc[[1]].copy(deep=True)
    nearby = candidate_points.iloc[[0]].copy(deep=True)
    nearby.loc[:, "geometry"] = Point(0.0, 0.2)
    points = gpd.GeoDataFrame(
        pd.concat([distant, nearby], ignore_index=False),
        crs="EPSG:4326",
    )

    with pytest.warns(
        revrtWarning, match=r"No features found for 1 point\(s\)"
    ):
        mapped = mapper.map_points(
            points,
            tmp_path / "dropped_points.gpkg",
            radius=0.01,
            expand_radius=False,
        )

    assert len(mapped) == 1
    assert mapped.index.tolist() == [0]
    assert mapped["end_feat_id"].tolist() == [1]


def test_point_to_feature_mapper_preserves_existing_region_ids(
    transmission_features, transmission_regions, candidate_points, tmp_path
):
    """Existing region identifiers remain untouched during setup"""

    features_fp, _ = transmission_features
    regions = transmission_regions.copy(deep=True)
    regions["rid"] = [5, 6]
    mapper = PointToFeatureMapper(
        "EPSG:4326", features_fp, regions=regions, batch_size=5
    )

    mapped = mapper.map_points(
        candidate_points.copy(deep=True),
        tmp_path / "existing_rid.gpkg",
        radius="search_radius",
    )

    assert mapped["rid"].tolist() == [5, 6]


def test_clip_to_radius_expands_until_features_found(
    transmission_features, transmission_regions, candidate_points
):
    """_clip_to_radius gradually expands buffers until features are found"""

    features_fp, __ = transmission_features
    mapper = PointToFeatureMapper(
        "EPSG:4326", features_fp, regions=transmission_regions
    )
    point = candidate_points.iloc[1].copy(deep=True)
    point[mapper._rid_column] = 1
    region_features = mapper._clip_to_region(point)

    empty = mapper._clip_to_radius(
        point,
        "search_radius",
        input_features=region_features,
        expand_radius=False,
    )
    expanded = mapper._clip_to_radius(
        point,
        "search_radius",
        input_features=region_features,
        expand_radius=True,
    )

    assert len(empty) == 0
    assert len(expanded) >= 1


def test_clip_to_radius_returns_input_on_null_radius(
    transmission_features, transmission_regions, candidate_points
):
    """_clip_to_radius returns input features unchanged when radius is None"""

    features_fp, features = transmission_features
    mapper = PointToFeatureMapper(
        "EPSG:4326", features_fp, regions=transmission_regions
    )
    point = candidate_points.iloc[0]

    result = mapper._clip_to_radius(point, None, features)

    assert result is features


def test_clip_to_radius_returns_empty_features(
    transmission_features, transmission_regions, candidate_points
):
    """Empty inputs are returned early by _clip_to_radius"""

    features_fp, _ = transmission_features
    mapper = PointToFeatureMapper(
        "EPSG:4326", features_fp, regions=transmission_regions
    )
    point = candidate_points.iloc[0]
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    result = mapper._clip_to_radius(point, 0.1, input_features=empty)

    assert result is empty


def test_convert_lat_lon_to_row_col_custom_columns(cost_grid):
    """Custom column mapping is honored when converting to row/col indices"""

    crs, transform, _ = cost_grid
    lon, lat = xy(transform, 1, 2, offset="center")
    points = pd.DataFrame({"lat_val": [lat], "lon_val": [lon]})

    converted = convert_lat_lon_to_row_col(
        points,
        crs,
        transform,
        lat_col="lat_val",
        lon_col="lon_val",
        out_row_name="row_idx",
        out_col_name="col_idx",
    )

    np.testing.assert_array_equal(
        converted["row_idx"].to_numpy(), np.array([1])
    )
    np.testing.assert_array_equal(
        converted["col_idx"].to_numpy(), np.array([2])
    )


def test_make_rev_sc_points_returns_expected_grid(cost_grid):
    """make_rev_sc_points builds a supply-curve grid with derived geometry"""

    crs, transform, shape = cost_grid
    sc_points = make_rev_sc_points(
        shape[0], shape[1], crs, transform, resolution=2
    )

    assert len(sc_points) == 6
    assert sc_points.index.name == "gid"
    assert {"start_row", "start_col", "latitude", "longitude"}.issubset(
        sc_points.columns
    )
    assert sc_points.geometry.iloc[0].geom_type == "Point"


def test_init_streaming_writer_appends_suffix(tmp_path):
    """_init_streaming_writer enforces GeoPackage output extension"""

    raw_fp = tmp_path / "stream_output"
    with pytest.warns(
        revrtWarning,
        match="Output feature file should have a '.gpkg' extension",
    ):
        writer = _init_streaming_writer(raw_fp)

    assert writer.out_fp.suffix == ".gpkg"


def test_add_voltage_polarity_expands_routes_by_combination():
    """_add_voltage_polarity should duplicate routes per input combination"""

    route_table = pd.DataFrame(
        {
            "route_id": ["route-a", "route-b"],
            "start_row": [0, 1],
            "start_col": [2, 3],
        }
    )

    updated = _add_voltage_polarity(
        route_table,
        voltages=[138, 230],
        polarities=["ac", "dc"],
    )

    assert len(updated) == len(route_table) * 4
    assert updated["route_id"].value_counts().to_dict() == {
        "route-a": 4,
        "route-b": 4,
    }

    actual = {
        (row.route_id, row.voltage, row.polarity, row.start_row, row.start_col)
        for row in updated.itertuples(index=False)
    }
    expected = {
        ("route-a", 138, "ac", 0, 2),
        ("route-a", 138, "dc", 0, 2),
        ("route-a", 230, "ac", 0, 2),
        ("route-a", 230, "dc", 0, 2),
        ("route-b", 138, "ac", 1, 3),
        ("route-b", 138, "dc", 1, 3),
        ("route-b", 230, "ac", 1, 3),
        ("route-b", 230, "dc", 1, 3),
    }

    assert actual == expected


@pytest.mark.parametrize(
    ("voltages", "polarities"),
    [
        (None, None),
        ([], []),
        (None, []),
        ([], None),
        ([138], None),
        ([138], []),
        (None, ["ac"]),
        ([], ["ac"]),
    ],
)
def test_add_voltage_polarity_handles_none_and_empty_inputs(
    voltages, polarities
):
    """_add_voltage_polarity should return a routing table for empty inputs"""

    route_table = pd.DataFrame(
        {
            "route_id": ["route-a", "route-b"],
            "start_row": [0, 1],
            "start_col": [2, 3],
        }
    )

    updated = _add_voltage_polarity(route_table, voltages, polarities)

    assert isinstance(updated, pd.DataFrame)
    if voltages:
        assert updated["voltage"].tolist() == [138, 138]
    else:
        assert "voltage" not in updated.columns

    if polarities:
        assert updated["polarity"].tolist() == ["ac", "ac"]
    else:
        assert "polarity" not in updated.columns

    expected = route_table.copy(deep=True)
    if voltages:
        expected["voltage"] = 138
    if polarities:
        expected["polarity"] = "ac"
    pd.testing.assert_frame_equal(updated.reset_index(drop=True), expected)


def test_filter_points_outside_cost_domain_only_start_indices(cost_grid):
    """Bounds filtering works when end indices are absent"""

    _, _, shape = cost_grid
    route_points = pd.DataFrame(
        {
            "start_row": [0, -1],
            "start_col": [0, shape[1]],
        }
    )

    with pytest.warns(
        revrtWarning,
        match=(
            r"The following 1 feature\(s\) are outside of the cost "
            r"domain and will be dropped"
        ),
    ):
        filtered = filter_points_outside_cost_domain(route_points, shape)

    assert filtered.index.tolist() == [0]


def test_compute_lens_splits_segment_lengths_across_route_cells():
    """Per-cell and total route lengths are computed consistently"""

    route = np.array([[0, 0], [0, 1], [1, 2]], dtype=float)

    lens, total_path_length = compute_lens(route, cell_size=2000)

    expected_lens = np.array(
        [0.5, (1 + math.sqrt(2)) / 2, math.sqrt(2) / 2]
    )
    np.testing.assert_allclose(lens, expected_lens)
    assert math.isclose(total_path_length, 2 * (1 + math.sqrt(2)))


def test_compute_lens_returns_zero_lengths_for_single_cell_route():
    """A route with one cell has zero per-cell and total length"""

    route = np.array([[3, 4]])

    lens, total_path_length = compute_lens(route, cell_size=90)

    np.testing.assert_array_equal(lens, np.array([0.0]))
    assert total_path_length == 0


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
