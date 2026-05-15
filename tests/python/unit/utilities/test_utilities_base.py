"""Tests for base reVRt utilities"""

import logging
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import dask.array as da
import geopandas as gpd
from rasterio.crs import CRS
from shapely.geometry import box, LineString, Point

from revrt.utilities import (
    buffer_routes,
    check_geotiff,
    delete_data_file,
    features_to_route_table,
    LayeredFile,
    log_array_backend,
)
from revrt.utilities.base import _crs_match
from revrt.exceptions import revrtProfileCheckError, revrtValueError
from revrt.warn import revrtWarning


@pytest.fixture
def sample_paths():
    """Sample paths for buffering tests"""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "A": ["a", "b"],
            "voltage": [12.0, 24],
        },
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
        crs="ESRI:102008",
    )


def test_buffer_no_row_input(sample_paths):
    """Test that no ROW input raises error"""

    with pytest.raises(
        revrtValueError,
        match="Must provide either `row_widths` or `row_width_ranges` input!",
    ):
        buffer_routes(sample_paths)


def test_buffer_routes(sample_paths):
    """Test buffering routes by row width with exact integer value"""

    row_widths = {"12": 10, "24": 20, "36": 30}
    routes = buffer_routes(sample_paths, row_widths)
    assert "geometry" in routes
    assert routes.geometry.is_valid.all()
    assert all(routes.geometry.type == "Polygon")

    # account for rounded corners
    assert 19**2 < routes.iloc[0].geometry.area < 20**2
    assert routes.iloc[1].geometry.area == 20 * 20


def test_buffer_routes_range(sample_paths):
    """Test buffering routes by row width with range of voltages"""

    row_width_ranges = [
        {"min": 0, "max": 18, "width": 10},
        {"min": 18, "max": 30, "width": 20},
    ]
    routes = buffer_routes(sample_paths, row_width_ranges=row_width_ranges)
    assert "geometry" in routes
    assert routes.geometry.is_valid.all()
    assert all(routes.geometry.type == "Polygon")

    # account for rounded corners
    assert 19**2 < routes.iloc[0].geometry.area < 20**2
    assert routes.iloc[1].geometry.area == 20 * 20


def test_buffer_routes_value_takes_precedence_over_range(sample_paths):
    """Test buffering routes by row width with values and ranges"""

    row_width_ranges = [
        {"min": 0, "max": 18, "width": 10},
        {"min": 18, "max": 30, "width": 20},
    ]
    row_widths = {"24": 16, "36": 30}
    routes = buffer_routes(
        sample_paths, row_widths=row_widths, row_width_ranges=row_width_ranges
    )
    assert "geometry" in routes
    assert routes.geometry.is_valid.all()
    assert all(routes.geometry.type == "Polygon")

    # account for rounded corners
    assert 19**2 < routes.iloc[0].geometry.area < 20**2
    assert routes.iloc[1].geometry.area == 20 * 16


def test_buffer_routes_dne_voltage():
    """Test buffering routes by row width with exact integer value"""

    sample_paths = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4],
            "A": ["a", "b", "c", "d"],
            "voltage": [12.0, 20, 24, 30],
        },
        geometry=[
            box(-5, -5, 5, 5),
            LineString([(10, -7), (10, 13)]),
            LineString([(9, -8), (9, 13)]),
            LineString([(11, -9), (12, 13)]),
        ],
        crs="ESRI:102008",
    )

    row_widths = {"12": 10, "24": 20, "36": 30}

    with pytest.warns(
        revrtWarning, match="2 route\\(s\\) will be dropped due to missing"
    ):
        routes = buffer_routes(sample_paths, row_widths)

    assert "geometry" in routes
    assert routes.geometry.is_valid.all()
    assert all(routes.geometry.type == "Polygon")

    assert len(routes) == 2
    assert set(routes.id) == {1, 3}

    # account for rounded corners
    assert 19**2 < routes.iloc[0].geometry.area < 20**2
    assert routes.iloc[1].geometry.area == 21 * 20


def test_buffer_routes_range_dne_voltage(sample_paths):
    """Test buffering routes by row width with range of voltages"""

    sample_paths = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4],
            "A": ["a", "b", "c", "d"],
            "voltage": [12.0, 20, 24, 30],
        },
        geometry=[
            box(-5, -5, 5, 5),
            LineString([(10, -7), (10, 13)]),
            LineString([(9, -8), (9, 13)]),
            LineString([(11, -9), (12, 13)]),
        ],
        crs="ESRI:102008",
    )

    row_width_ranges = [
        {"min": 0, "max": 18, "width": 10},
        {"min": 18, "max": 30, "width": 20},
    ]
    with pytest.warns(
        revrtWarning, match="1 route\\(s\\) will be dropped due to missing"
    ):
        routes = buffer_routes(sample_paths, row_width_ranges=row_width_ranges)

    assert "geometry" in routes
    assert routes.geometry.is_valid.all()
    assert all(routes.geometry.type == "Polygon")

    assert len(routes) == 3
    assert set(routes.id) == {1, 2, 3}

    # account for rounded corners
    assert 19**2 < routes.iloc[0].geometry.area < 20**2
    assert routes.iloc[1].geometry.area == 20 * 20
    assert routes.iloc[2].geometry.area == 21 * 20


def test_check_geotiff_bad_bands(sample_tiff_fp, sample_tiff_props, tmp_path):
    """Test check_geotiff with bad number of bands"""
    x0, y0, width, height, cell_size, transform = sample_tiff_props

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")

    test_tiff_fp = tmp_path / "test.tif"
    data = np.arange(width * height * 2, dtype=np.float32)
    da = xr.DataArray(
        data.reshape((2, height, width)),
        dims=("band", "y", "x"),
        coords={
            "x": x0 + np.arange(width) * cell_size + cell_size / 2,
            "y": y0 - np.arange(height) * cell_size - cell_size / 2,
        },
        name="test_band",
    )

    da = da.rio.write_crs("EPSG:4326")
    da.rio.write_transform(transform)
    da.rio.to_raster(test_tiff_fp, driver="GTiff")

    with pytest.raises(
        revrtProfileCheckError, match="contains more than one band"
    ):
        check_geotiff(test_fp, test_tiff_fp)


def test_check_geotiff_bad_shape(sample_tiff_fp, sample_tiff_props, tmp_path):
    """Test check_geotiff with bad number of bands"""
    x0, y0, width, height, cell_size, transform = sample_tiff_props

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")

    test_tiff_fp = tmp_path / "test.tif"
    data = np.arange(width * height * 4, dtype=np.float32)
    da = xr.DataArray(
        data.reshape((width * 2, height * 2)),
        dims=("y", "x"),
        coords={
            "x": x0 + np.arange(height * 2) * cell_size + cell_size / 2,
            "y": y0 - np.arange(width * 2) * cell_size - cell_size / 2,
        },
        name="test_band",
    )

    da = da.rio.write_crs("EPSG:4326")
    da.rio.write_transform(transform)
    da.rio.to_raster(test_tiff_fp, driver="GTiff")

    with pytest.raises(
        revrtProfileCheckError, match=r"Shape of layer data .* do not match.*"
    ):
        check_geotiff(test_fp, test_tiff_fp)


def test_check_geotiff_bad_transform(
    sample_tiff_fp, sample_tiff_props, tmp_path
):
    """Test check_geotiff with bad number of bands"""
    x0, y0, width, height, cell_size, __ = sample_tiff_props

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")

    test_tiff_fp = tmp_path / "test.tif"
    data = np.arange(width * height, dtype=np.float32)
    da = xr.DataArray(
        data.reshape((height, width)),
        dims=("y", "x"),
        coords={
            "x": 2 * x0 + np.arange(width) * cell_size + cell_size / 2,
            "y": 2 * y0 - np.arange(height) * cell_size - cell_size / 2,
        },
        name="test_band",
    )

    da = da.rio.write_crs("EPSG:4326")
    da.rio.write_transform()
    da.rio.to_raster(test_tiff_fp, driver="GTiff")

    with pytest.raises(
        revrtProfileCheckError,
        match=r'Geospatial "transform" .* do not match.*',
    ):
        check_geotiff(test_fp, test_tiff_fp)


def test_check_geotiff_equivalent_crs_wkt():
    """Test check_geotiff accepts equivalent CRS WKT encodings"""
    layered_file_crs = (
        'PROJCS["NAD83 / Conus Albers",GEOGCS["NAD83",'
        'DATUM["North_American_Datum_1983",'
        'SPHEROID["GRS 1980",6378137,298.257222101,'
        'AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,'
        'AUTHORITY["EPSG","9122"]]],'
        'PROJECTION["Albers_Conic_Equal_Area"],'
        'PARAMETER["latitude_of_center",23],'
        'PARAMETER["longitude_of_center",-96],'
        'PARAMETER["standard_parallel_1",29.5],'
        'PARAMETER["standard_parallel_2",45.5],'
        'PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],UNIT["metre",1,'
        'AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],'
        'AXIS["Northing",NORTH]]'
    )
    geotiff_crs = (
        'LOCAL_CS["NAD83 / Conus Albers",UNIT["metre",1,'
        'AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],'
        'AXIS["Northing",NORTH]]'
    )
    assert _crs_match(layered_file_crs, geotiff_crs)


def test_check_geotiff_equivalent_rasterio_crs_roundtrip():
    """Test CRS matching accepts rasterio round-trip differences"""
    layered_file_crs = CRS.from_wkt(
        'PROJCS["unknown",GEOGCS["unknown",'
        'DATUM["Unknown_based_on_GRS80_ellipsoid",'
        'SPHEROID["GRS 1980",6378137,298.257222101004,'
        'AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,'
        'AUTHORITY["EPSG","9122"]]],'
        'PROJECTION["Transverse_Mercator"],'
        'PARAMETER["latitude_of_origin",41.0833333333333],'
        'PARAMETER["central_meridian",-71.5],'
        'PARAMETER["scale_factor",0.99999375],'
        'PARAMETER["false_easting",100000],'
        'PARAMETER["false_northing",0],UNIT["metre",1,'
        'AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],'
        'AXIS["Northing",NORTH]]'
    )
    geotiff_crs = CRS.from_wkt(
        'PROJCS["unknown",GEOGCS["unknown",'
        'DATUM["Unknown_based_on_GRS80_ellipsoid",'
        'SPHEROID["GRS 1980",6378137,298.257222101004,'
        'AUTHORITY["EPSG","7019"]]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],'
        'PROJECTION["Transverse_Mercator"],'
        'PARAMETER["latitude_of_origin",41.0833333333333],'
        'PARAMETER["central_meridian",-71.5],'
        'PARAMETER["scale_factor",0.99999375],'
        'PARAMETER["false_easting",100000],'
        'PARAMETER["false_northing",0],UNIT["metre",1,'
        'AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],'
        'AXIS["Northing",NORTH]]'
    )

    assert layered_file_crs.equals(geotiff_crs)
    assert _crs_match(layered_file_crs, geotiff_crs)


@pytest.mark.parametrize("test_as_dir", [True, False])
def test_delete_data_file(tmp_path, test_as_dir):
    """Test not overwriting when creating a new file"""

    test_fp = tmp_path / "test.zarr"
    assert not test_fp.exists()

    delete_data_file(test_fp)
    assert not test_fp.exists()

    if test_as_dir:
        test_fp.mkdir()
    else:
        test_fp.touch()
    assert test_fp.exists()

    delete_data_file(test_fp)
    assert not test_fp.exists()

    delete_data_file(test_fp)
    assert not test_fp.exists()


def test_features_to_route_table_generates_pairs():
    """features_to_route_table builds expected permutations"""

    features = gpd.GeoDataFrame(
        {"name": ["north", "central", "south"]},
        geometry=[
            Point(-105.0, 39.5),
            Point(-104.5, 39.75),
            Point(-104.0, 40.0),
        ],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    route_table = features_to_route_table(features)

    assert route_table.columns.tolist() == [
        "rid",
        "end_lat",
        "end_lon",
        "index",
        "start_lat",
        "start_lon",
        "start_index",
    ]
    assert route_table["rid"].tolist() == [0, 1, 2]

    expected_pairs = {(0, 1), (0, 2), (1, 2)}
    observed_pairs = set(
        zip(route_table["start_index"], route_table["index"], strict=True)
    )
    assert observed_pairs == expected_pairs

    original_coords = [
        (39.5, -105.0),
        (39.75, -104.5),
        (40.0, -104.0),
    ]
    for row in route_table.itertuples(index=False):
        start_lat, start_lon = original_coords[row.start_index]
        end_lat, end_lon = original_coords[row.index]
        assert row.start_lat == pytest.approx(start_lat)
        assert row.start_lon == pytest.approx(start_lon)
        assert row.end_lat == pytest.approx(end_lat)
        assert row.end_lon == pytest.approx(end_lon)


@pytest.mark.parametrize(
    ("data", "expected_storage", "expected_backend"),
    [
        (
            xr.DataArray(
                np.arange(6, dtype=np.int16).reshape((2, 3)),
                dims=("y", "x"),
            ),
            "NumPy",
            "ndarray",
        ),
        (
            xr.DataArray(
                da.from_array(
                    np.arange(6, dtype=np.int16).reshape((2, 3)),
                    chunks=(1, 3),
                ),
                dims=("y", "x"),
            ),
            "Dask",
            "Array",
        ),
    ],
)
def test_log_array_backend_reports_xarray_storage(
    caplog, data, expected_storage, expected_backend
):
    """log_array_backend reports backend details for xarray inputs"""

    with caplog.at_level(logging.DEBUG, logger="revrt.utilities.base"):
        log_array_backend("test_layer", data, stage="processed")

    assert caplog.messages == [
        (
            "processed layer test_layer is "
            f"{expected_storage}-backed ({expected_backend}) "
            f"with dtype {data.dtype}, and shape {data.shape}"
        )
    ]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
