"""Tests for reVRt utilities"""

import os
import json
import math
import shutil
import platform
import traceback
import contextlib
from pathlib import Path

import pytest
import pyproj
import rioxarray
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from geopandas.testing import assert_geodataframe_equal
from pyproj.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling
from rasterio.transform import from_origin

import revrt
from revrt._cli import main
from revrt.utilities import (
    LayeredFile,
    file_full_path,
    load_data_using_layer_file_profile,
    save_data_using_layer_file_profile,
)
from revrt.utilities.handlers import chunked_read_gpkg, num_feats_in_gpkg
from revrt.exceptions import (
    revrtFileExistsError,
    revrtFileNotFoundError,
    revrtKeyError,
    revrtValueError,
)
from revrt.warn import revrtWarning


@pytest.fixture(scope="module")
def sample_tiff_fp_2x(sample_tiff_props, tmp_path_factory):
    """Return path to TIFF file used for tests"""

    x0, y0, width, height, cell_size, transform = sample_tiff_props
    data = np.arange(width * height, dtype=np.float32) * 2
    da = xr.DataArray(
        data.reshape((height, width)),
        dims=("y", "x"),
        coords={
            "x": x0 + np.arange(width) * cell_size + cell_size / 2,
            "y": y0 - np.arange(height) * cell_size - cell_size / 2,
        },
        name="test_band",
    )

    da = da.rio.write_crs("EPSG:4326")
    da.rio.write_transform(transform)

    out_fp = tmp_path_factory.mktemp("data") / "sample_2x.tif"
    da.rio.to_raster(out_fp, driver="GTiff")
    return out_fp


@pytest.fixture(scope="module")
def test_tl_fp(test_utility_data_dir):
    """Return path to transmission layers test file"""
    return test_utility_data_dir / "transmission_layers.zarr"


@pytest.fixture(scope="module")
def test_tiff_fp(test_utility_data_dir):
    """Return path to TIFF file used for tests"""
    return test_utility_data_dir / "ri_transmission_barriers.tif"


@pytest.fixture
def gpkg_with_nan_features(tmp_path):
    """Create GeoPackage with NaNs for chunked_read_gpkg tests"""

    coords = [
        (-71.431, 41.824),
        (-71.512, 41.743),
        (-71.612, 41.682),
        (-71.389, 41.598),
        (-71.274, 41.553),
    ]
    data = {
        "feature_id": [101, 102, 103, 104, 105],
        "voltage_kv": [69.0, np.nan, 115.0, 230.0, np.nan],
        "all_nan": [np.nan] * len(coords),
        "name": [
            "Hope Line",
            "Smithfield Tap",
            "Providence Loop",
            "Cranston Spur",
            "Warwick Link",
        ],
    }

    gdf = gpd.GeoDataFrame(
        data,
        geometry=[Point(x, y) for x, y in coords],
        crs="EPSG:4326",
    )

    gpkg_fp = tmp_path / "nan_features.gpkg"
    gdf.to_file(gpkg_fp, driver="GPKG", index=False)
    stored = gpd.read_file(gpkg_fp).reset_index(drop=True)
    stored["all_nan"] = stored["all_nan"].astype(float)
    return gpkg_fp, stored


def _validate_top_level_ds_props(ds, transform):
    """Validate top level dataset properties"""

    assert ds.rio.crs == CRS("EPSG:4326")
    assert ds.rio.transform() == transform
    assert set(ds.coords) == {
        "band",
        "latitude",
        "longitude",
        "spatial_ref",
        "x",
        "y",
    }
    assert ds.rio.grid_mapping == "spatial_ref"
    assert "nodata" not in ds.attrs
    assert "count" not in ds.attrs


def _validate_random_data_layer(width, height, transform, layer):
    """Validate data layer made of random numbers"""
    assert layer.shape == (1, height, width)
    assert layer.dtype == np.dtype("float32")
    assert layer.min() >= 0
    assert layer.max() <= 1
    assert layer.rio.crs == CRS("EPSG:4326")
    assert layer.rio.transform() == transform
    assert layer.rio.grid_mapping == "spatial_ref"


def test_methods_without_file():
    """Test methods without file"""

    lf = LayeredFile("test_file.zarr")
    with pytest.raises(revrtValueError, match="format is not supported"):
        lf.create_new("test_file.txt")

    with pytest.raises(revrtFileNotFoundError, match="not found on disk"):
        lf.create_new("test_file.zarr")

    with pytest.raises(revrtFileNotFoundError, match=r"File .* not found"):
        lf.write_layer(np.array([1, 2, 3]), layer_name="test_layer")


def test_no_layers():
    """Test getting layers for DNE file"""

    lf = LayeredFile("test_file.zarr")
    with pytest.raises(revrtFileNotFoundError, match=r"File .* not found"):
        __ = lf.layers


def test_not_overwrite_when_create_new_file(tmp_path):
    """Test not overwriting when creating a new file"""

    test_fp = tmp_path / "test.zarr"
    test_fp.touch()
    lf = LayeredFile(test_fp)
    with pytest.raises(
        revrtFileExistsError, match="exits and overwrite=False"
    ):
        lf.create_new(test_fp, overwrite=False)


def test_layered_file_handler_props(test_tl_fp):
    """Test LayeredFile property attributes"""

    lf = LayeredFile(test_tl_fp)

    assert repr(lf) == f"LayeredFile({test_tl_fp})"
    assert str(lf) == "LayeredFile with 8 layers"
    assert lf.shape == (1434, 972)

    expected_profile = {
        "width": 972,
        "height": 1434,
        "crs": CRS(
            "+proj=tmerc +lat_0=41.0833333333333 "  # cspell:disable-line
            "+lon_0=-71.5 +k=0.99999375 +x_0=100000 +y_0=0 "
            "+ellps=GRS80 +units=m +no_defs=True"  # cspell:disable-line
        ),
        "transform": Affine(
            90.0, 0.0, 65848.6171875, 0.0, -90.0, 103948.140625
        ),
    }
    assert lf.profile == expected_profile

    expected_layers = {
        "band",
        "x",
        "y",
        "latitude",
        "longitude",
        "spatial_ref",
        "ISO_regions",
        "tie_line_costs_1500MW",
        "tie_line_costs_400MW",
        "tie_line_multipliers",
        "tie_line_costs_3000MW",
        "tie_line_costs_205MW",
        "tie_line_costs_102MW",
        "transmission_barrier",
    }
    assert set(lf.layers) == expected_layers

    expected_data_layers = expected_layers - {
        "band",
        "x",
        "y",
        "latitude",
        "longitude",
        "spatial_ref",
    }
    assert set(lf.data_layers) == expected_data_layers


def test_layered_file_handler_get_layer(test_tl_fp):
    """Test LayeredFile get_layer method"""

    lf = LayeredFile(test_tl_fp)

    layer_profile, layer = lf["tie_line_costs_400MW"]
    assert layer.shape == (1, 1434, 972)
    assert layer.dtype == np.dtype("float32")
    assert layer.min() == -1
    assert layer.max() > 1e6

    # Test that it's there but don't compare on it
    for key in ["crs", "nodata"]:
        assert key in layer_profile
        layer_profile.pop(key, None)

    expected_profile = {
        "width": 972,
        "height": 1434,
        "count": 1,
        "dtype": np.dtype("float32"),
        "transform": Affine(
            90.0, 0.0, 65848.6171875, 0.0, -90.0, 103948.140625
        ),
    }
    assert layer_profile == expected_profile


def test_layered_file_layer_profile(test_tl_fp):
    """Test LayeredFile layer_profile method"""

    lf = LayeredFile(test_tl_fp)

    layer_profile = lf.layer_profile("tie_line_costs_400MW")

    # Test that it's there but don't compare on it
    for key in ["crs", "nodata"]:
        assert key in layer_profile
        layer_profile.pop(key, None)

    expected_profile = {
        "width": 972,
        "height": 1434,
        "count": 1,
        "dtype": np.dtype("float32"),
        "transform": Affine(
            90.0, 0.0, 65848.6171875, 0.0, -90.0, 103948.140625
        ),
    }
    assert layer_profile == expected_profile


def test_layered_file_handler_get_dne_layer(test_tl_fp):
    """Test getting a non-existent layer"""
    lf = LayeredFile(test_tl_fp)

    with pytest.raises(
        revrtKeyError, match="'non_existent_layer' is not present in"
    ):
        lf["non_existent_layer"]


def test_create_new_file(tmp_path, sample_tiff_fp, sample_tiff_props):
    """Test creating a new file"""
    *__, transform = sample_tiff_props

    test_fp = tmp_path / "test.zarr"
    assert not test_fp.exists()

    lf = LayeredFile(test_fp)
    lf.create_new(sample_tiff_fp, overwrite=True, chunk_x=100, chunk_y=50)

    assert test_fp.exists()

    with (
        xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds,
        rioxarray.open_rasterio(sample_tiff_fp) as xds,
    ):
        lat_lon = xds.rio.reproject("EPSG:4326")
        assert np.isclose(ds["longitude"].min(), lat_lon.x.min())
        assert np.isclose(ds["longitude"].max(), lat_lon.x.max())
        assert np.isclose(ds["latitude"].min(), lat_lon.y.min())
        assert np.isclose(ds["latitude"].max(), lat_lon.y.max())
        assert (*ds["band"].shape, *ds["y"].shape, *ds["x"].shape) == xds.shape
        assert ds["latitude"].shape == xds.shape[1:]
        assert ds["longitude"].shape == xds.shape[1:]

        _validate_top_level_ds_props(ds, transform)
        assert ds.attrs["chunks"] == {"x": 100, "y": 50}


def test_create_new_file_from_zarr(
    tmp_path, sample_tiff_fp, sample_tiff_props
):
    """Test creating a new file form a zarr template"""
    *__, transform = sample_tiff_props

    test_fp = tmp_path / "test.zarr"
    assert not test_fp.exists()

    lf = LayeredFile(test_fp)
    lf.create_new(sample_tiff_fp, overwrite=True, chunk_x=100, chunk_y=50)

    assert test_fp.exists()

    test_fp_2 = tmp_path / "test_2.zarr"
    assert not test_fp_2.exists()

    lf2 = LayeredFile(test_fp_2)
    lf2.create_new(test_fp, chunk_x=50, chunk_y=100)

    assert test_fp_2.exists()

    with (
        xr.open_dataset(
            test_fp, consolidated=False, engine="zarr"
        ) as ds_truth,
        xr.open_dataset(
            test_fp_2, consolidated=False, engine="zarr"
        ) as ds_test,
    ):
        assert np.allclose(ds_truth["longitude"], ds_test["longitude"])
        assert np.allclose(ds_truth["latitude"], ds_test["latitude"])
        assert ds_truth.sizes == ds_test.sizes
        assert set(ds_truth.coords) == set(ds_test.coords)

        _validate_top_level_ds_props(ds_test, transform)
        assert ds_test.attrs["chunks"] == {"x": 50, "y": 100}


def test_cleanup_on_file_create_error(tmp_path, monkeypatch):
    """Test cleanup on file create error"""

    orig_func = revrt.utilities.handlers._save_ds_as_zarr_with_encodings

    def _func_that_errors(out_ds, x, y, lat, lon, out_fp):
        orig_func(out_ds, x, y, lat, lon, out_fp)
        assert out_fp.exists()

        msg = "A test error"
        raise revrtValueError(msg)

    monkeypatch.setattr(
        revrt.utilities.handlers,
        "_save_ds_as_zarr_with_encodings",
        _func_that_errors,
    )

    test_fp = tmp_path / "test.zarr"
    assert not test_fp.exists()

    lf = LayeredFile(test_fp)
    with contextlib.suppress(revrtValueError):
        lf.create_new("test_file.txt")

    assert not test_fp.exists()


def test_write_layer(sample_tiff_fp, sample_tiff_props, tmp_path):
    """Test writing a layer to file"""
    __, __, width, height, __, transform = sample_tiff_props

    test_fp = tmp_path / "test.zarr"
    assert not test_fp.exists()

    lf = LayeredFile(test_fp)
    lf.create_new(sample_tiff_fp, overwrite=True)

    assert test_fp.exists()

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        _validate_top_level_ds_props(ds, transform)
        assert "test_layer" not in ds

    new_data = (
        np.random.default_rng()
        .random(width * height)
        .reshape((height, width))
        .astype(np.float32)
    )
    lf.write_layer(new_data, "test_layer")
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        _validate_top_level_ds_props(ds, transform)
        assert "test_layer" in ds
        _validate_random_data_layer(width, height, transform, ds["test_layer"])
        assert np.allclose(ds["test_layer"], new_data)
        assert "nodata" not in ds["test_layer"].attrs
        assert np.isnan(ds["test_layer"].rio.nodata)
        assert np.isnan(ds["test_layer"].rio.encoded_nodata)

    with pytest.raises(
        revrtKeyError, match="'test_layer' is already present in"
    ):
        lf.write_layer(new_data, "test_layer")

    new_data_2 = (
        np.random.default_rng()
        .random(width * height)
        .reshape((1, height, width))
        .astype(np.float32)
    )
    new_data_2 -= new_data
    new_data_2 -= new_data_2.min()
    new_data_2 /= new_data_2.max()

    lf.write_layer(new_data_2, "test_layer", overwrite=True)
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        _validate_top_level_ds_props(ds, transform)
        assert "test_layer" in ds
        _validate_random_data_layer(width, height, transform, ds["test_layer"])
        assert not np.allclose(ds["test_layer"], new_data)
        assert np.allclose(ds["test_layer"], new_data_2)
        assert "nodata" not in ds["test_layer"].attrs
        assert np.isnan(ds["test_layer"].rio.nodata)
        assert np.isnan(ds["test_layer"].rio.encoded_nodata)

    lf.write_layer(
        new_data, "original_layer", description="My desc", nodata=255
    )
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        _validate_top_level_ds_props(ds, transform)
        assert "original_layer" in ds
        _validate_random_data_layer(
            width, height, transform, ds["original_layer"]
        )
        assert np.allclose(ds["original_layer"], new_data)
        assert not np.allclose(ds["original_layer"], new_data_2)
        assert ds["original_layer"].attrs["description"] == "My desc"
        assert np.isclose(ds["original_layer"].attrs["nodata"], 255)
        assert np.isclose(ds["original_layer"].rio.encoded_nodata, 255)

    with pytest.raises(
        revrtValueError,
        match=r"Shape of provided data .* does not match shape of LayeredFile",
    ):
        lf.write_layer(new_data_2[:, :1, :1], "test_layer_2")


def test_write_tiff_to_layer_file(
    sample_tiff_fp, sample_tiff_fp_2x, sample_tiff_props, tmp_path
):
    """Test writing GeoTIFFs to a layered file"""
    *__, transform = sample_tiff_props

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer", nodata=255)

    with (
        xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds,
        rioxarray.open_rasterio(sample_tiff_fp) as tif,
    ):
        _validate_top_level_ds_props(ds, transform)
        assert "test_layer" in ds
        assert "test_layer_2" not in ds
        assert np.allclose(ds["test_layer"], tif)
        assert "nodata" in ds["test_layer"].attrs
        assert np.isclose(ds["test_layer"].rio.encoded_nodata, 255)

    lf.write_geotiff_to_file(sample_tiff_fp_2x, "test_layer_2")
    with (
        xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds,
        rioxarray.open_rasterio(sample_tiff_fp_2x) as tif,
    ):
        _validate_top_level_ds_props(ds, transform)
        assert "test_layer" in ds
        assert "test_layer_2" in ds
        assert np.allclose(ds["test_layer_2"], tif)
        assert "nodata" not in ds["test_layer_2"].attrs
        assert np.isnan(ds["test_layer_2"].rio.nodata)
        assert np.isnan(ds["test_layer_2"].rio.encoded_nodata)

    with pytest.warns(revrtWarning, match="Attempting to set ``nodata``"):
        lf.write_geotiff_to_file(
            sample_tiff_fp_2x,
            "test_layer",
            overwrite=True,
            description="My desc",
            nodata=100,
            check_tiff=False,
        )

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        _validate_top_level_ds_props(ds, transform)
        assert "test_layer" in ds
        assert "test_layer_2" in ds
        assert np.allclose(ds["test_layer_2"], ds["test_layer"])
        assert "nodata" in ds["test_layer"].attrs
        assert np.isclose(ds["test_layer"].rio.encoded_nodata, 255)


def test_layer_to_geotiff_file(sample_tiff_fp, tmp_path):
    """Test writing a layer to GeoTIFF file"""
    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")

    out_tiff_fp = tmp_path / "out.tif"
    lf.layer_to_geotiff("test_layer", out_tiff_fp)

    with (
        rioxarray.open_rasterio(sample_tiff_fp) as truth_tif,
        rioxarray.open_rasterio(out_tiff_fp) as test_tif,
    ):
        assert np.allclose(truth_tif, test_tif)
        assert np.allclose(truth_tif.rio.transform(), test_tif.rio.transform())
        assert truth_tif.rio.crs == test_tif.rio.crs


def test_extract_layers(sample_tiff_fp, sample_tiff_fp_2x, tmp_path):
    """Test extracting layers to GeoTIFF files"""
    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")
    lf.write_geotiff_to_file(sample_tiff_fp_2x, "test_layer_2")

    out_tiff_fp_1 = tmp_path / "out_1.tif"
    out_tiff_fp_2 = tmp_path / "out_2.tif"
    lf.extract_layers(
        {"test_layer": out_tiff_fp_1, "test_layer_2": out_tiff_fp_2}
    )

    with (
        rioxarray.open_rasterio(sample_tiff_fp) as truth_tif,
        rioxarray.open_rasterio(out_tiff_fp_1) as test_tif,
    ):
        assert np.allclose(truth_tif, test_tif)
        assert np.allclose(truth_tif.rio.transform(), test_tif.rio.transform())
        assert truth_tif.rio.crs == test_tif.rio.crs

    with (
        rioxarray.open_rasterio(sample_tiff_fp_2x) as truth_tif,
        rioxarray.open_rasterio(out_tiff_fp_2) as test_tif,
    ):
        assert np.allclose(truth_tif, test_tif)
        assert np.allclose(truth_tif.rio.transform(), test_tif.rio.transform())
        assert truth_tif.rio.crs == test_tif.rio.crs


@pytest.mark.parametrize("use_sub_dir", [True, False])
def test_extract_all_layers(
    sample_tiff_fp, sample_tiff_fp_2x, tmp_path, use_sub_dir
):
    """Test extracting layers to GeoTIFF files"""
    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")
    lf.write_geotiff_to_file(sample_tiff_fp_2x, "test_layer_2")

    out_dir = tmp_path / "sub_dir" if use_sub_dir else tmp_path
    lf.extract_all_layers(out_dir)

    assert len(list(out_dir.glob("*.tif"))) == 2

    with (
        rioxarray.open_rasterio(sample_tiff_fp) as truth_tif,
        rioxarray.open_rasterio(out_dir / "test_layer.tif") as test_tif,
    ):
        assert np.allclose(truth_tif, test_tif)
        assert np.allclose(truth_tif.rio.transform(), test_tif.rio.transform())
        assert truth_tif.rio.crs == test_tif.rio.crs

    with (
        rioxarray.open_rasterio(sample_tiff_fp_2x) as truth_tif,
        rioxarray.open_rasterio(out_dir / "test_layer_2.tif") as test_tif,
    ):
        assert np.allclose(truth_tif, test_tif)
        assert np.allclose(truth_tif.rio.transform(), test_tif.rio.transform())
        assert truth_tif.rio.crs == test_tif.rio.crs


@pytest.mark.parametrize("use_full_shape", [True, False])
@pytest.mark.parametrize("nodata", [None, 1024])
def test_write_tiff_using_layer_profile(
    sample_tiff_fp, sample_tiff_props, tmp_path, use_full_shape, nodata
):
    """Test writing data to GeoTIFF file using layer profile"""
    __, __, width, height, __, __ = sample_tiff_props

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")

    new_data = np.arange(width * height, dtype=np.float32)
    if use_full_shape:
        new_data = new_data.reshape((1, height, width)) * 3
    else:
        new_data = new_data.reshape((height, width)) * 3

    out_tiff_fp = tmp_path / "test.tif"
    assert not out_tiff_fp.exists()
    save_data_using_layer_file_profile(
        test_fp, new_data, out_tiff_fp, nodata=nodata
    )

    assert out_tiff_fp.exists()
    with rioxarray.open_rasterio(out_tiff_fp) as tif:
        assert np.allclose(tif, new_data)
        if nodata:
            assert np.isclose(tif.rio.nodata, nodata)


def test_write_tiff_using_layer_profile_bad_shape(
    sample_tiff_fp, sample_tiff_props, tmp_path
):
    """Test writing data to file using layer profile with bad shape"""
    __, __, width, height, __, __ = sample_tiff_props

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")

    new_data = np.arange(width * height * 4, dtype=np.float32)
    new_data = new_data.reshape((height * 2, width * 2))

    out_tiff_fp = tmp_path / "test.tif"

    with pytest.raises(
        revrtValueError,
        match=r"Shape of provided data .* does not match destination shape:",
    ):
        save_data_using_layer_file_profile(test_fp, new_data, out_tiff_fp)


def test_write_tiff_using_layer_profile_bool(
    sample_tiff_fp, sample_tiff_props, tmp_path
):
    """Test writing bool data to GeoTIFF file using layer profile"""
    __, __, width, height, __, __ = sample_tiff_props

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")

    new_data = np.full(width * height, True, dtype="bool")
    new_data = new_data.reshape((1, height, width))

    out_tiff_fp = tmp_path / "test.tif"
    assert not out_tiff_fp.exists()
    save_data_using_layer_file_profile(test_fp, new_data, out_tiff_fp)

    assert out_tiff_fp.exists()
    with rioxarray.open_rasterio(out_tiff_fp) as tif:
        assert np.allclose(tif, 1)
        assert tif.dtype == np.dtype("uint8")


def test_load_data_using_layer_file_profile(
    sample_tiff_fp, sample_tiff_props, tmp_path
):
    """Test loading data from GeoTIFF file using layer profile"""
    x0, y0, width, height, __, __ = sample_tiff_props

    proj = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )
    x0, y0 = proj.transform(x0, y0)

    target_crs = "EPSG:3857"
    target_shape = (height, width)
    target_transform = from_origin(x0, y0, 111320, 111320)

    out_fp = tmp_path / "test.tif"
    with rioxarray.open_rasterio(sample_tiff_fp) as tif:
        tif.rio.reproject(
            dst_crs=target_crs,
            shape=target_shape,
            transform=target_transform,
            num_threads=4,
            resampling=Resampling.nearest,
            INIT_DEST=0,
        ).rio.to_raster(out_fp, driver="GTiff")

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")

    test_tif = load_data_using_layer_file_profile(test_fp, sample_tiff_fp)
    test_tif_2 = load_data_using_layer_file_profile(test_fp, out_fp)

    with rioxarray.open_rasterio(sample_tiff_fp) as tif:
        assert test_tif.rio.crs == tif.rio.crs
        assert test_tif_2.rio.crs == tif.rio.crs
        assert np.allclose(test_tif, tif)

        assert test_tif_2.shape == tif.shape
        assert test_tif_2.any()
        assert not np.isnan(test_tif_2).all()
        assert np.allclose(test_tif_2[:, :-1], tif[:, :-1], atol=7)

    test_tif.close()
    test_tif_2.close()


@pytest.mark.parametrize("in_layer_dir", [True, False])
@pytest.mark.parametrize("band", [None, 0])
def test_load_data_using_file_full_path(
    sample_tiff_fp, tmp_path, in_layer_dir, band
):
    """Test loading data using layered transmission file profile"""

    layer_dir = tmp_path / "layers"
    layer_dir.mkdir()

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")

    if in_layer_dir:
        in_fp = "test.tif"
        shutil.copy(sample_tiff_fp, layer_dir / in_fp)
    else:
        in_fp = sample_tiff_fp

    test_tif = load_data_using_layer_file_profile(
        test_fp, in_fp, layer_dirs=[layer_dir], band_index=band
    )
    with rioxarray.open_rasterio(sample_tiff_fp) as tif:
        assert test_tif.rio.crs == tif.rio.crs
        if band is not None:
            assert test_tif.shape == tif.shape[1:]
            assert np.allclose(test_tif, tif[0])
        else:
            assert test_tif.shape == tif.shape
            assert np.allclose(test_tif, tif)
        assert np.allclose(tif.rio.transform(), test_tif.rio.transform())

    test_tif.close()

    with pytest.raises(revrtFileNotFoundError, match="Unable to find file"):
        file_full_path("DNE", layer_dir)


@pytest.mark.parametrize("as_list", [True, False])
def test_layers_to_file(sample_tiff_fp, sample_tiff_fp_2x, tmp_path, as_list):
    """Test adding multiple layers to file at once"""
    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)

    if as_list:
        layers = [sample_tiff_fp, sample_tiff_fp_2x]
        descriptions = None
        tl1_name = sample_tiff_fp.stem
        tl2_name = sample_tiff_fp_2x.stem
    else:
        tl1_name = "test_layer"
        tl2_name = "test_layer_2"
        layers = {
            tl1_name: sample_tiff_fp,
            tl2_name: sample_tiff_fp_2x,
        }
        descriptions = {tl1_name: "desc_1", tl2_name: "desc_2"}

    lf.layers_to_file(layers, descriptions=descriptions)

    with (
        xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds,
        rioxarray.open_rasterio(sample_tiff_fp) as truth_tif,
        rioxarray.open_rasterio(sample_tiff_fp_2x) as truth_tif_2,
    ):
        assert tl1_name in set(ds.variables)
        assert tl2_name in set(ds.variables)
        assert ds[tl1_name].rio.crs == truth_tif.rio.crs
        assert ds[tl2_name].rio.crs == truth_tif_2.rio.crs
        assert ds[tl1_name].rio.transform() == truth_tif.rio.transform()
        assert ds[tl2_name].rio.transform() == truth_tif_2.rio.transform()
        assert np.allclose(ds[tl1_name], truth_tif)
        assert np.allclose(ds[tl2_name], truth_tif_2)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
@pytest.mark.parametrize("as_list", [True, False])
def test_cli_layers_to_file(
    run_gaps_cli_with_expected_file,
    tmp_path,
    sample_tiff_fp,
    sample_tiff_fp_2x,
    as_list,
):
    """Test layers-to-file CLI"""

    out_file_fp = tmp_path / "test-cli.zarr"
    assert not out_file_fp.exists()
    config = {"fp": str(out_file_fp)}

    if as_list:
        config["layers"] = [str(sample_tiff_fp), str(sample_tiff_fp_2x)]
        tl1_name = sample_tiff_fp.stem
        tl2_name = sample_tiff_fp_2x.stem
    else:
        tl1_name = "test_layer"
        tl2_name = "test_layer_2"
        config["layers"] = {
            tl1_name: str(sample_tiff_fp),
            tl2_name: str(sample_tiff_fp_2x),
        }
        config["descriptions"] = {tl1_name: "desc_1", tl2_name: "desc_2"}

    run_gaps_cli_with_expected_file(
        "layers-to-file", config, tmp_path, glob_pattern="*.zarr"
    )

    with (
        xr.open_dataset(out_file_fp, consolidated=False, engine="zarr") as ds,
        rioxarray.open_rasterio(sample_tiff_fp) as truth_tif,
        rioxarray.open_rasterio(sample_tiff_fp_2x) as truth_tif_2,
    ):
        assert tl1_name in set(ds.variables)
        assert tl2_name in set(ds.variables)
        assert ds[tl1_name].rio.crs == truth_tif.rio.crs
        assert ds[tl2_name].rio.crs == truth_tif_2.rio.crs
        assert ds[tl1_name].rio.transform() == truth_tif.rio.transform()
        assert ds[tl2_name].rio.transform() == truth_tif_2.rio.transform()
        assert np.allclose(ds[tl1_name], truth_tif)
        assert np.allclose(ds[tl2_name], truth_tif_2)

        assert ds[tl1_name].attrs["description"] == config.get(
            "descriptions", {}
        ).get(tl1_name)
        assert ds[tl2_name].attrs["description"] == config.get(
            "descriptions", {}
        ).get(tl2_name)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_layers_from_file_single(
    cli_runner, tmp_path, sample_tiff_fp, sample_tiff_fp_2x
):
    """Test layers-to-file CLI"""

    out_file_fp = tmp_path / "test-cli.zarr"
    lf = LayeredFile(out_file_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")
    lf.write_geotiff_to_file(sample_tiff_fp_2x, "test_layer_2")

    config = {
        "fp": str(out_file_fp),
        "out_layer_dir": str(tmp_path / "test"),
        "layers": ["test_layer_2"],
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    result = cli_runner.invoke(main, ["layers-from-file", "-c", config_path])
    msg = f"Failed with error {traceback.print_exception(*result.exc_info)}"
    assert result.exit_code == 0, msg

    out_tiff_fp_1 = tmp_path / "test" / "test_layer.tif"
    out_tiff_fp_2 = tmp_path / "test" / "test_layer_2.tif"

    assert not out_tiff_fp_1.exists()
    assert out_tiff_fp_2.exists()

    with (
        rioxarray.open_rasterio(sample_tiff_fp_2x) as truth_tif,
        rioxarray.open_rasterio(out_tiff_fp_2) as test_tif,
    ):
        assert np.allclose(truth_tif, test_tif)
        assert np.allclose(truth_tif.rio.transform(), test_tif.rio.transform())
        assert truth_tif.rio.crs == test_tif.rio.crs


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_layers_from_file_all(
    cli_runner, tmp_path, sample_tiff_fp, sample_tiff_fp_2x
):
    """Test layers-from-file CLI"""

    out_file_fp = tmp_path / "test-cli.zarr"
    lf = LayeredFile(out_file_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")
    lf.write_geotiff_to_file(sample_tiff_fp_2x, "test_layer_2")

    config = {"fp": str(out_file_fp)}

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    result = cli_runner.invoke(main, ["layers-from-file", "-c", config_path])
    msg = f"Failed with error {traceback.print_exception(*result.exc_info)}"
    assert result.exit_code == 0, msg

    out_tiff_fp_1 = tmp_path / "test_layer.tif"
    out_tiff_fp_2 = tmp_path / "test_layer_2.tif"

    assert len(list(tmp_path.glob("*.tif"))) == 2
    assert out_tiff_fp_1.exists()
    assert out_tiff_fp_2.exists()

    with (
        rioxarray.open_rasterio(sample_tiff_fp) as truth_tif,
        rioxarray.open_rasterio(out_tiff_fp_1) as test_tif,
    ):
        assert np.allclose(truth_tif, test_tif)
        assert np.allclose(truth_tif.rio.transform(), test_tif.rio.transform())
        assert truth_tif.rio.crs == test_tif.rio.crs

    with (
        rioxarray.open_rasterio(sample_tiff_fp_2x) as truth_tif,
        rioxarray.open_rasterio(out_tiff_fp_2) as test_tif,
    ):
        assert np.allclose(truth_tif, test_tif)
        assert np.allclose(truth_tif.rio.transform(), test_tif.rio.transform())
        assert truth_tif.rio.crs == test_tif.rio.crs


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_layers_from_file_strips_required_path_whitespace(
    cli_runner, tmp_path, sample_tiff_fp, sample_tiff_fp_2x
):
    """layers-from-file CLI strips whitespace on required path inputs"""

    out_file_fp = tmp_path / "test-cli-whitespace.zarr"
    lf = LayeredFile(out_file_fp)
    lf.write_geotiff_to_file(sample_tiff_fp, "test_layer")
    lf.write_geotiff_to_file(sample_tiff_fp_2x, "test_layer_2")

    out_layer_dir = tmp_path / "trimmed_output"
    config = {
        "fp": f"  {out_file_fp}  ",
        "out_layer_dir": f"  {out_layer_dir}  ",
        "layers": ["test_layer"],
    }

    config_path = tmp_path / "config_whitespace.json"
    config_path.write_text(json.dumps(config))

    result = cli_runner.invoke(main, ["layers-from-file", "-c", config_path])
    msg = f"Failed with error {traceback.print_exception(*result.exc_info)}"
    assert result.exit_code == 0, msg
    assert (out_layer_dir / "test_layer.tif").exists()


@pytest.mark.parametrize(
    "chunk_size",
    [
        pytest.param(2, id="chunk-size-less"),
        pytest.param(5, id="chunk-size-equal"),
        pytest.param(10, id="chunk-size-greater"),
    ],
)
def test_chunked_read_gpkg_chunk_sizes(chunk_size, gpkg_with_nan_features):
    """Test chunked_read_gpkg yields expected GeoDataFrame chunks"""

    gpkg_fp, expected = gpkg_with_nan_features
    total_rows = len(expected)

    chunks = list(chunked_read_gpkg(gpkg_fp, chunk_size))
    assert chunks
    assert all(isinstance(chunk, gpd.GeoDataFrame) for chunk in chunks)

    chunk_lengths = [len(chunk) for chunk in chunks]
    assert all(length > 0 for length in chunk_lengths)

    expected_chunk_count = math.ceil(total_rows / chunk_size)
    assert len(chunks) == expected_chunk_count

    if chunk_size < total_rows:
        assert all(length == chunk_size for length in chunk_lengths[:-1])
        assert chunk_lengths[-1] <= chunk_size
    else:
        assert len(chunks) == 1
        assert chunk_lengths[0] == total_rows

    assert all(chunk.crs == expected.crs for chunk in chunks)
    assert any(chunk["voltage_kv"].isna().any() for chunk in chunks)
    assert any(chunk["all_nan"].isna().all() for chunk in chunks)

    reconstructed = gpd.GeoDataFrame(
        pd.concat(chunks, ignore_index=True),
        geometry="geometry",
        crs=expected.crs,
    ).reset_index(drop=True)

    for column in ("voltage_kv", "all_nan"):
        reconstructed[column] = reconstructed[column].astype(float)
    reconstructed = reconstructed[expected.columns]

    assert list(reconstructed["feature_id"]) == list(expected["feature_id"])
    assert_geodataframe_equal(reconstructed, expected, check_like=False)

    assert (
        reconstructed["voltage_kv"].isna().sum()
        == expected["voltage_kv"].isna().sum()
    )
    assert reconstructed["all_nan"].isna().all()


def test_num_feats_in_gpkg_normal(test_routing_data_dir):
    """Test counting features in a GeoPackage file"""
    test_fp = test_routing_data_dir / "ri_regions.gpkg"
    assert num_feats_in_gpkg(test_fp) == len(gpd.read_file(test_fp))


def test_num_feats_empty_gpkg(tmp_path):
    """Test counting features in an empty GeoPackage file"""
    test_fp = tmp_path / "ri_regions.gpkg"
    gpd.GeoDataFrame(
        columns=["id", "name", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(test_fp, driver="GPKG")
    assert num_feats_in_gpkg(test_fp) == 0


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
