"""Integration tests for reVRt handlers"""

import os
import json
import platform
import traceback
from pathlib import Path

import pytest
import rioxarray
import numpy as np
import xarray as xr
from pyproj.crs import CRS
from rasterio.transform import Affine

from revrt._cli import main
from revrt.utilities import LayeredFile


_EXPECTED_TRANSFORM = Affine(
    90.0, 0.0, 65848.6171875, 0.0, -90.0, 103948.140625
)
_EXPECTED_CRS = CRS(
    "+proj=tmerc +lat_0=41.0833333333333 "  # cspell:disable-line
    "+lon_0=-71.5 +k=0.99999375 +x_0=100000 +y_0=0 "
    "+ellps=GRS80 +units=m +no_defs=True"  # cspell:disable-line
)


@pytest.fixture(scope="module")
def test_ri_exclusions_fp(test_utility_data_dir):
    """Return path to Rhode Island exclusion layers test file"""
    return test_utility_data_dir / "ri_exclusions.zarr"


@pytest.fixture(scope="module")
def ri_tb_tiff_fp(test_data_dir):
    """Return path to transmission barrier TIFF file used for tests"""
    return test_data_dir / "utilities" / "ri_transmission_barriers.tif"


@pytest.fixture(scope="module")
def ri_regions_fp(test_data_dir):
    """Return path to regions TIFF file used for tests"""
    return test_data_dir / "utilities" / "ri_regions.tif"


def _validate_top_level_ds_props(ds):
    """Validate top level dataset properties"""
    assert ds.rio.crs == _EXPECTED_CRS
    assert np.allclose(ds.rio.transform(), _EXPECTED_TRANSFORM)
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


def test_create_new_file(tmp_path, ri_tb_tiff_fp):
    """Test creating a new file"""

    test_fp = tmp_path / "test.zarr"
    assert not test_fp.exists()

    lf = LayeredFile(test_fp)
    lf.create_new(ri_tb_tiff_fp, overwrite=True)

    assert test_fp.exists()

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        _validate_top_level_ds_props(ds)
        assert "iso_regions" not in ds

    with (
        xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds,
        rioxarray.open_rasterio(ri_tb_tiff_fp) as xds,
    ):
        lat_lon = xds.rio.reproject("EPSG:4326")
        assert np.isclose(ds["longitude"].min(), lat_lon.x.min())
        assert np.isclose(ds["longitude"].max(), lat_lon.x.max())
        assert np.isclose(ds["latitude"].min(), lat_lon.y.min())
        assert np.isclose(ds["latitude"].max(), lat_lon.y.max())
        assert (*ds["band"].shape, *ds["y"].shape, *ds["x"].shape) == xds.shape
        assert ds["latitude"].shape == xds.shape[1:]
        assert ds["longitude"].shape == xds.shape[1:]


def test_write_layer(tmp_path, ri_regions_fp):
    """Test writing layer data to ``LayeredFile``"""

    test_fp = tmp_path / "test.zarr"
    assert not test_fp.exists()

    lf = LayeredFile(test_fp).create_new(ri_regions_fp, overwrite=True)

    assert test_fp.exists()

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        _validate_top_level_ds_props(ds)
        assert "iso_regions" not in ds

    with rioxarray.open_rasterio(ri_regions_fp) as tif:
        data = tif.squeeze().values
        truth_crs = tif.rio.crs
        truth_transform = tif.rio.transform()

    lf.write_layer(data, "iso_regions", description="ISO")

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        assert "iso_regions" in ds
        assert np.allclose(ds["iso_regions"], data[None])
        assert ds["iso_regions"].attrs["description"] == "ISO"
        assert np.allclose(ds["iso_regions"].rio.transform(), truth_transform)
        assert ds["iso_regions"].rio.crs == truth_crs


def test_extract_layer(tmp_path, ri_regions_fp):
    """Test extracting layer data from ``LayeredFile``"""

    test_fp = tmp_path / "test.zarr"
    assert not test_fp.exists()

    lf = LayeredFile(test_fp)
    lf.create_new(ri_regions_fp)

    assert test_fp.exists()

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        _validate_top_level_ds_props(ds)
        assert "iso_regions" not in ds

    lf.write_geotiff_to_file(ri_regions_fp, "iso_regions")

    test_profile, test_values = lf["iso_regions"]
    with rioxarray.open_rasterio(ri_regions_fp) as tif:
        assert np.allclose(test_values, tif)
        assert test_profile["crs"] == tif.rio.crs
        assert np.allclose(test_profile["transform"], tif.rio.transform())


def test_extract_layer_to_geotiff(tmp_path, ri_regions_fp):
    """Test extracting layer data from file"""

    test_fp = tmp_path / "test.zarr"
    assert not test_fp.exists()

    lf = LayeredFile(test_fp)
    lf.create_new(ri_regions_fp)

    assert test_fp.exists()

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        _validate_top_level_ds_props(ds)
        assert "iso_regions" not in ds

    lf.write_geotiff_to_file(ri_regions_fp, "iso_regions")

    out_tiff_fp = tmp_path / "out.tif"
    lf.layer_to_geotiff("iso_regions", out_tiff_fp)

    with (
        rioxarray.open_rasterio(ri_regions_fp) as truth_tif,
        rioxarray.open_rasterio(out_tiff_fp) as test_tif,
    ):
        assert np.allclose(truth_tif, test_tif)
        assert np.allclose(truth_tif.rio.transform(), test_tif.rio.transform())
        assert truth_tif.rio.crs == test_tif.rio.crs


def test_write_geotiff_to_layer_file(tmp_path, ri_tb_tiff_fp, ri_regions_fp):
    """Test writing layer directly from GeoTIFF"""

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.create_new(ri_tb_tiff_fp)

    lf.write_geotiff_to_file(
        ri_regions_fp,
        "iso_regions",
        check_tiff=True,
        description="ISO",
        overwrite=True,
    )

    with (
        rioxarray.open_rasterio(ri_regions_fp) as tif,
        xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds,
    ):
        assert np.allclose(ds["iso_regions"], tif)
        assert np.allclose(
            ds["iso_regions"].rio.transform(), tif.rio.transform()
        )
        assert ds["iso_regions"].rio.crs == tif.rio.crs


def test_extract_all_layers_to_geotiff(tmp_path, ri_regions_fp):
    """Test extracting all layer data from ``LayeredFile``"""

    test_fp = tmp_path / "test.zarr"

    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(ri_regions_fp, "iso_regions")

    lf.extract_all_layers(tmp_path)

    with (
        rioxarray.open_rasterio(ri_regions_fp) as truth_tif,
        rioxarray.open_rasterio(tmp_path / "iso_regions.tif") as test_tif,
    ):
        assert np.allclose(truth_tif, test_tif)
        assert np.allclose(truth_tif.rio.transform(), test_tif.rio.transform())
        assert truth_tif.rio.crs == test_tif.rio.crs


@pytest.mark.parametrize(
    "layer",
    # cspell:disable-next-line
    ["ri_padus", "ri_reeds_regions", "ri_smod", "ri_srtm_slope"],
)
def test_layer_to_geotiff(
    tmp_path, test_ri_exclusions_fp, test_utility_data_dir, layer
):
    """Test extraction of layer and creation of GeoTIFF"""
    out_tiff_fp = tmp_path / f"test_{layer}.tif"
    lf = LayeredFile(test_ri_exclusions_fp)
    lf.layer_to_geotiff(layer, out_tiff_fp)

    truth_tiff_fp = test_utility_data_dir / f"{layer}.tif"

    with (
        rioxarray.open_rasterio(truth_tiff_fp) as truth_tif,
        rioxarray.open_rasterio(out_tiff_fp) as test_tif,
    ):
        assert np.allclose(truth_tif, test_tif)
        assert np.allclose(truth_tif.rio.transform(), test_tif.rio.transform())


@pytest.mark.parametrize(
    "tif",
    [
        "ri_padus.tif",
        "ri_reeds_regions.tif",
        "ri_smod.tif",  # cspell:disable-line
        "ri_srtm_slope.tif",  # cspell:disable-line
    ],
)
def test_geotiff_to_layer_file(tif, tmp_path, test_utility_data_dir):
    """Test creation of ``LayeredFile`` from GeoTIFF"""
    in_tiff_fp = test_utility_data_dir / tif
    layer = in_tiff_fp.stem

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.write_geotiff_to_file(in_tiff_fp, layer)

    with (
        rioxarray.open_rasterio(in_tiff_fp, masked=True) as truth_tif,
        xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds,
    ):
        assert np.allclose(ds[layer], truth_tif, equal_nan=True)
        assert np.allclose(
            ds[layer].rio.transform(), truth_tif.rio.transform()
        )
        assert ds[layer].rio.crs == truth_tif.rio.crs


@pytest.mark.parametrize("as_list", [True, False])
def test_roundtrip(as_list, tmp_path, test_utility_data_dir):
    """Test creation of ``LayeredFile`` from GeoTIFF and back again"""
    layers = [
        test_utility_data_dir / "ri_padus.tif",
        test_utility_data_dir / "ri_reeds_regions.tif",
        test_utility_data_dir / "ri_smod.tif",  # cspell:disable-line
        test_utility_data_dir / "ri_srtm_slope.tif",  # cspell:disable-line
    ]
    layer_names = {fp.stem: fp for fp in layers}
    descriptions = {fp.stem: f"desc_{i}" for i, fp in enumerate(layers)}

    if not as_list:
        layers = layer_names

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp)
    lf.layers_to_file(layers, descriptions=descriptions)
    lf.extract_all_layers(tmp_path)

    for layer, truth_fp in layer_names.items():
        test_tiff_fp = tmp_path / f"{layer}.tif"
        with (
            rioxarray.open_rasterio(truth_fp, masked=True) as truth_tif,
            rioxarray.open_rasterio(test_tiff_fp, masked=True) as test_tif,
            xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds,
        ):
            assert np.allclose(ds[layer], truth_tif, equal_nan=True)
            assert np.allclose(test_tif, truth_tif, equal_nan=True)
            assert np.allclose(
                ds[layer].rio.transform(), truth_tif.rio.transform()
            )
            assert np.allclose(
                test_tif.rio.transform(), truth_tif.rio.transform()
            )
            assert ds[layer].rio.crs == truth_tif.rio.crs
            assert test_tif.rio.crs == truth_tif.rio.crs
            assert ds[layer].attrs["description"] == descriptions[layer]


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
@pytest.mark.parametrize("as_list", [True, False])
def test_roundtrip_cli(cli_runner, tmp_path, test_utility_data_dir, as_list):
    """Test CLI with round-trip data conversion"""

    out_file_fp = tmp_path / "test-cli.zarr"
    assert not out_file_fp.exists()
    config = {"fp": str(out_file_fp)}

    config["layers"] = [
        test_utility_data_dir / "ri_padus.tif",
        test_utility_data_dir / "ri_reeds_regions.tif",
        test_utility_data_dir / "ri_smod.tif",  # cspell:disable-line
        test_utility_data_dir / "ri_srtm_slope.tif",  # cspell:disable-line
    ]
    layer_names = {fp.stem: fp for fp in config["layers"]}
    config["descriptions"] = {
        fp.stem: f"desc_{i}" for i, fp in enumerate(config["layers"])
    }

    if not as_list:
        config["layers"] = {k: str(v) for k, v in layer_names.items()}
    else:
        config["layers"] = list(map(str, config["layers"]))

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    result = cli_runner.invoke(main, ["layers-to-file", "-c", config_path])
    msg = f"Failed with error {traceback.print_exception(*result.exc_info)}"
    assert result.exit_code == 0, msg

    config_extract_path = tmp_path / "config_extract.json"
    config_extract_path.write_text(json.dumps({"fp": str(out_file_fp)}))

    result = cli_runner.invoke(
        main, ["layers-from-file", "-c", config_extract_path]
    )
    msg = f"Failed with error {traceback.print_exception(*result.exc_info)}"
    assert result.exit_code == 0, msg

    for layer, truth_fp in layer_names.items():
        test_tiff_fp = tmp_path / f"{layer}.tif"
        with (
            rioxarray.open_rasterio(truth_fp, masked=True) as truth_tif,
            rioxarray.open_rasterio(test_tiff_fp, masked=True) as test_tif,
            xr.open_dataset(
                out_file_fp, consolidated=False, engine="zarr"
            ) as ds,
        ):
            assert np.allclose(ds[layer], truth_tif, equal_nan=True)
            assert np.allclose(test_tif, truth_tif, equal_nan=True)
            assert np.allclose(
                ds[layer].rio.transform(), truth_tif.rio.transform()
            )
            assert np.allclose(
                test_tif.rio.transform(), truth_tif.rio.transform()
            )
            assert ds[layer].rio.crs == truth_tif.rio.crs
            assert test_tif.rio.crs == truth_tif.rio.crs
            assert (
                ds[layer].attrs["description"] == config["descriptions"][layer]
            )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
