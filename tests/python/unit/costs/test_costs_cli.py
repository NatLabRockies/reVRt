"""Test masks for cost layer creation"""

import os
import json
import logging
import platform
import shutil
import traceback
from types import SimpleNamespace
from pathlib import Path

import pytest
import rioxarray
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

from revrt._cli import main
from revrt.constants import BARRIER_H5_LAYER_NAME, METERS_IN_MILE
from revrt.costs.cli import build_masks, build_routing_layer_file
from revrt.costs.layer_creator import LayerCreator
from revrt.costs.masks import Masks
from revrt.models.cost_layers import LayerBuildConfig
from revrt.exceptions import revrtConfigurationError
from revrt.warn import revrtWarning
from revrt.utilities import LayeredFile, load_data_using_layer_file_profile


@pytest.fixture(scope="module")
def tiff_layers_for_testing(sample_tiff_props, tmp_path_factory):
    """Test TIFF layers for testing LayerCreator"""
    layer_dir = tmp_path_factory.mktemp("layers")

    x0, y0, width, height, cell_size, transform = sample_tiff_props

    layers = {
        "friction_1.tif": np.array([[ind] * width for ind in range(height)]),
        "fi_1.tif": np.array([list(range(width))] * height),
    }

    for layer_fn, data in layers.items():
        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={
                "x": x0 + np.arange(width) * cell_size + cell_size / 2,
                "y": y0 - np.arange(height) * cell_size - cell_size / 2,
            },
            name="test_band",
        )

        da = da.rio.write_crs("ESRI:102008")
        da.rio.write_transform(transform)
        da.rio.to_raster(layer_dir / layer_fn, driver="GTiff")

    return layer_dir, layers


@pytest.fixture(scope="module")
def masks_for_testing(sample_tiff_props, tmp_path_factory):
    """Masks for testing build function"""
    *__, width, height, ___, transform = sample_tiff_props

    masks_dir = tmp_path_factory.mktemp("masks")
    land_mask_fp = masks_dir / "test_basic_shape_mask.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    masks = Masks(
        shape=(height, width),
        crs="ESRI:102008",
        transform=transform,
        masks_dir=masks_dir,
    )
    masks.create(land_mask_fp, save_tiff=True, reproject_vector=True)
    return masks


@pytest.fixture(scope="module")
def expected_masks():
    """Return hardcoded mask arrays matching the sample land geometry"""
    landfall_mask = np.array(
        [
            [False, False, False, False, False, False],
            [False, False, False, True, True, False],
            [False, False, True, False, True, False],
            [False, False, True, False, True, False],
            [False, False, True, True, False, False],
        ],
        dtype=bool,
    )

    dry_mask = np.array(
        [
            [False, False, False, False, False, False],
            [False, False, False, False, True, False],
            [False, False, False, True, True, False],
            [False, False, False, True, True, False],
            [False, False, True, True, False, False],
        ],
        dtype=bool,
    )

    wet_mask = np.array(
        [
            [True, True, True, True, True, True],
            [True, True, True, False, True, True],
            [True, True, False, False, True, True],
            [True, True, False, False, True, True],
            [True, True, True, True, True, True],
        ],
        dtype=bool,
    )

    dry_plus_mask = np.array(
        [
            [False, False, False, False, False, False],
            [False, False, False, True, True, False],
            [False, False, True, True, True, False],
            [False, False, True, True, True, False],
            [False, False, True, True, False, False],
        ],
        dtype=bool,
    )

    wet_plus_mask = np.array(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
            [True, True, True, False, True, True],
            [True, True, True, False, True, True],
            [True, True, True, True, True, True],
        ],
        dtype=bool,
    )

    class ExpectedMasks:
        """Container for hardcoded mask arrays used in tests"""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return ExpectedMasks(
        landfall_mask=landfall_mask,
        dry_mask=dry_mask,
        wet_mask=wet_mask,
        dry_plus_mask=dry_plus_mask,
        wet_plus_mask=wet_plus_mask,
    )


@pytest.fixture(scope="module")
def basic_land_mask(tmp_path_factory):
    """Write a basic union-of-boxes polygon to disk as a land mask"""

    land_mask_fp = tmp_path_factory.mktemp("masks") / "land_mask.gpkg"
    geometry = unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])
    basic_shape = gpd.GeoDataFrame(geometry=[geometry], crs="ESRI:102008")
    basic_shape.to_file(land_mask_fp, driver="GPKG")
    return land_mask_fp


def _copy_test_layers(source_dir, out_dir):
    """Copy test input TIFFs into a temp directory"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer_fp in source_dir.iterdir():
        shutil.copy2(layer_fp, out_dir / layer_fp.name)


def _overwrite_test_tiff(layer_fp, data):
    """Overwrite a test TIFF while preserving its spatial metadata"""
    with rioxarray.open_rasterio(layer_fp) as tif:
        coords = {"x": tif.x.values, "y": tif.y.values}
        crs = tif.rio.crs
        transform = tif.rio.transform()
        name = tif.name

    da = xr.DataArray(
        np.asarray(data, dtype=np.float32),
        dims=("y", "x"),
        coords=coords,
        name=name,
    )
    da = da.rio.write_crs(crs)
    da.rio.write_transform(transform)
    da.rio.to_raster(layer_fp, driver="GTiff")


def _template_metadata(template_file):
    """Return shape, CRS, and affine transform for template"""
    template_file = Path(template_file)
    if template_file.suffix == ".zarr":
        open_func = xr.open_dataset
        kwargs = {"consolidated": False, "engine": "zarr"}
    else:
        open_func = rioxarray.open_rasterio
        kwargs = {}

    with open_func(template_file, chunks="auto", **kwargs) as fh:
        shape = fh.rio.shape
        crs = fh.rio.crs
        transform = fh.rio.transform()

    return shape, crs, transform


def _load_masks_from_disk(masks_dir, template_file):
    """Load masks written to disk for comparison"""
    shape, crs, transform = _template_metadata(template_file)
    layered_fp = (
        Path(masks_dir).parent / f"{Path(masks_dir).name}_template.zarr"
    )
    lf = LayeredFile(layered_fp)
    lf.create_new(template_file, overwrite=True)

    masks = Masks(
        shape=shape,
        crs=crs,
        transform=transform,
        masks_dir=masks_dir,
    )
    masks.load(layered_fp)
    return masks


def test_build_masks_writes_expected_outputs_from_geotiff(
    tmp_path, sample_extra_fp, basic_land_mask, expected_masks
):
    """build_masks writes all mask GeoTIFFs matching in-memory expectations"""

    masks_dir = tmp_path / "masks"
    build_masks(
        land_mask_shp_fp=basic_land_mask,
        template_file=sample_extra_fp,
        masks_dir=masks_dir,
        reproject_vector=False,
    )

    for fname in (
        Masks.LANDFALL_MASK_FNAME,
        Masks.RAW_LAND_MASK_FNAME,
        Masks.LAND_MASK_FNAME,
        Masks.OFFSHORE_MASK_FNAME,
    ):
        assert (masks_dir / fname).exists()

    actual_masks = _load_masks_from_disk(masks_dir, sample_extra_fp)

    assert np.array_equal(
        actual_masks.landfall_mask, expected_masks.landfall_mask
    )
    assert np.array_equal(actual_masks.dry_mask, expected_masks.dry_mask)
    assert np.array_equal(actual_masks.wet_mask, expected_masks.wet_mask)
    assert np.array_equal(
        actual_masks.dry_plus_mask, expected_masks.dry_plus_mask
    )
    assert np.array_equal(
        actual_masks.wet_plus_mask, expected_masks.wet_plus_mask
    )


def test_build_masks_writes_expected_outputs_from_zarr(
    tmp_path, sample_extra_fp, basic_land_mask, expected_masks
):
    """build_masks handles zarr templates via xarray.open_dataset"""

    template_zarr = tmp_path / "template.zarr"
    LayeredFile(template_zarr).create_new(sample_extra_fp)

    masks_dir = tmp_path / "masks_zarr"
    build_masks(
        land_mask_shp_fp=basic_land_mask,
        template_file=template_zarr,
        masks_dir=masks_dir,
        reproject_vector=False,
    )

    for fname in (
        Masks.LANDFALL_MASK_FNAME,
        Masks.RAW_LAND_MASK_FNAME,
        Masks.LAND_MASK_FNAME,
        Masks.OFFSHORE_MASK_FNAME,
    ):
        assert (masks_dir / fname).exists()

    actual_masks = _load_masks_from_disk(masks_dir, template_zarr)

    assert np.array_equal(
        actual_masks.landfall_mask, expected_masks.landfall_mask
    )
    assert np.array_equal(actual_masks.dry_mask, expected_masks.dry_mask)
    assert np.array_equal(actual_masks.wet_mask, expected_masks.wet_mask)
    assert np.array_equal(
        actual_masks.dry_plus_mask, expected_masks.dry_plus_mask
    )
    assert np.array_equal(
        actual_masks.wet_plus_mask, expected_masks.wet_plus_mask
    )


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_build_masks_cli_creates_expected_outputs(
    tmp_path, sample_extra_fp, cli_runner, basic_land_mask, expected_masks
):
    """CLI build-masks command writes expected mask rasters"""

    masks_dir = tmp_path / "masks_cli"
    config = {
        "land_mask_shp_fp": str(basic_land_mask),
        "template_file": str(sample_extra_fp),
        "masks_dir": str(masks_dir),
        "reproject_vector": False,
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    result = cli_runner.invoke(main, ["build-masks", "-c", str(config_path)])
    msg = f"Failed with error {traceback.print_exception(*result.exc_info)}"
    assert result.exit_code == 0, msg

    for fname in (
        Masks.LANDFALL_MASK_FNAME,
        Masks.RAW_LAND_MASK_FNAME,
        Masks.LAND_MASK_FNAME,
        Masks.OFFSHORE_MASK_FNAME,
    ):
        assert (masks_dir / fname).exists()

    actual_masks = _load_masks_from_disk(masks_dir, sample_extra_fp)

    assert np.array_equal(
        actual_masks.landfall_mask, expected_masks.landfall_mask
    )
    assert np.array_equal(actual_masks.dry_mask, expected_masks.dry_mask)
    assert np.array_equal(actual_masks.wet_mask, expected_masks.wet_mask)
    assert np.array_equal(
        actual_masks.dry_plus_mask, expected_masks.dry_plus_mask
    )
    assert np.array_equal(
        actual_masks.wet_plus_mask, expected_masks.wet_plus_mask
    )


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_build_masks_cli_strips_required_path_whitespace(
    tmp_path, sample_extra_fp, cli_runner, basic_land_mask
):
    """CLI build-masks strips whitespace on required path inputs"""

    masks_dir = tmp_path / "masks_cli_whitespace"
    config = {
        "land_mask_shp_fp": f"  {basic_land_mask}  ",
        "template_file": f"  {sample_extra_fp}  ",
        "masks_dir": f"  {masks_dir}  ",
        "reproject_vector": False,
    }

    config_path = tmp_path / "config_whitespace.json"
    config_path.write_text(json.dumps(config))

    result = cli_runner.invoke(main, ["build-masks", "-c", str(config_path)])
    msg = f"Failed with error {traceback.print_exception(*result.exc_info)}"
    assert result.exit_code == 0, msg
    assert (masks_dir / Masks.LANDFALL_MASK_FNAME).exists()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_build_routing_layer_file_cli_strips_required_path_whitespace(
    run_gaps_cli_with_expected_file,
    tmp_path,
    sample_iso_fp,
    sample_nlcd_fp,
    sample_slope_fp,
    sample_extra_fp,
    tiff_layers_for_testing,
    masks_for_testing,
):
    """CLI build-routing-layer-file strips whitespace on required paths"""

    test_fp = tmp_path / "trimmed_test.zarr"
    out_tiff_dir = tmp_path / "trimmed_out_tiffs"
    layer_dir, __ = tiff_layers_for_testing

    config = {
        "execution_control": {"max_workers": 1},
        "routing_file": f"  {test_fp}  ",
        "template_file": str(sample_extra_fp),
        "input_layer_dir": str(layer_dir),
        "output_tiff_dir": f"  {out_tiff_dir}  ",
        "masks_dir": f"    {masks_for_testing._masks_dir}  ",
        "layers": [
            {
                "layer_name": "fi_1",
                "include_in_file": False,
                "build": {
                    "  fi_1.tif ": {"extent": "wet+", "pass_through": True}
                },
            }
        ],
        "dry_costs": {
            "iso_region_tiff": str(sample_iso_fp),
            "nlcd_tiff": str(sample_nlcd_fp),
            "slope_tiff": str(sample_slope_fp),
            "extra_tiffs": [str(sample_extra_fp)],
        },
    }

    run_gaps_cli_with_expected_file(
        "build-routing-layer-file",
        config,
        tmp_path,
        glob_pattern="trimmed_test.zarr",
    )

    assert test_fp.exists()
    assert (out_tiff_dir / "fi_1.tif").exists()


def test_build_config_missing_action(tmp_path):
    """Test correct error is raised for config with no actions"""
    tiff_fp = tmp_path / "nonexistent.tif"
    tiff_fp.touch()

    with pytest.raises(
        revrtConfigurationError,
        match=r"At least one of .* must be in the config file",
    ):
        build_routing_layer_file(
            routing_file=tmp_path / "test.zarr", template_file=tiff_fp
        )


@pytest.mark.parametrize("mw", [None, 1, 2])
def test_build_basic_all(
    tmp_path,
    sample_iso_fp,
    sample_nlcd_fp,
    sample_slope_fp,
    sample_extra_fp,
    tiff_layers_for_testing,
    masks_for_testing,
    mw,
):
    """Test basic building of layers, dry costs, and merging"""
    test_fp = tmp_path / "test.zarr"
    out_tiff_dir = tmp_path / "out_tiffs"
    layer_dir, layers = tiff_layers_for_testing

    assert not test_fp.exists()
    assert not out_tiff_dir.exists()

    config = {
        "routing_file": test_fp,
        "template_file": sample_extra_fp,
        "input_layer_dir": layer_dir,
        "output_tiff_dir": out_tiff_dir,
        "masks_dir": masks_for_testing._masks_dir,
        "layers": [
            {
                "layer_name": "fi_1",
                "include_in_file": False,
                "build": {
                    "fi_1.tif": {"extent": "wet+", "pass_through": True}
                },
            },
            {
                "layer_name": "friction",
                "build": {
                    "friction_1.tif": {
                        "extent": "dry+",
                        "map": {x: x for x in range(20)},
                    }
                },
            },
        ],
        "dry_costs": {
            "iso_region_tiff": sample_iso_fp,
            "nlcd_tiff": sample_nlcd_fp,
            "slope_tiff": sample_slope_fp,
            "extra_tiffs": [sample_extra_fp],
        },
        "merge_friction_and_barriers": {
            "friction_layer": "friction",
            "barrier_layer": "fi_1",
            "barrier_multiplier": 100,
        },
    }

    build_routing_layer_file(**config, max_workers=mw)

    assert test_fp.exists()
    assert out_tiff_dir.exists()
    assert (out_tiff_dir / "fi_1.tif").exists()
    assert (out_tiff_dir / "friction.tif").exists()
    assert (out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif").exists()

    expected_datasets = [
        "sample_nlcd",
        "sample_iso",
        "sample_slope",
        "sample_extra_data",
        "friction",
        "dry_multipliers",
        "tie_line_costs_102MW",
        "tie_line_costs_205MW",
        "tie_line_costs_400MW",
        "tie_line_costs_1500MW",
        "tie_line_costs_3000MW",
    ]
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_datasets:
            assert ds_name in ds

        assert "fi_1" not in ds
        assert "friction_1" not in ds
        assert np.allclose(
            ds["friction"],
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask,
        )

    with rioxarray.open_rasterio(
        out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif", chunks="auto"
    ) as ds:
        assert np.allclose(
            ds,
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask
            + layers["fi_1.tif"] * masks_for_testing.wet_plus_mask * 100,
        )


def test_build_dry_only(
    tmp_path,
    sample_iso_fp,
    sample_nlcd_fp,
    sample_slope_fp,
    sample_extra_fp,
    masks_for_testing,
):
    """Test building only dry costs"""
    test_fp = tmp_path / "test.zarr"
    out_tiff_dir = tmp_path / "out_tiffs"

    assert not test_fp.exists()
    assert not out_tiff_dir.exists()

    config = {
        "routing_file": str(test_fp),
        "template_file": str(sample_extra_fp),
        "output_tiff_dir": str(out_tiff_dir),
        "masks_dir": str(masks_for_testing._masks_dir),
        "dry_costs": {
            "iso_region_tiff": str(sample_iso_fp),
            "nlcd_tiff": str(sample_nlcd_fp),
            "slope_tiff": str(sample_slope_fp),
            "extra_tiffs": [str(sample_extra_fp)],
        },
    }

    with pytest.warns(revrtWarning, match="Dry mask not found"):
        build_routing_layer_file(**config)

    assert test_fp.exists()
    assert out_tiff_dir.exists()
    assert not (out_tiff_dir / "fi_1.tif").exists()
    assert not (out_tiff_dir / "friction.tif").exists()
    assert not (out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif").exists()

    expected_datasets = [
        "sample_nlcd",
        "sample_iso",
        "sample_slope",
        "sample_extra_data",
        "dry_multipliers",
        "tie_line_costs_102MW",
        "tie_line_costs_205MW",
        "tie_line_costs_400MW",
        "tie_line_costs_1500MW",
        "tie_line_costs_3000MW",
    ]
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_datasets:
            assert ds_name in ds

        assert "fi_1" not in ds
        assert "friction_1" not in ds
        assert "friction" not in ds


def test_build_routing_layer_file_ignores_dask_close_timeout(
    monkeypatch, caplog, tmp_path
):
    """build_routing_layer_file should not fail if closing client times out"""

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.status = "created"
            self._Client__loop = object()

        def close(self, timeout):
            msg = f"timed out after {timeout} seconds"
            raise TimeoutError(msg)

    monkeypatch.setattr("revrt.costs.cli._validated_config", SimpleNamespace)
    monkeypatch.setattr(
        "revrt.costs.cli._build_routing_layer_file", lambda *__, **___: None
    )
    monkeypatch.setattr("revrt.costs.cli.dask.distributed.Client", FakeClient)
    monkeypatch.setattr(
        "revrt.costs.cli.dask.distributed.Lock", lambda name: name
    )

    with caplog.at_level(logging.WARNING, logger="revrt.utilities.monitoring"):
        build_routing_layer_file(
            routing_file=tmp_path / "routing.zarr",
            template_file=tmp_path / "template.tif",
            layers=[{"layer_name": "friction", "build": {}}],
            max_workers=2,
        )

    assert "Timed out closing Dask client" in caplog.text


def test_build_layers_only(
    tmp_path, sample_extra_fp, tiff_layers_for_testing, masks_for_testing
):
    """Test building only layers"""
    test_fp = tmp_path / "test.zarr"
    out_tiff_dir = tmp_path / "out_tiffs"
    layer_dir, layers = tiff_layers_for_testing

    assert not test_fp.exists()
    assert not out_tiff_dir.exists()

    config = {
        "routing_file": str(test_fp),
        "template_file": str(sample_extra_fp),
        "input_layer_dir": str(layer_dir),
        "output_tiff_dir": str(out_tiff_dir),
        "masks_dir": str(masks_for_testing._masks_dir),
        "layers": [
            {
                "layer_name": "fi_1",
                "include_in_file": False,
                "build": {
                    "fi_1.tif": {"extent": "wet+", "pass_through": True}
                },
            },
            {
                "layer_name": "friction",
                "build": {
                    "friction_1.tif": {
                        "extent": "dry+",
                        "map": {x: x for x in range(20)},
                    }
                },
            },
        ],
    }

    build_routing_layer_file(**config)

    assert test_fp.exists()
    assert out_tiff_dir.exists()
    assert (out_tiff_dir / "fi_1.tif").exists()
    assert (out_tiff_dir / "friction.tif").exists()
    assert not (out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif").exists()

    expected_missing_datasets = [
        "sample_nlcd",
        "sample_iso",
        "sample_slope",
        "sample_extra_data",
        "dry_multipliers",
        "tie_line_costs_102MW",
        "tie_line_costs_205MW",
        "tie_line_costs_400MW",
        "tie_line_costs_1500MW",
        "tie_line_costs_3000MW",
    ]
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_missing_datasets:
            assert ds_name not in ds

        assert "friction" in ds
        assert "fi_1" not in ds
        assert "friction_1" not in ds
        assert np.allclose(
            ds["friction"],
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask,
        )

    # Test adding one more layer
    config = {
        "routing_file": str(test_fp),
        "template_file": str(sample_extra_fp),
        "input_layer_dir": str(layer_dir),
        "output_tiff_dir": str(out_tiff_dir),
        "masks_dir": str(masks_for_testing._masks_dir),
        "layers": [
            {
                "layer_name": "fi_1",
                "include_in_file": False,
                "build": {
                    "fi_1.tif": {"extent": "wet+", "pass_through": True}
                },
            },
            {
                "layer_name": "friction",
                "build": {
                    "friction_1.tif": {
                        "extent": "dry+",
                        "map": {x: x for x in range(20)},
                    }
                },
            },
            {
                "layer_name": "fi_1",
                "include_in_file": True,
                "build": {"fi_1.tif": {"pass_through": True}},
            },
        ],
    }

    build_routing_layer_file(**config)

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_missing_datasets:
            assert ds_name not in ds

        assert "friction" in ds
        assert "fi_1" in ds
        assert "friction_1" not in ds
        assert np.allclose(
            ds["friction"],
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask,
        )
        assert np.allclose(ds["fi_1"], layers["fi_1.tif"])


def test_build_layers_rebuilds_when_build_config_changes(
    tmp_path, sample_extra_fp, tiff_layers_for_testing, masks_for_testing
):
    """Changed build config should rebuild an existing layer"""
    test_fp = tmp_path / "test.zarr"
    input_layer_dir = tmp_path / "layers"
    out_tiff_dir = tmp_path / "out_tiffs"
    source_layer_dir, layers = tiff_layers_for_testing
    _copy_test_layers(source_layer_dir, input_layer_dir)

    base_config = {
        "routing_file": str(test_fp),
        "template_file": str(sample_extra_fp),
        "input_layer_dir": str(input_layer_dir),
        "output_tiff_dir": str(out_tiff_dir),
        "masks_dir": str(masks_for_testing._masks_dir),
        "layers": [
            {
                "layer_name": "friction",
                "build": {
                    "friction_1.tif": {
                        "extent": "dry+",
                        "map": {x: x for x in range(20)},
                    }
                },
            }
        ],
    }

    build_routing_layer_file(**base_config)

    updated_layer = np.full_like(layers["friction_1.tif"], 99)
    _overwrite_test_tiff(input_layer_dir / "friction_1.tif", updated_layer)

    build_routing_layer_file(**base_config)

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        assert np.allclose(
            ds["friction"],
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask,
        )
        assert LayerCreator.BUILD_CONFIG_ATTR in ds["friction"].attrs
        assert LayerCreator.CPM_CONFIG_ATTR in ds["friction"].attrs

    rebuild_config = {
        **base_config,
        "layers": [
            {
                "layer_name": "friction",
                "build": {
                    "friction_1.tif": {
                        "extent": "all",
                        "pass_through": True,
                    }
                },
            }
        ],
    }

    build_routing_layer_file(**rebuild_config)

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        assert np.allclose(ds["friction"], updated_layer)
        assert LayerCreator.BUILD_CONFIG_ATTR in ds["friction"].attrs
        assert LayerCreator.CPM_CONFIG_ATTR in ds["friction"].attrs


def test_build_layers_rebuilds_without_stored_build_config(
    tmp_path, sample_extra_fp, tiff_layers_for_testing, masks_for_testing
):
    """Existing layers without build metadata should be rebuilt"""
    test_fp = tmp_path / "test.zarr"
    input_layer_dir = tmp_path / "layers"
    out_tiff_dir = tmp_path / "out_tiffs"
    source_layer_dir, layers = tiff_layers_for_testing
    _copy_test_layers(source_layer_dir, input_layer_dir)

    config = {
        "routing_file": str(test_fp),
        "template_file": str(sample_extra_fp),
        "input_layer_dir": str(input_layer_dir),
        "output_tiff_dir": str(out_tiff_dir),
        "masks_dir": str(masks_for_testing._masks_dir),
        "layers": [
            {
                "layer_name": "friction",
                "build": {
                    "friction_1.tif": {
                        "extent": "dry+",
                        "map": {x: x for x in range(20)},
                    }
                },
            }
        ],
    }

    lf_handler = LayeredFile(test_fp)
    lf_handler.create_new(sample_extra_fp)
    LayerCreator(
        lf_handler,
        masks_for_testing,
        input_layer_dir=input_layer_dir,
        output_tiff_dir=out_tiff_dir,
    ).build(
        "friction",
        {
            "friction_1.tif": LayerBuildConfig(
                extent="dry+", map={x: x for x in range(20)}
            )
        },
        write_to_file=False,
    )
    legacy_data = load_data_using_layer_file_profile(
        layer_fp=lf_handler.fp,
        geotiff=out_tiff_dir / "friction.tif",
        band_index=0,
    )
    lf_handler.write_layer(legacy_data, "friction", overwrite=True)

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        assert ds["friction"].attrs.get(LayerCreator.BUILD_CONFIG_ATTR) is None
        assert ds["friction"].attrs.get(LayerCreator.CPM_CONFIG_ATTR) is None

    updated_layer = np.full_like(layers["friction_1.tif"], 7)
    _overwrite_test_tiff(input_layer_dir / "friction_1.tif", updated_layer)

    build_routing_layer_file(**config)

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        assert np.allclose(
            ds["friction"], updated_layer * masks_for_testing.dry_plus_mask
        )
        assert LayerCreator.BUILD_CONFIG_ATTR in ds["friction"].attrs
        assert LayerCreator.CPM_CONFIG_ATTR in ds["friction"].attrs


def test_build_layers_rebuilds_when_costs_per_mile_flag_changes(
    tmp_path,
    sample_extra_fp,
    sample_tiff_props,
    tiff_layers_for_testing,
    masks_for_testing,
):
    """Changing values_are_costs_per_mile should rebuild a layer"""
    test_fp = tmp_path / "test.zarr"
    input_layer_dir = tmp_path / "layers"
    out_tiff_dir = tmp_path / "out_tiffs"
    source_layer_dir, layers = tiff_layers_for_testing
    _copy_test_layers(source_layer_dir, input_layer_dir)

    updated_layer = np.full_like(layers["friction_1.tif"], 10)
    _overwrite_test_tiff(input_layer_dir / "friction_1.tif", updated_layer)

    base_config = {
        "routing_file": str(test_fp),
        "template_file": str(sample_extra_fp),
        "input_layer_dir": str(input_layer_dir),
        "output_tiff_dir": str(out_tiff_dir),
        "masks_dir": str(masks_for_testing._masks_dir),
        "layers": [
            {
                "layer_name": "friction",
                "values_are_costs_per_mile": False,
                "build": {
                    "friction_1.tif": {
                        "extent": "all",
                        "pass_through": True,
                    }
                },
            }
        ],
    }

    build_routing_layer_file(**base_config)

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        assert np.allclose(ds["friction"], updated_layer)

    rebuild_config = {
        **base_config,
        "layers": [
            {
                "layer_name": "friction",
                "values_are_costs_per_mile": True,
                "build": {
                    "friction_1.tif": {
                        "extent": "all",
                        "pass_through": True,
                    }
                },
            }
        ],
    }

    build_routing_layer_file(**rebuild_config)

    cell_size = sample_tiff_props[4]
    expected = updated_layer / METERS_IN_MILE * cell_size
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        assert np.allclose(ds["friction"], expected)
        assert LayerCreator.BUILD_CONFIG_ATTR in ds["friction"].attrs
        assert LayerCreator.CPM_CONFIG_ATTR in ds["friction"].attrs


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_build_basic_from_cli(
    run_gaps_cli_with_expected_file,
    tmp_path,
    sample_iso_fp,
    sample_nlcd_fp,
    sample_slope_fp,
    sample_extra_fp,
    tiff_layers_for_testing,
    masks_for_testing,
):
    """Test basic building from command line"""
    test_fp = tmp_path / "test.zarr"
    out_tiff_dir = tmp_path / "out_tiffs"
    layer_dir, layers = tiff_layers_for_testing

    assert not out_tiff_dir.exists()

    config = {
        "execution_control": {"max_workers": 1},
        "routing_file": str(test_fp),
        "template_file": str(sample_extra_fp),
        "input_layer_dir": str(layer_dir),
        "output_tiff_dir": str(out_tiff_dir),
        "masks_dir": str(masks_for_testing._masks_dir),
        "layers": [
            {
                "layer_name": "fi_1",
                "include_in_file": False,
                "build": {
                    "fi_1.tif": {"extent": "wet+", "pass_through": True}
                },
            },
            {
                "layer_name": "friction",
                "build": {
                    "friction_1.tif": {
                        "extent": "dry+",
                        "map": {x: x for x in range(20)},
                    }
                },
            },
        ],
        "dry_costs": {
            "iso_region_tiff": str(sample_iso_fp),
            "nlcd_tiff": str(sample_nlcd_fp),
            "slope_tiff": str(sample_slope_fp),
            "extra_tiffs": [str(sample_extra_fp)],
        },
        "merge_friction_and_barriers": {
            "friction_layer": "friction",
            "barrier_layer": "fi_1",
            "barrier_multiplier": 100,
        },
    }

    test_fp = run_gaps_cli_with_expected_file(
        "build-routing-layer-file", config, tmp_path, glob_pattern="test.zarr"
    )

    assert out_tiff_dir.exists()
    assert (out_tiff_dir / "fi_1.tif").exists()
    assert (out_tiff_dir / "friction.tif").exists()
    assert (out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif").exists()

    expected_datasets = [
        "sample_nlcd",
        "sample_iso",
        "sample_slope",
        "sample_extra_data",
        "friction",
        "dry_multipliers",
        "tie_line_costs_102MW",
        "tie_line_costs_205MW",
        "tie_line_costs_400MW",
        "tie_line_costs_1500MW",
        "tie_line_costs_3000MW",
    ]
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_datasets:
            assert ds_name in ds

        assert "fi_1" not in ds
        assert "friction_1" not in ds
        assert np.allclose(
            ds["friction"],
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask,
        )

    with rioxarray.open_rasterio(
        out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif", chunks="auto"
    ) as ds:
        assert np.allclose(
            ds,
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask
            + layers["fi_1.tif"] * masks_for_testing.wet_plus_mask * 100,
        )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
