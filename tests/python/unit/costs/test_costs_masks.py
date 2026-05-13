"""Test masks for cost layer creation"""

from pathlib import Path

import pytest
import numpy as np
import geopandas as gpd
import rioxarray
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio.enums import Resampling
from rasterio.transform import Affine

from revrt.costs.masks import Masks
from revrt.exceptions import revrtAttributeError
from revrt.utilities import LayeredFile, load_data_using_layer_file_profile


def test_no_masks():
    """Test error when no masks"""
    masks = Masks((3, 3), "EPSG:4326", None, ".")
    with pytest.raises(revrtAttributeError, match="No mask available"):
        _ = masks.dry_mask

    with pytest.raises(revrtAttributeError, match="No mask available"):
        _ = masks.wet_mask

    with pytest.raises(revrtAttributeError, match="No mask available"):
        _ = masks.dry_plus_mask

    with pytest.raises(revrtAttributeError, match="No mask available"):
        _ = masks.wet_plus_mask

    with pytest.raises(revrtAttributeError, match="No mask available"):
        _ = masks.landfall_mask


def test_basic_shapefile_to_masks(tmp_path):
    """Test basic shapefile to masks"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    masks = Masks(
        shape=(5, 6),
        crs="ESRI:102008",
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        masks_dir=tmp_path,
    )

    masks.create(land_mask_fp, save_tiff=False, reproject_vector=False)
    assert len(list(tmp_path.glob("*.tif"))) == 0

    assert np.allclose(
        masks.landfall_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 1, 1, 1, 0],
            ]
        ),
    )
    assert masks.landfall_mask.dtype == bool

    assert np.allclose(
        masks.wet_mask,
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
            ]
        ),
    )
    assert masks.wet_mask.dtype == bool

    assert np.allclose(
        masks.dry_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    )
    assert masks.dry_mask.dtype == bool

    assert np.allclose(
        masks.dry_plus_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
            ]
        ),
    )
    assert masks.dry_plus_mask.dtype == bool

    assert np.allclose(
        masks.wet_plus_mask,
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
    )
    assert masks.wet_plus_mask.dtype == bool


def test_loading_basic_masks(tmp_path):
    """Test basic loading of masks"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    layer_file_fp = tmp_path / "test_masks_layer_file.zarr"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    masks = Masks(
        shape=(5, 6),
        crs="ESRI:102008",
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        masks_dir=tmp_path,
    )

    masks.create(land_mask_fp, save_tiff=True, reproject_vector=False)
    assert len(list(tmp_path.glob("*.tif"))) == 4
    for fn in [
        Masks.LANDFALL_MASK_FNAME,
        Masks.RAW_LAND_MASK_FNAME,
        Masks.LAND_MASK_FNAME,
        Masks.OFFSHORE_MASK_FNAME,
    ]:
        assert (tmp_path / fn).exists()

    lf = LayeredFile(layer_file_fp)
    lf.create_new(tmp_path / Masks.LANDFALL_MASK_FNAME)

    new_masks = Masks(
        shape=(5, 6),
        crs="ESRI:102008",
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        masks_dir=tmp_path,
    )
    new_masks.load(layer_file_fp)

    assert np.allclose(
        new_masks.landfall_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 1, 1, 1, 0],
            ]
        ),
    )
    assert new_masks.landfall_mask.shape == (5, 6)
    assert new_masks.landfall_mask.dtype == bool

    assert np.allclose(
        new_masks.wet_mask,
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
            ]
        ),
    )
    assert new_masks.wet_mask.shape == (5, 6)
    assert new_masks.wet_mask.dtype == bool

    assert np.allclose(
        new_masks.dry_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    )
    assert new_masks.dry_mask.shape == (5, 6)
    assert new_masks.dry_mask.dtype == bool

    assert np.allclose(
        new_masks.dry_plus_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
            ]
        ),
    )
    assert new_masks.dry_plus_mask.shape == (5, 6)
    assert new_masks.dry_plus_mask.dtype == bool

    assert np.allclose(
        new_masks.wet_plus_mask,
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
    )
    assert new_masks.wet_plus_mask.shape == (5, 6)
    assert new_masks.wet_plus_mask.dtype == bool


def test_load_resets_cached_combined_masks(tmp_path):
    """Test load refreshes cached combined masks from existing files"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    layer_file_fp = tmp_path / "test_masks_layer_file.zarr"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    masks = Masks(
        shape=(5, 6),
        crs="ESRI:102008",
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        masks_dir=tmp_path,
    )

    masks._dry_mask = np.zeros((5, 6), dtype=bool)
    masks._wet_mask = np.zeros((5, 6), dtype=bool)
    masks._landfall_mask = np.zeros((5, 6), dtype=bool)

    assert not masks.dry_plus_mask.any()
    assert not masks.wet_plus_mask.any()

    masks.create(land_mask_fp, save_tiff=True, reproject_vector=False)

    lf = LayeredFile(layer_file_fp)
    lf.create_new(tmp_path / Masks.LANDFALL_MASK_FNAME)

    masks.load(layer_file_fp)

    assert np.allclose(
        masks.dry_plus_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
            ]
        ),
    )
    assert np.allclose(
        masks.wet_plus_mask,
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
    )


def test_loading_masks_with_different_crs(tmp_path):
    """Test loading masks when mask files use a different CRS"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    layer_file_fp = tmp_path / "test_masks_layer_file_3857.zarr"
    template_fp = tmp_path / "template_3857.tif"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="EPSG:4326",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    masks = Masks(
        shape=(5, 6),
        crs="EPSG:4326",
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        masks_dir=tmp_path,
    )
    masks.create(land_mask_fp, save_tiff=True, reproject_vector=False)

    with rioxarray.open_rasterio(tmp_path / Masks.LANDFALL_MASK_FNAME) as tif:
        tif.rio.reproject(
            dst_crs="EPSG:3857", resampling=Resampling.nearest
        ).rio.to_raster(template_fp, driver="GTiff")

    lf = LayeredFile(layer_file_fp)
    lf.create_new(template_fp)

    loaded_masks = Masks(
        shape=(5, 6),
        crs="EPSG:3857",
        transform=Affine.identity(),
        masks_dir=tmp_path,
    )
    loaded_masks.load(layer_file_fp)

    expected_dry = (
        load_data_using_layer_file_profile(
            layer_file_fp, tmp_path / Masks.LAND_MASK_FNAME, band_index=0
        )
        == 1
    )
    expected_wet = (
        load_data_using_layer_file_profile(
            layer_file_fp, tmp_path / Masks.OFFSHORE_MASK_FNAME, band_index=0
        )
        == 1
    )
    expected_landfall = (
        load_data_using_layer_file_profile(
            layer_file_fp, tmp_path / Masks.LANDFALL_MASK_FNAME, band_index=0
        )
        == 1
    )

    assert np.array_equal(loaded_masks.dry_mask, expected_dry)
    assert np.array_equal(loaded_masks.wet_mask, expected_wet)
    assert np.array_equal(loaded_masks.landfall_mask, expected_landfall)
    assert np.array_equal(
        loaded_masks.dry_plus_mask,
        np.logical_or(expected_dry, expected_landfall),
    )
    assert np.array_equal(
        loaded_masks.wet_plus_mask,
        np.logical_or(expected_wet, expected_landfall),
    )
    assert str(loaded_masks.dry_mask.rio.crs) == "EPSG:3857"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
