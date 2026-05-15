"""Tests for base reVRt utilities"""

from pathlib import Path

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import GeometryCollection, LineString, box
from shapely.ops import unary_union
from rasterio.transform import Affine
from rasterio.windows import from_bounds

from revrt.utilities.raster import (
    integer_dimension_window,
    rasterize_shape_file,
    simplify_shapes,
    _simplify_tolerance,
)


@pytest.mark.parametrize("at", [True, False])
def test_basic_rasterize_shape_file(tmp_path, at):
    """Test basic shapefile rasterization"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    out = rasterize_shape_file(
        land_mask_fp,
        width=6,
        height=5,
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        buffer_dist=None,
        all_touched=at,
        dest_crs=None,
        burn_value=1,
        boundary_only=False,
        dtype="uint8",
    )

    assert out.shape == (5, 6)
    assert out.dtype == "uint8"
    assert out.max() == 1
    assert out.min() == 0
    assert out.sum() == 11 if at else 7


def test_basic_rasterize_shape_file_with_opts(tmp_path):
    """Test basic shapefile rasterization with other options"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    out = rasterize_shape_file(
        land_mask_fp,
        width=6,
        height=5,
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        buffer_dist=None,
        all_touched=False,
        burn_value=9,
        boundary_only=True,
        dtype="uint8",
    )

    assert np.allclose(
        out,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 9, 9, 0],
                [0, 0, 9, 9, 9, 0],
                [0, 0, 9, 0, 9, 0],
                [0, 0, 9, 9, 9, 0],
            ]
        ),
    )

    out = rasterize_shape_file(
        land_mask_fp,
        width=6,
        height=5,
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        buffer_dist=5,
        all_touched=False,
        burn_value=9,
        boundary_only=True,
        dtype="uint8",
    )

    assert np.allclose(
        out,
        np.array(
            [
                [0, 0, 9, 9, 9, 9],
                [0, 9, 9, 0, 0, 9],
                [0, 9, 0, 0, 0, 9],
                [0, 9, 0, 0, 0, 9],
                [0, 9, 0, 0, 0, 9],
            ]
        ),
    )

    # Make sure buffer in previous step doesn't affect re-runs
    out = rasterize_shape_file(
        land_mask_fp,
        width=6,
        height=5,
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        buffer_dist=None,
        all_touched=False,
        burn_value=9,
        boundary_only=True,
        dtype="uint8",
    )

    assert np.allclose(
        out,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 9, 9, 0],
                [0, 0, 9, 9, 9, 0],
                [0, 0, 9, 0, 9, 0],
                [0, 0, 9, 9, 9, 0],
            ]
        ),
    )


def test_rasterize_with_reproject(tmp_path):
    """Test basic shapefile rasterization with reprojecting"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    out = rasterize_shape_file(
        land_mask_fp,
        width=6,
        height=5,
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        buffer_dist=None,
        all_touched=False,
        dest_crs="EPSG:4326",
        burn_value=1,
        boundary_only=False,
        dtype="uint8",
    )

    assert out.shape == (5, 6)
    assert out.dtype == "uint8"
    assert out.max() == 0
    assert out.min() == 0
    assert out.sum() == 0


@pytest.mark.parametrize("tile_size", [None, 2, 2048])
def test_rasterize_shape_file_uses_tiles(tmp_path, tile_size):
    """Rasterization is stable when forced to use multiple tiles"""
    land_mask_fp = tmp_path / "test_tiled_shape_mask.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[box(0, 0, 12, 12)], crs="ESRI:102008"
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    out = rasterize_shape_file(
        land_mask_fp,
        width=4,
        height=4,
        transform=Affine(3.0, 0.0, 0.0, 0.0, -3.0, 12.0),
        tile_size=tile_size,
        buffer_dist=None,
        all_touched=False,
        dest_crs=None,
        burn_value=1,
        boundary_only=False,
        dtype="uint8",
    )

    assert np.array_equal(out, np.ones((4, 4), dtype="uint8"))


def test_rasterize_shape_file_uses_nan_fill_when_fill_is_none(tmp_path):
    """Rasterization uses NaN for untouched cells when fill is None"""
    land_mask_fp = tmp_path / "test_shape_mask_nan_fill.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[box(0, 0, 6, 6)], crs="ESRI:102008"
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    out = rasterize_shape_file(
        land_mask_fp,
        width=4,
        height=4,
        transform=Affine(3.0, 0.0, 0.0, 0.0, -3.0, 12.0),
        buffer_dist=None,
        all_touched=False,
        dest_crs=None,
        burn_value=1,
        boundary_only=False,
        dtype="float32",
        fill=None,
    )

    assert out.dtype == np.dtype("float32")
    assert np.array_equal(
        np.isnan(out),
        np.array(
            [
                [True, True, True, True],
                [True, True, True, True],
                [False, False, True, True],
                [False, False, True, True],
            ]
        ),
    )
    assert np.array_equal(
        np.nan_to_num(out, nan=0.0),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype="float32",
        ),
    )


def test_rasterize_shape_file_uses_explicit_fill_value(tmp_path):
    """Rasterization uses the provided fill value for untouched cells"""
    land_mask_fp = tmp_path / "test_shape_mask_fill_value.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[box(0, 0, 6, 6)], crs="ESRI:102008"
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    out = rasterize_shape_file(
        land_mask_fp,
        width=4,
        height=4,
        transform=Affine(3.0, 0.0, 0.0, 0.0, -3.0, 12.0),
        buffer_dist=None,
        all_touched=False,
        dest_crs=None,
        burn_value=1,
        boundary_only=False,
        dtype="int16",
        fill=7,
    )

    assert out.dtype == np.dtype("int16")
    assert np.array_equal(
        out,
        np.array(
            [
                [7, 7, 7, 7],
                [7, 7, 7, 7],
                [1, 1, 7, 7],
                [1, 1, 7, 7],
            ],
            dtype="int16",
        ),
    )


def test_simplify_tolerance_uses_half_cell_size():
    """Simplification tolerance is half of the largest raster cell size"""
    transform = Affine(4.0, 0.0, 0.0, 0.0, -6.0, 0.0)

    assert _simplify_tolerance(transform) == pytest.approx(3.0)


def test_simplify_shapes_returns_copy_and_drops_empty_geometries():
    """simplify_shapes returns a simplified copy without empty shapes"""
    gdf = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (1, 0.5), (2, 0)]),
            GeometryCollection(),
        ],
        crs="ESRI:102008",
    )

    simplified = simplify_shapes(gdf, Affine(2.0, 0.0, 0.0, 0.0, -2.0, 0.0))

    assert len(simplified) == 1
    assert simplified.geometry.iloc[0].equals(LineString([(0, 0), (2, 0)]))


def test_integer_dimension_window_rounds_offsets():
    """integer_dimension_window rounds offsets and dimensions"""

    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    bounds = (0.25, -3.75, 6.6, 1.4)

    window = integer_dimension_window(bounds, transform)

    assert window.col_off == 0
    assert window.row_off == -2
    assert window.width == 8
    assert window.height == 7


def test_integer_dimension_window_enforces_min_size():
    """integer_dimension_window enforces minimum positive size"""

    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    bounds = (5.0, -5.0, 5.0, -5.0)

    raw = from_bounds(*bounds, transform=transform)
    assert raw.width == 0
    assert raw.height == 0

    window = integer_dimension_window(bounds, transform)

    assert window.width == 2
    assert window.height == 2


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
