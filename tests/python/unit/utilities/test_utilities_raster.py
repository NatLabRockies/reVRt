"""Tests for base reVRt utilities"""

from pathlib import Path

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import box
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
