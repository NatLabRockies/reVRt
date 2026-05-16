"""Test revrt spatial characterization CLI"""

import os
import json
import logging
import platform
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from rasterio.transform import Affine
from shapely.geometry import box, LineString

from revrt.spatial_characterization.stats import (
    Stat,
    FractionalStat,
    _PCT_PREFIX,
)
from revrt.spatial_characterization.cli import (
    buffered_route_characterizations,
    _route_characterizations_from_config,
)
from revrt._cli import main


@pytest.fixture
def sample_raster():
    """Sample raster data for testing"""
    return xr.DataArray(
        np.array(
            [
                [1, 1, 5],
                [4, 5, 5],
                [9, 9, 9],
            ],
            dtype=np.float64,
        ),
        dims=("y", "x"),
        attrs={
            "transform": Affine(10.0, 0.0, -15, 0.0, -10.0, 15),
            "crs": "ESRI:102008",
        },
    )


def test_buffered_route_characterizations(tmp_path, sample_raster):
    """Test running stats through buffered characterizations function"""
    raster_fp = tmp_path / "test.tif"
    zones_fp = tmp_path / "test.gpkg"

    zones = gpd.GeoDataFrame(
        {"id": [1, 2], "A": ["a", "b"]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    out_stats = buffered_route_characterizations(
        raster_fp,
        zones_fp,
        row_widths={1: 200, 2: 8},
        row_width_key="id",
        stats="*",
    )

    assert len(out_stats) == len(zones)

    sub_arr = sample_raster.isel(x=2)
    assert np.allclose(
        out_stats[Stat.COUNT], [sample_raster.count(), sub_arr.count()]
    )
    assert np.allclose(
        out_stats[Stat.MIN], [sample_raster.min(), sub_arr.min()]
    )
    assert np.allclose(
        out_stats[Stat.MAX], [sample_raster.max(), sub_arr.max()]
    )
    assert np.allclose(
        out_stats[Stat.MEAN], [sample_raster.mean(), sub_arr.mean()]
    )
    assert np.allclose(
        out_stats[Stat.SUM], [sample_raster.sum(), sub_arr.sum()]
    )
    assert np.allclose(
        out_stats[Stat.STD], [sample_raster.std(), sub_arr.std()]
    )
    assert np.allclose(
        out_stats[Stat.MEDIAN], [sample_raster.median(), sub_arr.median()]
    )
    assert np.allclose(out_stats[Stat.MAJORITY], 5)
    assert np.allclose(out_stats[Stat.MINORITY], [4, 9])
    assert np.allclose(out_stats[Stat.UNIQUE], [4, 2])
    assert np.allclose(out_stats[Stat.RANGE], [8, 4])
    assert np.allclose(out_stats[Stat.NODATA], 0)
    assert np.allclose(
        out_stats[f"{Stat.PIXEL_COUNT}_1.0"], [2, np.nan], equal_nan=True
    )
    assert np.allclose(
        out_stats[f"{Stat.PIXEL_COUNT}_4.0"], [1, np.nan], equal_nan=True
    )
    assert np.allclose(out_stats[f"{Stat.PIXEL_COUNT}_5.0"], [3, 2])
    assert np.allclose(out_stats[f"{Stat.PIXEL_COUNT}_9.0"], [3, 1])

    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_1.0"],
        [2, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_4.0"],
        [1, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_5.0"],
        [3, 0.8 + 0.64],
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_9.0"],
        [3, 0.16],
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_1.0"],
        [200, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_4.0"],
        [100, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_5.0"], [300, 80 + 64]
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_9.0"], [300, 16]
    )
    assert np.allclose(
        out_stats[FractionalStat.VALUE_MULTIPLIED_BY_FRACTIONAL_AREA],
        [sample_raster.sum() * 100, 5 * (80 + 64) + 9 * 16],
    )


def test_buffered_route_characterizations_with_multiplier(
    tmp_path, sample_raster
):
    """Test running stats with a scalar multiplier"""
    raster_fp = tmp_path / "test.tif"
    zones_fp = tmp_path / "test.gpkg"

    zones = gpd.GeoDataFrame(
        {"id": [1, 2], "A": ["a", "b"]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    out_stats = buffered_route_characterizations(
        raster_fp,
        zones_fp,
        row_widths={1: 200, 2: 8},
        multiplier_scalar=3,
        row_width_key="id",
        stats="*",
    )

    assert len(out_stats) == len(zones)

    scaled_raster = sample_raster * 3
    sub_arr = scaled_raster.isel(x=2)
    assert np.allclose(
        out_stats[Stat.COUNT], [scaled_raster.count(), sub_arr.count()]
    )
    assert np.allclose(
        out_stats[Stat.MIN], [scaled_raster.min(), sub_arr.min()]
    )
    assert np.allclose(
        out_stats[Stat.MAX], [scaled_raster.max(), sub_arr.max()]
    )
    assert np.allclose(
        out_stats[Stat.MEAN], [scaled_raster.mean(), sub_arr.mean()]
    )
    assert np.allclose(
        out_stats[Stat.SUM], [scaled_raster.sum(), sub_arr.sum()]
    )
    assert np.allclose(
        out_stats[Stat.STD], [scaled_raster.std(), sub_arr.std()]
    )
    assert np.allclose(
        out_stats[Stat.MEDIAN], [scaled_raster.median(), sub_arr.median()]
    )
    assert np.allclose(out_stats[Stat.MAJORITY], 15)
    assert np.allclose(out_stats[Stat.MINORITY], [12, 27])
    assert np.allclose(out_stats[Stat.UNIQUE], [4, 2])
    assert np.allclose(out_stats[Stat.RANGE], [24, 12])
    assert np.allclose(out_stats[Stat.NODATA], 0)
    assert np.allclose(
        out_stats[f"{Stat.PIXEL_COUNT}_3.0"], [2, np.nan], equal_nan=True
    )
    assert np.allclose(
        out_stats[f"{Stat.PIXEL_COUNT}_12.0"], [1, np.nan], equal_nan=True
    )
    assert np.allclose(out_stats[f"{Stat.PIXEL_COUNT}_15.0"], [3, 2])
    assert np.allclose(out_stats[f"{Stat.PIXEL_COUNT}_27.0"], [3, 1])

    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_3.0"],
        [2, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_12.0"],
        [1, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_15.0"],
        [3, 0.8 + 0.64],
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_27.0"],
        [3, 0.16],
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_3.0"],
        [200, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_12.0"],
        [100, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_15.0"], [300, 80 + 64]
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_27.0"], [300, 16]
    )
    assert np.allclose(
        out_stats[FractionalStat.VALUE_MULTIPLIED_BY_FRACTIONAL_AREA],
        [scaled_raster.sum() * 100, 15 * (80 + 64) + 27 * 16],
    )


def test_buffered_route_characterizations_strips_required_path_whitespace(
    cli_runner, cli_error_message, tmp_cwd, sample_raster
):
    """Route characterization strips whitespace on required paths"""

    raster_fp = tmp_cwd / "test_whitespace.tif"
    zones_fp = tmp_cwd / "test_whitespace.gpkg"

    zones = gpd.GeoDataFrame(
        {"id": [1], "A": ["a"]},
        geometry=[box(-5, -5, 5, 5)],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    config = {
        "layers": {
            "geotiff_fp": f"  {raster_fp}  ",
            "route_fp": f"\n{zones_fp}\t",
            "row_width_key": "id",
            "stats": "*",
        },
        "row_widths": {"1": 200},
    }
    config_fp = tmp_cwd / "config_whitespace.json"
    config_fp.write_text(json.dumps(config))

    result = cli_runner.invoke(
        main, ["route-characterization", "-c", config_fp.as_posix()]
    )
    msg = f"Failed with error {cli_error_message(result)}"
    assert result.exit_code == 0, msg

    out_files = sorted(tmp_cwd.glob("*.csv"))
    assert len(out_files) == 1
    out_fp = out_files[0]
    assert out_fp.name == "characterized_test_whitespace_test_whitespace.csv"

    out_stats = pd.read_csv(out_fp)

    assert len(out_stats) == 1


def test_buffered_route_characterizations_percentile(tmp_path, sample_raster):
    """Test running percentile stats"""
    raster_fp = tmp_path / "test.tif"
    zones_fp = tmp_path / "test.gpkg"

    zones = gpd.GeoDataFrame(
        {"id": [1, 2], "A": [50, 42]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    out_stats = buffered_route_characterizations(
        raster_fp,
        zones_fp,
        row_widths={"50": 200, 42: 8},
        row_width_key="A",
        stats=[f"{_PCT_PREFIX}50", f"{_PCT_PREFIX}95"],
    )

    assert len(out_stats) == len(zones)

    sub_arr = sample_raster.isel(x=2)
    assert np.allclose(
        out_stats[f"{_PCT_PREFIX}50"],
        [np.percentile(sample_raster, 50), np.percentile(sub_arr, 50)],
    )
    assert np.allclose(
        out_stats[f"{_PCT_PREFIX}95"],
        [np.percentile(sample_raster, 95), np.percentile(sub_arr, 95)],
    )


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
@pytest.mark.parametrize("use_top_level_default", [True, False])
def test_cli_command_minimal(
    run_gaps_cli_with_expected_file,
    tmp_cwd,
    sample_raster,
    use_top_level_default,
):
    """Test running from config with minimal user inputs"""
    raster_fp = tmp_cwd / "test_raster.tif"
    zones_fp = tmp_cwd / "test_zones.gpkg"

    zones = gpd.GeoDataFrame(
        {"voltage": [1, 2], "A": ["a", "b"]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    if use_top_level_default:
        config = {
            "execution_control": {"option": "local"},
            "default_route_fp": str(zones_fp),
            "layers": {"geotiff_fp": str(raster_fp)},
            "row_widths": {"1": 200, "2": 8},
        }
    else:
        config = {
            "execution_control": {"option": "local"},
            "layers": {
                "geotiff_fp": str(raster_fp),
                "route_fp": str(zones_fp),
            },
            "row_widths": {"1": 200, "2": 8},
        }

    out_fp = run_gaps_cli_with_expected_file(
        "route-characterization", config, tmp_cwd, glob_pattern="*.csv"
    )

    assert out_fp.name == "characterized_test_raster_test_zones.csv"

    out_stats = pd.read_csv(out_fp)

    sub_arr = sample_raster.isel(x=2)
    assert np.allclose(
        out_stats[Stat.COUNT], [sample_raster.count(), sub_arr.count()]
    )
    assert np.allclose(
        out_stats[Stat.MIN], [sample_raster.min(), sub_arr.min()]
    )
    assert np.allclose(
        out_stats[Stat.MAX], [sample_raster.max(), sub_arr.max()]
    )
    assert np.allclose(
        out_stats[Stat.MEAN], [sample_raster.mean(), sub_arr.mean()]
    )
    assert np.allclose(out_stats["voltage"], [1, 2])
    assert out_stats["A"].to_list() == ["a", "b"]


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
@pytest.mark.parametrize("use_top_level_default", [True, False])
def test_cli_command_multiple_rasters(
    tmp_cwd, sample_raster, cli_runner, use_top_level_default
):
    """Test running from config with multiple raster inputs"""
    raster_fp = tmp_cwd / "raster.tif"
    zones_fp = tmp_cwd / "lcp.gpkg"

    zones = gpd.GeoDataFrame(
        {"voltage": [1, 2], "A": ["a", "b"]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    row_widths = {"1": 200, "2": 8}
    row_widths_fp = tmp_cwd / "row_widths.json"
    with row_widths_fp.open("w", encoding="utf-8") as f:
        json.dump(row_widths, f)

    if use_top_level_default:
        config = {
            "execution_control": {"option": "local"},
            "default_route_fp": str(zones_fp),
            "layers": [
                {
                    "geotiff_fp": str(raster_fp),
                    "stats": "count min",
                },
                {
                    "geotiff_fp": str(raster_fp),
                    "prefix": "test_",
                    "stats": "max mean",
                    "copy_properties": ["A"],
                },
            ],
            "row_widths": str(row_widths_fp),
        }
    else:
        config = {
            "execution_control": {"option": "local"},
            "layers": [
                {
                    "geotiff_fp": str(raster_fp),
                    "route_fp": str(zones_fp),
                    "stats": "count min",
                },
                {
                    "geotiff_fp": str(raster_fp),
                    "route_fp": str(zones_fp),
                    "prefix": "test_",
                    "stats": "max mean",
                    "copy_properties": ["A"],
                },
            ],
            "row_widths": str(row_widths_fp),
        }
    config_fp = tmp_cwd / "config.json"
    config_fp.write_text(json.dumps(config))

    assert not list(tmp_cwd.glob("*.csv"))
    cli_runner.invoke(
        main, ["route-characterization", "-c", config_fp.as_posix()]
    )

    out_files = sorted(tmp_cwd.glob("*.csv"))
    assert len(out_files) == 2

    out_fp = Path(out_files[0])
    assert out_fp.name == "characterized_raster_lcp_j0.csv"

    out_stats = pd.read_csv(out_fp)
    sub_arr = sample_raster.isel(x=2)

    assert np.allclose(
        out_stats[Stat.COUNT], [sample_raster.count(), sub_arr.count()]
    )
    assert np.allclose(
        out_stats[Stat.MIN], [sample_raster.min(), sub_arr.min()]
    )
    assert np.allclose(out_stats["voltage"], [1, 2])
    assert out_stats["A"].to_list() == ["a", "b"]
    assert not any(c in out_stats for c in [Stat.MAX, Stat.MEAN])

    out_fp = Path(out_files[1])
    assert out_fp.name == "characterized_raster_lcp_j1.csv"

    out_stats = pd.read_csv(out_fp)

    assert np.allclose(
        out_stats[f"test_{Stat.MAX}"], [sample_raster.max(), sub_arr.max()]
    )
    assert np.allclose(
        out_stats[f"test_{Stat.MEAN}"], [sample_raster.mean(), sub_arr.mean()]
    )
    assert out_stats["A"].to_list() == ["a", "b"]
    assert not any(
        c in out_stats
        for c in [
            Stat.COUNT,
            Stat.MIN,
            f"test_{Stat.COUNT}",
            f"test_{Stat.MIN}",
            "voltage",
        ]
    )


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_local_overrides_top_level(
    run_gaps_cli_with_expected_file, tmp_cwd, sample_raster
):
    """Test that local route_fp overrides top-level default_route_fp"""
    raster_fp = tmp_cwd / "test_raster.tif"
    zones_fp = tmp_cwd / "test_zones.gpkg"

    zones = gpd.GeoDataFrame(
        {"voltage": [1, 2], "A": ["a", "b"]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    config = {
        "execution_control": {"option": "local"},
        "default_route_fp": "./does_not_exist.gpkg",
        "layers": {
            "geotiff_fp": str(raster_fp),
            "route_fp": str(zones_fp),
        },
        "row_widths": {"1": 200, "2": 8},
    }

    out_fp = run_gaps_cli_with_expected_file(
        "route-characterization", config, tmp_cwd, glob_pattern="*.csv"
    )

    assert out_fp.name == "characterized_test_raster_test_zones.csv"

    out_stats = pd.read_csv(out_fp)

    sub_arr = sample_raster.isel(x=2)
    assert np.allclose(
        out_stats[Stat.COUNT], [sample_raster.count(), sub_arr.count()]
    )
    assert np.allclose(
        out_stats[Stat.MIN], [sample_raster.min(), sub_arr.min()]
    )
    assert np.allclose(
        out_stats[Stat.MAX], [sample_raster.max(), sub_arr.max()]
    )
    assert np.allclose(
        out_stats[Stat.MEAN], [sample_raster.mean(), sub_arr.mean()]
    )
    assert np.allclose(out_stats["voltage"], [1, 2])
    assert out_stats["A"].to_list() == ["a", "b"]


def test_route_characterizations_ignores_dask_close_timeout(
    monkeypatch, caplog, tmp_path
):
    """route characterization should not fail if Dask close times out"""

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.status = "created"
            self._Client__loop = object()

        def close(self, timeout):
            msg = f"timed out after {timeout} seconds"
            raise TimeoutError(msg)

    class FakeDataFrame:
        def to_csv(self, out_fp, index=False):
            assert index is False
            Path(out_fp).write_text("value\n1\n", encoding="utf-8")

    def fake_buffered_route_characterizations(**kwargs):
        assert kwargs["parallel"] is True
        return FakeDataFrame()

    monkeypatch.setattr(
        "revrt.spatial_characterization.cli.Client", FakeClient
    )
    monkeypatch.setattr(
        "revrt.spatial_characterization.cli.buffered_route_characterizations",
        fake_buffered_route_characterizations,
    )

    with caplog.at_level(logging.WARNING, logger="revrt.utilities.monitoring"):
        out_fp = _route_characterizations_from_config(
            _stat_kwargs={
                "geotiff_fp": tmp_path / "raster.tif",
                "route_fp": tmp_path / "routes.gpkg",
            },
            _row_widths={"1": 100},
            _row_width_ranges=None,
            out_dir=tmp_path,
            max_workers=2,
        )

    assert Path(out_fp).exists()
    assert "Timed out closing Dask client" in caplog.text


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
