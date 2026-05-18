"""Tests for CLI utility commands"""

import csv
import json
import os
import platform
import traceback
from pathlib import Path

import pytest
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, Polygon

from revrt._cli import main
from revrt.utilities import cli
from revrt.exceptions import revrtValueError
from revrt.utilities.handlers import LayeredFile
from revrt.warn import revrtWarning


def _cli_error_message(result):
    """Return CLI error message for assertion context"""

    if not result.exc_info:
        return ""
    return "".join(traceback.format_exception(*result.exc_info))


def _write_template_raster(path):
    """Helper function to write a template raster for tests"""
    transform = from_origin(0, 1, 1, 1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(np.ones((1, 1), dtype=np.uint8), 1)


def _write_template_zarr(path):
    """Helper function to write a template Zarr file for tests"""
    template_tif = Path(path).with_suffix(".tif")
    _write_template_raster(template_tif)
    layered = LayeredFile(path)
    layered.create_new(
        template_tif,
        overwrite=True,
        chunk_x=1,
        chunk_y=1,
    )
    height = layered.profile["height"]
    width = layered.profile["width"]
    values = np.zeros((1, height, width), dtype=np.uint8)
    layered.write_layer(values, "layer", overwrite=True)
    template_tif.unlink(missing_ok=True)


def test_layers_from_file_selects_specific_layers(tmp_path):
    """Ensure layers_from_file extracts chosen layers"""

    template = tmp_path / "template.tif"
    _write_template_raster(template)

    layered_path = tmp_path / "input.zarr"
    layered = LayeredFile(layered_path)
    layered.create_new(template, overwrite=True, chunk_x=1, chunk_y=1)

    values = {
        "alpha": np.ones((1, 1, 1), dtype=np.uint8),
        "beta": np.full((1, 1, 1), 2, dtype=np.uint8),
    }
    for layer, data in values.items():
        layered.write_layer(data, layer)

    layers_to_extract = list(values)
    result = cli.layers_from_file(
        layered_path,
        tmp_path,
        layers=layers_to_extract,
        profile_kwargs={"compress": "LZW"},
    )

    expected = [tmp_path / f"{layer}.tif" for layer in layers_to_extract]
    assert result == [str(path) for path in expected]

    for layer, expected_value in zip(layers_to_extract, (1, 2), strict=True):
        tif_path = tmp_path / f"{layer}.tif"
        assert tif_path.exists()
        with rasterio.open(tif_path) as dataset:
            assert dataset.read(1)[0, 0] == expected_value
            compress = dataset.profile.get("compress", "")
            assert compress.lower() == "lzw"


def test_layers_from_file_extracts_all_layers(tmp_path):
    """Ensure layers_from_file calls extract_all_layers when needed"""

    template = tmp_path / "template.tif"
    _write_template_raster(template)

    layered_path = tmp_path / "input.zarr"
    layered = LayeredFile(layered_path)
    layered.create_new(template, overwrite=True, chunk_x=1, chunk_y=1)

    values = {"one": 1, "two": 3}
    for layer, fill_value in values.items():
        data = np.full((1, 1, 1), fill_value, dtype=np.uint8)
        layered.write_layer(data, layer)

    result = cli.layers_from_file(layered_path, tmp_path)

    expected_paths = {layer: tmp_path / f"{layer}.tif" for layer in values}
    assert sorted(result) == sorted(
        str(path) for path in expected_paths.values()
    )

    for layer, fill_value in values.items():
        tif_path = expected_paths[layer]
        assert tif_path.exists()
        with rasterio.open(tif_path) as dataset:
            assert dataset.read(1)[0, 0] == fill_value


def test_preprocess_layers_from_file_config(tmp_path):
    """Ensure preprocess injects layer dir"""

    config = {"foo": 1}
    out_dir = tmp_path / "layers"

    processed = cli._preprocess_layers_from_file_config(
        config, tmp_path, out_dir
    )

    assert processed["_out_layer_dir"] == str(out_dir)


def test_convert_pois_to_lines_creates_fake_trans_line(tmp_path):
    """Validate POI conversion includes fake transmission entry"""

    template = tmp_path / "template.tif"
    _write_template_raster(template)

    poi_csv = tmp_path / "pois.csv"
    pd.DataFrame(
        {
            "POI Name": ["Test"],
            "State": ["CO"],
            "Voltage (kV)": [115],
            "Lat": [40.0],
            "Long": [-105.0],
        }
    ).to_csv(poi_csv, index=False)

    out_file = tmp_path / "out.gpkg"
    cli.convert_pois_to_lines(poi_csv, template, out_file)

    frame = gpd.read_file(out_file)
    assert set(frame["category"]) == {"Substation", "TransLine"}
    assert 9999 in frame["gid"].tolist()


def test_map_ss_to_rr_filters_low_voltage(tmp_path):
    """Ensure low voltage substations are dropped with warning"""

    features = gpd.GeoDataFrame(
        {
            "category": ["Substation", "Substation", "Line", "Line"],
            "gid": [1, 2, 10, 11],
            "trans_gids": [json.dumps([10]), json.dumps([11]), None, None],
            "voltage": [0, 0, 115, 34],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(5.0, 0.0),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(5.0, 0.0), (5.0, 1.0)]),
        ],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region_id": ["A", "B"]},
        geometry=[
            Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            Polygon([(4, -1), (4, 1), (6, 1), (6, -1)]),
        ],
        crs="EPSG:4326",
    )

    features_path = tmp_path / "features.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    features.to_file(features_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")

    out_file = tmp_path / "subs.gpkg"
    with pytest.warns(revrtWarning):
        cli.map_ss_to_rr(
            features_path,
            regions_path,
            "region_id",
            out_fpath=out_file,
        )

    frame = gpd.read_file(out_file)
    assert frame["trans_gid"].tolist() == [1]
    assert frame["region_id"].tolist() == ["A"]
    assert frame.loc[0, "min_volts"] == 115
    assert frame.loc[0, "max_volts"] == 115


def test_map_ss_to_rr_without_drops(tmp_path):
    """Ensure substations remain when voltage threshold is met"""

    features = gpd.GeoDataFrame(
        {
            "category": ["Substation", "Substation", "Line", "Line"],
            "gid": [1, 2, 10, 11],
            "trans_gids": [json.dumps([10]), json.dumps([11]), None, None],
            "voltage": [0, 0, 115, 230],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(5.0, 0.0),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(5.0, 0.0), (5.0, 1.0)]),
        ],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region_id": ["A", "B"]},
        geometry=[
            Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            Polygon([(4, -1), (4, 1), (6, 1), (6, -1)]),
        ],
        crs="EPSG:4326",
    )

    features_path = tmp_path / "features.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    out_file = tmp_path / "remaining_subs.gpkg"

    features.to_file(features_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")

    cli.map_ss_to_rr(
        features_path,
        regions_path,
        "region_id",
        out_fpath=out_file,
    )

    frame = gpd.read_file(out_file)
    assert frame["trans_gid"].tolist() == [1, 2]
    assert frame["region_id"].tolist() == ["A", "B"]


def test_ss_from_conn_csv(tmp_path):
    """Validate CSV extraction filters null records"""
    csv_path = tmp_path / "connections.csv"
    pd.DataFrame(
        {
            "poi_gid": [1, 1, np.nan],
            "poi_lat": [10.0, 10.0, 40.0],
            "poi_lon": [100.0, 100.0, 120.0],
            "region_id": ["A", "A", "B"],
        }
    ).to_csv(csv_path, index=False)

    out_file = tmp_path / "out.gpkg"
    cli.ss_from_conn(
        str(csv_path),
        str(out_file),
        "region_id",
        batch_size=2,
    )

    frame = gpd.read_file(out_file)
    assert len(frame) == 1
    assert pd.api.types.is_integer_dtype(frame["poi_gid"])


def test_ss_from_conn_gpkg(tmp_path):
    """Validate GeoPackage extraction matches CSV logic"""

    connections = gpd.GeoDataFrame(
        {
            "poi_gid": [1],
            "poi_lat": [10.0],
            "poi_lon": [100.0],
            "region_id": ["A"],
        },
        geometry=[Point(100.0, 10.0)],
        crs="EPSG:4326",
    )

    connections_path = tmp_path / "connections.gpkg"
    out_file = tmp_path / "out.gpkg"
    connections.to_file(connections_path, driver="GPKG")

    cli.ss_from_conn(
        str(connections_path), str(out_file), "region_id", batch_size=1
    )

    frame = gpd.read_file(out_file)
    assert len(frame) == 1


def test_ss_from_conn_batches_deduplicate(tmp_path):
    """Ensure batching preserves uniqueness across chunk boundaries"""

    csv_path = tmp_path / "connections.csv"
    pd.DataFrame(
        {
            "poi_gid": [1, 2, 1, 3],
            "poi_lat": [10.0, 20.0, 10.0, 30.0],
            "poi_lon": [100.0, 200.0, 100.0, 300.0],
        }
    ).to_csv(csv_path, index=False)

    out_file = tmp_path / "out.gpkg"
    cli.ss_from_conn(
        str(csv_path),
        str(out_file),
        region_identifier_column=None,
        batch_size=2,
    )

    frame = gpd.read_file(out_file)
    assert frame.shape[0] == 3


def test_ss_from_conn_invalid_extension():
    """Ensure invalid connection file extensions raise error"""

    with pytest.raises(revrtValueError, match="Unknown file ending"):
        cli.ss_from_conn("connections.txt", "out.gpkg", "region_id")


def test_ss_from_conn_invalid_batch_size(tmp_path):
    """Ensure invalid batch sizes raise validation errors"""

    csv_path = tmp_path / "connections.csv"
    pd.DataFrame(
        {
            "poi_gid": [1],
            "poi_lat": [10.0],
            "poi_lon": [100.0],
        }
    ).to_csv(csv_path, index=False)

    out_file = tmp_path / "out.gpkg"
    with pytest.raises(revrtValueError, match="batch_size"):
        cli.ss_from_conn(str(csv_path), str(out_file), batch_size=0)


def test_add_rr_to_nn_with_zarr_template(tmp_path):
    """Ensure add_rr_to_nn reads CRS from Zarr template"""

    network = gpd.GeoDataFrame(
        {"value": [1, 2]},
        geometry=[Point(0.0, 0.0), Point(10.0, 0.0)],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region": ["west", "east"]},
        geometry=[
            Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            Polygon([(9, -1), (9, 1), (11, 1), (11, -1)]),
        ],
        crs="EPSG:4326",
    )

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    template_path = tmp_path / "template.zarr"
    out_file = tmp_path / "out.gpkg"

    network.to_file(network_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")
    _write_template_zarr(template_path)

    cli.add_rr_to_nn(
        network_path,
        regions_path,
        "region",
        crs_template_file=template_path,
        out_fpath=out_file,
    )

    frame = gpd.read_file(out_file)
    assert frame["region"].tolist() == ["west", "east"]


def test_add_rr_to_nn_with_existing_region_column(tmp_path):
    """Validate add_rr_to_nn warns when region column exists"""

    network = gpd.GeoDataFrame(
        {"region": ["existing", "existing"]},
        geometry=[Point(0.0, 0.0), Point(1.0, 1.0)],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region": ["west"]},
        geometry=[Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])],
        crs="EPSG:4326",
    )

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    template_path = tmp_path / "template.zarr"
    out_file = tmp_path / "out.gpkg"

    network.to_file(network_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")
    _write_template_zarr(template_path)

    with pytest.warns(revrtWarning):
        cli.add_rr_to_nn(
            network_path,
            regions_path,
            "region",
            crs_template_file=template_path,
            out_fpath=out_file,
        )

    assert not out_file.exists()


def test_add_rr_to_nn_with_tif_template(tmp_path):
    """Ensure add_rr_to_nn reads CRS from GeoTIFF template"""

    network = gpd.GeoDataFrame(
        {"value": [1]},
        geometry=[Point(0.0, 0.0)],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region": ["west"]},
        geometry=[Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])],
        crs="EPSG:4326",
    )

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    template_path = tmp_path / "template.tif"
    out_file = tmp_path / "out.gpkg"

    network.to_file(network_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")
    _write_template_raster(template_path)

    cli.add_rr_to_nn(
        network_path,
        regions_path,
        "region",
        crs_template_file=template_path,
        out_fpath=out_file,
    )

    frame = gpd.read_file(out_file)
    assert frame["region"].tolist() == ["west"]


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_layers_to_file_command(
    run_gaps_cli_with_expected_file, tmp_path, sample_tiff_fp
):
    """Ensure layers-to-file CLI writes expected layers"""

    doubled_tiff = tmp_path / "sample_doubled.tif"
    with rioxarray.open_rasterio(sample_tiff_fp) as base:
        doubled = base.copy()
        doubled.values = base.values * 2
        doubled.rio.to_raster(doubled_tiff)

    out_file_fp = tmp_path / "cli_layers.zarr"
    config = {
        "fp": str(out_file_fp),
        "layers": {
            "sample": str(sample_tiff_fp),
            "double": str(doubled_tiff),
        },
    }

    run_gaps_cli_with_expected_file(
        "layers-to-file", config, tmp_path, glob_pattern="*.zarr"
    )

    with (
        xr.open_dataset(out_file_fp, consolidated=False, engine="zarr") as ds,
        rioxarray.open_rasterio(sample_tiff_fp) as sample_da,
        rioxarray.open_rasterio(doubled_tiff) as double_da,
    ):
        assert "sample" in ds
        assert "double" in ds
        assert ds["sample"].rio.crs == sample_da.rio.crs
        assert ds["double"].rio.crs == double_da.rio.crs
        assert ds["sample"].rio.transform() == sample_da.rio.transform()
        assert ds["double"].rio.transform() == double_da.rio.transform()
        assert np.allclose(ds["sample"].values, sample_da.values)
        assert np.allclose(ds["double"].values, double_da.values)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_layers_from_file_command(cli_runner, tmp_path, sample_tiff_fp):
    """Ensure layers-from-file CLI extracts expected GeoTIFFs"""

    doubled_tiff = tmp_path / "sample_doubled.tif"
    with rioxarray.open_rasterio(sample_tiff_fp) as base:
        doubled = base.copy()
        doubled.values = base.values * 2
        doubled.rio.to_raster(doubled_tiff)

    layered_fp = tmp_path / "source_layers.zarr"
    layered = LayeredFile(layered_fp)
    layered.create_new(sample_tiff_fp, overwrite=True)
    layered.write_geotiff_to_file(str(sample_tiff_fp), "sample")
    layered.write_geotiff_to_file(str(doubled_tiff), "double")

    out_dir = tmp_path / "layer_outputs"
    config = {
        "fp": str(layered_fp),
        "out_layer_dir": str(out_dir),
    }

    config_path = tmp_path / "config_layers_from_file.json"
    config_path.write_text(json.dumps(config))

    result = cli_runner.invoke(
        main, ["layers-from-file", "-c", str(config_path)]
    )
    msg = f"Failed with error {_cli_error_message(result)}"
    assert result.exit_code == 0, msg

    sample_tiff = out_dir / "sample.tif"
    double_tiff = out_dir / "double.tif"
    assert sample_tiff.exists()
    assert double_tiff.exists()

    with (
        rioxarray.open_rasterio(sample_tiff_fp) as sample_da,
        rioxarray.open_rasterio(doubled_tiff) as double_da_src,
        rioxarray.open_rasterio(sample_tiff) as sample_out,
        rioxarray.open_rasterio(double_tiff) as double_out,
    ):
        assert np.allclose(sample_out.values, sample_da.values)
        assert np.allclose(double_out.values, double_da_src.values)
        assert sample_out.rio.transform() == sample_da.rio.transform()
        assert double_out.rio.transform() == double_da_src.rio.transform()
        assert sample_out.rio.crs == sample_da.rio.crs
        assert double_out.rio.crs == double_da_src.rio.crs


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_convert_pois_to_lines_command(
    run_gaps_cli_with_expected_file, tmp_path, sample_tiff_fp
):
    """Ensure convert-pois-to-lines CLI creates GeoPackage"""

    poi_rows = [
        {
            "POI Name": "alpha",
            "State": "CO",
            "Voltage (kV)": 230,
            "Lat": 35.0,
            "Long": -110.0,
        },
        {
            "POI Name": "beta",
            "State": "NM",
            "Voltage (kV)": 345,
            "Lat": 36.0,
            "Long": -109.0,
        },
    ]

    poi_csv = tmp_path / "poi.csv"
    with poi_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "POI Name",
                "State",
                "Voltage (kV)",
                "Lat",
                "Long",
            ],
        )
        writer.writeheader()
        writer.writerows(poi_rows)

    out_gpkg = tmp_path / "pois.gpkg"
    config = {
        "poi_csv_f": str(poi_csv),
        "template_f": str(sample_tiff_fp),
        "out_f": str(out_gpkg),
    }

    run_gaps_cli_with_expected_file(
        "convert-pois-to-lines", config, tmp_path, glob_pattern="pois.gpkg"
    )

    pois = gpd.read_file(out_gpkg)
    assert pois.crs and pois.crs.to_string().upper() == "EPSG:4326"

    pois = pois.sort_values("gid").reset_index(drop=True)
    assert pois["POI Name"].tolist() == ["alpha", "beta", "fake"]
    assert pois["category"].tolist() == [
        "Substation",
        "Substation",
        "TransLine",
    ]
    assert pois["State"].iloc[:2].tolist() == ["CO", "NM"]
    assert pd.isna(pois.loc[2, "State"])

    expected_voltage_kv = [230, 345]
    assert [int(pois.loc[i, "Voltage (kV)"]) for i in range(2)] == (
        expected_voltage_kv
    )
    assert pd.isna(pois.loc[2, "Voltage (kV)"])

    assert list(pois["ac_cap"]) == [9999999] * 3
    assert list(pois["voltage"]) == [500, 500, 500]
    assert list(pois["trans_gids"].iloc[:2]) == ["[9999]"] * 2
    assert pd.isna(pois.loc[2, "trans_gids"])
    assert [int(gid) for gid in pois["gid"]] == [0, 1, 9999]

    expected_geometries = [
        LineString([(-110.0, 35.0), (-60.0, 85.0)]),
        LineString([(-109.0, 36.0), (-59.0, 86.0)]),
        LineString([(0.0, 0.0), (100000.0, 100000.0)]),
    ]
    for actual_geom, expected_geom in zip(
        pois.geometry.to_list(), expected_geometries, strict=True
    ):
        assert actual_geom.equals(expected_geom)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_convert_pois_to_lines_strips_required_path_whitespace(
    run_gaps_cli_with_expected_file, tmp_path, sample_tiff_fp
):
    """convert-pois-to-lines CLI strips whitespace on required paths"""

    poi_csv = tmp_path / "poi_whitespace.csv"
    pd.DataFrame(
        {
            "POI Name": ["alpha"],
            "State": ["CO"],
            "Voltage (kV)": [230],
            "Lat": [35.0],
            "Long": [-110.0],
        }
    ).to_csv(poi_csv, index=False)

    out_gpkg = tmp_path / "pois_trimmed.gpkg"
    config = {
        "poi_csv_f": f"  {poi_csv}  ",
        "template_f": f"  {sample_tiff_fp}  ",
        "out_f": f"  {out_gpkg}  ",
    }

    run_gaps_cli_with_expected_file(
        "convert-pois-to-lines",
        config,
        tmp_path,
        glob_pattern="pois_trimmed.gpkg",
    )

    assert out_gpkg.exists()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_map_ss_to_rr_command(run_gaps_cli_with_expected_file, tmp_path):
    """Ensure map-ss-to-rr CLI filters and maps regions"""

    features = gpd.GeoDataFrame(
        {
            "category": ["Substation", "Substation", "Line", "Line"],
            "gid": [1, 2, 10, 11],
            "trans_gids": [
                json.dumps([10]),
                json.dumps([11]),
                None,
                None,
            ],
            "voltage": [0, 0, 230, 34],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(5.0, 0.0),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(5.0, 0.0), (5.0, 1.0)]),
        ],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region_id": ["A", "B"]},
        geometry=[
            Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            Polygon([(4, -1), (4, 1), (6, 1), (6, -1)]),
        ],
        crs="EPSG:4326",
    )

    features_path = tmp_path / "features.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    features.to_file(features_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")

    config = {
        "features_fpath": str(features_path),
        "regions_fpath": str(regions_path),
        "region_identifier_column": "region_id",
        "minimum_substation_voltage_kv": 100,
    }

    out_path = run_gaps_cli_with_expected_file(
        "map-ss-to-rr", config, tmp_path
    )

    mapped = gpd.read_file(out_path)
    assert mapped["trans_gid"].tolist() == [1]
    assert mapped["region_id"].tolist() == ["A"]
    assert mapped.loc[0, "min_volts"] == 230
    assert mapped.loc[0, "max_volts"] == 230


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_map_ss_to_rr_strips_required_path_whitespace(
    run_gaps_cli_with_expected_file, tmp_path
):
    """map-ss-to-rr CLI strips whitespace on required path inputs"""

    features = gpd.GeoDataFrame(
        {
            "category": ["Substation", "Line"],
            "gid": [1, 10],
            "trans_gids": [json.dumps([10]), None],
            "voltage": [0, 230],
        },
        geometry=[Point(0.0, 0.0), LineString([(0.0, 0.0), (0.0, 1.0)])],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region_id": ["A"]},
        geometry=[Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])],
        crs="EPSG:4326",
    )

    features_path = tmp_path / "features_trimmed.gpkg"
    regions_path = tmp_path / "regions_trimmed.gpkg"
    features.to_file(features_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")

    config = {
        "features_fpath": f"  {features_path}  ",
        "regions_fpath": f"  {regions_path}  ",
        "region_identifier_column": "region_id",
    }

    out_path = run_gaps_cli_with_expected_file(
        "map-ss-to-rr", config, tmp_path
    )

    assert Path(out_path).exists()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_ss_from_conn_command(run_gaps_cli_with_expected_file, tmp_path):
    """Ensure ss-from-conn CLI extracts substations"""

    csv_path = tmp_path / "connections.csv"
    pd.DataFrame(
        {
            "poi_gid": [1, 1, np.nan],
            "poi_lat": [10.0, 10.0, 40.0],
            "poi_lon": [100.0, 100.0, 120.0],
            "region_id": ["A", "A", "B"],
        }
    ).to_csv(csv_path, index=False)

    config = {"connections_fpath": str(csv_path)}

    out_path = run_gaps_cli_with_expected_file(
        "ss-from-conn", config, tmp_path
    )

    subs = gpd.read_file(out_path)
    assert len(subs) == 1
    assert subs.loc[0, "poi_gid"] == 1


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_ss_from_conn_strips_required_path_whitespace(
    run_gaps_cli_with_expected_file, tmp_path
):
    """ss-from-conn CLI strips whitespace on required path inputs"""

    csv_path = tmp_path / "connections_trimmed.csv"
    pd.DataFrame(
        {
            "poi_gid": [1],
            "poi_lat": [10.0],
            "poi_lon": [100.0],
            "region_id": ["A"],
        }
    ).to_csv(csv_path, index=False)

    config = {"connections_fpath": f"  {csv_path}  "}

    out_path = run_gaps_cli_with_expected_file(
        "ss-from-conn", config, tmp_path
    )

    assert Path(out_path).exists()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_add_rr_to_nn_command(
    run_gaps_cli_with_expected_file, tmp_path, sample_tiff_fp
):
    """Ensure add-rr-to-nn CLI annotates network nodes"""

    network = gpd.GeoDataFrame(
        {"value": [1, 2]},
        geometry=[Point(0.0, 0.0), Point(10.0, 0.0)],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region": ["west", "east"]},
        geometry=[
            Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            Polygon([(9, -1), (9, 1), (11, 1), (11, -1)]),
        ],
        crs="EPSG:4326",
    )

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    network.to_file(network_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")

    config = {
        "network_nodes_fpath": str(network_path),
        "regions_fpath": str(regions_path),
        "region_identifier_column": "region",
        "crs_template_file": str(sample_tiff_fp),
    }

    out_path = run_gaps_cli_with_expected_file(
        "add-rr-to-nn", config, tmp_path
    )

    nodes = gpd.read_file(out_path).sort_values("value").reset_index(drop=True)
    assert nodes["region"].tolist() == ["west", "east"]


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_add_rr_to_nn_strips_required_path_whitespace(
    run_gaps_cli_with_expected_file, tmp_path, sample_tiff_fp
):
    """add-rr-to-nn CLI strips whitespace on required path inputs"""

    network = gpd.GeoDataFrame(
        {"value": [1]},
        geometry=[Point(0.0, 0.0)],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region": ["west"]},
        geometry=[Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])],
        crs="EPSG:4326",
    )

    network_path = tmp_path / "network_trimmed.gpkg"
    regions_path = tmp_path / "regions_trimmed.gpkg"
    network.to_file(network_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")

    config = {
        "network_nodes_fpath": f"  {network_path}  ",
        "regions_fpath": f"  {regions_path}  ",
        "region_identifier_column": "region",
        "crs_template_file": str(sample_tiff_fp),
    }

    out_path = run_gaps_cli_with_expected_file(
        "add-rr-to-nn",
        config,
        tmp_path,
    )

    assert Path(out_path).exists()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
