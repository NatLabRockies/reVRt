"""reVRt routing CLI build costs tests"""

import os
import json
import platform
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import rasterio
from rasterio.transform import from_origin

from revrt._cli import main
from revrt.utilities import LayeredFile

from revrt.routing.cli.build_costs import (
    build_final_routing_layers_command,
    build_final_routing_layers,
)


@pytest.fixture(scope="module")
def sample_layered_data(tmp_path_factory):
    """Create layered routing data mimicking point_to_point tests"""

    data_dir = tmp_path_factory.mktemp("routing_cli_data")

    layered_fp = data_dir / "test_layered.zarr"
    layer_file = LayeredFile(layered_fp)

    height, width = (7, 8)
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    layer_values = [
        np.array(
            [
                [
                    [7, 7, 8, 0, 9, 9, 9, 0],
                    [8, 1, 2, 2, 9, 9, 9, 0],
                    [9, 1, 3, 3, 9, 1, 2, 3],
                    [9, 1, 2, 1, 9, 1, 9, 0],
                    [9, 9, 9, 1, 9, 1, 9, 0],
                    [9, 9, 9, 1, 1, 1, 9, 0],
                    [9, 9, 9, 9, 9, 9, 9, 0],
                ]
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [
                    [8, 7, 6, 5, 5, 6, 7, 9],
                    [7, 1, 1, 2, 3, 3, 2, 8],
                    [6, 2, 9, 6, 5, 2, 1, 7],
                    [7, 3, 8, 1, 2, 3, 2, 6],
                    [8, 4, 7, 2, 8, 4, 3, 5],
                    [9, 5, 6, 3, 4, 4, 3, 4],
                    [9, 6, 7, 4, 5, 5, 4, 3],
                ]
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [
                    [6, 6, 6, 6, 6, 7, 8, 9],
                    [5, 2, 2, 3, 4, 5, 6, 8],
                    [4, 3, 7, 7, 6, 4, 5, 7],
                    [5, 4, 6, 2, 3, 4, 4, 6],
                    [6, 5, 5, 3, 7, 5, 5, 5],
                    [7, 6, 6, 4, 5, 5, 4, 4],
                    [8, 7, 7, 5, 6, 5, 4, 3],
                ]
            ],
            dtype=np.float32,
        ),
    ]

    for ind, routing_layer in enumerate(layer_values, start=1):
        da = xr.DataArray(
            routing_layer,
            dims=("band", "y", "x"),
            coords={"y": y_coords, "x": x_coords},
        )
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.write_transform(transform)

        geotiff_fp = data_dir / f"layer_{ind}.tif"
        da.rio.to_raster(geotiff_fp, driver="GTiff")

        layer_file.write_geotiff_to_file(
            geotiff_fp, f"layer_{ind}", overwrite=True
        )

    return layered_fp


def test_build_final_routing_layers_command_writes_expected_layers(
    sample_layered_data, tmp_path
):
    """build_final_routing_layers_command should persist aggregated outputs"""

    config = {
        "cost_fpath": str(sample_layered_data),
        "cost_layers": [
            {"layer_name": "layer_1", "multiplier_scalar": 1.5},
            {"layer_name": "layer_2", "multiplier_scalar": 0.5},
        ],
        "cost_multiplier_scalar": 2.0,
        "ignore_invalid_costs": True,
    }

    config_fp = tmp_path / "lcp_config.json"
    config_fp.write_text(json.dumps(config))
    out_dir = tmp_path / "outputs"

    outputs = build_final_routing_layers_command.runner(
        lcp_config_fp=config_fp,
        out_dir=out_dir,
        polarity=None,
        voltage=None,
    )

    assert len(outputs) == 2
    cost_fp, final_fp = [Path(fp) for fp in outputs]
    assert cost_fp.exists()
    assert final_fp.exists()

    with xr.open_dataset(
        sample_layered_data, consolidated=False, engine="zarr"
    ) as ds:
        layer_one = ds["layer_1"].isel(band=0).astype(np.float32).load()
        layer_two = ds["layer_2"].isel(band=0).astype(np.float32).load()

    expected_vals = (layer_one * 1.5 + layer_two * 0.5) * 2.0
    expected_vals = expected_vals.to_numpy()

    with rasterio.open(cost_fp) as src:
        agg_costs = src.read(1)

    with rasterio.open(final_fp) as src:
        final_layer = src.read(1)

    assert agg_costs.shape == expected_vals.shape
    assert final_layer.shape == expected_vals.shape
    assert np.allclose(agg_costs, expected_vals)
    assert np.allclose(final_layer, expected_vals)


def test_build_final_routing_layers_command_applies_explicit_barriers(
    sample_layered_data, tmp_path
):
    """build-final-routing-layers applies explicit barriers"""

    config = {
        "cost_fpath": str(sample_layered_data),
        "cost_layers": [
            {"layer_name": "layer_2"},
        ],
        "barrier_layers": [
            {"layer_name": "layer_1", "barrier_values": "==0"},
        ],
        "ignore_invalid_costs": False,
    }

    config_fp = tmp_path / "barrier_lcp_config.json"
    config_fp.write_text(json.dumps(config))
    out_dir = tmp_path / "barrier_outputs"

    outputs = build_final_routing_layers_command.runner(
        lcp_config_fp=config_fp,
        out_dir=out_dir,
        polarity=None,
        voltage=None,
    )

    cost_fp, final_fp = [Path(fp) for fp in outputs]
    with xr.open_dataset(
        sample_layered_data, consolidated=False, engine="zarr"
    ) as ds:
        expected_costs = ds["layer_2"].isel(band=0).astype(np.float32).load()
        expected_final = expected_costs.to_numpy().copy()
        expected_final[ds["layer_1"].isel(band=0).to_numpy() == 0] = np.nan

    with rasterio.open(cost_fp) as src:
        agg_costs = src.read(1)

    with rasterio.open(final_fp) as src:
        final_layer = src.read(1)

    assert np.allclose(agg_costs, expected_costs)
    assert np.array_equal(final_layer, expected_final, equal_nan=True)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_build_final_routing_layers_command(
    cli_runner, sample_layered_data, tmp_path
):
    """CLI build-final-routing-layers command should produce routed rasters"""

    lcp_config = {
        "cost_fpath": str(sample_layered_data),
        "cost_layers": [
            {"layer_name": "layer_1", "multiplier_scalar": 1.5},
            {"layer_name": "layer_2", "multiplier_scalar": 0.5},
        ],
        "cost_multiplier_scalar": 2.0,
        "ignore_invalid_costs": True,
    }

    lcp_config_fp = tmp_path / "cli_lcp_config.json"
    lcp_config_fp.write_text(json.dumps(lcp_config))

    cli_config = {"lcp_config_fp": str(lcp_config_fp)}

    cli_config_fp = tmp_path / "cli_command_config.json"
    cli_config_fp.write_text(json.dumps(cli_config))

    result = cli_runner.invoke(
        main, ["build-final-routing-layers", "-c", str(cli_config_fp)]
    )
    assert result.exit_code == 0, result.output

    cost_fp = tmp_path / "agg_costs.tif"
    final_fp = tmp_path / "final_routing_layer.tif"
    assert cost_fp.exists()
    assert final_fp.exists()

    with xr.open_dataset(
        sample_layered_data, consolidated=False, engine="zarr"
    ) as ds:
        layer_one = ds["layer_1"].isel(band=0).astype(np.float32).load()
        layer_two = ds["layer_2"].isel(band=0).astype(np.float32).load()

    expected_vals = (layer_one * 1.5 + layer_two * 0.5) * 2.0

    with rasterio.open(cost_fp) as src:
        agg_costs = src.read(1)

    with rasterio.open(final_fp) as src:
        final_layer = src.read(1)

    assert np.allclose(agg_costs, expected_vals)
    assert np.allclose(final_layer, expected_vals)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_build_route_costs_strips_required_path_whitespace(
    cli_runner, sample_layered_data, tmp_path
):
    """build-final-routing-layers CLI strips whitespace on path inputs"""

    lcp_config = {
        "cost_fpath": str(sample_layered_data),
        "cost_layers": [{"layer_name": "layer_1"}],
        "ignore_invalid_costs": True,
    }

    lcp_config_fp = tmp_path / "cli_trimmed_lcp_config.json"
    lcp_config_fp.write_text(json.dumps(lcp_config))

    cli_config = {"lcp_config_fp": f"  {lcp_config_fp}  "}

    cli_config_fp = tmp_path / "cli_trimmed_command_config.json"
    cli_config_fp.write_text(json.dumps(cli_config))

    result = cli_runner.invoke(
        main, ["build-final-routing-layers", "-c", str(cli_config_fp)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "agg_costs.tif").exists()
    assert (tmp_path / "final_routing_layer.tif").exists()


def test_build_final_routing_layers_command_metadata():
    """build_final_routing_layers_command should expose CLI settings"""

    assert (
        build_final_routing_layers_command.name == "build-final-routing-layers"
    )
    assert (
        build_final_routing_layers_command.runner is build_final_routing_layers
    )
    assert build_final_routing_layers_command.add_collect is False
    assert tuple(build_final_routing_layers_command.preprocessor_args) == (
        "config",
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
