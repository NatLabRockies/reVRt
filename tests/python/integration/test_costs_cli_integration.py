"""reVRt routing creator CLI tests"""

import os
import shutil
import platform
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import dask.array as da


TEST_DEFAULT_MULTS = {
    "iso": "default",
    "land_use": {
        "cropland": 1,
        "forest": 4,
        "suburban": 5,
        "urban": 6,
        "wetland": 7,
    },
    "slope": {"hill_mult": 2, "hill_slope": 2, "mtn_mult": 5, "mtn_slope": 8},
}


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli(run_gaps_cli_with_expected_file, tmp_path, test_utility_data_dir):
    """Test routing layer builder CLI"""
    tb_tiff = test_utility_data_dir / "ri_transmission_barriers.tif"
    regions_tiff = test_utility_data_dir / "ri_regions.tif"
    nlcd_tiff = test_utility_data_dir / "ri_nlcd.tif"
    slope_tiff = test_utility_data_dir / "ri_srtm_slope.tif"

    temp_iso = tmp_path / "ISO_regions.tif"
    temp_tb = tmp_path / "transmission_barrier.tif"
    layer_mapping = {
        "iso_region_tiff": (regions_tiff, temp_iso),
        "transmission_barrier": (tb_tiff, temp_tb),
    }

    for src, dst in layer_mapping.values():
        shutil.copy(src, dst)

    test_fp = tmp_path / "test.zarr"
    config = {
        "execution_control": {"max_workers": 1},
        "routing_file": str(test_fp),
        "template_file": str(regions_tiff),
        "output_tiff_dir": str(tmp_path / "out_tiffs"),
        "dry_costs": {
            "iso_region_tiff": str(temp_iso),
            "nlcd_tiff": str(nlcd_tiff),
            "slope_tiff": str(slope_tiff),
            "default_mults": TEST_DEFAULT_MULTS,
            "extra_tiffs": [str(temp_tb)],
        },
    }

    run_gaps_cli_with_expected_file(
        "build-routing-layer-file", config, tmp_path, glob_pattern="*.zarr"
    )

    to_check_ds = [
        "tie_line_multipliers",
        "transmission_barrier",
        "tie_line_costs_102MW",
        "tie_line_costs_205MW",
        "tie_line_costs_400MW",
        "tie_line_costs_1500MW",
        "tie_line_costs_3000MW",
    ]
    baseline_fp = test_utility_data_dir / "transmission_layers.zarr"
    with (
        xr.open_dataset(
            baseline_fp, consolidated=False, engine="zarr"
        ) as ds_truth,
        xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds_test,
    ):
        for layer in to_check_ds:
            truth = ds_truth[layer]

            # layer renamed
            if layer == "tie_line_multipliers":
                layer = "dry_multipliers"  # noqa

            test = ds_test[layer]
            truth = da.where(np.isclose(truth, -1), 0, truth)

            assert np.allclose(truth, test)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
