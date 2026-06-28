"""reVrt tests for routing utilities"""

from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio.transform import from_origin

from revrt.utilities import LayeredFile
from revrt.routing.base import RoutingScenario
from revrt.routing.processing import BatchRouteProcessor


@pytest.fixture(scope="module")
def sample_large_layered_data(tmp_path_factory):
    """Sample (large) layered data files to use across tests"""
    data_dir = tmp_path_factory.mktemp("routing_data")

    layered_fp = data_dir / "test_large_layered.zarr"
    layer_file = LayeredFile(layered_fp)

    height, width = (1000, 1000)
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    layer_1 = -1 * np.ones((1, height, width), dtype=np.float32)
    layer_1[0, 5, 0] = 1000
    layer_1[0, 5, 900] = 100

    for ind, routing_layer in enumerate(
        [
            layer_1,
        ],
        start=1,
    ):
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


@pytest.mark.parametrize("invalid_costs_block_routing", [True, False])
def test_soft_barrier_with_large_dataset(
    sample_large_layered_data, invalid_costs_block_routing, tmp_path
):
    """Test that soft barriers work as expected in point-to-many routing"""
    scenario = RoutingScenario(
        cost_fpath=sample_large_layered_data,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
            }
        },
        invalid_costs_block_routing=invalid_costs_block_routing,
    )

    out_gpkg = tmp_path / "routes.gpkg"

    route_computer = BatchRouteProcessor(
        routing_scenario=scenario,
        route_definitions=[([(5, 0, "default")], [(5, 900, "default")])],
    )
    route_computer.process(out_fp=out_gpkg, save_paths=True)

    if invalid_costs_block_routing:
        assert not out_gpkg.exists()
    else:
        output = gpd.read_file(out_gpkg)
        assert len(output) == 1
        route = output.iloc[0]
        assert route["cost"] == pytest.approx(550.0)
        assert route["length_km"] == pytest.approx(0.9)
        x, y = route["geometry"].xy
        assert np.allclose(x, [0.5, 900.5])
        assert np.allclose(y, 994.5)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
