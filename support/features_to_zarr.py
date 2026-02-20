"""Convert the standard HDF5 features to Zarr format

The transmission cost was originally pre-processed and saved in HDF5 format.
This script converts that HDF5 into a Zarr dataset and then verify if the
values are indeed all the same, including any possible NaN should match.
"""

from pathlib import Path

from dask.diagnostics import ProgressBar
from dask.distributed import LocalCluster
import netCDF4
import numpy as np
import xarray as xr
import zarr

cluster = LocalCluster(
    n_workers=2,
    memory_limit=0.15,
    processes=True,
    threads_per_worker=4,
    # security=True,
)
ProgressBar().register()

h5filename = Path(
    "/projects/rev/data/transmission/north_america/conus/fy25/nrel_build/build/costs/outputs/transmission_costs.h5"
)
output_dir = Path(
    "/projects/rev/data/transmission/north_america/conus/fy25/dev"
)
outfilename = output_dir / h5filename.name.replace(".h5", ".zarr")


ds = xr.open_mfdataset(
    h5filename, decode_cf=False, mask_and_scale=False, engine="netcdf4"
)
ds = ds.rename({"phony_dim_1": "y", "phony_dim_2": "x"})

ds = ds.set_coords(["latitude", "longitude"])

ds.attrs = {}

compressors = zarr.codecs.BloscCodec(
    cname="zstd", clevel=9, shuffle=zarr.codecs.BloscShuffle.shuffle
)
# from numcodecs.blosc import Blosc
# compressor = Blosc(cname="zstd", clevel=9, shuffle=2)
encoding = {}
for v in ds:
    if ds[v].dtype == "float32":
        ds[v] = ds[v].squeeze()
        """
        ds[v].attrs = {
            # "fill_value": netCDF4.default_fillvals["f4"],
        }
        """
        ds[v].encoding = {
            "fill_value": netCDF4.default_fillvals["f4"],
            "_FillValue": netCDF4.default_fillvals["f4"],
        }
        encoding[v] = {
            # "fill_value": netCDF4.default_fillvals["f4"],
            "compressors": compressors,
            # "dtype": "f4",
            # "_FillValue": netCDF4.default_fillvals["f4"],
        }
    else:
        print(ds[v].dtype)


# subset = ds.isel(x=range(3000), y=range(3000))
# subset[["fmv_dollar_per_acre"]].chunk({"x": 2_000, "y": 1_000}).to_zarr(
#     outfilename,
#     mode="w",
#     zarr_format=3,
#     # encoding=encoding["fmv_dollar_per_acre"],
#     )

ds.chunk({"x": 2_000, "y": 1_000}).to_zarr(
    outfilename,
    mode="w",
    zarr_format=3,
    # If activate encoding, the created zarr has fill_value with 0.0 instead of the default_fill_value
    # encoding=encoding,
    # write_empty_chunks=False,
)

# ==== Verify the created zarr file ====
zs = xr.open_zarr(outfilename)

for v in ds:
    if np.allclose(ds[v], zs[v], equal_nan=True):
        print(f"All equal: {v}")
    else:
        delta = zs[v] - ds[v]
        delta.compute()
        print(f"==== {v} ====")
        print(
            f"min: {delta.min().compute()}, median: {delta.median().compute()}, max: {delta.max().compute()}"
        )
