"""Build friction or barrier layers from raster and vector data"""

import logging
from pathlib import Path
from warnings import warn

import numpy as np
import dask.array as da

from revrt.costs.base import BaseLayerCreator
from revrt.utilities import (
    file_full_path,
    load_data_using_layer_file_profile,
    save_data_using_layer_file_profile,
    log_mem,
)
from revrt.utilities.raster import rasterize_shape_file
from revrt.constants import DEFAULT_DTYPE, ALL, METERS_IN_MILE
from revrt.exceptions import revrtAttributeError, revrtValueError
from revrt.warn import revrtWarning

logger = logging.getLogger(__name__)
TIFF_EXTENSIONS = {".tif", ".tiff"}
SHP_EXTENSIONS = {".shp", ".gpkg", ".geojson"}


class LayerCreator(BaseLayerCreator):
    """Build layer based on tiff and user config"""

    def __init__(
        self,
        io_handler,
        masks,
        input_layer_dir=".",
        output_tiff_dir=".",
        dtype=DEFAULT_DTYPE,
    ):
        """

        Parameters
        ----------
        io_handler : LayeredFile
            Layered file IO handler.
        masks : Masks
            Masks instance that can be used to retrieve multiple types
            of masks.
        input_layer_dir : path-like, optional
            Directory to search for input layers in, if not found in
            current directory. By default, ``'.'``.
        output_tiff_dir : path-like, optional
            Directory where cost layers should be saved as GeoTIFF.
            By default, ``"."``.
        dtype : np.dtype, optional
            Data type for final dataset. By default, ``float32``.
        """
        self._masks = masks
        super().__init__(
            io_handler=io_handler,
            input_layer_dir=input_layer_dir,
            output_tiff_dir=output_tiff_dir,
            dtype=dtype,
        )

    def build(
        self,
        layer_name,
        build_config,
        values_are_costs_per_mile=False,
        write_to_file=True,
        description=None,
        tiff_chunks="file",
        nodata=None,
        lock=None,
        **profile_kwargs,
    ):
        """Combine multiple GeoTIFFs and vectors to a raster layer

        Parameters
        ----------
        layer_name : str
            Name of layer to use in H5 and for output tiff.
        build_config : LayerBuildComponents
            Dict of LayerBuildConfig keyed by GeoTIFF/vector filenames.
        values_are_costs_per_mile : bool, default=False
            Option to convert values into costs per cell under the
            assumption that the resulting values are costs in $/mile.
            By default, ``False``, which writes raw values to TIFF/H5.
        write_to_file : bool, default=True
            Option to write the layer to file after creation.

            .. IMPORTANT::
                This will overwrite existing layers with the same name
                already in the file.

            By default, ``True``.
        description : str, optional
            Optional description to store with this layer in the H5
            file. By default, ``None``.
        tiff_chunks : int or str, default="file"
            Chunk size to use when reading the GeoTIFF file. This will
            be passed down as the ``chunks`` argument to
            :func:`rioxarray.open_rasterio`. By default, ``"file"``.
        nodata : int or float, optional
            Optional nodata value for output rasters. This value will
            be added to the layer's attributes meta dictionary under the
            "nodata" key.
        lock : bool or `dask.distributed.Lock`, optional
            Lock to use to write data to GeoTIFF using dask. If not
            supplied, a single process is used for writing data to disk.
            By default, ``None``.
        **profile_kwargs
            Additional keyword arguments to pass into writing output
            rasters. The following attributes ar ignored (they are set
            using properties of the
            :class:`~revrt.utilities.handlers.LayeredFile`):

                - nodata
                - transform
                - crs
                - count
                - width
                - height

        """
        tiff_filename = self._process_and_write_as_tiff(
            layer_name=layer_name,
            build_config=build_config,
            values_are_costs_per_mile=values_are_costs_per_mile,
            tiff_chunks=tiff_chunks,
            nodata=nodata,
            lock=lock,
            **profile_kwargs,
        )
        if write_to_file:
            out = load_data_using_layer_file_profile(
                layer_fp=self._io_handler.fp,
                geotiff=tiff_filename,
                tiff_chunks=tiff_chunks,
                layer_dirs=[self.input_layer_dir, self.output_tiff_dir],
                band_index=0,
            )
            log_mem()
            logger.debug("Writing %r to '%s'", layer_name, self._io_handler.fp)
            self._io_handler.write_layer(
                out, layer_name, description=description, overwrite=True
            )
            log_mem()

    def _process_and_write_as_tiff(
        self,
        layer_name,
        build_config,
        values_are_costs_per_mile=False,
        tiff_chunks="file",
        nodata=None,
        lock=None,
        **profile_kwargs,
    ):
        layer_name = layer_name.replace(".tif", "").replace(".tiff", "")
        logger.debug("Combining %s layers", layer_name)
        log_mem()
        result = da.zeros(self.shape, dtype=self._dtype, chunks=self.chunks)
        fi_layers = {}
        logger.debug("Initialized zeros")
        log_mem()

        for fname, config in build_config.items():
            if config.forced_inclusion:
                fi_layers[fname] = config
                continue

            logger.debug("Processing %s with config %s", fname, config)
            if Path(fname).suffix.lower() in TIFF_EXTENSIONS:
                temp = self._process_raster_layer(
                    fname, config, tiff_chunks=tiff_chunks
                )
                result += temp
            elif Path(fname).suffix.lower() in SHP_EXTENSIONS:
                temp = self._process_vector_layer(fname, config)
                result += temp
            else:
                msg = f"Unsupported file extension on {fname!r}"
                raise revrtValueError(msg)

            log_mem()

        result = self._process_forced_inclusions(
            result, fi_layers, tiff_chunks=tiff_chunks
        )
        logger.debug("After forced inclusions")
        log_mem()
        if values_are_costs_per_mile:
            result = result / METERS_IN_MILE * self.cell_size
            log_mem()

        result = result.astype(self._dtype)
        out_filename = self.output_tiff_dir / f"{layer_name}.tif"
        logger.debug(
            "Writing combined %s layers to %s", layer_name, out_filename
        )
        log_mem()
        save_data_using_layer_file_profile(
            layer_fp=self._io_handler.fp,
            data=result,
            geotiff=out_filename,
            nodata=nodata,
            lock=lock,
            **profile_kwargs,
        )
        return out_filename

    def _process_raster_layer(self, fname, config, tiff_chunks="file"):
        """Create the desired layer from the input file"""
        _check_tiff_layer_config(config, fname)
        data = load_data_using_layer_file_profile(
            layer_fp=self._io_handler.fp,
            geotiff=fname,
            tiff_chunks=tiff_chunks,
            layer_dirs=[self.input_layer_dir, self.output_tiff_dir],
            band_index=0,
        )
        return self._process_raster_data(data, config)

    def _process_raster_data(self, data, config):
        """Create the desired layer from the data array"""
        if config.global_value is not None:
            return self._process_global_raster_value(config)

        if config.bins is not None:
            return self._process_raster_bins(config, data)

        if config.pass_through:
            return self._pass_through_raster(config, data)

        return self._process_raster_map(config, data)

    def _process_global_raster_value(self, config):
        """Create the desired layer from the global value"""
        temp = da.full(
            self.shape,
            fill_value=config.global_value,
            dtype=self._dtype,
            chunks=self.chunks,
        )
        return self._apply_mask(config, temp)

    def _process_raster_bins(self, config, data):
        """Create the desired layer from the input file using bins"""
        _validate_bin_range(config.bins)
        _validate_bin_continuity(config.bins)

        processed = da.zeros(self.shape, dtype=self._dtype, chunks=self.chunks)
        if config.extent != ALL:
            mask = self._get_mask(config.extent)

        for i, interval in enumerate(config.bins):
            logger.debug(
                "Calculating layer values for bin %d/%d: %r",
                i + 1,
                len(config.bins),
                interval,
            )
            temp = da.where(
                np.logical_and(data >= interval.min, data < interval.max),
                interval.value,
                0,
            )

            if config.extent == ALL:
                processed += temp
                continue

            processed = da.where(mask, processed + temp, processed)

        return processed

    def _pass_through_raster(self, config, data):
        """Process raster by passing it through without modification"""
        return self._apply_mask(config, data)

    def _process_raster_map(self, config, data):
        """Create the desired layer from the input file using a map"""
        temp = da.zeros(self.shape, dtype=self._dtype, chunks=self.chunks)
        for key, val in config.map.items():
            temp = da.where(data == key, val, temp)

        return self._apply_mask(config, temp)

    def _process_vector_layer(self, fname, config):
        """Rasterize a vector layer"""
        if config.rasterize is None:
            msg = (
                f"{fname!r} is a vector but the config is missing "
                f'key "rasterize": {config}'
            )
            raise revrtValueError(msg)

        kwargs = {
            k: v for k, v in self._io_handler.profile.items() if k != "crs"
        }
        if config.rasterize.reproject:
            kwargs["dest_crs"] = self._io_handler.profile["crs"]

        fname = file_full_path(fname, self.input_layer_dir)
        temp = rasterize_shape_file(
            fname,
            buffer_dist=config.rasterize.buffer,
            burn_value=config.rasterize.value,
            all_touched=config.rasterize.all_touched,
            dtype=self._dtype,
            **kwargs,
        )

        return self._apply_mask(config, temp)

    def _apply_mask(self, config, data):
        """Apply the mask to the data based on the config extent"""
        if config.extent == ALL:
            return data

        mask = self._get_mask(config.extent)
        return da.where(mask, data, 0)

    def _process_forced_inclusions(self, data, fi_layers, tiff_chunks="file"):
        """Use forced inclusion (FI) layers to remove barriers/friction

        Any value > 0 in the FI layers will result in a 0 in the
        corresponding cell in the returned raster.
        """
        fi = da.zeros(self.shape, dtype=self._dtype, chunks=self.chunks)

        for fname, config in fi_layers.items():
            if Path(fname).suffix.lower() not in TIFF_EXTENSIONS:
                msg = (
                    f"Forced inclusion file {fname!r} does not end with .tif."
                    " GeoTIFFs are the only format allowed for forced "
                    "inclusions."
                )
                raise revrtValueError(msg)

            global_value_given = config.global_value is not None
            map_given = config.map is not None
            range_given = config.bins is not None
            rasterize_given = config.rasterize is not None
            bad_input_given = (
                global_value_given
                or map_given
                or range_given
                or rasterize_given
            )
            if bad_input_given:
                msg = (
                    "`global_value`, `map`, `bins`, and `rasterize` are "
                    "not allowed if `forced_inclusion` is True, but one "
                    f"was found in config: {fname!r}: {config}"
                )
                raise revrtValueError(msg)

            # Past guard clauses, process FI
            if config.extent != ALL:
                mask = self._get_mask(config.extent)

            temp = load_data_using_layer_file_profile(
                layer_fp=self._io_handler.fp,
                geotiff=fname,
                tiff_chunks=tiff_chunks,
                layer_dirs=[self.input_layer_dir, self.output_tiff_dir],
                band_index=0,
            )
            temp = _backend_array(temp)

            if config.extent == ALL:
                fi += temp
            else:
                fi = da.where(mask, fi + temp, fi)

        return da.where(fi > 0, 0, data)

    def _get_mask(self, extent):
        """Get mask by requested extent"""
        if extent == ALL:
            msg = f"Mask for extent of {extent!r} is unnecessary"
            raise revrtAttributeError(msg)

        if extent == "wet":
            mask = self._masks.wet_mask
        elif extent == "wet+":
            mask = self._masks.wet_plus_mask
        elif extent == "dry":
            mask = self._masks.dry_mask
        elif extent == "dry+":
            mask = self._masks.dry_plus_mask
        elif extent == "landfall":
            mask = self._masks.landfall_mask
        else:
            msg = f"Unknown mask type: {extent!r}"
            raise revrtAttributeError(msg)

        return mask


def _check_tiff_layer_config(config, fname):
    """Check if a LayerBuildConfig is valid for a GeoTIFF"""
    if config.rasterize is not None:
        msg = (
            f"'rasterize' is only for vectors. Found in {fname!r} config: "
            f"{config}"
        )
        raise revrtValueError(msg)

    mutex_entries = [config.map, config.bins, config.global_value]
    num_entries = sum(entry is not None for entry in mutex_entries)
    num_entries += int(config.pass_through)
    if num_entries > 1:
        msg = (
            "Keys 'global_value', 'map', 'bins', and "
            "'pass_through' are mutually exclusive but "
            f"more than one was found in {fname!r} raster config: {config}"
        )
        raise revrtValueError(msg)

    if num_entries < 1:
        msg = (
            "Either 'global_value', 'map', 'bins', and "
            "'pass_through' must be specified for a raster, "
            f"but none were found in {fname!r} config: {config}"
        )
        raise revrtValueError(msg)


def _validate_bin_range(bins):
    """Check for correctness in bin range"""
    for input_bin in bins:
        if input_bin.min > input_bin.max:
            msg = f"Min is greater than max for bin config {input_bin}."
            raise revrtAttributeError(msg)

        if input_bin.min == float("-inf") and input_bin.max == float("inf"):
            msg = (
                "Bin covers all possible values, did you forget to set "
                f"min or max? {input_bin}"
            )
            warn(msg, revrtWarning)


def _validate_bin_continuity(bins):
    """Warn user of potential gaps in bin range continuity"""
    sorted_bins = sorted(bins, key=lambda x: x.min)
    last_max = float("-inf")
    for i, input_bin in enumerate(sorted_bins):
        if input_bin.min < last_max:
            last_bin = sorted_bins[i - 1] if i > 0 else "-infinity"
            msg = f"Overlapping bins detected between bin {last_bin} and {bin}"
            warn(msg, revrtWarning)

        if input_bin.min > last_max:
            last_bin = sorted_bins[i - 1] if i > 0 else "-infinity"
            msg = f"Gap detected between bin {last_bin} and {input_bin}"
            warn(msg, revrtWarning)

        if i + 1 == len(sorted_bins) and input_bin.max < float("inf"):
            msg = f"Gap detected between bin {input_bin} and infinity"
            warn(msg, revrtWarning)

        last_max = input_bin.max


def _backend_array(data):
    """Return the NumPy/Dask backing array for accumulation"""
    if isinstance(data, xr.DataArray):
        return data.data

    return data
