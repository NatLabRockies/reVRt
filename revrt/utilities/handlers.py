"""Handler for file containing GeoTIFF layers"""

import json
import sqlite3
import logging
import operator
import functools
from pathlib import Path
from functools import cached_property

import zarr
import dask
import fiona
import rioxarray
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from revrt.exceptions import (
    revrtFileExistsError,
    revrtFileNotFoundError,
    revrtKeyError,
    revrtValueError,
)
from revrt.utilities.base import (
    delete_data_file,
    expand_dim_if_needed,
    transform_xy,
    save_data_array_to_geotiff,
    load_data_using_layer_file_profile,
)
from revrt.utilities.monitoring import log_mem, log_runtime


logger = logging.getLogger(__name__)
ZARR_COMPRESSORS = zarr.codecs.BloscCodec(  # cspell:disable-line
    cname="zstd",
    clevel=9,  # cspell:disable-line
    shuffle=zarr.codecs.BloscShuffle.shuffle,  # cspell:disable-line
)
"""[NOT PUBLIC API] Compressors to use when writing to Zarr file"""


class IncrementalWriter:
    """Stream results to disk by appending result in chunks to a file

    File can be of type CSV or GeoPackage. A new file is created if one
    does not exist.
    """

    def __init__(self, out_fp):
        """

        Parameters
        ----------
        out_fp : path-like
            Path to output file.
        """
        self.out_fp = Path(out_fp)
        self._columns = None

    def preprocess_chunk(self, chunk):  # ruff:ignore[no-self-use]
        """Preprocess chunk before saving

        By default this method just passes the data through (underlying
        assumption is that the chunk data is already in a DataFrame).
        If the underlying data is not in a DataFrame, users can subclass
        and overwrite this method to convert the data into a DataFrame
        for writing.

        Parameters
        ----------
        chunk : pandas.DataFrame or geopandas.GeoDataFrame
            A chunk of data that will eventually be written to file.

        Returns
        -------
        pandas.DataFrame or geopandas.GeoDataFrame
            A dataframe holding the chunk of data to be written to file.
        """
        return chunk

    @property
    def _is_gpkg(self):
        """bool: Check if output file is GeoPackage"""
        return self.out_fp.suffix.lower() == ".gpkg"

    def save(self, result):
        """Write result chunk to file

        Parameters
        ----------
        result : pandas.DataFrame or geopandas.GeoDataFrame
            A chunk of data that will eventually be written to file.
        """
        result = self.preprocess_chunk(result)
        if self._is_gpkg:
            self._save_gpkg(result)
            return
        self._save_csv(result)

    def _save_gpkg(self, data):
        """Save route result to GeoPackage file"""
        if self.out_fp.exists():
            if self._columns is None:
                self._columns = gpd.read_file(self.out_fp, rows=1).columns
            data = data.reindex(columns=self._columns)

        data.to_file(self.out_fp, driver="GPKG", mode="a")

    def _save_csv(self, data):
        """Save route result to CSV file"""
        if self.out_fp.exists():
            if self._columns is None:
                self._columns = pd.read_csv(self.out_fp, nrows=0).columns
            data = data.reindex(columns=self._columns)

        data.to_csv(
            self.out_fp, mode="a", index=False, header=not self.out_fp.exists()
        )


class LayeredFile:
    """Handler for file containing GeoTIFF layers"""

    SUPPORTED_FILE_ENDINGS = {".zarr", ".tif", ".tiff"}
    """Supported template file endings"""

    LATITUDE = "latitude"
    """Name of latitude values layer in LayeredFile"""

    LONGITUDE = "longitude"
    """Name of longitude values layer in LayeredFile"""

    def __init__(self, fp):
        """

        Parameters
        ----------
        fp : path-like
            Path to layered file on disk.
        """
        self.fp = Path(fp)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.fp})"

    def __str__(self):
        num_layers = len(self.data_layers)
        if num_layers == 1:  # pragma: no cover
            return f"{self.__class__.__name__} with 1 layer"
        return f"{self.__class__.__name__} with {num_layers:,d} layers"

    def __getitem__(self, layer):
        # This method is ported for backward compatibility, but it's
        # unlikely to be useful in practice since it loads the entire
        # layer data all at once
        if layer not in self.layers:
            msg = f"{layer!r} is not present in {self.fp}"
            raise revrtKeyError(msg)

        logger.debug("\t- Extracting %s from %s", layer, self.fp)
        with xr.open_dataset(self.fp, consolidated=False, engine="zarr") as ds:
            profile = _layer_profile_from_open_ds(layer, ds)
            values = ds[layer].to_numpy()

        return profile, values

    @cached_property
    def profile(self):
        """dict: Template layer profile"""
        with xr.open_dataset(self.fp, consolidated=False, engine="zarr") as ds:
            return {
                "width": ds.rio.width,
                "height": ds.rio.height,
                "crs": ds.rio.crs,
                "transform": ds.rio.transform(),
            }

    @property
    def shape(self):
        """tuple: Template layer shape"""
        return self.profile["height"], self.profile["width"]

    @property
    def cell_size(self):
        """float: Size of cell in layer file"""
        return abs(self.profile["transform"].a)

    @property
    def layers(self):
        """list: All available layers in file"""
        if not self.fp.exists():
            msg = f"File {self.fp} not found"
            raise revrtFileNotFoundError(msg)

        with xr.open_dataset(self.fp, consolidated=False, engine="zarr") as ds:
            return list(ds.variables)

    @property
    def data_layers(self):
        """list: Available data layers in file"""
        return [
            layer_name
            for layer_name in self.layers
            if layer_name
            not in {
                "band",
                "x",
                "y",
                "spatial_ref",
                self.LATITUDE,
                self.LONGITUDE,
            }
        ]

    def layer_profile(self, layer):
        """Get layer profile as dictionary

        Parameters
        ----------
        layer : str
            Name of layer in file to get profile for.

        Returns
        -------
        dict
            Dictionary containing layer profile information, including
            the following keys:

                - "nodata": NoData value for layer
                - "width": width of layer
                - "height": height of layer
                - "crs": :class:`pyproj.crs.CRS` object for layer
                - "count": number of bands in layer
                - "dtype": data type of layer
                - "transform": :class:`affine.Affine` transform for
                               layer

        Raises
        ------
        revrtKeyError
            If layer is not present in file.
        """
        if layer not in self.layers:
            msg = f"{layer!r} is not present in {self.fp}"
            raise revrtKeyError(msg)

        with xr.open_dataset(self.fp, consolidated=False, engine="zarr") as ds:
            return _layer_profile_from_open_ds(layer, ds)

    def layer_attrs(self, layer):
        """Get layer attributes

        Parameters
        ----------
        layer : str
            Name of layer in file to get attributes for.

        Returns
        -------
        dict
            Dictionary of attribute metadata for the requested layer
            as stored in the Zarr variable attrs.

        Raises
        ------
        revrtKeyError
            If layer is not present in file.
        """
        if layer not in self.layers:
            msg = f"{layer!r} is not present in {self.fp}"
            raise revrtKeyError(msg)

        with xr.open_dataset(self.fp, consolidated=False, engine="zarr") as ds:
            return dict(ds[layer].attrs)

    def create_new(
        self,
        template_file,
        overwrite=False,
        chunk_x=2048,
        chunk_y=2048,
        read_chunks="auto",
    ):
        """Create a new layered file

        Parameters
        ----------
        template_file : path-like, optional
            Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or Zarr
            (``*.zarr``) file containing the profile and transform to be
            used for the layered file. If ``None``, then the `fp`
            input is used as the template. By default, ``None``.
        overwrite : bool, optional
            Overwrite file if is exists. By default, ``False``.
        chunk_x, chunk_y : int, default=2048
            Chunk size of x and y dimension for newly-created layered
            file. By default, ``2048``.
        read_chunks : int or str, default="auto"
            Chunk size to use when reading the template file. This will
            be passed down as the ``chunks`` argument to
            :func:`rioxarray.open_rasterio` or
            :func:`xarray.open_dataset`, depending on what template file
            is passed in. By default, ``"auto"``.

        Returns
        -------
        LayeredFile
            This :class:`~revrt.utilities.handlers.LayeredFile` object
            with a corresponding file on disk.
        """
        if self.fp.exists() and not overwrite:
            msg = f"File {self.fp!r} exits and overwrite=False"
            raise revrtFileExistsError(msg)

        template_file = str(template_file).strip()
        _validate_template(template_file)

        logger.debug("\t- Initializing %s from %s", self.fp, template_file)
        with log_runtime(f"Layered file creation ({self.fp})"):
            try:
                _init_zarr_file_from_template(
                    template_file,
                    self.fp,
                    chunk_x=chunk_x,
                    chunk_y=chunk_y,
                    read_chunks=read_chunks,
                )
                logger.info(
                    "Layered file %s created from %s!", self.fp, template_file
                )
            except Exception:  # pragma: no cover
                logger.exception("Error initializing %s", self.fp)
                if self.fp.exists():
                    delete_data_file(self.fp)
                raise

        return self

    def write_layer(
        self,
        values,
        layer_name,
        description=None,
        overwrite=False,
        nodata=None,
        **layer_attrs,
    ):
        """Write a layer to the file

        Parameters
        ----------
        values : array-like
            Layer data (can be numpy array, xarray.DataArray, or
            dask.array).
        layer_name : str
            Name of layer to be written to file.
        description : str, optional
            Description of layer being added. By default, ``None``.
        overwrite : bool, default=False
            Option to overwrite layer data if layer already exists in
            :class:`~revrt.utilities.handlers.LayeredFile`.
            By default, ``False``.
        nodata : int or float, optional
            Optional nodata value for the raster layer. This value will
            be added to the layer's attributes meta dictionary under the
            "nodata" key.

            .. WARNING::
               ``rioxarray`` does not recognize the "nodata" value when
               reading from a zarr file (because zarr uses the
               ``_FillValue`` encoding internally). To get the correct
               "nodata" value back when reading a
               :class:`~revrt.utilities.handlers.LayeredFile`,
               you can either 1) read from ``da.rio.encoded_nodata`` or
               2) check the layer's attributes for the ``"nodata"`` key,
               and if present, use ``da.rio.write_nodata`` to write the
               nodata value so that ``da.rio.nodata`` gives the right
               value.

        **layer_attrs
            Additional attributes to store on the written layer.

        Raises
        ------
        revrtFileNotFoundError
            If :class:`~revrt.utilities.handlers.LayeredFile` does not
            exist.
        revrtKeyError
            If layer with the same name already exists and
            ``overwrite=False``.
        """
        if not self.fp.exists():
            msg = (
                f"File {self.fp} not found. Please create the file before "
                "adding layers."
            )
            raise revrtFileNotFoundError(msg)

        logger.info("Writing layer %s to %s", layer_name, self.fp)
        with log_runtime(f"Writing layer {layer_name} to {self.fp}"):
            values = expand_dim_if_needed(values)

            if values.shape[1:] != self.shape:
                msg = (
                    f"Shape of provided data {values.shape[1:]} does "
                    f"not match shape of LayeredFile: {self.shape}"
                )
                raise revrtValueError(msg)

            with xr.open_dataset(
                self.fp, consolidated=False, engine="zarr"
            ) as ds:
                attrs = ds.attrs
                coords = ds.coords
                layer_exists = layer_name in ds

            if layer_exists:
                msg = f"{layer_name!r} is already present in {self.fp}"
                if not overwrite:
                    msg = f"{msg} and 'overwrite=False'"
                    raise revrtKeyError(msg)
                self._delete_layer(layer_name)

            chunks = (1, attrs["chunks"]["y"], attrs["chunks"]["x"])

            da = xr.DataArray(values, dims=("band", "y", "x"), attrs=attrs)
            da = da.chunk(attrs["chunks"])
            da = da.assign_coords(coords)
            da.attrs["count"] = 1
            da.attrs["description"] = description
            da.attrs.update(layer_attrs)
            if nodata is not None:
                nodata = da.dtype.type(nodata)
                da = da.rio.write_nodata(nodata)
                da.attrs["nodata"] = nodata

            ds_to_add = xr.Dataset({layer_name: da}, attrs=attrs)
            encoding = {layer_name: da.encoding or {}}
            encoding[layer_name].update(
                {
                    "compressors": ZARR_COMPRESSORS,
                    "dtype": da.dtype,
                    "chunks": chunks,
                }
            )

            log_mem()
            ds_to_add.to_zarr(
                self.fp,
                mode="a-",
                encoding=encoding,
                zarr_format=3,
                consolidated=False,
                compute=True,
            )

    def _delete_layer(self, layer_name):
        """Delete a layer from the backing Zarr file"""
        group = zarr.open_group(self.fp, mode="a")
        del group[layer_name]

    def write_geotiff_to_file(
        self,
        geotiff,
        layer_name,
        check_tiff=True,
        description=None,
        overwrite=True,
        nodata=None,
        tiff_chunks="auto",
    ):
        """Transfer GeoTIFF to layered file

        Parameters
        ----------
        geotiff : path-like
            Path to GeoTIFF file.
        layer_name : str
            Name of layer to be written to file.
        check_tiff : bool, optional
            Option to check GeoTIFF profile, CRS, and shape against
            layered file profile, CRS, and shape. By default, ``True``.
        description : str, optional
            Description of layer being added. By default, ``None``.
        overwrite : bool, default=False
            Option to overwrite layer data if layer already exists in
            :class:`~revrt.utilities.handlers.LayeredFile`.

            .. IMPORTANT::
              When overwriting data, the encoding (and therefore things
              like data type, nodata value, etc) is not allowed to
              change. If you need to overwrite an existing layer with a
              new type of data, manually remove it from the file first.

            By default, ``False``.
        nodata : int or float, optional
            Optional nodata value for the raster layer. This value will
            be added to the layer's attributes meta dictionary under the
            "nodata" key.

            .. WARNING::
               ``rioxarray`` does not recognize the "nodata" value when
               reading from a zarr file (because zarr uses the
               ``_FillValue`` encoding internally). To get the correct
               "nodata" value back when reading a
               :class:`~revrt.utilities.handlers.LayeredFile`,
               you can either 1) read from ``da.rio.encoded_nodata`` or
               2) check the layer's attributes for the ``"nodata"`` key,
               and if present, use ``da.rio.write_nodata`` to write the
               nodata value so that ``da.rio.nodata`` gives the right
               value.

        tiff_chunks : int or str, default="auto"
            Chunk size to use when reading the GeoTIFF file. This will
            be passed down as the ``chunks`` argument to
            :func:`rioxarray.open_rasterio`. By default, ``"auto"``.
        """
        if not self.fp.exists():
            logger.info("%s not found - creating from %s...", self.fp, geotiff)
            self.create_new(geotiff)

        logger.info(
            "%s being extracted from %s and written to %s",
            layer_name,
            self.fp,
            geotiff,
        )
        with log_runtime(f"Writing GeoTIFF to {geotiff}"):
            tif = load_data_using_layer_file_profile(
                self.fp,
                geotiff,
                tiff_chunks=tiff_chunks,
                check_tiff=check_tiff,
                fillna=None,
            )
            logger.debug("\t- Writing data from %s to %s", geotiff, self.fp)
            try:
                self.write_layer(
                    tif,
                    layer_name,
                    description=description,
                    overwrite=overwrite,
                    nodata=nodata,
                )
            finally:
                tif.close()

    def layer_to_geotiff(
        self, layer, geotiff, ds_chunks="auto", lock=None, **profile_kwargs
    ):
        """Extract layer from file and write to GeoTIFF file

        Parameters
        ----------
        layer : str
            Layer to extract,
        geotiff : path-like
            Path to output GeoTIFF file.
        ds_chunks : int or str, default="auto"
            Chunk size to use when reading the
            :class:`~revrt.utilities.handlers.LayeredFile`.
            This will be passed down as the ``chunks`` argument to
            :func:`xarray.open_dataset`. By default, ``"auto"``.
        lock : bool or `dask.distributed.Lock`, optional
            Lock to use to write data using dask. If not supplied, a
            single process is used for writing data to the GeoTIFF.
            By default, ``None``.
        **profile_kwargs
            Additional keyword arguments to pass into writing the
            raster. The following attributes ar ignored (they are set
            using properties of the source
            :class:`~revrt.utilities.handlers.LayeredFile`):

                - nodata
                - transform
                - crs
                - count
                - width
                - height

        """
        logger.debug("\t- Writing %s from %s to %s", layer, self.fp, geotiff)
        with xr.open_dataset(
            self.fp, chunks=ds_chunks, consolidated=False, engine="zarr"
        ) as ds:
            save_data_array_to_geotiff(
                ds[layer], geotiff, lock=lock, **profile_kwargs
            )

    def layers_to_file(
        self,
        layers,
        check_tiff=True,
        descriptions=None,
        overwrite=False,
        nodata=None,
    ):
        """Transfer GeoTIFF layers into layered file

        If layered file does not exist, it is created and populated.

        Parameters
        ----------
        layers : list or dict
            Dictionary mapping layer names to GeoTIFFs filepaths. Each
            GeoTIFF will be loaded into the
            :class:`~revrt.utilities.handlers.LayeredFile` user
            the layer name. If a list of GeoTIFFs filepaths is provided,
            the file name stems are used as the layer names.
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against layered
            file profile and coordinates. By default, ``True``.
        description : dict, optional
            Mapping of layer name to layer description of layers.
            By default, ``None``, which does not store any descriptions.
        overwrite : bool, default=False
            Option to overwrite layer data if layer already exists in
            :class:`~revrt.utilities.handlers.LayeredFile`.

            .. IMPORTANT::
              When overwriting data, the encoding (and therefore things
              like data type, nodata value, etc) is not allowed to
              change. If you need to overwrite an existing layer with a
              new type of data, manually remove it from the file first.

            By default, ``False``.
        nodata : int or float, optional
            Optional nodata value for the raster layer. This value will
            be added to the layer's attributes meta dictionary under the
            "nodata" key.

            .. WARNING::
               ``rioxarray`` does not recognize the "nodata" value when
               reading from a zarr file (because zarr uses the
               ``_FillValue`` encoding internally). To get the correct
               "nodata" value back when reading a
               :class:`~revrt.utilities.handlers.LayeredFile`,
               you can either 1) read from ``da.rio.encoded_nodata`` or
               2) check the layer's attributes for the ``"nodata"`` key,
               and if present, use ``da.rio.write_nodata`` to write the
               nodata value so that ``da.rio.nodata`` gives the right
               value.

        Returns
        -------
        str
            String representation of path to output layered file.
        """
        if isinstance(layers, list):
            layers = {Path(fp).stem: fp for fp in layers}

        if descriptions is None:
            descriptions = {}

        logger.info("Moving layers to %s", self.fp)
        for layer_name, geotiff in layers.items():
            logger.info("- Transferring %s", layer_name)
            description = descriptions.get(layer_name)

            self.write_geotiff_to_file(
                geotiff,
                layer_name,
                check_tiff=check_tiff,
                description=description,
                overwrite=overwrite,
                nodata=nodata,
            )

        return str(self.fp)

    def extract_layers(
        self, layers, ds_chunks="auto", lock=None, **profile_kwargs
    ):
        """Extract layers from file and save to disk as GeoTIFFs

        Parameters
        ----------
        layers : dict
            Dictionary mapping layer names to GeoTIFF files to create.
        ds_chunks : int or str, default="auto"
            Chunk size to use when reading the
            :class:`~revrt.utilities.handlers.LayeredFile`.
            This will be passed down as the ``chunks`` argument to
            :func:`xarray.open_dataset`. By default, ``"auto"``.
        lock : bool or `dask.distributed.Lock`, optional
            Lock to use to write data using dask. If not supplied, a
            single process is used for writing data to the GeoTIFFs.
            By default, ``None``.
        **profile_kwargs
            Additional keyword arguments to pass into writing the
            raster. The following attributes ar ignored (they are set
            using properties of the source
            :class:`~revrt.utilities.handlers.LayeredFile`):

                - nodata
                - transform
                - crs
                - count
                - width
                - height
        """
        logger.info("Extracting layers from %s", self.fp)
        for layer_name, geotiff in layers.items():
            logger.info("- Extracting %s", layer_name)
            self.layer_to_geotiff(
                layer_name,
                geotiff,
                ds_chunks=ds_chunks,
                lock=lock,
                **profile_kwargs,
            )

    def extract_all_layers(
        self, out_dir, ds_chunks="auto", lock=None, **profile_kwargs
    ):
        """Extract all layers from file and save to disk as GeoTIFFs

        Parameters
        ----------
        out_dir : path-like
            Path to output directory into which layers should be saved
            as GeoTIFFs. This directory will be created if it does not
            already exist.
        ds_chunks : int or str, default="auto"
            Chunk size to use when reading the :class:`LayeredFile`.
            This will be passed down as the ``chunks`` argument to
            :func:`xarray.open_dataset`. By default, ``"auto"``.
        lock : bool or `dask.distributed.Lock`, optional
            Lock to use to write data using dask. If not supplied, a
            single process is used for writing data to the GeoTIFFs.
            By default, ``None``.
        **profile_kwargs
            Additional keyword arguments to pass into writing the
            raster. The following attributes ar ignored (they are set
            using properties of the source
            :class:`~revrt.utilities.handlers.LayeredFile`):

                - nodata
                - transform
                - crs
                - count
                - width
                - height

        Returns
        -------
        dict
            Dictionary mapping layer names to GeoTIFF files created.
        """
        out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        layers = {
            layer_name: out_dir / f"{layer_name}.tif"
            for layer_name in self.data_layers
        }
        self.extract_layers(
            layers, ds_chunks=ds_chunks, lock=lock, **profile_kwargs
        )
        return layers


def serialize_layer_build_dict(build_dict):
    """Serialize layer config for storing in file attrs

    Parameters
    ----------
    build_dict : dict
        Dictionary mapping layer names to layer build configs.

    Returns
    -------
    str
        Serialized layer build config dictionary as a JSON string.
    """
    normalized_config = {
        str(layer_name): component_config.model_dump(
            mode="json", exclude_none=True
        )
        for layer_name, component_config in build_dict.items()
    }
    return json.dumps(normalized_config, sort_keys=True)


def num_feats_in_gpkg(filename):
    """Lightweight func to get number of features in GeoPackage file

    This function does not load the entire GeoPackage into memory.
    Instead, it queries the internal SQLite database to get the number
    of features.

    Parameters
    ----------
    filename : path-like
        Path to GeoPackage file.

    Returns
    -------
    int
        Number of features in the GeoPackage file.
    """
    with sqlite3.connect(filename) as con:
        cursor = con.cursor()
        cursor.execute(
            "SELECT table_name, column_name FROM gpkg_geometry_columns;"
        )
        try:
            geom_table_suffix = "_".join(cursor.fetchall()[0])
        except IndexError:
            return 0  # No geometry columns found

        geom_table = f"rtree_{geom_table_suffix}"

        q = f"SELECT COUNT(distinct id) FROM {geom_table};"  # ruff:ignore[hardcoded-sql-expression]
        cursor.execute(q)
        return cursor.fetchall()[0][0]


def gpkg_crs(data_fp):
    """Get CRS of GeoPackage file

    Parameters
    ----------
    data_fp : path-like
        Path to GeoPackage file.

    Returns
    -------
    pyproj.crs.CRS
        CRS of GeoPackage file.
    """
    with fiona.Env(), fiona.open(data_fp) as src:
        return src.crs


def chunked_read_gpkg(data_fp, chunk_size):
    """Read GeoPackage file in chunks

    Parameters
    ----------
    data_fp : path-like
        Path to GeoPackage file to read in chunks.
    chunk_size : int
        Number of rows per chunk.

    Yields
    ------
    geopandas.GeoDataFrame
        Chunk of data as GeoDataFrame.
    """
    with fiona.Env(), fiona.open(data_fp) as src:
        total_rows = len(src)

    logger.debug(
        "\t- Processing %s with %d rows in chunks of %d",
        data_fp,
        total_rows,
        chunk_size,
    )

    for start in range(0, total_rows, chunk_size):
        df = gpd.read_file(data_fp, rows=slice(start, start + chunk_size))
        if df.empty:
            continue

        logger.debug(
            "\t\t- Processing %d rows starting at index %d",
            len(df),
            start,
        )
        yield df


def _layer_profile_from_open_ds(layer, ds):
    """Get layer profile from open dataset"""
    return {
        "nodata": ds[layer].attrs.get("nodata", ds[layer].rio.encoded_nodata),
        "width": ds.rio.width,
        "height": ds.rio.height,
        "crs": ds.rio.crs,
        "count": ds[layer].rio.count,
        "dtype": ds[layer].dtype,
        "transform": ds.rio.transform(),
    }


def _validate_template(template_file):
    """Validate template file"""
    template_file = Path(template_file)
    valid_file_ending = any(
        template_file.suffix == fe for fe in LayeredFile.SUPPORTED_FILE_ENDINGS
    )
    if not valid_file_ending:
        msg = (
            f"Template file {template_file!r} format is not "
            "supported! File must end in one of: "
            f"{LayeredFile.SUPPORTED_FILE_ENDINGS}"
        )
        raise revrtValueError(msg)

    if not template_file.exists():
        msg = f"Template file {template_file!r} not found on disk!"
        raise revrtFileNotFoundError(msg)


def _init_zarr_file_from_template(
    template_file, out_fp, chunk_x, chunk_y, read_chunks="auto"
):
    """Initialize Zarr file from GeoTIFF template"""
    template_file = Path(template_file)
    if template_file.suffix == ".zarr":
        return _init_zarr_file_from_zarr_template(
            template_file, out_fp, chunk_x, chunk_y, ds_chunks=read_chunks
        )

    return _init_zarr_file_from_tiff_template(
        template_file, out_fp, chunk_x, chunk_y, tiff_chunks=read_chunks
    )


def _init_zarr_file_from_zarr_template(
    template_file, out_fp, chunk_x, chunk_y, ds_chunks="auto"
):
    """Initialize Zarr file from a Zarr template"""
    with xr.open_dataset(
        template_file, chunks=ds_chunks, consolidated=False, engine="zarr"
    ) as ds:
        transform = ds.rio.transform()
        src_crs = ds.rio.crs

        out_ds = _compile_ds(
            ds["x"].data,
            ds["y"].data,
            ds[LayeredFile.LATITUDE].data,
            ds[LayeredFile.LONGITUDE].data,
            transform,
            src_crs,
            chunk_x,
            chunk_y,
        )
        _save_ds_as_zarr_with_encodings(
            out_ds, chunk_x=chunk_x, chunk_y=chunk_y, out_fp=out_fp
        )


def _init_zarr_file_from_tiff_template(
    template_file, out_fp, chunk_x, chunk_y, tiff_chunks="auto"
):
    """Initialize Zarr file from GeoTIFF template"""
    with rioxarray.open_rasterio(template_file, chunks=tiff_chunks) as geo:
        transform = geo.rio.transform()
        src_crs = geo.rio.crs

        x, y, lat, lon = _compute_lat_lon(
            geo.sizes["y"],
            geo.sizes["x"],
            src_crs,
            transform,
            chunk_x=chunk_x,
            chunk_y=chunk_y,
        )

        out_ds = _compile_ds(
            x, y, lat, lon, transform, src_crs, chunk_x, chunk_y
        )
        _save_ds_as_zarr_with_encodings(
            out_ds, chunk_x=chunk_x, chunk_y=chunk_y, out_fp=out_fp
        )


def _compute_lat_lon(ny, nx, src_crs, transform, chunk_x=2048, chunk_y=2048):
    """Compute latitude and longitude arrays from transform and CRS"""
    xx = dask.array.arange(nx, chunks=chunk_x, dtype="float32") + 0.5
    yy = dask.array.arange(ny, chunks=chunk_y, dtype="float32") + 0.5
    x_mesh, y_mesh = dask.array.meshgrid(xx, yy)  # shapes (y, x), chunked

    x = transform.c + xx * transform.a
    y = transform.f + yy * transform.e

    x_mesh_transformed = (
        transform.c + x_mesh * transform.a + y_mesh * transform.b
    )
    y_mesh_transformed = (
        transform.f + x_mesh * transform.d + y_mesh * transform.e
    )

    lon, lat = dask.array.map_blocks(
        _proj_to_lon_lat,
        x_mesh_transformed,
        y_mesh_transformed,
        src_crs.to_wkt(),
        dtype="float32",
        new_axis=(0,),  # we add a new leading axis of length 2
        chunks=((2,), *x_mesh_transformed.chunks),  # chunk sizes for [2, y, x]
    )
    logger.debug(
        "Array shapes:\n\t- x=%r\n\t- y=%r\n\t- lon=%r\n\t- lat=%r",
        x.shape,
        y.shape,
        lon.shape,
        lat.shape,
    )
    return x, y, lat, lon


def _compile_ds(x, y, lat, lon, transform, src_crs, chunk_x, chunk_y):
    """Create an xarray Dataset with coordinates and attributes"""
    attrs = {"chunks": {"y": chunk_y, "x": chunk_x}}

    out_ds = xr.Dataset(attrs=attrs)
    out_ds = out_ds.assign_coords(
        band=(("band"), [0]),
        y=(("y"), y.astype(np.float32)),
        x=(("x"), x.astype(np.float32)),
        longitude=(("y", "x"), lon),
        latitude=(("y", "x"), lat),
    )

    out_ds = out_ds.rio.write_crs(src_crs)
    out_ds = out_ds.rio.write_transform(transform)
    return out_ds.rio.write_grid_mapping()


def _save_ds_as_zarr_with_encodings(out_ds, chunk_x, chunk_y, out_fp):
    """Write dataset to Zarr file with encodings"""
    encoding = {
        "y": {"dtype": "float32", "chunks": (chunk_y,)},
        "x": {"dtype": "float32", "chunks": (chunk_x,)},
        "longitude": {
            "compressors": ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": (chunk_y, chunk_x),
        },
        "latitude": {
            "compressors": ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": (chunk_y, chunk_x),
        },
    }
    logger.debug("Writing data to '%s' with encoding:\n%r", out_fp, encoding)
    out_ds.to_zarr(
        out_fp, mode="w", encoding=encoding, zarr_format=3, consolidated=False
    )


def _proj_to_lon_lat(xx_block, yy_block, src):
    """Block-wise transform to lon/lat; returns array shape [2, y, x]"""
    # create transformer inside the block to avoid pickling issues
    lon, lat = transform_xy(
        src, "EPSG:4326", xx_block.ravel(), yy_block.ravel()
    )
    out = np.empty(
        (2, functools.reduce(operator.mul, xx_block.shape)), dtype="float32"
    )
    out[0] = lon
    out[1] = lat

    return out.reshape((2, *xx_block.shape))
