"""Abstract base class for layer builders"""

from pathlib import Path
from abc import ABC, abstractmethod
from functools import cached_property

import xarray as xr

from revrt.constants import DEFAULT_DTYPE


class BaseLayerCreator(ABC):
    """Abstract Base Class to create transmission routing layers"""

    def __init__(
        self,
        io_handler,
        input_layer_dir=".",
        output_tiff_dir=".",
        dtype=DEFAULT_DTYPE,
    ):
        """
        Parameters
        ----------
        io_handler : LayeredFile
            Transmission layer IO handler
        input_layer_dir : path-like, optional
            Directory to search for input layers in, if not found in
            current directory. By default, ``'.'``.
        output_tiff_dir : path-like, optional
            Directory where cost layers should be saved as GeoTIFF.
            By default, ``"."``.
        dtype : np.dtype, optional
            Data type for final dataset. By default, ``float32``.
        """
        self._io_handler = io_handler

        self.input_layer_dir = (
            Path(str(input_layer_dir).strip()).expanduser().resolve()
        )
        self._output_tiff_dir = (
            Path(str(output_tiff_dir).strip()).expanduser().resolve()
        )
        self._dtype = dtype

    @property
    def shape(self):
        """tuple: Layer shape"""
        return self._io_handler.shape

    @property
    def cell_size(self):
        """float: Size of cell in layer file"""
        return self._io_handler.cell_size

    @cached_property
    def output_tiff_dir(self):
        """:term:`path-like <path-like object>` GeoTIFFs output dir"""
        self._output_tiff_dir.mkdir(exist_ok=True, parents=True)
        return self._output_tiff_dir

    @cached_property
    def chunks(self):
        """tuple: Layer chunks (no band dimension)"""

        with xr.open_dataset(
            self._io_handler.fp, consolidated=False, engine="zarr"
        ) as ds:
            attrs = ds.attrs

        return attrs["chunks"]["y"], attrs["chunks"]["x"]

    @abstractmethod
    def build(self, *args, **kwargs):
        """Build layer"""
        raise NotImplementedError
