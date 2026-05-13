"""Class to create, load, and store masks to determine land and sea"""

import logging
from pathlib import Path

import numpy as np

from revrt.exceptions import (
    revrtAttributeError,
    revrtFileNotFoundError,
    revrtValueError,
)
from revrt.utilities.raster import rasterize_shape_file
from revrt.utilities import (
    load_data_using_layer_file_profile,
    save_data_using_custom_props,
)


logger = logging.getLogger(__name__)
_MASK_MSG = "No mask available. Please run create() or load() first."


class Masks:
    """Create, load, and store mask data layers"""

    LANDFALL_MASK_FNAME = "landfall_mask.tif"
    """One pixel width line at shore"""
    RAW_LAND_MASK_FNAME = "raw_land_mask.tif"
    """Rasterized land vector"""
    LAND_MASK_FNAME = "land_mask.tif"
    """Raw mask - landfall mask"""
    OFFSHORE_MASK_FNAME = "offshore_mask.tif"
    """Offshore mask filename"""

    def __init__(self, shape, crs, transform, masks_dir="."):
        """

        Parameters
        ----------
        shape : tuple
            Shape of mask rasters (height, width).
        crs : str or dict
            Coordinate reference system of mask rasters.
        transform : affine.Affine
            Affine transform of mask rasters.
        masks_dir : path-like, optional
            Directory for storing/finding mask GeoTIFFs.
            By default, ``"."``.
        """
        self.shape = shape
        self.crs = crs
        self.transform = transform

        self._masks_dir = Path(masks_dir)
        self._masks_dir.mkdir(parents=True, exist_ok=True)

        self._landfall_mask = None
        self._dry_mask = None
        self._wet_mask = None
        self._dry_plus_mask = None
        self._wet_plus_mask = None

    @property
    def landfall_mask(self):
        """:term:`array_like`: Landfalls cells bool mask; 1 cell wide"""
        if self._landfall_mask is None:
            raise revrtAttributeError(_MASK_MSG)
        return self._landfall_mask

    @property
    def wet_mask(self):
        """:term:`array_like`: Wet cells bool mask; no landfall cells"""
        if self._wet_mask is None:
            raise revrtAttributeError(_MASK_MSG)
        return self._wet_mask

    @property
    def dry_mask(self):
        """:term:`array_like`: Dry cells bool mask; no landfall cells"""
        if self._dry_mask is None:
            raise revrtAttributeError(_MASK_MSG)
        return self._dry_mask

    @property
    def dry_plus_mask(self):
        """:term:`array_like`: Dry cells bool mask; *with* landfall"""
        if self._dry_plus_mask is None:
            self._dry_plus_mask = np.logical_or(
                self.dry_mask, self.landfall_mask
            )
        return self._dry_plus_mask

    @property
    def wet_plus_mask(self):
        """:term:`array_like`: Wet cells mask, *with* landfall cells"""
        if self._wet_plus_mask is None:
            self._wet_plus_mask = np.logical_or(
                self.wet_mask, self.landfall_mask
            )
        return self._wet_plus_mask

    def create(
        self,
        land_mask_shp_fp,
        save_tiff=True,
        reproject_vector=True,
        lock=None,
    ):
        """Create mask layers from a polygon land vector file

        Parameters
        ----------
        land_mask_shp_fp : str
            Full path to land polygon GPKG or shp file
        save_tiff : bool, optional
            Save mask as tiff if true. By default, ``True``.
        reproject_vector : bool, optional
            Reproject CRS of vector to match template raster if True.
            By default, ``True``.
        lock : bool or `dask.distributed.Lock`, optional
            Lock to use to write data using dask. If not supplied, a
            single process is used for writing data to the mask
            GeoTIFFs.
            By default, ``None``.
        """
        logger.debug("Creating masks from %s", land_mask_shp_fp)

        dest_crs = self.crs if reproject_vector else None
        height, width = self.shape

        # Raw land is all land cells, include landfall cells
        raw_land = rasterize_shape_file(
            land_mask_shp_fp,
            width,
            height,
            self.transform,
            all_touched=True,
            dest_crs=dest_crs,
            dtype="uint8",
        )

        raw_land_mask = raw_land == 1

        # Offshore mask is inversion of raw land mask
        self._wet_mask = ~raw_land_mask

        landfall = rasterize_shape_file(
            land_mask_shp_fp,
            width,
            height,
            self.transform,
            dest_crs=dest_crs,
            all_touched=True,
            boundary_only=True,
            dtype="uint8",
        )
        self._landfall_mask = landfall == 1

        # XOR landfall and raw land to get all land cells except
        # landfall cells
        self._dry_mask = np.logical_xor(self.landfall_mask, raw_land_mask)
        self._clear_combined_mask_cache()

        logger.debug("Created all masks")

        if save_tiff:
            logger.debug("Saving masks to GeoTIFF")
            self._save_mask(raw_land_mask, self.RAW_LAND_MASK_FNAME, lock=lock)
            self._save_mask(self.wet_mask, self.OFFSHORE_MASK_FNAME, lock=lock)
            self._save_mask(self.dry_mask, self.LAND_MASK_FNAME, lock=lock)
            self._save_mask(
                self.landfall_mask, self.LANDFALL_MASK_FNAME, lock=lock
            )
            logger.debug("Completed saving all masks")

    def _save_mask(self, data, fname, lock):
        """Save mask to GeoTiff"""
        full_fname = self._masks_dir / fname
        save_data_using_custom_props(
            data,
            full_fname,
            shape=self.shape,
            crs=self.crs,
            transform=self.transform,
            lock=lock,
        )

    def _clear_combined_mask_cache(self):
        """Clear cached masks derived from base mask layers"""
        self._dry_plus_mask = None
        self._wet_plus_mask = None

    def load(self, layer_fp, validate=False):
        """Load the mask layers from GeoTIFFs

        Parameters
        ----------
        layer_fp : path-like
            Path to :class:`~revrt.utilities.handlers.LayeredFile` on
            disk for which masks were created. The masks will be of the
            same shape/crs/transform as this file.
        validate : bool, optional
            Whether to validate that loaded masks have appropriate
            values. This breaks the lazy (Dask) loading of the masks, so
            it is not recommended use if you know your masks are valid.
            By default, ``False``.

        Notes
        -----
        This does not need to be called if :meth:`Masks.create()`
        was run previously. Mask files must be in the current directory.
        """
        logger.debug("Loading masks")
        self._dry_mask = self._load_mask(
            self.LAND_MASK_FNAME, layer_fp, validate
        )
        self._wet_mask = self._load_mask(
            self.OFFSHORE_MASK_FNAME, layer_fp, validate
        )
        self._landfall_mask = self._load_mask(
            self.LANDFALL_MASK_FNAME, layer_fp, validate
        )
        self._clear_combined_mask_cache()
        logger.debug("Successfully loaded wet, dry, and landfall masks")

    def _load_mask(self, fname, layer_fp, validate=False):
        """Load mask from GeoTIFF with sanity checking"""
        full_fname = self._masks_dir / fname

        if not full_fname.exists():
            msg = (
                f"Mask file at {full_fname} not found. "
                "Please create masks first."
            )
            raise revrtFileNotFoundError(msg)

        raster = load_data_using_layer_file_profile(
            layer_fp, full_fname, band_index=0
        )

        if validate:
            raster_max = raster.max().compute().item()
            raster_min = raster.min().compute().item()

            if raster_max != 1:  # pragma: no cover
                msg = (
                    f"Maximum value in mask file {fname} is {raster_max} but"
                    " should be 1. Mask file appears to be corrupt. Please "
                    "recreate it."
                )
                raise revrtValueError(msg)

            if raster_min != 0:  # pragma: no cover
                msg = (
                    f"Minimum value in mask file {fname} is {raster_min} but"
                    " should be 0. Mask file appears to be corrupt. Please "
                    "recreate it."
                )
                raise revrtValueError(msg)

        return raster == 1
