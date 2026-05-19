"""Module to build and save dry (land) cost raster layers"""

import logging
from pathlib import Path

import dask.array as da

from revrt.costs.config import TransmissionConfig
from revrt.costs.base import BaseLayerCreator
from revrt.constants import DEFAULT_DTYPE, METERS_IN_MILE
from revrt.utilities import (
    load_data_using_layer_file_profile,
    save_data_using_layer_file_profile,
)
from revrt.exceptions import revrtValueError


logger = logging.getLogger(__name__)

DRY_MULTIPLIER_TIFF = "dry_multipliers.tif"
DEFAULT_HILL_MULTIPLIER = 1
"""Default hill slope multiplier value (1)"""
DEFAULT_MTN_MULTIPLIER = 1
"""Default mountain slope multiplier value (1)"""
DEFAULT_HILL_SLOPE = 2
"""Default hill slope cutoff value (2)

Slope values above this (inclusive) are considered hills"""
DEFAULT_MTN_SLOPE = 8
"""Default mountain slope cutoff value (8)"""
NLCD_LAND_USE_CLASSES = {
    "cropland": [80, 81],
    "forest": [41, 42, 43],
    "wetland": [90, 95],
    "suburban": [21, 22, 23],
    "urban": [24],
}
"""NLCD categories relevant to routing"""
WATER_NLCD_CODE = 11
"""NLCD category value for water (11)"""
WATER_MULTIPLIER = 10.0
"""Multiplier value for water cells based on NLCD (10)"""


class DryCostsCreator(BaseLayerCreator):
    """Class to create and save dry transmission cost layers"""

    def build(  # noqa: PLR0913, PLR0917
        self,
        iso_region_tiff,
        nlcd_tiff,
        slope_tiff,
        transmission_config=None,
        mask=None,
        default_multipliers=None,
        extra_tiffs=None,
        tiff_chunks="file",
        descriptions=None,
        nodata=None,
        lock=None,
        **profile_kwargs,
    ):
        """Build cost rasters using base line costs and multipliers

        This function also allows you to save to GeoTIFF.

        Cells without a know ISO region are left with a cost value of 0.

        Parameters
        ----------
        iso_region_tiff : path-like
            Path to the ISO region GeoTIFF.
        nlcd_tiff : path-like
            Path to the National Land Coverage Database GeoTIFF.
        slope_tiff : path-like
            Path to the slope GeoTIFF. Slope must be in decimal percent.
        transmission_config : dict or path-like, optional
            Dictionary or path to JSON file containing dictionary with
            transmission cost configuration values. Valid configuration
            keys are:

                - "base_line_costs"
                - "iso_lookup"
                - "iso_multipliers"
                - "land_use_classes"
                - "new_substation_costs"
                - "power_classes"
                - "power_to_voltage"
                - "transformer_costs"
                - "upgrade_substation_costs"

            Each of these keys should point to a dictionary or a path to
            a separate path-like file containing a dictionary of
            configurations for each section. BY default, ``None``, which
            uses the default cost configs for all values.
        mask : ndarray
            Boolean array representing mask where dry cost values should
            be applied. BY default, ``None``, which does not apply a
            mask to the cost array.
        default_multipliers : dict or IsoMultipliers, optional
            Multipliers for regions not specified in the
            `iso_region_tiff`. Must be a dictionary of the form::

                "land_use": {
                    "cropland": 1.03,
                    "forest": 1.2,
                    "suburban": 1.08,
                    "urban": 1.2,
                    "wetland": 1.8
                },
                "slope": {
                    "hill_mult": 1.1,
                    "hill_slope": 2,
                    "mtn_mult": 1.21,
                    "mtn_slope": 8
                }

            All keys are optional. By default, ``None``.
        extra_tiffs : list, optional
            Optional list of extra GeoTIFFs to add to cost layer file
            (e.g. a transmission barrier file). By default, ``None``,
            which does not add any extra layers.
        descriptions : dict, optional
            Optional mapping  where keys are layer names and values are
            descriptions to add to the layer's attributes meta
            dictionary under the "description" key.
            By default, ``None``, which does not add any description.
        nodata : dict, optional
            Optional mapping where keys are layer names and values are
            the nodata value for the output raster of that layer. This
            value will be added to the layer's attributes meta
            dictionary under the "nodata" key. By default, ``None``.
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
        xc = TransmissionConfig(config=transmission_config)
        nodata = nodata or {}

        layers = self._load_layers(
            iso_region_tiff, slope_tiff, nlcd_tiff, tiff_chunks=tiff_chunks
        )
        multipliers = self._compute_multipliers(
            xc["iso_multipliers"],
            layers[iso_region_tiff],
            layers[slope_tiff],
            layers[nlcd_tiff],
            iso_lookup=xc["iso_lookup"],
            land_use_classes=xc["land_use_classes"],
            default_multipliers=default_multipliers,
        )

        logger.debug("Saving multipliers array GeoTIFF")
        multiplier_tiff_fp = self.output_tiff_dir / DRY_MULTIPLIER_TIFF
        save_data_using_layer_file_profile(
            layer_fp=self._io_handler.fp,
            data=multipliers,
            geotiff=multiplier_tiff_fp,
            nodata=nodata.get(multiplier_tiff_fp.stem, None),
            lock=lock,
            **profile_kwargs,
        )
        layers[multiplier_tiff_fp.stem] = multipliers

        for tiff_fp, data in layers.items():
            self._write_single_layer(
                tiff_fp,
                data=data,
                nodata=nodata,
                descriptions=descriptions,
                tiff_chunks=tiff_chunks,
            )

        for tiff_fp in extra_tiffs or []:
            self._write_single_layer(
                tiff_fp,
                nodata=nodata,
                descriptions=descriptions,
                tiff_chunks=tiff_chunks,
            )

        for power_class, capacity in xc["power_classes"].items():
            logger.info(
                "Calculating costs for class %s using a %sMW line",
                power_class,
                capacity,
            )
            blc_arr = self._compute_base_line_costs(
                capacity,
                xc["base_line_costs"],
                layers[iso_region_tiff],
                xc["iso_lookup"],
            )

            base_costs_tiff = f"base_line_costs_{capacity}MW.tif"
            out_fp = self.output_tiff_dir / base_costs_tiff
            save_data_using_layer_file_profile(
                layer_fp=self._io_handler.fp,
                data=blc_arr,
                geotiff=out_fp,
                nodata=nodata.get(out_fp.stem),
                lock=lock,
                **profile_kwargs,
            )

            costs_arr = blc_arr * multipliers

            dry_layer_name = f"tie_line_costs_{capacity}MW"
            tie_line_costs_tiff = f"{dry_layer_name}.tif"
            out_fp = self.output_tiff_dir / tie_line_costs_tiff
            if mask is not None:
                costs_arr = da.where(mask, costs_arr, 0)
            save_data_using_layer_file_profile(
                layer_fp=self._io_handler.fp,
                data=costs_arr,
                geotiff=out_fp,
                nodata=nodata.get(out_fp.stem),
                lock=lock,
                **profile_kwargs,
            )

            self._write_single_layer(
                out_fp,
                layer_name=dry_layer_name,
                nodata=nodata,
                descriptions=descriptions,
                tiff_chunks=tiff_chunks,
            )

    def _load_layers(
        self, iso_region_tiff, slope_tiff, nlcd_tiff, tiff_chunks="file"
    ):
        """Load ISO region, slope and land use rasters"""
        logger.debug("Loading ISO region, slope and land use rasters")
        iso_layer = load_data_using_layer_file_profile(
            layer_fp=self._io_handler.fp,
            geotiff=iso_region_tiff,
            tiff_chunks=tiff_chunks,
            layer_dirs=[self.input_layer_dir, self.output_tiff_dir],
            band_index=0,
        )
        slope_layer = load_data_using_layer_file_profile(
            layer_fp=self._io_handler.fp,
            geotiff=slope_tiff,
            tiff_chunks=tiff_chunks,
            layer_dirs=[self.input_layer_dir, self.output_tiff_dir],
            band_index=0,
        )
        nlcd_layer = load_data_using_layer_file_profile(
            layer_fp=self._io_handler.fp,
            geotiff=nlcd_tiff,
            tiff_chunks=tiff_chunks,
            layer_dirs=[self.input_layer_dir, self.output_tiff_dir],
            band_index=0,
        )
        logger.debug("Loading complete")
        return {
            iso_region_tiff: iso_layer,
            slope_tiff: slope_layer,
            nlcd_tiff: nlcd_layer,
        }

    def _write_single_layer(
        self,
        tiff_fp,
        data=None,
        layer_name=None,
        nodata=None,
        descriptions=None,
        tiff_chunks="file",
    ):
        """Write a single layer to the layered file"""
        nodata = nodata or {}
        descriptions = descriptions or {}

        if data is None:
            data = load_data_using_layer_file_profile(
                layer_fp=self._io_handler.fp,
                geotiff=tiff_fp,
                tiff_chunks=tiff_chunks,
                layer_dirs=[self.input_layer_dir, self.output_tiff_dir],
            )
        if not layer_name:
            layer_name = Path(tiff_fp).stem

        logger.debug(
            "Writing %s to layer file: %s", layer_name, self._io_handler.fp
        )
        self._io_handler.write_layer(
            data,
            layer_name,
            nodata=nodata.get(layer_name),
            description=descriptions.get(layer_name),
            overwrite=True,
        )

    def _compute_multipliers(
        self,
        iso_multipliers,
        iso_layer,
        slope_layer,
        land_use_layer,
        iso_lookup,
        land_use_classes=None,
        default_multipliers=None,
    ):
        """Create costs multiplier raster"""
        multipliers = da.ones(
            self.shape, dtype=self._dtype, chunks=self.chunks
        )
        regions_mask = da.full(
            self.shape, False, dtype=bool, chunks=self.chunks
        )
        land_use_classes = land_use_classes or NLCD_LAND_USE_CLASSES

        for r_conf in iso_multipliers:
            iso_name = r_conf["iso"]
            logger.info("Processing multipliers for region %s", iso_name)

            iso = iso_lookup[iso_name]
            logger.debug("ISO %s has id %s", iso_name, iso)

            mask = iso_layer == iso
            regions_mask |= mask

            if "land_use" in r_conf:
                r_lu = da.where(mask, land_use_layer, da.nan)
                lum = compute_land_use_multipliers(
                    r_lu,
                    r_conf["land_use"],
                    land_use_classes,
                    chunks=self.chunks,
                )
                multipliers = da.where(mask, multipliers * lum, multipliers)

            if "slope" in r_conf:
                r_slope = da.where(mask, slope_layer, da.nan)
                slope_multipliers = compute_slope_multipliers(
                    r_slope,
                    chunks=self.chunks,
                    config=r_conf["slope"],
                )
                multipliers = da.where(
                    mask, multipliers * slope_multipliers, multipliers
                )

        # Calculate multipliers for regions not defined in `config`
        logger.debug("Processing default region")
        if default_multipliers is not None:
            default_mask = ~regions_mask

            if "land_use" in default_multipliers:
                region_land_use = da.where(
                    default_mask, land_use_layer, da.nan
                )
                lum_dict = default_multipliers["land_use"]
                lum = compute_land_use_multipliers(
                    region_land_use,
                    lum_dict,
                    land_use_classes,
                    chunks=self.chunks,
                )
                multipliers = da.where(
                    default_mask, multipliers * lum, multipliers
                )

            if "slope" in default_multipliers:
                region_slope = da.where(default_mask, slope_layer, da.nan)
                slope_multipliers = compute_slope_multipliers(
                    region_slope,
                    chunks=self.chunks,
                    config=default_multipliers["slope"],
                )
                multipliers = da.where(
                    default_mask, multipliers * slope_multipliers, multipliers
                )

        # Set water multiplier last so we don't get super high
        # multipliers at water body boundaries next to steep slopes
        return da.where(
            land_use_layer == WATER_NLCD_CODE, WATER_MULTIPLIER, multipliers
        )

    def _compute_base_line_costs(
        self, capacity, base_line_costs, iso_layer, iso_lookup
    ):
        """Get base line cost per cell raster for a given voltage"""
        base_cost = da.zeros(self.shape, dtype=self._dtype, chunks=self.chunks)

        for iso in base_line_costs:
            logger.info("Processing costs for %s for %sMW", iso, capacity)
            iso_code = iso_lookup[iso]
            cost_per_mile = base_line_costs[iso][str(capacity)]
            cost_per_cell = cost_per_mile / METERS_IN_MILE * self.cell_size

            logger.debug(
                "Base line $/mile is %s, $/cell is %s",
                cost_per_mile,
                cost_per_cell,
            )
            mask = iso_layer == iso_code
            base_cost = da.where(mask, cost_per_cell, base_cost)

        return base_cost


def compute_slope_multipliers(slope, chunks, config=None):
    """Create slope multiplier raster for a region

    Unspecified slopes are left at 1.0

    Parameters
    ----------
    slope : array-like
        Slope raster clipped to a region - "Terrain slope in % of grade"
    chunks : tuple
        Dask chunks to use when creating multipliers array. Should be
        of the same shape as `slope`.
    config : dict or None
        Multipliers and slope cut offs for hilly and mountain terrain.
        The following keys are allowed:

            - 'hill_mult' : Multiplier for hilly terrain.
            - 'mtn_slope' : Multiplier for mountainous terrain.
            - 'hill_slope' : Slope at and above which a cell is
                             classified as hilly.
            - 'mtn_slope' : Slope at and above which a cell is
                            classified as mountainous.

        If ``None``, uses default values. By default, ``None``.

    Returns
    -------
    array-like
        Slope multiplier raster. Minimum value for any cell is 1.
    """
    config = config or {}

    hill_multiplier = config.get("hill_mult", DEFAULT_HILL_MULTIPLIER)
    mtn_multiplier = config.get("mtn_mult", DEFAULT_MTN_MULTIPLIER)
    hill_slope = config.get("hill_slope", DEFAULT_HILL_SLOPE)
    mtn_slope = config.get("mtn_slope", DEFAULT_MTN_SLOPE)

    hilly = (slope >= hill_slope) & (slope < mtn_slope)
    mountainous = slope >= mtn_slope

    multipliers = da.ones(slope.shape, dtype=DEFAULT_DTYPE, chunks=chunks)
    multipliers = da.where(hilly, hill_multiplier, multipliers)

    return da.where(mountainous, mtn_multiplier, multipliers)


def compute_land_use_multipliers(
    land_use, multipliers, land_use_classes, chunks
):
    """Convert NLCD raster to land use multiplier raster

    Land classes without specified multipliers are left at 1.

    Parameters
    ----------
    land_use : array-like
        NLCD land user raster clipped to a region.
    multipliers : LandUseMultipliers
        Multiplier for for land classes, E.g. {'forest': 1.5}.
    land_use_classes : dict
        NLCD land use codes corresponding to use classes for
        multipliers.
    chunks : tuple
        Dask chunks to use when creating multipliers array. Should be
        of the same shape as `land_use`.

    Returns
    -------
    array-like
        Land use multiplier raster. Minimum value for any cell is 1.
    """
    multiplier_raster = da.ones(
        land_use.shape, dtype=DEFAULT_DTYPE, chunks=chunks
    )

    # Determine mask arrays for NLCD values and multiplier to apply
    indices = []
    for class_, multiplier in multipliers.items():
        if class_ not in land_use_classes:
            msg = f"Class {class_} not in land_use_classes: {land_use_classes}"
            raise revrtValueError(msg)

        nlcd_values = land_use_classes[class_]
        if not isinstance(nlcd_values, list):
            msg = f"NLCD values must be in list form; Got: {nlcd_values}"
            raise revrtValueError(msg)

        for nlcd_value in nlcd_values:
            index = land_use == nlcd_value
            indices.append((index, multiplier, nlcd_value))

    # Apply multipliers to appropriate cells
    for i in indices:
        multiplier_raster = da.where(i[0], i[1], multiplier_raster)

    return multiplier_raster
