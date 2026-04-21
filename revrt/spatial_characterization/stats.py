"""revrt statistic computation functions"""

import sys
from itertools import chain
from functools import partial
from enum import StrEnum, auto
from collections import defaultdict

import numpy as np
import rasterio.features
from shapely.geometry import shape

from revrt.exceptions import revrtValueError, revrtNotImplementedError


_PCT_PREFIX = "percentile_"


def _not_implemented(stat_name, *__, **___):
    """Default function for stats which may not have a computation"""
    msg = f"Default computation unavailable for {stat_name!r}"
    raise revrtNotImplementedError(msg)


def _calc_count(processed_raster, **__):
    """Compute number of non-NaN elements"""
    try:
        return int(processed_raster.count())
    except (AttributeError, TypeError):
        return int(np.count_nonzero(~np.isnan(processed_raster)))


def _calc_min(processed_raster, **__):
    """Compute minimum value in raster"""
    return float(np.nanmin(processed_raster))


def _calc_max(processed_raster, **__):
    """Compute maximum value in raster"""
    return float(np.nanmax(processed_raster))


def _calc_mean(processed_raster, out_dtype, **__):
    """Compute mean value of raster"""
    return float(np.nanmean(processed_raster, dtype=out_dtype))


def _calc_sum(processed_raster, out_dtype, **__):
    """Compute sum of raster"""
    return float(np.nansum(processed_raster, dtype=out_dtype))


def _calc_std(processed_raster, **__):
    """Compute std of raster"""
    return float(np.nanstd(processed_raster))


def _calc_median(processed_raster, **__):
    """Compute median of raster"""
    return float(np.nanmedian(processed_raster))


def _calc_majority(pixel_count, **__):
    """Compute value that makes up majority of raster"""
    return max(pixel_count, key=pixel_count.get)


def _calc_minority(pixel_count, **__):
    """Compute value that makes up minority of raster"""
    return min(pixel_count, key=pixel_count.get)


def _calc_unique(pixel_count, **__):
    """Compute number of unique values in raster"""
    return len(pixel_count)


def _calc_range(processed_raster, feature_stats, **__):
    """Get the range from stats using min/max, compute if not present"""
    try:
        r_min = feature_stats["min"]
    except KeyError:
        r_min = float(np.nanmin(processed_raster))
    try:
        r_max = feature_stats["max"]
    except KeyError:
        r_max = float(np.nanmax(processed_raster))
    return r_max - r_min


def _calc_nodata(masked_raster, nodata, **__):
    """Compute number of nodata pixels in raster"""
    try:
        return float(
            np.sum(
                (masked_raster == nodata)
                | (masked_raster == masked_raster.attrs.get("nodata"))
            )
        )
    except AttributeError:
        return float(np.sum(masked_raster == nodata))


class Stat(StrEnum):
    """Enum of basic computable statistics"""

    COUNT = auto()
    """Compute number of non-NaN elements"""

    MIN = auto()
    """Compute minimum value in zone"""

    MAX = auto()
    """Compute maximum value in zone"""

    MEAN = auto()
    """Compute mean value of zone"""

    SUM = auto()
    """Compute sum of zone"""

    STD = auto()
    """Compute std of zone"""

    MEDIAN = auto()
    """Compute median of zone"""

    MAJORITY = auto()
    """Compute pixel value that makes up majority of zone"""

    MINORITY = auto()
    """Compute pixel value that makes up minority of zone"""

    UNIQUE = auto()
    """Compute number of unique pixel values in zone"""

    RANGE = auto()
    """Compute range of pixel values in zone"""

    NODATA = auto()
    """Compute number of nodata pixels in zone"""

    PIXEL_COUNT = auto()
    """Compute count for each pixel value in zone"""

    def __new__(cls, value):  # noqa: PLR0912, C901
        """Create new enum member"""

        obj = str.__new__(cls, value)
        obj._value_ = value
        match value:
            case "count":
                obj.compute = _calc_count
                obj.requires_pixel_count = True
            case "min":
                obj.compute = _calc_min
                obj.requires_pixel_count = False
            case "max":
                obj.compute = _calc_max
                obj.requires_pixel_count = False
            case "mean":
                obj.compute = _calc_mean
                obj.requires_pixel_count = False
            case "sum":
                obj.compute = _calc_sum
                obj.requires_pixel_count = False
            case "std":
                obj.compute = _calc_std
                obj.requires_pixel_count = False
            case "median":
                obj.compute = _calc_median
                obj.requires_pixel_count = False
            case "majority":
                obj.compute = _calc_majority
                obj.requires_pixel_count = True
            case "minority":
                obj.compute = _calc_minority
                obj.requires_pixel_count = True
            case "unique":
                obj.compute = _calc_unique
                obj.requires_pixel_count = True
            case "range":
                obj.compute = _calc_range
                obj.requires_pixel_count = False
            case "nodata":
                obj.compute = _calc_nodata
                obj.requires_pixel_count = False
            case "pixel_count":
                obj.compute = partial(_not_implemented, value)
                obj.requires_pixel_count = True
            case _:  # pragma: no cover
                obj.compute = partial(_not_implemented, value)
                obj.requires_pixel_count = False

        return obj


class FractionalStat(StrEnum):
    """Enum of fractional pixel statistics"""

    FRACTIONAL_PIXEL_COUNT = auto()
    """Compute fractional pixel count for each pixel value in zone"""

    FRACTIONAL_AREA = auto()
    """Compute fractional area for each pixel value in zone"""

    VALUE_MULTIPLIED_BY_FRACTIONAL_AREA = auto()
    """Compute fractional pixel * area for each pixel value in zone"""


class ComputableStats:
    """A class to represent computable statistics for zonal stats"""

    def __init__(
        self, base_stats=None, percentiles=None, fractional_stats=None
    ):
        """

        Parameters
        ----------
        base_stats : iterable of str, optional
            Iterable of "base" statistics to compute (i.e. not
            fractional or percentile). By default, ``None``.
        percentiles : dict, optional
            Dictionary mapping the percentile stat name to the int or
            float representing the percentile that can be passed to
            :func:`numpy.percentile`. By default, ``None``.
        fractional_stats : iterable of str, optional
            One or mre stats from the :class:`FractionalStat` enum to
            compute. By default, ``None``.
        """
        self.base_stats = base_stats or []
        self.percentiles = percentiles or {}
        self.fractional_stats = fractional_stats or []
        self.all = {
            *self.base_stats,
            *self.percentiles,
            *self.fractional_stats,
        }

    def __contains__(self, key):
        return key in self.all

    @property
    def empty(self):
        """dict: Dict with empty stat values"""
        out = {
            k: (0 if k in {Stat.COUNT, Stat.PIXEL_COUNT} else None)
            for k in chain(self.base_stats, self.fractional_stats)
        }
        for pct in self.percentiles:
            out[pct] = None
        return out

    def lazy_pixel_count(self, processed_raster):
        """Compute pixel counts from a raster

        The output dictionary will be empty if this stats collection
        does not contain any stats that require a pixel count.

        Parameters
        ----------
        processed_raster : array-like
            List or array of data to compute pixel counts from. This
            collection can contain NaN values, which will be ignored in
            the final output.

        Returns
        -------
        dict
            Dictionary where keys are the unique array entries and the
            values are the counts for each entry. This dictionary will
            be empty if this stats collection does not contain any stats
            that require a pixel count.
        """
        if not any(stat.requires_pixel_count for stat in self.base_stats):
            return {}

        keys, counts = np.unique(processed_raster, return_counts=True)
        return {
            k.item(): c.item()
            for k, c in zip(keys, counts, strict=False)
            if not np.isnan(k)
        }

    def computed_base_stats(
        self, processed_raster, category_map=None, **kwargs
    ):
        """Compute statistics on array

        Parameters
        ----------
        processed_raster : array-like
            Array to compute statistics for. This collection can
            contain NaN values, which will be ignored in the final
            output.
        category_map : dict, optional
            Dictionary mapping raster values to new names. If given,
            this mapping will be applied to the pixel count dictionary,
            so you can use it to map raster values to human-readable
            category names. By default, ``None``.
        **kwargs
            Extra keyword-argument pairs to pass to statistic
            computation functions. Currently this is only necessary for
            the `nodata` stat, which requires the following extra
            parameters:

                - `masked_raster`: Raster masked to zone, but still
                                   containing the "nodata" value (i.e.
                                   it's not masked out with NaNs).
                - `nodata`: Value representing `nodata` that will be
                            counted for this statistic.

        Returns
        -------
        dict
            Dictionary of computed statistics.
        """
        category_map = category_map or {}

        pixel_count = self.lazy_pixel_count(processed_raster)
        pixel_count = {
            category_map.get(k, k): v for k, v in pixel_count.items()
        }
        out_dtype = _out_dtype_from_raster(processed_raster)

        feature_stats = {}
        for stat in self.base_stats:
            if stat == Stat.PIXEL_COUNT:
                feature_stats[stat] = pixel_count
                continue

            feature_stats[stat] = stat.compute(
                processed_raster=processed_raster,
                pixel_count=pixel_count,
                feature_stats=feature_stats,
                out_dtype=out_dtype,
                **kwargs,
            )

        return feature_stats

    def computed_fractional_stats(
        self, zone, processed_raster, transform, nodata=None, category_map=None
    ):
        """Compute fractional statistics on array

        Parameters
        ----------
        zone : shapely.GeometryType
            Geometry object representing the zone to compute statistics
            over.
        processed_raster : array-like
            Array to compute statistics for. This collection can
            contain NaN values, which will be ignored in the final
            output.
        transform : affine.Affine
            Affine transform object representing the raster
            transformation. This is used to compute the raster shapes
            for statistics computations as well as the pixel area
            for each pixel value in the raster.
        nodata : int or float, optional
            Value representing "nodata" in the array. These values in
            the raster will be ignored and will not contribute to the
            statistics and will not be reported. By default, ``None``.
        category_map : dict, optional
            Optional mapping for raster values. If given, the outputs
            will be labeled using the categories in this mapping instead
            of the raster values directly. The map does not have to
            be exhaustive - missing values will just be reported using
            the raw array value. By default, ``None``.

        Returns
        -------
        dict
            Dictionary of computed fractional statistics. The keys are
            the names of the statistics in the :class:`FractionalStat`
            enum and the values are the computed statistics.
        """
        if not self.fractional_stats:
            return {}

        frac_stats = _fractional_stats(
            zone, processed_raster, transform, nodata, category_map
        )
        return {stat: frac_stats[stat] for stat in self.fractional_stats}

    def computed_percentiles(self, processed_raster):
        """Generate percentile statistics for array

        Parameters
        ----------
        processed_raster : array-like
            Array to compute percentiles for.

        Yields
        ------
        str
            Name of percentile stat
        int or float
            Value representing that percentile in the array.
        """
        return {
            stat: np.nanpercentile(processed_raster, pct)
            for stat, pct in self.percentiles.items()
        }

    @classmethod
    def from_iter(cls, stats=None):
        """Create a ComputableStats object from an iterable of stats

        Parameters
        ----------
        stats : str or iterable of str, optional
            Names of all statistics to compute. Statistics must be one
            of the members of :class:`Stat` or :class:`FractionalStat`,
            or must start with the ``percentile_`` prefix and end with
            an int or float representing the percentile to compute (e.g.
            ``percentile_10.5``). If only one statistic is to be
            computed, you can provide it directly as a string.
            Otherwise, provide a list of statistic names or a string
            with the names separated by a space. You can also provide
            the string ``"ALL"`` or ``"*"`` to specify that all
            statistics should be computed. If no input, empty input, or
            ``None`` is provided, then only the base stats ("count",
            "min", "max", "mean") are configured. To summarize, all of
            the following are valid inputs:

                - ``stats="*"`` or ``stats="ALL"`` or ``stats="All"``
                - ``stats="min"``
                - ``stats="min max"``
                - ``stats=["min"]``
                - ``stats=["min", "max", "percentile_10.5"]``

            By default, ``None``.

        Returns
        -------
        ComputableStats
            An initialized :class:`ComputableStats` object.

        Raises
        ------
        revrtValueError
            If one or more input stats are not known.
        """
        if not stats:
            return cls(base_stats=[Stat.COUNT, Stat.MIN, Stat.MAX, Stat.MEAN])

        if isinstance(stats, str):
            if stats.casefold() in {"*", "all"}:
                return cls(
                    base_stats=list(Stat),
                    fractional_stats=list(FractionalStat),
                )

            stats = stats.split()

        allowed_base_stats = {*Stat}
        allowed_fractional_stats = {*FractionalStat}
        base_stats = []
        percentiles = {}
        fractional_stats = []
        for stat in stats:
            stat_name = stat.casefold()
            if stat_name.startswith(_PCT_PREFIX):
                percentiles[stat_name] = _get_percentile(stat_name)
            elif stat_name in allowed_fractional_stats:
                fractional_stats.append(stat_name)
            elif stat_name in allowed_base_stats:
                base_stats.append(Stat(stat_name))
            else:
                valid_stats = {str(s) for s in Stat}
                valid_stats |= {str(s) for s in FractionalStat}
                msg = (
                    f"Stat {stat!r} not valid; must be one of:\n"
                    f"{valid_stats!r}"
                )
                raise revrtValueError(msg)

        return cls(base_stats, percentiles, fractional_stats)


def _get_percentile(stat):
    """Get the percentile value from user string input"""
    q = float(stat.replace(_PCT_PREFIX, ""))
    if not (0 <= q <= 100):  # noqa: PLR2004
        msg = f"Percentiles must be between 0 and 100 (inclusive). Got: {q}"
        raise revrtValueError(msg)
    return q


def _out_dtype_from_raster(raster):  # pragma: no cover
    """Get the output dtype for stats like mean and std

    If we're on a 64 bit platform and the array is an integer type, this
    function ensures that the array is cast to 64 bit to avoid overflow
    for certain numpy ops
    """
    if sys.maxsize > 2**32 and issubclass(raster.dtype.type, np.integer):
        return "int64"
    return None  # numpy default


def _fractional_stats(
    zonal_polygon, window_array, window_transform, nodata, category_map
):
    """Compute fractional statistics on array"""
    value_multiplied_by_area = 0.0
    category_map = category_map or {}
    returned_value_pixel_count_dict = defaultdict(float)
    returned_value_pixel_area_dict = defaultdict(float)
    pixel_area = _compute_pixel_area(window_transform)
    window_array = np.asarray(window_array)
    valid_mask = (window_array != nodata) & (~np.isnan(window_array))

    raster_shapes = rasterio.features.shapes(
        window_array,
        mask=valid_mask,
        connectivity=4,
        transform=window_transform,
    )
    for geom, value in raster_shapes:
        raster_poly = shape(geom)
        intersection = raster_poly.intersection(zonal_polygon)

        if not intersection.is_empty:
            intersection_area = intersection.area
            returned_value_pixel_area_dict[value] += intersection_area

            value_multiplied_by_area += value * intersection_area

            pixel_fractional_count = intersection_area / pixel_area
            returned_value_pixel_count_dict[value] += pixel_fractional_count

    returned_value_pixel_count_dict = {
        category_map.get(k, k): v
        for k, v in returned_value_pixel_count_dict.items()
    }

    returned_value_pixel_area_dict = {
        category_map.get(k, k): v
        for k, v in returned_value_pixel_area_dict.items()
    }

    return {
        FractionalStat.FRACTIONAL_PIXEL_COUNT: returned_value_pixel_count_dict,
        FractionalStat.FRACTIONAL_AREA: returned_value_pixel_area_dict,
        FractionalStat.VALUE_MULTIPLIED_BY_FRACTIONAL_AREA: (
            value_multiplied_by_area
        ),
    }


def _compute_pixel_area(affine):
    """Compute the pixel area from the affine transform"""
    width, _, _, _, height, *_ = affine
    return abs(width) * abs(height)
