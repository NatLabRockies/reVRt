"""Definition of friction, barrier, and costs processing config files"""

import re
from pathlib import Path
from typing import Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, DirectoryPath, FilePath, field_validator

from revrt.constants import ALL, BARRIER_H5_LAYER_NAME
from revrt.exceptions import revrtConfigurationError


_BARRIER_VALUE_PATTERN = re.compile(
    r"^\s*(!=|>=|<=|==|>|<)\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
_BARRIER_OPERATOR_MAP = {
    "!=": "ne",
    ">": "gt",
    ">=": "ge",
    "<": "lt",
    "<=": "le",
    "==": "eq",
}


Extents = Literal["all", "wet", "wet+", "landfall", "dry+", "dry"]
"""Terms for specifying masks

Defined as follows:

    - 'all': Full extent, including offshore, onshore, and landfall
    - 'wet': offshore extent only
    - 'wet+': offshore extent + landfall extent
    - 'landfall': landfall extent (area between wet and dry extents)
    - 'dry+': onshore extent + landfall extent
    - 'dry': onshore extent only

"""


class LandUseMultipliers(TypedDict, total=False):
    """Land use multipliers"""

    cropland: float
    """Cost multiplier for cropland"""

    forest: float
    """Cost multiplier for forest"""

    suburban: float
    """Cost multiplier for suburban areas"""

    urban: float
    """Cost multiplier for urban areas"""

    wetland: float
    """Cost multiplier for wetlands

    This value is independent of the water multiplier.
    """


class SlopeMultipliers(TypedDict, total=False):
    """Slope multipliers and cutoffs"""

    hill_mult: float
    """Cost multiplier for hills"""

    hill_slope: float
    """Lowest slope that qualifies as a hill (decimal percent)"""

    mtn_mult: float
    """Cost multiplier for mountains"""

    mtn_slope: float
    """Lowest slope that qualifies as a mountain (decimal percent)"""


class IsoMultipliers(TypedDict):
    """Multiplier config for one ISO"""

    iso: str
    """Name of ISO these multipliers are for"""

    land_use: LandUseMultipliers
    """Land use multipliers"""

    slope: SlopeMultipliers
    """Slope multipliers and cutoffs"""


class RangeConfig(BaseModel, extra="forbid"):
    """Config for defining a range

    When you define a range, you can add a value to assign to cells
    matching that range. Cells with values >= than `min` and < `max`
    will be assigned `value`. One or both of `min` and `max` can be
    specified.
    """

    min: float = float("-inf")
    """Minimum value to get a cost assigned (inclusive)"""

    max: float = float("inf")
    """Maximum value to get a cost assigned (exclusive)"""

    value: float
    """Value to assign to the range defined by `min` and `max`"""


class Rasterize(BaseModel, extra="forbid"):
    """Config to rasterize a vector layer and apply a value to it"""

    value: float
    """Value to burn in to raster"""

    buffer: float | None = None
    """Value to buffer by (can be negative)"""

    reproject: bool = True
    """Reproject vector to raster CRS if ``True``"""

    all_touched: bool = False
    """Rasterize all cells touched by vector if ``True``"""

    tile_size: int | None = None
    """Tile size to use for rasterization

    If not specified, a default value of 2048 will be used.
    """

    simply_before_rasterize: bool = False
    """Simplify geometries before rasterization if ``True``

    Geometries are simplified to half of the raster cell size when this
    is ``True``. This can help speed up rasterization and reduce memory
    usage for very complex vectors, but may result in less accurate
    rasterization. By default, ``False``.
    """

    fill: float | int | None = 0
    """Value used to fill raster cells not burned by vector

    If None, uses np.nan. By default, ``0``.
    """


class LayerBuildConfig(BaseModel, extra="forbid"):
    """Friction and barrier layers config model

    The inputs `global_value`, `map`, `bins`, `rasterize`, and
    `forced_inclusion` are exclusive, but exactly one must be specified.
    """

    extent: Extents = ALL
    """Extent to apply map or range to

    Must be one of the following:

        - 'all': Full extent, including offshore, onshore, and landfall
        - 'wet': offshore extent only
        - 'wet+': offshore extent + landfall extent
        - 'landfall': landfall extent (area between wet and dry extents)
        - 'dry+': onshore extent + landfall extent
        - 'dry': onshore extent only

    By default, 'all'.
    """

    global_value: float | None = None
    """Global value to use for entire layer extent"""

    map: dict[float, float] | None = None
    """Values in raster (keys) and values to use layer"""

    bins: list[RangeConfig] | None = None
    """Ranges of raster values

    This input can be one or more ranges of raster values to apply to
    barrier/friction. The value of overlapping ranges are added
    together.
    """

    pass_through: bool | None = False
    """Pass cost data through without extra processing"""

    rasterize: Rasterize | None = None
    """Rasterize a vector and save as layer"""

    forced_inclusion: bool = False
    """Force inclusion

    If `forced_inclusion` is ``True``, any cells with a value > 0 will
    force the final value of corresponding cells to 0. Multiple forced
    inclusions are allowed.
    """


def parse_barrier_values(barrier_values):
    """Parse barrier comparison text into an operator and threshold"""
    match = _BARRIER_VALUE_PATTERN.fullmatch(barrier_values)
    if match is None:
        msg = (
            "Barrier values must use one of the supported comparison "
            "operators ('==', '!=', '>', '>=', '<', '<=') followed by a "
            f"number. Got: {barrier_values!r}"
        )
        raise revrtConfigurationError(msg)

    operator, threshold = match.groups()
    return _BARRIER_OPERATOR_MAP[operator], float(threshold)


class BarrierLayer(BaseModel, extra="forbid"):
    """Config for a routing barrier layer"""

    layer_name: str
    """Name of layer in H5/Zarr file"""

    barrier_values: str
    """Comparison definition describing barrier cells"""

    barrier_importance: int | None = None
    """Optional rank used when relaxing soft barriers"""

    @field_validator("barrier_values")
    @classmethod
    def _validate_barrier_values(cls, barrier_values):
        parse_barrier_values(barrier_values)
        return barrier_values

    @field_validator("barrier_importance")
    @classmethod
    def _validate_barrier_importance(cls, barrier_importance):
        if barrier_importance is not None and barrier_importance <= 0:
            msg = (
                "Barrier importance must be a positive integer when "
                f"provided. Got: {barrier_importance!r}"
            )
            raise revrtConfigurationError(msg)

        return barrier_importance

    def to_routing_dict(self):
        """Convert barrier config to the normalized routing payload"""
        barrier_operator, barrier_threshold = parse_barrier_values(
            self.barrier_values
        )

        return {
            "layer_name": self.layer_name,
            "barrier_operator": barrier_operator,
            "barrier_threshold": barrier_threshold,
            "barrier_importance": self.barrier_importance,
        }


class DryCosts(BaseModel, extra="forbid"):
    """Config items required to generate dry costs"""

    iso_region_tiff: FilePath
    """Filename of ISO region GeoTIFF"""

    nlcd_tiff: FilePath
    """File name of NLCD GeoTiff"""

    slope_tiff: FilePath
    """File name of slope GeoTiff"""

    cost_configs: FilePath | None = None
    """Path to json file with transmission cost configuration values

    Path to json file containing dictionary with transmission cost
    configuration values. Valid configuration keys are:

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
    a separate json file containing a dictionary of
    configurations for each section.
    """

    default_mults: IsoMultipliers | None = None
    """Multipliers to be used for default region

    This input should be a dictionary with three keys:

        - "iso": This key is ignored, but is required. Can set to
          "default" and move on.
        - "land_use": A dictionary where keys are the land use types
          (e.g. "cropland", "forest", "wetland", etc.) and values are
          the multipliers for those land uses.
        - "slope": A dictionary where keys are the slope
          types/multipliers (e.g. "hill_mult", "hill_slope",
          "mtn_mult", "mtn_slope", etc.) and values are the
          slopes/multipliers.


    """

    extra_tiffs: list[FilePath] | None = None
    """Optional list of extra GeoTIFFs to add to cost H5 file"""


class MergeFrictionBarriers(BaseModel, extra="forbid"):
    """Config to combine friction and barriers and save to file

    All barrier values are multiplied by a factor before merging with
    friction. The multiplier should be large enough that all barriers
    have a higher value than any possible friction.
    """

    friction_layer: str
    """Name of friction layer

    A file with this name plus a '.tif' extension must have just been
    created or had already existed in the tiff directory.
    """

    barrier_layer: str
    """Name of barrier layer

    A file with this name plus a '.tif' extension must have just been
    created or had already existed in the tiff directory.
    """

    output_layer_name: str | None = BARRIER_H5_LAYER_NAME
    """Name of combined output layer

    By default, :obj:`~revrt.constants.BARRIER_H5_LAYER_NAME`.
    """

    barrier_multiplier: float = 1e6
    """Value to multiply barrier layer by during merge with friction

    The multiplier should be large enough that all barriers have
    a higher value than any possible friction.
    """


LayerBuildComponents = dict[str, LayerBuildConfig]
"""Mapping of layer components to use for building the final layer

Keys are GeoTIFF or vector filepaths. Values are the
:class:`LayerBuildConfig` to use for that file.
"""


class LayerConfig(BaseModel):
    """Config for friction, barrier, and costs processing"""

    layer_name: str
    """Name of layer in H5 file"""

    description: str | None = None
    """Optional description to store in attrs for layer"""

    include_in_file: bool | None = True
    """Flag to specify whether layer should be stored in the file"""

    values_are_costs_per_mile: bool | None = False
    """Option to specify that the values given represent $/mile

    If ``True``, the values will be converted to $/cell_size,
    which is what is ultimately used for routing.
    """

    build: LayerBuildComponents
    """Mapping of layer components used to build this layer

    Keys are GeoTIFF or vector filepaths. Values are the
    :class:`LayerBuildConfig` to use for that file.
    """


Layers = list[LayerConfig]
"""Layer configs to build and potentially add to file"""


class TransmissionLayerCreationConfig(BaseModel):
    """Config for transmission layer creation"""

    template_file: FilePath
    """Template GeoTIFF/Zarr file for shape, profile, and transform"""

    routing_file: Path
    """Layer file to store results in"""

    input_layer_dir: DirectoryPath = Path()
    """Directory to look for GeoTIFFs in, in addition to '.'"""

    masks_dir: Path = Path()
    """Optional path for mask GeoTIFFs"""

    output_tiff_dir: Path = Path()
    """Directory to store output tiff files in"""

    layers: Layers | None = None
    """Optional configuration for layers to be built

    At least one of `layers`, `dry_costs`, or
    `merge_friction_and_barriers` must be defined.
    """

    dry_costs: DryCosts | None = None
    """Optional dry cost layer

    At least one of `layers`, `dry_costs`, or
    `merge_friction_and_barriers` must be defined.
    """

    merge_friction_and_barriers: MergeFrictionBarriers | None = None
    """Optional config to merge friction barriers

    At least one of `layers`, `dry_costs`, or
    `merge_friction_and_barriers` must be defined.
    """
