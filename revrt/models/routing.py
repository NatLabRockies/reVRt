"""Pydantic models for routing inputs"""

from typing import Literal
from functools import partial

from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from revrt.exceptions import revrtConfigurationError
from revrt.models.cost_layers import BarrierLayer
from revrt.utilities.parsing import (
    parse_comparison_values,
    normalize_str_list_input,
)


type DriverOptionRules = dict[str, float | Literal["excluded"]]
"""Per-option driver multipliers

Keys are routing-option names and values are either numeric friction
multipliers to apply while routing in that option or the keyword
``"excluded"``, which means the option is not allowed at all in
the zone.
"""

type MultiplierLayerInput = list[str] | None
"""One or more layer names used as spatial multipliers"""

type TransitionCostValue = float | dict[str, float | dict[str, float]]
"""A scalar transition cost or a voltage/polarity-dependent cost"""


class RoutingCostLayer(BaseModel, extra="forbid"):
    """Config for one cost layer in a routing option

    Cost layers are summed to build the routing cost surface for an
    option. Each layer may be rescaled before aggregation and may also
    opt out of final-cost reporting while still influencing routing.

    **The rest of this docstring is inserted by Pydantic and can be
    ignored.**
    """

    layer_name: str
    """Name of layer in layered file containing cost data"""

    multiplier_layer: MultiplierLayerInput = None
    """Optional spatial multiplier layer(s) applied before summation

    A string or iterable is accepted and normalized to a list.
    All supplied layers are multiplied together.
    """

    _normalize_multiplier_layer = field_validator(
        "multiplier_layer", mode="before"
    )(partial(normalize_str_list_input, name="multiplier_layer"))

    multiplier_scalar: float = 1
    """Optional scalar multiplier applied before summation"""

    is_invariant: bool = False
    """Skip path-length scaling when ``True``

    Use this for layers whose values are already lump-sum route costs,
    such as fixed-dollar costs to cross into a region.
    """

    include_in_final_cost: bool = True
    """Include this layer in final route cost output when ``True``"""

    include_in_report: bool = True
    """Report this layer's cost and distance outputs if ``True``"""

    apply_row_mult: bool = False
    """Apply the transmission ``row_width`` multiplier when ``True``

    The routing table input should resolve a voltage value for each
    routing option, either from the shared `voltage` column or from
    `voltage_<option>` column. The ``row_width`` dictionary may specify
    one multiplier for every option at a voltage, such as
    ``{"138": 1.15}``, or values by routing option, such as
    ``{"500": {"overhead": 1.15, "underground": 2}}``. An
    option value may instead map spatial layer names to scalar values.
    For example, ``{"500": {"underground": {"rural": 2, "urban": 20}}}``
    produces one cost term per spatial layer. Existing
    ``multiplier_layer`` inputs apply to every generated term. Every
    resolved voltage must be listed. For an option mapping, every
    routing option using this multiplier must be listed.
    """

    apply_polarity_mult: bool = False
    """Apply the voltage and polarity multiplier when ``True``

    The routing table input should resolve both a voltage and a polarity
    value for each routing option, either from shared columns or from
    `voltage_<option>` / `polarity_<option>` columns, and the
    transmission config must provide each combination in
    ``voltage_polarity_mult``. A scalar polarity value applies to every
    routing option. To define values per routing option, use a mapping
    of routing-option names to scalar values. A routing-option value may
    instead map spatial layer names to scalar values. In that case, each
    layer is expanded into one cost term per spatial layer. For example,
    ``{"500": {"dc": {"underground": {"rural": 50, "urban": 60}}}}``
    produces terms weighted by ``50 * rural`` and ``60 * urban``.
    Existing ``multiplier_layer`` inputs apply to each generated term.

    .. IMPORTANT::
      The configured multiplier is assumed to be in million dollars per
      mile and is converted to dollars per pixel before being applied.

    """


class RoutingFrictionLayer(BaseModel, extra="forbid"):
    """Config for one friction layer in a routing option

    Friction layers are aggregated separately and then multiplied onto
    the summed cost surface using
    ``C = (sum cost) * (1 + sum friction)``. Their values influence the
    selected route but are not included in the final route cost.

    **The rest of this docstring is inserted by Pydantic and can be
    ignored.**
    """

    multiplier_layer: MultiplierLayerInput = None
    """Spatial multiplier layer(s) applied to the aggregated costs

    A string or iterable is accepted and normalized to a list.
    All supplied layers are multiplied together.
    """

    _normalize_multiplier_layer = field_validator(
        "multiplier_layer", mode="before"
    )(partial(normalize_str_list_input, name="multiplier_layer"))

    multiplier_scalar: float = 1
    """Scalar multiplier applied before friction is aggregated"""

    include_in_report: bool = False
    """Report route distance and values for this layer when ``True``"""

    apply_row_mult: bool = False
    """Apply the transmission ``row_width`` multiplier when ``True``

    The routing table input should resolve a voltage value for each
    routing option, either from the shared `voltage` column or from
    `voltage_<option>` column. The ``row_width`` dictionary may specify
    one multiplier for every option at a voltage, such as
    ``{"138": 1.15}``, or values by routing option, such as
    ``{"500": {"overhead": 1.15, "underground": 2}}``. An
    option value may instead map spatial layer names to scalar values.
    For example, ``{"500": {"underground": {"rural": 2, "urban": 20}}}``
    produces one friction term per spatial layer. Existing
    ``multiplier_layer`` inputs apply to every generated term. Every
    resolved voltage must be listed. For an option mapping, every
    routing option using this multiplier must be listed.
    """

    apply_polarity_mult: bool = False
    """Apply the voltage and polarity multiplier when ``True``

    The routing table input should resolve both a voltage and a polarity
    value for each routing option, either from shared columns or from
    `voltage_<option>` / `polarity_<option>` columns, and the
    transmission config must provide each combination in
    ``voltage_polarity_mult``. A scalar polarity value applies to every
    routing option. To define values per routing option, use a mapping
    of routing-option names to scalar values. A routing-option value may
    instead map spatial layer names to scalar values. In that case, each
    layer is expanded into one friction term per spatial layer. For
    example,
    ``{"500": {"dc": {"underground": {"rural": 50, "urban": 60}}}}``
    produces terms weighted by ``50 * rural`` and ``60 * urban``.
    Existing ``multiplier_layer`` inputs apply to each generated term.

    .. IMPORTANT::
      The configured multiplier is assumed to be in million dollars per
      mile and is converted to dollars per pixel before being applied.

    """


class RoutingBarrierLayer(BarrierLayer):
    """Config for one hard or soft routing barrier

    **The rest of this docstring is inserted by Pydantic and can be
    ignored.**
    """

    layer_name: str
    """Name of layer containing barrier candidate values"""

    where: str
    """Comparison expression defining which values act as barriers

    Any pixel matching ``where`` is treated as blocked during routing.

    Supported operators are ``==``, ``!=``, ``>``, ``>=``, ``<``, and
    ``<=``, followed by a numeric threshold such as ``">=15"`` or
    ``"!=0"``.

    Multiple entries may reference the same layer with different
    ``"where"`` definitions.
    """

    barrier_importance: int | None = None
    """Optional positive rank used to relax soft barriers when needed

    When a route cannot be found, reVRt will iteratively drop the
    lowest-ranked soft barrier and retry routing until a route is found
    or all ranked barriers have been removed. If a barrier should never
    be relaxed, omit this field or set it to ``None``. This allows hard
    and soft barriers to be combined in the same routing run.
    """


class TrackedLayer(BaseModel, extra="forbid"):
    """Config for one route-characterization layer

    Tracked layers mirror the scaling behavior used by cost layers,
    but do not contribute to routing cost and do not influence the
    routing objective. They are only sampled along the chosen route and
    summarized into additional output columns.

    **The rest of this docstring is inserted by Pydantic and can be
    ignored.**
    """

    layer_name: str
    """Name of layer in layered file to aggregate along the route"""

    agg_method: str
    """Name of the ``dask.array`` aggregation applied to route values

    Examples of this input include ``"sum"``, ``"mean"``, ``"max"``,
    etc.
    """

    multiplier_layer: MultiplierLayerInput = None
    """Optional spatial multiplier layer(s) applied before aggregation

    A string or iterable is accepted and normalized to a list.
    All supplied layers are multiplied together.
    """

    _normalize_multiplier_layer = field_validator(
        "multiplier_layer", mode="before"
    )(partial(normalize_str_list_input, name="multiplier_layer"))

    multiplier_scalar: float = 1
    """Optional scalar multiplier applied before aggregation"""


class RoutingOptionConfig(BaseModel, extra="forbid"):
    """Config for one named routing option

    Routing options are keyed by name, such as ``"overhead"`` or
    ``"underground"``. Route tables may reference these option names in
    ``start_option`` and ``end_option`` columns.

    **The rest of this docstring is inserted by Pydantic and can be
    ignored.**
    """

    cost_layers: list[RoutingCostLayer] | None = Field(default=None)
    r"""Cost layers definitions

    The cost layers are built and then summed to create the option's
    base routing surface, which determines the cost output for each
    route. Specifically, the cost at each pixel is multiplied by the
    length that the route takes through the pixel, and all of these
    values are summed for each route to determine the final cost.

    .. IMPORTANT::
        If a pixel has a final cost of :math:`\leq 0` and the
        `invalid_costs_block_routing` is set to ``True``, the pixel is
        treated as a barrier (i.e. no paths can ever cross this pixel).


    """

    friction_layers: list[RoutingFrictionLayer] | None = Field(default=None)
    r"""Friction layer definitions


    Friction layers are multiplied onto the aggregated cost layer to
    influence routing but are NOT reported in final cost. These layers
    are first aggregated, and then the aggregated friction layer is
    applied to the aggregated cost. The cost at each pixel is therefore
    computed as:

    .. math::

        C = (\sum_{i} c_i) * (1 + \sum_{j} f_j)

    where :math:`C` is the final cost at each pixel, :math:`c_i` are
    the individual cost layers, and :math:`f_j` are the individual
    friction layers.

    .. NOTE:: :math:`\sum_{j} f_j` is always clamped to be
        :math:`\gt -1` to prevent zero or negative routing costs.
        In other words, :math:`(1 + \sum_{j} f_j) > 0` always holds.
        This means friction can scale costs to/away from zero but
        never cause the sign of the cost layer to flip (even if
        friction values themselves are negative). This means all
        "barrier" pixels (i.e. cost value :math:`\leq 0`) will remain
        barriers after friction is applied (assuming
        `invalid_costs_block_routing` is set to ``True``).


    """

    barrier_layers: list[RoutingBarrierLayer] | None = Field(default=None)
    """Barrier layer definitions

    Barrier layers define explicit routing barriers that routes should
    not cross. Unlike `friction_layers`, barrier layers do not add a
    penalty to the routing surface. Instead, any pixel matching a
    barrier definition is treated as blocked during routing. Barrier
    layers _can_ be relaxed to allow routing through them if no valid
    route can be found. This behavior can be configured.
    """

    cost_multiplier_layer: MultiplierLayerInput = None
    """Optional option-level multiplier layer(s) applied to final costs

    A string or iterable is accepted and normalized to a list.
    All supplied layers are multiplied together.
    """

    _normalize_cost_multiplier_layer = field_validator(
        "cost_multiplier_layer", mode="before"
    )(partial(normalize_str_list_input, name="cost_multiplier_layer"))

    cost_multiplier_scalar: float = 1
    """Optional option-level scalar multiplied onto final costs"""

    @model_validator(mode="before")
    @classmethod
    def _fix_missing_fields(cls, data):
        if not isinstance(data, dict):
            return data

        for key in ("cost_layers", "friction_layers", "barrier_layers"):
            if data.get(key) is None:
                data.pop(key, None)

        return data


type RoutingOptionsMap = dict[str, RoutingOptionConfig]
"""Mapping of routing-option names to configurations"""


class _RoutingOptionsInput(BaseModel, extra="forbid"):
    """Canonical container for routing-option configurations

    The ``routing_options`` mapping is keyed by the option names that
    route tables and driver rules reference, such as ``"overhead"`` or
    ``"underground"``.

    **The rest of this docstring is inserted by Pydantic and can be
    ignored.**
    """

    routing_options: RoutingOptionsMap
    """Mapping of routing-option names to option configurations"""

    @field_validator("routing_options")
    @classmethod
    def _validate_routing_options(cls, routing_options):
        return TypeAdapter(RoutingOptionsMap).validate_python(routing_options)


class DriverZoneConfig(BaseModel, extra="forbid"):
    """Config for one spatially varying driver rule zone

    Driver zones define route-option multipliers or exclusions within a
    subset of the raster identified by ``layer_name`` and ``where``.

    You may specify the zone's route-option multipliers as flat mapping
    of names to values instead of a nested ``options`` input.

    **The rest of this docstring is inserted by Pydantic and can be
    ignored.**
    """

    layer_name: str
    """Layer in the cost file used to define the spatial driver mask"""

    where: str
    """Comparison expression describing cells that belong to the zone

    Any pixel matching ``where`` is treated as part of the zone.

    Supported operators are ``==``, ``!=``, ``>``, ``>=``, ``<``, and
    ``<=``, followed by a numeric threshold such as ``">=15"`` or
    ``"!=0"``.
    """

    options: DriverOptionRules = Field(default_factory=dict)
    """Per-option rules within this zone

    The values can either be friction multipliers to apply while routing
    within this zone in that option or the keyword ``"excluded"``, which
    means the option is not allowed at all within this zone.
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce_flat_options(cls, data):
        if not isinstance(data, dict) or "options" in data:
            return data

        return {
            "layer_name": data["layer_name"],
            "where": data["where"],
            "options": {
                key: value
                for key, value in data.items()
                if key not in {"layer_name", "where"}
            },
        }

    @field_validator("where")
    @classmethod
    def _validate_where(cls, where):
        parse_comparison_values(where)
        return where

    @field_validator("options")
    @classmethod
    def _validate_options(cls, options):
        return TypeAdapter(DriverOptionRules).validate_python(options)


class DriverConfig(BaseModel, extra="forbid"):
    """Config for route-option driver rules

    Drivers adjust the relative desirability of routing options. Values
    may be numeric multipliers or the special keyword ``"excluded"`` to
    make an option unavailable.

    **The rest of this docstring is inserted by Pydantic and can be
    ignored.**
    """

    default: DriverOptionRules = Field(default_factory=dict)
    """Default per-option multipliers or exclusions applied everywhere

    Keys are names of the routing options and values are either numeric
    friction multipliers to apply while routing by default in that
    option or the keyword ``"excluded"``, which means the option is not
    allowed at all by default.
    """

    zones: list[DriverZoneConfig] = Field(default_factory=list)
    """Specific zones that override the default rules"""

    @field_validator("default")
    @classmethod
    def _validate_default(cls, default):
        return TypeAdapter(DriverOptionRules).validate_python(default)


class TransitionCostRule(BaseModel, extra="forbid"):
    """Config for one pairwise transition cost rule

    **The rest of this docstring is inserted by Pydantic and can be
    ignored.**
    """

    between: tuple[str, str]
    """Routing option names

    These two routing option names define the options between which this
    transition cost applies.
    """

    cost: float
    """The transition cost

    This is the transition cost (in $) applied when a route switches
    between the specified options.
    """


class TransitionCostsConfig(BaseModel, extra="forbid"):
    """Config for route-option transition costs

    **The rest of this docstring is inserted by Pydantic and can be
    ignored.**
    """

    default: float = 0
    """Fallback cost applied when no pairwise rule is configured"""

    pairwise: list[TransitionCostRule] = Field(default_factory=list)
    """Explicit transition costs between routing options"""


def validate_routing_options(routing_options):
    """[NOT PUBLIC API] Normalize routing options"""
    try:
        validated = _RoutingOptionsInput.model_validate(
            {"routing_options": routing_options}
        )
    except ValidationError as error:
        raise _validation_error_to_config_error(error) from error

    return validated.model_dump(exclude_none=True, exclude_unset=True)[
        "routing_options"
    ]


def validate_driver_configs(drivers, routing_options):
    """[NOT PUBLIC API] Normalize driver rules"""
    if drivers is None:
        return None

    try:
        validated = DriverConfig.model_validate(drivers)
    except ValidationError as error:
        raise _validation_error_to_config_error(error) from error

    _validate_driver_option_names(
        validated.default, routing_options, context="drivers.default"
    )
    for zone in validated.zones:
        _validate_driver_option_names(
            zone.options, routing_options, context="drivers.zones"
        )

    return _flatten_driver_config(validated)


def validate_transition_cost_configs(transition_costs, routing_options):
    """[NOT PUBLIC API] Normalize transition costs"""
    if transition_costs is None:
        return None

    try:
        validated = TransitionCostsConfig.model_validate(transition_costs)
    except ValidationError as error:
        raise _validation_error_to_config_error(error) from error

    normalized_rules = [
        {
            "between": [
                _validate_transition_option_ref(
                    rule.between[0], routing_options
                ),
                _validate_transition_option_ref(
                    rule.between[1], routing_options
                ),
            ],
            "cost": rule.cost,
        }
        for rule in validated.pairwise
    ]

    payload = validated.model_dump(exclude_none=True, exclude_unset=True)
    if "pairwise" in payload:
        payload["pairwise"] = normalized_rules
    return payload


def _flatten_driver_config(drivers):
    """Convert canonical driver config to the current runtime shape"""
    payload = drivers.model_dump(exclude_none=True, exclude_unset=True)
    if "zones" not in payload:
        return payload

    payload["zones"] = [
        {
            "layer_name": zone["layer_name"],
            "where": zone["where"],
            **zone.get("options", {}),
        }
        for zone in payload["zones"]
    ]
    return payload


def _validate_driver_option_names(option_rules, routing_option_names, context):
    """Ensure driver rules only reference known routing options"""
    known_options = set(routing_option_names)
    for option_name in option_rules:
        if option_name not in known_options:
            msg = (
                f"unknown routing option {option_name!r} in {context}. "
                f"Known options: {routing_option_names}"
            )
            raise revrtConfigurationError(msg)


def _validate_transition_option_ref(option_ref, routing_option_names):
    """Normalize a transition-cost option reference to an option name"""
    if option_ref not in routing_option_names:
        msg = (
            f"unknown routing option {option_ref!r} in transition_costs. "
            f"Known options: {routing_option_names}"
        )
        raise revrtConfigurationError(msg)
    return option_ref


def _validation_error_to_config_error(error):
    """Convert a Pydantic validation error to a configuration error"""
    first_error = error.errors()[0]
    msg = first_error.get("msg", str(error))
    if msg.startswith("Value error, "):
        msg = msg.removeprefix("Value error, ")
    return revrtConfigurationError(msg)


__all__ = [
    "DriverConfig",
    "DriverOptionRules",
    "DriverZoneConfig",
    "MultiplierLayerInput",
    "RoutingBarrierLayer",
    "RoutingCostLayer",
    "RoutingFrictionLayer",
    "RoutingOptionConfig",
    "RoutingOptionsMap",
    "TrackedLayer",
    "TransitionCostRule",
    "TransitionCostsConfig",
    "validate_driver_configs",
    "validate_routing_options",
    "validate_transition_cost_configs",
]
