"""reVRt build costs CLI command"""

import logging
from pathlib import Path

from gaps.cli import CLICommandFromFunction
from gaps.config import load_config

from revrt.costs.config import parse_config
from revrt.utilities import strip_path_keys, save_data_array_to_geotiff
from revrt.routing.cli.base import update_multipliers
from revrt.routing.base import RoutingScenario, RoutingLayerManager

logger = logging.getLogger(__name__)


def build_final_routing_layers(
    lcp_config_fp, output_dir, routing_option, polarity=None, voltage=None
):
    """Build out the final routing layers based on an LCP config file

    Given an LCP config file, build out the final routing layers used by
    ``reVRt``. The routing layers include the aggregated cost layer and
    the final routing layer that is used for computing routes. The
    layers that are created and added to the file are determined based
    on the input config file. The config file can specify three types of
    actions: building custom layers, building dry cost layers, and
    merging friction and barriers. At least one of these actions must be
    specified in the config file. See the documentation for more details
    on each type of action.

    Parameters
    ----------
    lcp_config_fp : path-like
        Path to LCP config file for which the routing layer should be
        created.
    output_dir : path-like
        Path to directory where to store the outputs.
    routing_option : str
        Routing option to use when building the routing layer. This
        input is used to determine which cost and friction layers to use
        when building the routing layer. The routing option must be one
        of the options specified in the config file.
    polarity : str, optional
        Polarity to use when building the routing layer. This input is
        required if any cost or friction layers that have
        `apply_polarity_mult` set to `True` - they will have the
        appropriate multiplier applied based on this polarity.
        By default, ``None``.
    voltage : str, optional
        Voltage to use when building the routing layer. This input is
        required if any cost or friction layers that have
        `apply_row_mult` or `apply_polarity_mult` set to `True` - they
        will have the appropriate multiplier applied based on this
        voltage. By default, ``None``.

    Returns
    -------
    list
        List of paths to the GeoTIFF files that were created.
    """
    # TODO: Add dask client here??
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(lcp_config_fp)
    transmission_config = parse_config(
        config=config.get("transmission_config")
    )

    route_layer_config = config["routing_options"][routing_option]
    route_cl = update_multipliers(
        route_layer_config["cost_layers"],
        polarity,
        voltage,
        transmission_config,
    )
    route_fl = update_multipliers(
        route_layer_config.get("friction_layers") or [],
        polarity,
        voltage,
        transmission_config,
    )

    routing_scenario = RoutingScenario(
        cost_fpath=config["cost_fpath"],
        cost_layers=route_cl,
        friction_layers=route_fl,
        barrier_layers=route_layer_config.get("barrier_layers"),
        cost_multiplier_layer=config.get("cost_multiplier_layer"),
        cost_multiplier_scalar=config.get("cost_multiplier_scalar", 1),
        ignore_invalid_costs=config.get("ignore_invalid_costs", False),
    )

    rl = RoutingLayerManager(routing_scenario)
    rl.build()

    cost_out_fp = out_dir / f"{out_dir.name}_agg_costs.tif"
    logger.debug("Writing costs to %s", cost_out_fp)
    save_data_array_to_geotiff(rl.cost, cost_out_fp, nodata=-1)

    frl_out_fp = out_dir / f"{out_dir.name}_final_routing_layer.tif"
    logger.debug("Writing final routing layer to %s", frl_out_fp)
    save_data_array_to_geotiff(rl.final_routing_layer, frl_out_fp, nodata=-1)

    return [str(cost_out_fp), str(frl_out_fp)]


def _preprocess_build_final_routing_layers(
    config, project_dir, output_directory=None
):
    """Preprocess config for build_routing_layer command

    Parameters
    ----------
    config : dict
        Dictionary containing the config for the command.
    project_dir : path-like
        Path to the project directory. This is used as the default
        output directory if `output_directory` is not provided.
    output_directory : path-like, optional
        Path to directory where output files should be stored. If not
        provided, the output files will be stored in the project
        directory. The directory name will be prepended to each output
        file. By default, ``None``.

    Returns
    -------
    dict
        Updated config dictionary.
    """
    config["output_dir"] = Path(output_directory or project_dir).as_posix()
    return strip_path_keys(config, keys_to_fix={"lcp_config_fp", "output_dir"})


build_final_routing_layers_command = CLICommandFromFunction(
    build_final_routing_layers,
    name="build-final-routing-layers",
    add_collect=False,
    config_preprocessor=_preprocess_build_final_routing_layers,
    skip_doc_params=["output_dir"],
)
