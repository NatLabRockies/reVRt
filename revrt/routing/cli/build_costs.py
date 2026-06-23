"""reVRt build costs CLI command"""

import logging
from pathlib import Path

from gaps.cli import CLICommandFromFunction
from gaps.config import load_config

from revrt.costs.config import parse_config
from revrt.utilities import strip_path_keys, save_data_array_to_geotiff
from revrt.routing.cli.base import update_route_options
from revrt.routing.base import RoutingScenario, RoutingLayerManager

logger = logging.getLogger(__name__)


def build_final_routing_layers(
    lcp_config_fp, output_dir, polarity=None, voltage=None
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

    route_options = update_route_options(
        config["routing_options"], polarity, voltage, transmission_config
    )

    routing_scenario = RoutingScenario(
        cost_fpath=config["cost_fpath"],
        routing_options=route_options,
        # cost_multiplier_layer=config.get("cost_multiplier_layer"),,
        ignore_invalid_costs=config.get("ignore_invalid_costs", False),
    )

    rl = RoutingLayerManager(routing_scenario)
    rl.build()

    out_ol = []
    for option, layer in rl.costs.items():
        cost_out_fp = out_dir / f"{out_dir.name}_{option}_agg_costs.tif"
        logger.debug("Writing costs to %s", cost_out_fp)
        save_data_array_to_geotiff(layer, cost_out_fp, nodata=-1)
        out_ol.append(str(cost_out_fp))

    out_frl = []
    for option, layer in rl.final_routing_layers.items():
        final_out_fp = (
            out_dir / f"{out_dir.name}_{option}_final_routing_layer.tif"
        )
        logger.debug(
            "Writing %r final routing layer to %s", option, final_out_fp
        )
        save_data_array_to_geotiff(layer, final_out_fp, nodata=-1)
        out_frl.append(str(final_out_fp))

    return out_ol + out_frl


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
