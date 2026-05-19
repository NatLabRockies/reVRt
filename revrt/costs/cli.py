"""Code to build cost layer file"""

import json
import logging
from pathlib import Path
from warnings import warn

import rioxarray
import dask.config
import xarray as xr
import dask.distributed
from gaps.cli import CLICommandFromFunction

from revrt.models.cost_layers import ALL, TransmissionLayerCreationConfig
from revrt.costs.layer_creator import LayerCreator
from revrt.costs.dry_costs_creator import DryCostsCreator
from revrt.costs.masks import Masks
from revrt.utilities import (
    LayeredFile,
    close_dask_client,
    load_data_using_layer_file_profile,
    save_data_using_layer_file_profile,
    dask_performance_report,
    log_runtime,
    strip_path_keys,
    serialize_layer_build_dict,
)
from revrt.exceptions import revrtAttributeError, revrtConfigurationError
from revrt.warn import revrtWarning


logger = logging.getLogger(__name__)
CONFIG_ACTIONS = ["layers", "dry_costs", "merge_friction_and_barriers"]


def build_masks(
    land_mask_shp_fp,
    template_file,
    masks_dir,
    reproject_vector=True,
    chunks="auto",
):
    """Build masks from land vector file

    Masks are used in the cost layer creation step to determine where
    costs should be applied (e.g. wet vs dry). The land mask is the base
    mask that is created by rasterizing the input land vector file. The
    landfall mask is derived from the land mask and is a one pixel width
    line at the shore. The offshore mask is derived from the land mask
    and is the inverse of the land mask (i.e. all non-land cells are
    offshore).

    Parameters
    ----------
    land_mask_shp_fp : path-like
        Path to land polygon GPKG or shp file.
    template_file : path-like
        Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or Zarr
        (``*.zarr``) file containing the profile and transform to be
        used for the masks.
    masks_dir : path-like
        Directory to output mask GeoTIFFs.
    reproject_vector : bool, default=True
        Option to reproject CRS of input land vector file to match CRS
        of template file. By default, ``True``.
    chunks : str, optional
        Chunk size to use when reading the template file. This will be
        passed down as the ``chunks`` argument to
        :func:`rioxarray.open_rasterio` or :func:`xarray.open_dataset`.
        By default, ``"auto"``.
    """
    land_mask_shp_fp = Path(land_mask_shp_fp)
    template_file = Path(template_file)
    masks_dir = Path(masks_dir)

    if template_file.suffix == ".zarr":
        open_func = xr.open_dataset
        kwargs = {"consolidated": False, "engine": "zarr"}
    else:
        open_func = rioxarray.open_rasterio
        kwargs = {}

    with open_func(template_file, chunks=chunks, **kwargs) as fh:
        masks = Masks(
            shape=fh.rio.shape,
            crs=fh.rio.crs,
            transform=fh.rio.transform(),
            masks_dir=masks_dir,
        )

    masks.create(
        land_mask_shp_fp=land_mask_shp_fp,
        save_tiff=True,
        reproject_vector=reproject_vector,
    )


def build_routing_layer_file(  # noqa: PLR0917, PLR0913
    routing_file,
    template_file=None,
    input_layer_dir=".",
    output_tiff_dir=".",
    masks_dir=".",
    layers=None,
    dry_costs=None,
    merge_friction_and_barriers=None,
    validate_masks=False,
    max_workers=1,
    memory_limit_per_worker="auto",
    create_kwargs=None,
    log_directory=None,
):
    """Create a layered file with cost, barrier, and friction layers

    This function creates a cost layers file that is ultimately used to
    compute routes between points. The layers that are created and added
    to the file are determined based on the input config file. If the
    layered file does not already exist, it will be created based on the
    provided template file. The config file can specify three types of
    actions: building custom layers, building dry cost layers, and
    merging friction and barriers. At least one of these actions must be
    specified in the config file. See the documentation for more details
    on each type of action.

    You can re-run this function on an existing file to add new layers
    without overwriting existing layers or needing to change your
    original config.

    Parameters
    ----------
    routing_file : path-like
        Path to GeoTIFF/Zarr file to store cost layers in. If the file
        does not exist, it will be created based on the `template_file`
        input.
    template_file : path-like, optional
        Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or Zarr
        (``*.zarr``) file containing the profile and transform to be
        used for the layered costs file. If ``None``, then the
        `routing_file`  is assumed to exist on disk already.
        By default, ``None``.
    input_layer_dir : path-like, optional
        Directory to search for input layers in, if not found in
        current directory. By default, ``'.'``.
    output_tiff_dir : path-like, optional
        Directory where cost layers should be saved as GeoTIFF.
        By default, ``"."``.
    masks_dir : path-like, optional
        Directory for storing/finding mask GeoTIFFs (wet, dry, landfall,
        wet+, dry+). By default, ``"."``.
    layers : list of LayerConfig, optional
        Configuration for layers to be built and added to the file.
        At least one of `layers`, `dry_costs`, or
        `merge_friction_and_barriers` must be defined.
        By default, ``None``.
    dry_costs : DryCosts, optional
        Configuration for dry cost layers to be built and added to the
        file. At least one of `layers`, `dry_costs`, or
        `merge_friction_and_barriers` must be defined.
        By default, ``None``.
    merge_friction_and_barriers : MergeFrictionBarriers, optional
        Configuration for merging friction and barriers and adding to
        the layered costs file. At least one of `layers`, `dry_costs`,
        or `merge_friction_and_barriers` must be defined.
        By default, ``None``
    validate_masks : bool, optional
        Whether to validate that any loaded masks have appropriate
        values. This breaks the lazy (Dask) loading of the masks, so it
        it is not recommended to use this if you know your masks are
        valid. By default, ``False``.
    max_workers : int, optional
        Number of parallel workers to use for file creation. If ``None``
        or >1, processing is performed in parallel using Dask.
        By default, ``1``.
    memory_limit_per_worker : str, float, int, or None, default="auto"
        Sets the memory limit *per worker*. This only applies if
        ``max_workers != 1``. If ``None`` or ``0``, no limit is applied.
        If ``"auto"``, the total system memory is split evenly between
        the workers. If a float, that fraction of the system memory is
        used *per worker*. If a string giving a number  of bytes (like
        "1GiB"), that amount is used *per worker*. If an int, that
        number of bytes is used *per worker*. By default, ``"auto"``
    create_kwargs : dict, optional
        Additional keyword arguments to pass to
        :meth:`~revrt.utilities.handlers.LayeredFile.create_new` when
        creating a new layered file. Do not include ``template_file``;
        it will be ignored. By default, ``None``.
    log_directory : path-like, optional
        Directory to save Dask performance reports in. If ``None``, Dask
        performance reports will not be generated. By default, ``None``.
    """
    config = _validated_config(
        routing_file=routing_file,
        template_file=template_file or routing_file,
        input_layer_dir=input_layer_dir,
        output_tiff_dir=output_tiff_dir,
        masks_dir=masks_dir,
        layers=layers,
        dry_costs=dry_costs,
        merge_friction_and_barriers=merge_friction_and_barriers,
    )
    logger.debug(
        "Using dask config:\n%s", json.dumps(dask.config.config, indent=4)
    )

    client = None
    lock = None
    if max_workers != 1:
        client = dask.distributed.Client(
            n_workers=max_workers,
            memory_limit=memory_limit_per_worker,
            dashboard_address=None,
        )
        logger.info(
            "Dask client created with %s workers and %s memory limit per "
            "worker",
            max_workers,
            memory_limit_per_worker,
        )
        lock = dask.distributed.Lock("rioxarray-write")

    try:
        with (
            dask_performance_report(
                "build_routing_layer_file",
                out_dir=log_directory if max_workers != 1 else None,
            ),
            log_runtime("Building routing layers"),
        ):
            _build_routing_layer_file(
                config,
                lock,
                validate_masks=validate_masks,
                create_kwargs=create_kwargs,
            )

    finally:
        if client is not None:
            close_dask_client(client)


def _build_routing_layer_file(
    config, lock, validate_masks=False, create_kwargs=None
):
    """Build routing layers based on config file"""
    lf_handler = _create_lf_if_not_exists(
        config.routing_file, config.template_file, create_kwargs
    )

    masks = _load_masks(config, lf_handler, validate_masks=validate_masks)

    builder = LayerCreator(
        lf_handler,
        masks,
        input_layer_dir=config.input_layer_dir,
        output_tiff_dir=config.output_tiff_dir,
    )
    _build_layers(config, builder, lf_handler, lock=lock)

    if config.dry_costs is not None:
        _build_dry_costs(config, masks, lf_handler, lock=lock)

    if config.merge_friction_and_barriers is not None:
        _combine_friction_and_barriers(config, lf_handler, lock=lock)


def _validated_config(**config_dict):
    """Validate use config inputs"""
    config = TransmissionLayerCreationConfig.model_validate(config_dict)
    if not any(config.model_dump()[key] is not None for key in CONFIG_ACTIONS):
        msg = f"At least one of {CONFIG_ACTIONS!r} must be in the config file"
        raise revrtConfigurationError(msg)

    return config


def _create_lf_if_not_exists(lf_fp, template_file, create_kwargs):
    """Create layered file if it doesn't already exist"""
    lf_handler = LayeredFile(fp=lf_fp)
    if lf_handler.fp.exists():
        return lf_handler

    create_kwargs = create_kwargs or {}
    create_kwargs.pop("template_file", None)
    logger.info(
        "%s not found. Creating new layered file with kwargs:\n%r",
        lf_handler.fp,
        create_kwargs,
    )
    lf_handler.create_new(template_file=template_file, **create_kwargs)
    return lf_handler


def _load_masks(config, lf_handler, validate_masks=False):
    """Load masks based on config file"""
    masks = Masks(
        shape=lf_handler.shape,
        crs=lf_handler.profile["crs"],
        transform=lf_handler.profile["transform"],
        masks_dir=config.masks_dir,
    )
    if not config.layers:
        return masks

    build_configs = [lc.build for lc in config.layers]
    need_masks = any(
        lc.extent != ALL for bc in build_configs for lc in bc.values()
    )
    if need_masks:
        masks.load(lf_handler.fp, validate_masks)

    return masks


def _build_layers(config, builder, lf_handler, lock):
    """Build layers from config file"""
    existing_layers = set(lf_handler.data_layers)

    for lc in config.layers or []:
        layer_exists = lc.layer_name in existing_layers
        if layer_exists and _should_skip_layer(lf_handler, lc):
            continue

        builder.build(
            lc.layer_name,
            lc.build,
            values_are_costs_per_mile=lc.values_are_costs_per_mile,
            write_to_file=lc.include_in_file or layer_exists,
            description=lc.description,
            lock=lock,
        )


def _should_skip_layer(lf_handler, lc):
    """Determine whether to skip building a layer"""
    existing_attrs = lf_handler.layer_attrs(lc.layer_name)
    existing_build_config = existing_attrs.get(LayerCreator.BUILD_CONFIG_ATTR)
    cpm = existing_attrs.get(LayerCreator.CPM_CONFIG_ATTR)
    serialized_build_config = serialize_layer_build_dict(lc.build)
    should_skip = (
        existing_build_config == serialized_build_config
        and cpm == lc.values_are_costs_per_mile
    )
    if should_skip:
        logger.info(
            "Layer %r already exists in %s with matching stored "
            "config. Skipping...",
            lc.layer_name,
            lf_handler.fp,
        )
        return True

    logger.info(
        "Layer %r exists in %s but build config does not match! Rebuilding...",
        lc.layer_name,
        lf_handler.fp,
    )
    logger.debug(
        "Existing config:\n%r\nNew config:\n%r",
        existing_build_config,
        serialize_layer_build_dict,
    )
    logger.debug(
        "Existing cpm:\n%r\nNew cpm:\n%r", cpm, lc.values_are_costs_per_mile
    )
    return False


def _build_dry_costs(config, masks, lf_handler, lock):
    """Build dry costs from config file"""
    dc = config.dry_costs

    dry_mask = None
    try:
        dry_mask = masks.dry_mask
    except revrtAttributeError:
        msg = "Dry mask not found! Computing dry costs for full extent!"
        warn(msg, revrtWarning)

    dcc = DryCostsCreator(
        lf_handler,
        input_layer_dir=config.input_layer_dir,
        output_tiff_dir=config.output_tiff_dir,
    )
    cost_configs = None if not dc.cost_configs else str(dc.cost_configs)
    dcc.build(
        iso_region_tiff=dc.iso_region_tiff,
        nlcd_tiff=dc.nlcd_tiff,
        slope_tiff=dc.slope_tiff,
        transmission_config=cost_configs,
        mask=dry_mask,
        default_mults=dc.default_mults,
        extra_tiffs=dc.extra_tiffs,
        lock=lock,
    )


def _combine_friction_and_barriers(config, io_handler, lock):
    """Combine friction and barriers and save to layered file"""

    logger.info("Loading friction and raw barriers")

    merge_config = config.merge_friction_and_barriers
    friction = load_data_using_layer_file_profile(
        io_handler.fp,
        f"{merge_config.friction_layer}.tif",
        layer_dirs=[config.output_tiff_dir, config.input_layer_dir],
    )
    barriers = load_data_using_layer_file_profile(
        io_handler.fp,
        f"{merge_config.barrier_layer}.tif",
        layer_dirs=[config.output_tiff_dir, config.input_layer_dir],
    )
    combined = friction + barriers * merge_config.barrier_multiplier

    out_fp = config.output_tiff_dir / f"{merge_config.output_layer_name}.tif"
    logger.debug("Saving combined barriers to %s", out_fp)
    save_data_using_layer_file_profile(
        layer_fp=io_handler.fp, data=combined, geotiff=out_fp, lock=lock
    )

    logger.info("Writing combined barriers to layered file")
    io_handler.write_layer(combined, merge_config.output_layer_name)


def _preprocess_build_masks(config):
    """Preprocess config for build_masks command"""
    return strip_path_keys(
        config, keys_to_fix={"land_mask_shp_fp", "template_file", "masks_dir"}
    )


def _preprocess_build_routing_layer_file(config):
    """Preprocess config for build_routing_layer_file command"""
    return strip_path_keys(config, keys_to_fix={"routing_file"})


build_masks_command = CLICommandFromFunction(
    build_masks,
    name="build-masks",
    add_collect=False,
    split_keys=None,
    config_preprocessor=_preprocess_build_masks,
)

build_routing_layer_file_command = CLICommandFromFunction(
    build_routing_layer_file,
    name="build-routing-layer-file",
    add_collect=False,
    split_keys=None,
    config_preprocessor=_preprocess_build_routing_layer_file,
)
