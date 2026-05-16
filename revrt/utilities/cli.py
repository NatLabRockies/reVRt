"""revrt utilities command line interface (CLI)"""

import json
import logging
from pathlib import Path
from warnings import warn
from functools import partial

import rasterio
import rioxarray
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point, LineString
from shapely.ops import transform as shapely_transform
from gaps.cli import CLICommandFromClass, CLICommandFromFunction

from revrt.utilities import region_mapper, transform_xy, strip_path_keys
from revrt.utilities.handlers import (
    IncrementalWriter,
    LayeredFile,
    chunked_read_gpkg,
)
from revrt.exceptions import revrtValueError
from revrt.warn import revrtWarning


logger = logging.getLogger(__name__)


def layers_from_file(fp, _out_layer_dir, layers=None, profile_kwargs=None):
    """Extract layers from a layered file on disk

    Layers are output as individual GeoTIFF files.

    Parameters
    ----------
    fp : path-like
        Path to layered file on disk.
    layers : list, optional
        List of layer names to extract. Layer names must match layers in
        the `fp`, otherwise an error will be raised. If ``None``,
        extracts all layers from the
        :class:`~revrt.utilities.handlers.LayeredFile`.
        By default, ``None``.
    profile_kwargs : dict, optional
        Additional keyword arguments to pass into writing each raster.
        The following attributes ar ignored (they are set using
        properties of the source
        :class:`~revrt.utilities.handlers.LayeredFile`):

                - nodata
                - transform
                - crs
                - count
                - width
                - height

        By default, ``None``.

    Returns
    -------
    list
        List of paths to the GeoTIFF files that were created.
    """
    # TODO: Add dask client here??
    out_layer_dir = Path(_out_layer_dir)
    out_layer_dir.mkdir(parents=True, exist_ok=True)

    profile_kwargs = profile_kwargs or {}

    if layers is not None:
        layers = {layer: out_layer_dir / f"{layer}.tif" for layer in layers}
        LayeredFile(fp).extract_layers(layers, **profile_kwargs)
    else:
        layers = LayeredFile(fp).extract_all_layers(
            out_layer_dir, **profile_kwargs
        )

    return [str(layer_fp) for layer_fp in layers.values()]


def _preprocess_layers_from_file_config(config, out_dir, out_layer_dir=None):
    """Preprocess user config

    Parameters
    ----------
    config : dict
        User configuration parsed as (nested) dict.
    out_dir : path-like
        Output directory as suggested by GAPs (typically the config
        directory).
    out_layer_dir : path-like, optional
        Path to output directory into which layers should be saved as
        GeoTIFFs. This directory will be created if it does not already
        exist. If not provided, will use the config directory as output.
        By default, ``None``.
    """
    config["_out_layer_dir"] = str(out_layer_dir or out_dir)
    return strip_path_keys(config, keys_to_fix={"fp", "_out_layer_dir"})


def convert_pois_to_lines(poi_csv_f, template_f, out_f):
    """Convert POIs in CSV to lines and save in a GPKG as substations

    This function also creates a fake transmission line to connect to
    the substations to satisfy LCP code requirements.

    Parameters
    ----------
    poi_csv_f : path-like
        Path to CSV file with POIs in it.
    template_f : path-like
        Path to template raster with CRS to use for GeoPackage.
    out_f : path-like
        Path and file name for GeoPackage.
    """
    logger.debug("Converting POIs in %s to lines in %s", poi_csv_f, out_f)
    with rasterio.open(template_f) as tif:
        crs = tif.crs

    df = pd.read_csv(poi_csv_f)[
        ["POI Name", "State", "Voltage (kV)", "Lat", "Long"]
    ]

    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Long, df.Lat))
    pts = pts.set_crs("EPSG:4326")
    x, y = transform_xy("EPSG:4326", crs, df.Long, df.Lat)
    pts = pts.set_geometry(gpd.points_from_xy(x, y), crs=crs)

    # Convert points to short lines
    new_geom = []
    for pt in pts.geometry:
        end = Point(pt.x + 50, pt.y + 50)
        line = LineString([pt, end])
        new_geom.append(line)
    lines = pts.set_geometry(new_geom, crs=crs)

    # Append some fake values to make the LCP code happy
    lines["ac_cap"] = 9999999
    lines["category"] = "Substation"
    lines["voltage"] = 500  # kV
    lines["trans_gids"] = "[9999]"

    # add a fake trans line for the subs to connect to
    trans_line = pd.DataFrame(
        {
            "POI Name": "fake",
            "ac_cap": 9999999,
            "category": "TransLine",
            "voltage": 500,  # kV
            "trans_gids": None,
        },
        index=[9999],
    )

    trans_line = gpd.GeoDataFrame(trans_line)
    geo = LineString([Point(0, 0), Point(100000, 100000)])
    trans_line = trans_line.set_geometry([geo], crs=crs)

    pois = pd.concat([lines, trans_line])
    pois["gid"] = pois.index

    pois.to_file(out_f, driver="GPKG")
    logger.debug("Complete")


def map_ss_to_rr(
    features_fpath,
    regions_fpath,
    region_identifier_column,
    out_fpath,
    substation_category_name="Substation",
    minimum_substation_voltage_kv=69,
):
    """
    Map substation locations to reinforcement regions.

    Reinforcement regions are user-defined. Typical regions are
    Balancing Areas, States, or Counties, though custom regions are also
    allowed. Each region must be supplied with a unique identifier in
    the input file.

    This method also removes substations that do not meet the min 69 kV
    voltage requirement and adds {'min_volts', 'max_volts'} fields to
    the remaining substations.

    .. IMPORTANT::
        This method DOES NOT clip the substations to the reinforcement
        regions boundary. All substations will be mapped to their
        closest region. It is your responsibility to remove any
        substations outside of the analysis region before calling this
        method.

    Doing the pre-processing step avoids any issues with substations
    being left out or double counted if they were simply clipped to the
    reinforcement region shapes.

    Parameters
    ----------
    features_fpath : path-like
        Path to GeoPackage with substation and transmission features.
    regions_fpath : path-like
        Path to reinforcement regions GeoPackage.
    region_identifier_column : str
        Name of column in reinforcement regions GeoPackage
        containing a unique identifier for each region.
    out_fpath : path-like, optional
        Name for output GeoPackage file.
    substation_category_name : str, default="Substation"
        Name of category in features GeoPackage that contains
        substation features. By default, "Substation".
    minimum_substation_voltage_kv : float, default=69
        Minimum voltage (kV) that a substation must have to be
        included in the output file. By default, 69 kV.
    """

    features = gpd.read_file(features_fpath)
    features = features.rename(columns={"gid": "trans_gid"})
    substations = (
        features[features.category == substation_category_name]
        .reset_index(drop=True)
        .dropna(axis="columns", how="all")
    )

    regions = gpd.read_file(regions_fpath).to_crs(features.crs)
    logger.info(
        "Mapping %d substation locations to %d reinforcement regions",
        substations.shape[0],
        regions.shape[0],
    )

    map_func = region_mapper(regions, region_identifier_column)
    centroids = _geometry_centroids(substations)
    substations[region_identifier_column] = centroids.apply(map_func)

    logger.info("Calculating min/max voltage for each substation...")
    bad_subs = np.zeros(len(substations), dtype=bool)
    for idx, row in substations.iterrows():
        lines = row["trans_gids"]
        if isinstance(lines, str):
            lines = json.loads(lines)

        lines_mask = features["trans_gid"].isin(lines)
        voltage = features.loc[lines_mask, "voltage"].to_numpy()

        if np.max(voltage) >= minimum_substation_voltage_kv:
            substations.loc[idx, "min_volts"] = np.min(voltage)
            substations.loc[idx, "max_volts"] = np.max(voltage)
        else:
            bad_subs[idx] = True

    if any(bad_subs):
        msg = (
            "The following sub-stations do not have the minimum "
            "required voltage of 69 kV and will be dropped:\n"
            f"{substations.loc[bad_subs, 'trans_gid']}"
        )
        warn(msg, revrtWarning)
        substations = substations.loc[~bad_subs].reset_index(drop=True)

    out_fpath = Path(out_fpath).with_suffix(".gpkg")
    logger.info("Writing substation output to %s", out_fpath)
    substations.to_file(out_fpath, driver="GPKG", index=False)


def ss_from_conn(
    connections_fpath,
    out_fpath,
    region_identifier_column=None,
    batch_size=10_000,
):
    """Extract substations from connections table output by LCP.

    Substations extracted by this method can be used for reinforcement
    calculations.

    This method also does minor filtering/formatting of the input
    connections table.

    Reinforcement regions are user-defined. Typical regions are
    Balancing Areas, States, or Counties, though custom regions are also
    allowed. Each region must be supplied with a unique identifier in
    the input file.

    .. Important:: This method DOES NOT clip the substations to the
      reinforcement regions boundary. All substations will be mapped to
      their closest region. It is your responsibility to remove any
      substations outside of the analysis region before calling this
      method.

    Doing the pre-processing step avoids any issues with substations
    being left out or double counted if they were simply clipped to the
    reinforcement region shapes.

    Parameters
    ----------
    connections_fpath : path-like
        Path to GeoPackage with substation and transmission features.
    out_fpath : path-like, optional
        Name for output GeoPackage file.
    region_identifier_column : str, optional
        Name of column in reinforcement regions GeoPackage containing a
        unique identifier for each region. If ``None``, no column is
        ported. By default, ``None``.
    batch_size : int, default=10_000
        Number of records to load into memory at once when extracting
        substations. By default, ``10_000``.

    Raises
    ------
    revrtValueError
        If `batch_size` is not a positive integer or if the connections
        file does not end with '.csv' or '.gpkg'.
    """

    if batch_size <= 0:
        msg = "`batch_size` must be a positive integer"
        raise revrtValueError(msg)

    logger.info("Processing connection info in batches of %d", batch_size)
    if connections_fpath.endswith(".csv"):
        chunk_iter = pd.read_csv(connections_fpath, chunksize=batch_size)
    elif connections_fpath.endswith(".gpkg"):
        chunk_iter = chunked_read_gpkg(connections_fpath, batch_size)
    else:
        msg = (
            "Unknown file ending for features file (must be "
            f"'.csv' or '.gpkg'): {connections_fpath}"
        )
        raise revrtValueError(msg)

    logger.info("Filtering out NaN's in connection info...")
    cols = ["poi_gid", "poi_lat", "poi_lon"]
    if region_identifier_column:
        cols.append(region_identifier_column)

    out_fpath = Path(out_fpath).with_suffix(".gpkg")
    if out_fpath.exists():
        out_fpath.unlink()

    seen_keys = set()
    writer = IncrementalWriter(out_fpath)
    logger.info("Extracting substation locations...")
    for chunk in chunk_iter:
        if chunk is None:
            continue

        subset = chunk.dropna(subset=["poi_gid", "poi_lat", "poi_lon"])
        if subset.empty:
            continue

        subset = subset[cols].drop_duplicates()
        records = _deduplicate_records_across_batches(subset, cols, seen_keys)
        if not records:
            continue

        substations = _records_to_geo_df_with_points(records, cols)
        logger.debug("Writing %d substation records", len(substations))
        writer.save(substations)

    logger.info("Substation extraction complete; output at %s", out_fpath)


def _deduplicate_records_across_batches(subset, cols, seen_keys):
    """Deduplicate records across batches based on specified columns"""
    records = []
    for record in subset.to_dict("records"):
        key = tuple(record[col] for col in cols)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        records.append(record)
    return records


def _records_to_geo_df_with_points(records, cols):
    """Convert list of records to GeoDataFrame with Point geometry"""
    pois = pd.DataFrame.from_records(records, columns=cols)
    pois["poi_gid"] = pois["poi_gid"].astype("Int64")
    geometry = gpd.points_from_xy(pois["poi_lon"], pois["poi_lat"])
    return gpd.GeoDataFrame(pois, crs="epsg:4326", geometry=geometry)


def add_rr_to_nn(
    network_nodes_fpath,
    regions_fpath,
    region_identifier_column,
    out_fpath,
    crs_template_file=None,
):
    """Add reinforcement region column to network node file.

    Reinforcement regions are user-defined. Typical regions are
    Balancing Areas, States, or Counties, though custom regions are also
    allowed. Each region must be supplied with a unique identifier in
    the input file.

    Parameters
    ----------
    network_nodes_fpath : path-like
        Path to network nodes GeoPackage. The `region_identifier_column`
        will be added to this file if it is missing.
    regions_fpath : path-like
        Path to reinforcement regions GeoPackage. The
        `region_identifier_column` from this file will be added to the
        `network_nodes_fpath` if needed.
    region_identifier_column : str
        Name of column in reinforcement regions GeoPackage
        containing a unique identifier for each region.
    out_fpath : path-like
        Name for output GeoPackage file.
    crs_template_file : path-like, optional
        Path to a file containing the CRS that should be used to
        perform the mapping. This input can be an exclusions Zarr file,
        a GeoTIFF, or any file tha can be read with GeoPandas. If
        ``None``, the `network_nodes_fpath` file CRS is used to
        perform the mapping. By default, ``None``.
    """
    crs_template_file = Path(crs_template_file or network_nodes_fpath)

    logger.info("Reading in network node info...")
    network_nodes_fpath = Path(network_nodes_fpath)
    network_nodes = gpd.read_file(network_nodes_fpath)

    logger.info("Reading in CRS template...")
    if crs_template_file.suffix == ".zarr":
        with xr.open_dataset(
            crs_template_file, consolidated=False, engine="zarr"
        ) as ds:
            crs = ds.rio.crs
    elif crs_template_file.suffix in {".tif", ".tiff"}:
        with rioxarray.open_rasterio(crs_template_file) as geo:
            crs = geo.rio.crs
    else:
        crs = gpd.read_file(crs_template_file).crs

    network_nodes = _safe_to_crs(network_nodes, crs)
    regions = _safe_to_crs(gpd.read_file(regions_fpath), crs)
    if region_identifier_column in network_nodes:
        msg = (
            f"Network nodes file {str(network_nodes_fpath)!r} was specified "
            f"but it already contains the {region_identifier_column!r} "
            "column. No data modified!"
        )
        warn(msg, revrtWarning)
        return

    logger.info("Adding regions to network nodes...")
    centroids = _geometry_centroids(network_nodes)
    map_func = region_mapper(regions, region_identifier_column)
    network_nodes[region_identifier_column] = centroids.apply(map_func)

    out_fpath = Path(out_fpath).with_suffix(".gpkg")
    logger.info("Writing updated network node data to %s", out_fpath)
    network_nodes.to_file(out_fpath, driver="GPKG")


def _safe_to_crs(features, crs):
    """Project a GeoDataFrame while avoiding single-feature warnings"""

    if features.crs is None or crs is None or features.crs == crs:
        return features

    if len(features) != 1:
        return features.to_crs(crs)

    transformer = Transformer.from_crs(features.crs, crs, always_xy=True)
    geometry = features.geometry.apply(
        lambda geom: shapely_transform(transformer.transform, geom)
    )
    return features.set_geometry(geometry, crs=crs)


def _geometry_centroids(features):
    """Return centroids without geographic CRS warnings"""

    if features.crs is None or not features.crs.is_geographic:
        return features.geometry.centroid

    projected_crs = features.estimate_utm_crs() or "EPSG:3857"
    projected = _safe_to_crs(features, projected_crs)
    centroids = projected.geometry.centroid

    if len(projected) != 1:
        return centroids.to_crs(features.crs)

    transformer = Transformer.from_crs(
        projected_crs, features.crs, always_xy=True
    )
    geometry = centroids.apply(
        lambda geom: shapely_transform(transformer.transform, geom)
    )
    return geometry.set_crs(features.crs, allow_override=True)


layers_to_file_command = CLICommandFromClass(
    LayeredFile, method="layers_to_file", add_collect=False
)
layers_from_file_command = CLICommandFromFunction(
    function=layers_from_file,
    add_collect=False,
    config_preprocessor=_preprocess_layers_from_file_config,
)
convert_pois_to_lines_command = CLICommandFromFunction(
    convert_pois_to_lines,
    name="convert-pois-to-lines",
    add_collect=False,
    split_keys=None,
    config_preprocessor=partial(
        strip_path_keys, keys_to_fix={"poi_csv_f", "template_f", "out_f"}
    ),
)
map_ss_to_rr_command = CLICommandFromFunction(
    function=map_ss_to_rr,
    add_collect=False,
    config_preprocessor=partial(
        strip_path_keys,
        keys_to_fix={"features_fpath", "regions_fpath", "out_fpath"},
    ),
)
ss_from_conn_command = CLICommandFromFunction(
    function=ss_from_conn,
    add_collect=False,
    config_preprocessor=partial(
        strip_path_keys, keys_to_fix={"connections_fpath", "out_fpath"}
    ),
)
add_rr_to_nn_command = CLICommandFromFunction(
    function=add_rr_to_nn,
    add_collect=False,
    config_preprocessor=partial(
        strip_path_keys,
        keys_to_fix={"network_nodes_fpath", "regions_fpath", "out_fpath"},
    ),
)
