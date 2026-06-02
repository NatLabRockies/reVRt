"""reVRt routing CLI utilities"""

import json
import getpass
import hashlib
import logging
import os
import tempfile
import contextlib
from pathlib import Path

import xarray as xr
import rioxarray  # noqa: F401
from slugify import slugify

from revrt.exceptions import revrtFileExistsError
from revrt.utilities.handlers import ZARR_COMPRESSORS


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def routing_layer_mover(
    save,
    cost_fpath,
    out_fp,
    route_attrs,
    job_name,
    route_cl,
    route_fl,
    route_bl,
):
    """Yield temporary routing-layer path and optionally persist it

    Parameters
    ----------
    save : bool
        Whether to capture and persist routing layer output.
    cost_fpath : path-like
        Path to the source layered cost dataset used for coordinates.
    out_fp : path-like
        Output route-table path. Routing layers are written under
        ``out_fp.parent / "extra_outputs"``.
    route_attrs : dict
        Route attribute mapping. The first route entry is used to
        derive ``polarity`` and ``voltage`` labels for the output name.
    job_name : str
        Job name included in the saved routing-layer filename.
    route_cl : list
        List of dictionaries representing cost layer definitions used to
        build the output hash suffix.
    route_fl : list
        List of dictionaries representing friction layer definitions
        used to build the output hash suffix.
    route_bl : list
        List of dictionaries representing barrier layer definitions
        used to build the output hash suffix.

    Yields
    ------
    path-like or None
        Temporary zarr directory path used as the routing-layer swap
        location. When ``save`` is True the directory contents are
        also persisted to a named output under ``out_fp.parent /
        "extra_outputs"``; otherwise the temporary directory is
        removed on context exit.
    """
    scratch_dir = _make_scratch_dir()
    tfc = tempfile.TemporaryDirectory(dir=scratch_dir, suffix=".zarr")

    with tfc as temp_zarr_file_str:
        logger.debug("Setting swap file location to %r", temp_zarr_file_str)
        temp_zarr_file = Path(temp_zarr_file_str)
        yield temp_zarr_file  # noqa

        if not save:
            return

        if not temp_zarr_file.exists():
            logger.warning(
                "Routing layer output does not exist at %s",
                temp_zarr_file,
            )
            return

        polarity, voltage = _extract_batch_group(route_attrs)
        saved_fp = _persist_routing_layer_output(
            src_fp=cost_fpath,
            out_dir=Path(out_fp).parent,
            tmp_routing_layer_fp=temp_zarr_file,
            job_name=job_name,
            polarity=polarity,
            voltage=voltage,
            cost_layers=route_cl,
            friction_layers=route_fl,
            barrier_layers=route_bl,
        )
        logger.info("Saved routing layer to %s", saved_fp)


def _make_scratch_dir():
    """Try making scratch dir in $TMPDIR and /tmp/scratch/$USER"""
    with contextlib.suppress(PermissionError, FileNotFoundError):
        return _create_routing_layer_tmp_dir()

    return None


def _create_routing_layer_tmp_dir():
    """Create a temporary directory in $TMPDIR"""
    user = _get_scratch_username()
    out_dir = Path(tempfile.gettempdir()) / "scratch" / user
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _get_scratch_username():
    """Resolve a filesystem-safe username"""
    for env_name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        if user := os.environ.get(env_name):
            return _sanitize_username(user)

    with contextlib.suppress(Exception):
        if user := getpass.getuser():
            return _sanitize_username(user)

    with contextlib.suppress(Exception):
        if user := Path.home().name:
            return _sanitize_username(user)

    return "unknown-user"


def _sanitize_username(user):
    """Keep username path-safe and non-empty"""
    clean = "".join(
        char if (char.isalnum() or char in "._-") else "_"
        for char in str(user)
    ).strip("._-")
    return clean or "unknown-user"


def _extract_batch_group(route_attrs):
    """Extract polarity and voltage for route batch"""
    if not route_attrs:
        return "unknown", "unknown"

    sample = next(iter(route_attrs.values()), {})
    polarity = str(sample.get("polarity", "unknown")).lower()
    voltage = str(sample.get("voltage", "unknown")).lower()
    return polarity, voltage


def _route_layer_hash(cost_layers, friction_layers, barrier_layers):
    """Compute short hash for layer definitions"""
    payload = json.dumps(
        {
            "routing_options": {
                "default": {
                    "cost_layers": cost_layers,
                    "friction_layers": friction_layers,
                    "barrier_layers": barrier_layers,
                }
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _unique_output_path(path):
    """Keep both files by appending numeric suffix"""
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    for attempt in range(1, 100_001):
        candidate = parent / f"{stem}_{attempt}{suffix}"
        if not candidate.exists():
            return candidate

    msg = (
        "Could not find unique non-existent output path with the form "
        f"'{stem}_<i>{suffix}' in {str(parent)!r} after 100,000 attempts"
    )
    raise revrtFileExistsError(msg)


def _persist_routing_layer_output(
    src_fp,
    out_dir,
    tmp_routing_layer_fp,
    job_name,
    polarity,
    voltage,
    cost_layers,
    friction_layers,
    barrier_layers,
):
    """Save routing layer output with coordinates"""
    extra_outputs = Path(out_dir) / "extra_outputs"
    extra_outputs.mkdir(parents=True, exist_ok=True)

    layer_hash = _route_layer_hash(
        cost_layers, friction_layers, barrier_layers
    )
    base_name = (
        f"{slugify(job_name)}_"
        f"p-{slugify(polarity)}_"
        f"v-{slugify(voltage)}_"
        f"h-{layer_hash}"
    )
    out_fp = _unique_output_path(extra_outputs / f"{base_name}.zarr")

    with (
        xr.open_dataset(src_fp, engine="zarr", consolidated=False) as src_ds,
        xr.open_dataset(
            tmp_routing_layer_fp, engine="zarr", consolidated=False
        ) as tmp_ds,
    ):
        coord_axes = [
            key
            for key in ("band", "y", "x", "latitude", "longitude")
            if key in src_ds
        ]
        fixed_ds = tmp_ds.assign_coords(
            {key: src_ds[key] for key in coord_axes}
        )

        if src_ds.rio.crs is not None:
            fixed_ds = fixed_ds.rio.write_crs(src_ds.rio.crs)
        fixed_ds = fixed_ds.rio.write_transform(src_ds.rio.transform())

        fixed_ds.attrs.update(src_ds.attrs)
        fixed_ds.to_zarr(
            out_fp,
            mode="w",
            encoding=_routing_layer_zarr_encoding(fixed_ds),
            zarr_format=3,
            consolidated=False,
        )

    return out_fp


def _routing_layer_zarr_encoding(ds):
    """Build Zarr encoding for persisted routing layers"""
    encoded_names = set(ds.data_vars) | {"latitude", "longitude"}
    return {
        name: {"compressors": ZARR_COMPRESSORS}
        for name in encoded_names
        if name in ds
    }
