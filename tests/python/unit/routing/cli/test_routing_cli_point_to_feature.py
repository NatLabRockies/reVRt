"""Unit tests for point-to-feature routing CLI module"""

import json
import os
import platform
from pathlib import Path

import pytest
import rioxarray  # noqa: F401
import xarray as xr
import pandas as pd
import geopandas as gpd
from gaps.utilities import TAG
from shapely.geometry import LineString
from rasterio.transform import xy

from revrt._cli import main
from revrt.exceptions import revrtConfigurationError
from revrt.warn import revrtWarning
from revrt.routing.cli.point_to_feature import (
    PointToFeatureRouteDefinitionConverter,
    _handle_route_table_and_features_input,
    _prep_config,
    compute_lcp_routes,
)


def _build_route_table(metadata, rows_cols, feature_ids):
    """Build route table DataFrame for testing"""
    records = []
    for idx, ((row, col), feat_id) in enumerate(
        zip(rows_cols, feature_ids, strict=True)
    ):
        lat = float(metadata["latitude"][row, col])
        lon = float(metadata["longitude"][row, col])
        records.append(
            {
                "route_id": f"route_{idx}",
                "start_row": row,
                "start_col": col,
                "start_lat": lat,
                "start_lon": lon,
                "end_lat": lat,
                "end_lon": lon,
                "end_feat_id": feat_id,
                "polarity": "ac",
                "voltage": 138,
            }
        )
    return pd.DataFrame.from_records(records)


def _run_route_features_cli(
    cli_runner, cli_error_message, config, tmp_path, config_name
):
    """Write config to disk and invoke the route-features CLI"""
    config_fp = tmp_path / config_name
    config_fp.write_text(json.dumps(config))

    result = cli_runner.invoke(main, ["route-features", "-c", str(config_fp)])
    msg = cli_error_message(result) or result.output
    assert result.exit_code == 0, msg
    return result


def _read_route_feature_outputs(run_dir):
    """Read route-features CSV outputs written by the CLI run"""
    output_files = sorted(Path(run_dir).glob("*_route_features*.csv"))
    assert output_files

    outputs = []
    for output_fp in output_files:
        df = pd.read_csv(output_fp)
        df = df[df["route_id"] != "route_id"].reset_index(drop=True)
        outputs.append(df)

    return outputs


def _write_route_feature_input_pairs(tmp_path, point_feature_dataset, tagged):
    """Create aligned route-table and feature file pairs for CLI tests"""
    feature_source = gpd.read_file(point_feature_dataset["features_fp"])
    pairs = [((1, 1), 1), ((2, 2), 2)]

    route_tables = []
    feature_files = []
    for idx, (row_col, feat_id) in enumerate(pairs):
        suffix = f"_{TAG}{idx}" if tagged else f"_{idx}"
        route_table_fp = tmp_path / f"route_table{suffix}.csv"
        feature_fp = tmp_path / f"mapped_features{suffix}.gpkg"

        route_table = _build_route_table(
            point_feature_dataset["metadata"], [row_col], [feat_id]
        )
        route_table.to_csv(route_table_fp, index=False)

        features = feature_source.loc[
            feature_source["end_feat_id"] == feat_id
        ].copy()
        features.to_file(feature_fp, driver="GPKG")

        route_tables.append(route_table_fp)
        feature_files.append(feature_fp)

    return route_tables, feature_files


@pytest.fixture(scope="module")
def point_feature_dataset(tmp_path_factory, revx_transmission_layers):
    """Create point-to-feature routing test dataset"""
    data_dir = tmp_path_factory.mktemp("point_to_feature_cli")

    with xr.open_dataset(
        revx_transmission_layers, consolidated=False, engine="zarr"
    ) as ds:
        transform = ds.rio.transform()
        metadata = {
            "crs": ds.rio.crs.to_string(),
            "transform": transform,
            "shape": ds.rio.shape,
            "latitude": ds["latitude"].to_numpy()[:5, :5],
            "longitude": ds["longitude"].to_numpy()[:5, :5],
        }

    features_fp = data_dir / "features.gpkg"
    cell_width = abs(transform.a)
    half_width = cell_width / 2
    feature_geoms = []
    for row, col in [(1, 1), (2, 2)]:
        x_center, y_center = xy(transform, row, col, offset="center")
        feature_geoms.append(
            LineString(
                [
                    (x_center - half_width, y_center),
                    (x_center + half_width, y_center),
                ]
            )
        )

    features = gpd.GeoDataFrame(
        {
            "end_feat_id": [1, 2],
            "category": ["north", "south"],
        },
        geometry=feature_geoms,
        crs=metadata["crs"],
    )
    features.to_file(features_fp, driver="GPKG")

    return {
        "cost_fp": revx_transmission_layers,
        "features_fp": features_fp,
        "metadata": metadata,
        "tmp_path": data_dir,
    }


def test_converter_maps_lat_lon_and_iterates(point_feature_dataset):
    """Test lat/lon mapping and iteration"""
    lat0 = float(point_feature_dataset["metadata"]["latitude"][1, 2])
    lon0 = float(point_feature_dataset["metadata"]["longitude"][1, 2])
    lat1 = float(point_feature_dataset["metadata"]["latitude"][2, 2])
    lon1 = float(point_feature_dataset["metadata"]["longitude"][2, 2])
    route_points = pd.DataFrame(
        {
            "start_lat": [lat0, lat1],
            "start_lon": [lon0, lon1],
            "end_lat": [lat0, lat1],
            "end_lon": [lon0, lon1],
            "end_feat_id": [1, 1],
            "polarity": ["ac", "ac"],
            "voltage": [138, 138],
        }
    )

    converter = PointToFeatureRouteDefinitionConverter(
        cost_fpath=point_feature_dataset["cost_fp"],
        route_points=route_points,
        features_fpath=point_feature_dataset["features_fp"],
        out_fp=point_feature_dataset["tmp_path"] / "routes.csv",
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}],
            }
        },
    )

    batches = list(converter)
    assert len(batches) == 1
    __, route_definitions, route_attrs = batches[0]
    assert len(route_definitions) == 1

    route_id, start_points, end_points = route_definitions[0]
    assert route_id == 0
    assert start_points == [
        (
            converter.route_points.iloc[0]["start_row"],
            converter.route_points.iloc[0]["start_col"],
            "default",
        ),
        (
            converter.route_points.iloc[1]["start_row"],
            converter.route_points.iloc[1]["start_col"],
            "default",
        ),
    ]
    assert end_points

    first_key = (route_id, start_points[0])
    assert route_attrs[first_key]["end_feat_id"] == 1

    tuple_repr = converter._route_as_tuple(converter.route_points.iloc[0])
    assert tuple_repr[2] == "default"
    assert tuple_repr[3] == "1"
    assert tuple_repr[4] == "ac"
    assert tuple_repr[5] == "138"


def test_converter_warns_when_feature_missing(point_feature_dataset):
    """Test warning when feature missing from features file"""
    transform = point_feature_dataset["metadata"]["transform"]
    half_width = abs(transform.a) / 2
    x_center, y_center = xy(transform, 4, 4, offset="center")
    missing_features = gpd.GeoDataFrame(
        {"end_feat_id": [5]},
        geometry=[
            LineString(
                [
                    (x_center - half_width, y_center),
                    (x_center + half_width, y_center),
                ]
            )
        ],
        crs=point_feature_dataset["metadata"]["crs"],
    )
    features_fp = point_feature_dataset["tmp_path"] / "missing_features.gpkg"
    missing_features.to_file(features_fp, driver="GPKG")

    route_points = pd.DataFrame(
        {
            "start_row": [1],
            "start_col": [1],
            "end_feat_id": [9],
            "polarity": ["dc"],
            "voltage": [230],
            "start_option": ["default"],
            "end_option": ["default"],
        }
    )

    converter = PointToFeatureRouteDefinitionConverter(
        cost_fpath=point_feature_dataset["cost_fp"],
        route_points=route_points,
        features_fpath=features_fp,
        out_fp=point_feature_dataset["tmp_path"] / "unused.csv",
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}],
            }
        },
    )

    with pytest.warns(revrtWarning, match="No features found"):
        route_definitions, route_attrs = (
            converter._convert_to_route_definitions(converter.route_points)
        )

    assert route_definitions == []
    assert route_attrs == {}


def test_compute_lcp_routes_returns_none_when_subset_empty(
    point_feature_dataset,
):
    """Test compute_lcp_routes returns ``None`` when no valid routes"""
    route_table = _build_route_table(
        point_feature_dataset["metadata"], [(1, 1)], [1]
    )
    route_table_fp = point_feature_dataset["tmp_path"] / "routes.csv"
    route_table.to_csv(route_table_fp, index=False)

    out_dir = point_feature_dataset["tmp_path"] / "empty_outputs"
    result = compute_lcp_routes(
        cost_fpath=point_feature_dataset["cost_fp"],
        route_table_fpath=route_table_fp,
        features_fpath=point_feature_dataset["features_fp"],
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}],
            }
        },
        out_dir=out_dir,
        job_name="empty",
        _split_params=(1, 1),
    )

    assert result is None
    assert out_dir.exists()


def test_compute_lcp_routes_creates_csv_output(point_feature_dataset):
    """Test compute_lcp_routes creates CSV output"""
    route_table = _build_route_table(
        point_feature_dataset["metadata"], [(1, 1), (2, 2)], [1, 2]
    )
    route_table_fp = point_feature_dataset["tmp_path"] / "routes.csv"
    route_table.to_csv(route_table_fp, index=False)

    out_dir = point_feature_dataset["tmp_path"] / "csv_outputs"
    transmission_config = {"row_width": {"138": 1.0}}
    tracked_layers = [
        {"layer_name": "tie_line_multipliers", "agg_method": "max"}
    ]

    csv_path = compute_lcp_routes(
        cost_fpath=point_feature_dataset["cost_fp"],
        route_table_fpath=route_table_fp,
        features_fpath=point_feature_dataset["features_fp"],
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}],
                "cost_multiplier_scalar": 3,
                "cost_multiplier_layer": "tie_line_multipliers",
            }
        },
        out_dir=out_dir,
        job_name="csv_run",
        transmission_config=transmission_config,
        tracked_layers=tracked_layers,
        invalid_costs_block_routing=True,
    )

    output_fp = Path(csv_path)
    assert output_fp.exists()

    df = pd.read_csv(output_fp)
    df = df[df["route_id"] != "route_id"].reset_index(drop=True)
    assert len(df) == 2
    assert set(df["end_feat_id"].unique()) == {1, 2}


def test_compute_lcp_routes_creates_geo_package_output(point_feature_dataset):
    """Test compute_lcp_routes creates GeoPackage output"""
    route_table = _build_route_table(
        point_feature_dataset["metadata"], [(1, 2)], [1]
    )
    route_table_fp = point_feature_dataset["tmp_path"] / "routes.csv"
    route_table.to_csv(route_table_fp, index=False)

    out_dir = point_feature_dataset["tmp_path"] / "gpkg_outputs"
    gpkg_path = compute_lcp_routes(
        cost_fpath=point_feature_dataset["cost_fp"],
        route_table_fpath=route_table_fp,
        features_fpath=point_feature_dataset["features_fp"],
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}],
            }
        },
        out_dir=out_dir,
        job_name="paths_run",
        save_paths=True,
    )

    output_fp = Path(gpkg_path)
    assert output_fp.exists()

    gdf = gpd.read_file(output_fp)
    assert "geometry" in gdf.columns
    assert not gdf.empty


def test_compute_lcp_routes_with_routing_options_creates_companion_gpkg(
    point_feature_dataset,
):
    """routing_options configs should emit full and per-option features"""

    route_table = _build_route_table(
        point_feature_dataset["metadata"], [(1, 2)], [1]
    )
    route_table_fp = point_feature_dataset["tmp_path"] / "routes_multi.csv"
    route_table.to_csv(route_table_fp, index=False)

    out_dir = point_feature_dataset["tmp_path"] / "gpkg_multi_outputs"
    gpkg_path = compute_lcp_routes(
        cost_fpath=point_feature_dataset["cost_fp"],
        route_table_fpath=route_table_fp,
        features_fpath=point_feature_dataset["features_fp"],
        out_dir=out_dir,
        job_name="feature_multi",
        routing_options={
            "overhead": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}]
            },
            "underground": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}]
            },
        },
        drivers={"default": {"overhead": 1, "underground": 10}},
        transition_costs={"default": 0},
        save_paths=True,
    )

    output_fp = Path(gpkg_path)
    split_fp = output_fp.with_name(f"{output_fp.stem}_routing_options.gpkg")

    assert output_fp.exists()
    assert split_fp.exists()
    assert not gpd.read_file(output_fp).empty
    split_gdf = gpd.read_file(split_fp)
    assert not split_gdf.empty
    assert set(split_gdf["routing_option"]) <= {"overhead", "underground"}


def test_compute_lcp_routes_saves_routing_layer(point_feature_dataset):
    """compute_lcp_routes should persist routing layers when requested"""

    route_table = _build_route_table(
        point_feature_dataset["metadata"], [(1, 1)], [1]
    )
    route_table_fp = point_feature_dataset["tmp_path"] / "routes_saved.csv"
    route_table.to_csv(route_table_fp, index=False)

    out_dir = point_feature_dataset["tmp_path"] / "saved_layer_outputs"
    csv_path = compute_lcp_routes(
        cost_fpath=point_feature_dataset["cost_fp"],
        route_table_fpath=route_table_fp,
        features_fpath=point_feature_dataset["features_fp"],
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}],
            }
        },
        out_dir=out_dir,
        job_name="feature_save_layer",
        save_routing_layer=True,
    )

    assert Path(csv_path).exists()

    extra_outputs = out_dir / "extra_outputs"
    saved_layers = sorted(extra_outputs.glob("*.zarr"))
    assert saved_layers

    with xr.open_dataset(
        saved_layers[0], engine="zarr", consolidated=False
    ) as ds:
        assert "cost" in ds
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords


def test_pipeline_inputs_are_split_and_aligned_by_shared_tag(monkeypatch):
    """Pipeline inputs should align CSVs and GPKGs by shared tag"""

    files = [
        "/tmp/route_table_j0.csv",  # noqa
        "/tmp/mapped_features_j0.gpkg",  # noqa
        "/tmp/route_table_j10.csv",  # noqa
        "/tmp/mapped_features_j10.gpkg",  # noqa
    ]

    monkeypatch.setattr(
        "revrt.routing.cli.point_to_feature.parse_previous_status",
        lambda *_args, **_kwargs: files,
    )

    config = {
        "route_table_fpath": "PIPELINE",
        "features_fpath": "PIPELINE",
    }

    result = _handle_route_table_and_features_input(
        config,
        "/tmp/project",  # noqa
        "route-features",
    )

    assert result["route_table_fpath"] == [
        "/tmp/route_table_j0.csv",  # noqa
        "/tmp/route_table_j10.csv",  # noqa
    ]
    assert result["features_fpath"] == [
        "/tmp/mapped_features_j0.gpkg",  # noqa
        "/tmp/mapped_features_j10.gpkg",  # noqa
    ]


def test_prep_config_normalizes_non_pipeline_paths():
    """Preprocessor should strip and wrap non-pipeline path inputs"""

    config = {
        "cost_fpath": "  /tmp/cost_layers.zarr  ",
        "route_table_fpath": "  /tmp/routes.csv  ",
        "features_fpath": "  /tmp/features.gpkg  ",
        "out_dir": "  /tmp/output_dir  ",
    }

    result = _prep_config(config, 1, "/tmp/project", "route-features")  # noqa

    assert result["cost_fpath"] == "/tmp/cost_layers.zarr"  # noqa
    assert result["route_table_fpath"] == ["/tmp/routes.csv"]  # noqa
    assert result["features_fpath"] == ["/tmp/features.gpkg"]  # noqa
    assert result["out_dir"] == "/tmp/output_dir"  # noqa
    assert result["_split_params"] == [(0, 1)]


def test_prep_config_normalizes_non_pipeline_sequences():
    """Preprocessor should strip whitespace from sequence path inputs"""

    config = {
        "route_table_fpath": ["  /tmp/routes_a.csv  ", " /tmp/routes_b.csv"],
        "features_fpath": [
            "  /tmp/features_a.gpkg  ",
            " /tmp/features_b.gpkg",
        ],
    }

    result = _prep_config(config, 1, "/tmp/project", "route-features")  # noqa

    assert result["route_table_fpath"] == [
        "/tmp/routes_a.csv",  # noqa
        "/tmp/routes_b.csv",  # noqa
    ]
    assert result["features_fpath"] == [
        "/tmp/features_a.gpkg",  # noqa
        "/tmp/features_b.gpkg",  # noqa
    ]


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            {
                "route_table_fpath": ["/tmp/routes.csv"],  # noqa
                "features_fpath": "/tmp/features.gpkg",  # noqa
            },
            "must both be sequences or both be strings",
        ),
        (
            {
                "route_table_fpath": [
                    "/tmp/routes_a.csv",  # noqa
                    "/tmp/routes_b.csv",  # noqa
                ],
                "features_fpath": ["/tmp/features_a.gpkg"],  # noqa
            },
            "must be the same length",
        ),
    ],
)
def test_non_pipeline_inputs_require_matching_shapes(config, match):
    """Non-pipeline paths must have matching string/sequence shapes"""

    with pytest.raises(revrtConfigurationError, match=match):
        _handle_route_table_and_features_input(
            config,
            "/tmp/project",  # noqa
            "route-features",
        )


def test_pipeline_inputs_require_both_pipeline_sentinels():
    """Pipeline mode requires both route-table and feature sentinels"""

    config = {
        "route_table_fpath": "PIPELINE",
        "features_fpath": "/tmp/features.gpkg",  # noqa
    }

    with pytest.raises(
        revrtConfigurationError,
        match="must be set to 'PIPELINE' for pipeline runs",
    ):
        _handle_route_table_and_features_input(
            config,
            "/tmp/project",  # noqa
            "route-features",
        )


@pytest.mark.parametrize(
    ("files", "match"),
    [
        (
            [
                "/tmp/route_table_j0.csv",  # noqa
                "/tmp/route_table_j1.csv",  # noqa
                "/tmp/mapped_features_j0.gpkg",  # noqa
                "/tmp/mapped_features_j9.gpkg",  # noqa
            ],
            "Could not align pipeline route-table CSV outputs",
        ),
        (
            [
                "/tmp/route_table.csv",  # noqa
                "/tmp/route_table_j1.csv",  # noqa
                "/tmp/mapped_features_j0.gpkg",  # noqa
                "/tmp/mapped_features_j1.gpkg",  # noqa
            ],
            "ambiguously tagged",
        ),
    ],
)
def test_pipeline_inputs_validate_previous_outputs(monkeypatch, files, match):
    """Pipeline mode should reject ambiguous or misaligned outputs"""

    monkeypatch.setattr(
        "revrt.routing.cli.point_to_feature.parse_previous_status",
        lambda *_args, **_kwargs: files,
    )

    config = {
        "route_table_fpath": "PIPELINE",
        "features_fpath": "PIPELINE",
    }

    with pytest.raises(revrtConfigurationError, match=match):
        _handle_route_table_and_features_input(
            config,
            "/tmp/project",  # noqa
            "route-features",
        )


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_route_features_cli_executes(
    run_gaps_cli_with_expected_file, point_feature_dataset
):
    """Test point-to-feature routing CLI execution"""
    route_table = _build_route_table(
        point_feature_dataset["metadata"], [(1, 1)], [1]
    )
    route_table_fp = point_feature_dataset["tmp_path"] / "routes.csv"
    route_table.to_csv(route_table_fp, index=False)

    out_dir = point_feature_dataset["tmp_path"] / "cli_outputs"
    config = {
        "cost_fpath": str(point_feature_dataset["cost_fp"]),
        "route_table_fpath": str(route_table_fp),
        "features_fpath": str(point_feature_dataset["features_fp"]),
        "routing_options": {
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}],
            }
        },
        "out_dir": str(out_dir),
        "job_name": "cli_run",
        "save_paths": False,
    }

    out_fp = run_gaps_cli_with_expected_file(
        "route-features", config, point_feature_dataset["tmp_path"]
    )

    df = pd.read_csv(out_fp)
    df = df[df["route_id"] != "route_id"].reset_index(drop=True)
    assert len(df) == 1


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_route_features_cli_strips_required_path_whitespace(
    run_gaps_cli_with_expected_file, point_feature_dataset, tmp_path
):
    """route-features CLI strips whitespace on required path inputs"""

    route_table = _build_route_table(
        point_feature_dataset["metadata"], [(1, 1)], [1]
    )
    route_table_fp = point_feature_dataset["tmp_path"] / "trimmed_routes.csv"
    route_table.to_csv(route_table_fp, index=False)

    config = {
        "cost_fpath": f"  {point_feature_dataset['cost_fp']}  ",
        "route_table_fpath": f"  {route_table_fp}  ",
        "features_fpath": f"  {point_feature_dataset['features_fp']}  ",
        "routing_options": {
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}],
            }
        },
        "save_paths": False,
    }
    out_fp = run_gaps_cli_with_expected_file(
        "route-features", config, tmp_path
    )

    assert Path(out_fp).exists()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_route_features_cli_accepts_string_input_paths(
    cli_runner, cli_error_message, point_feature_dataset, tmp_path
):
    """route-features CLI should accept direct string path inputs"""

    route_table = _build_route_table(
        point_feature_dataset["metadata"], [(1, 1)], [1]
    )
    route_table_fp = tmp_path / "string_route_table.csv"
    route_table.to_csv(route_table_fp, index=False)

    out_dir = tmp_path / "string_outputs"
    config = {
        "cost_fpath": str(point_feature_dataset["cost_fp"]),
        "route_table_fpath": str(route_table_fp),
        "features_fpath": str(point_feature_dataset["features_fp"]),
        "routing_options": {
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}],
            }
        },
        "out_dir": str(out_dir),
        "job_name": "string_cli_run",
        "save_paths": False,
    }

    _run_route_features_cli(
        cli_runner,
        cli_error_message,
        config,
        tmp_path,
        "route_features_string_config.json",
    )

    outputs = _read_route_feature_outputs(tmp_path)
    assert len(outputs) == 1
    assert len(outputs[0]) == 1
    assert outputs[0]["end_feat_id"].tolist() == [1]


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_route_features_cli_accepts_list_input_paths(
    cli_runner, cli_error_message, point_feature_dataset, tmp_path
):
    """route-features CLI should accept aligned list path inputs"""

    route_tables, feature_files = _write_route_feature_input_pairs(
        tmp_path, point_feature_dataset, tagged=False
    )

    out_dir = tmp_path / "list_outputs"
    config = {
        "cost_fpath": str(point_feature_dataset["cost_fp"]),
        "route_table_fpath": [str(fp) for fp in route_tables],
        "features_fpath": [str(fp) for fp in feature_files],
        "routing_options": {
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}],
            }
        },
        "out_dir": str(out_dir),
        "job_name": "list_cli_run",
        "save_paths": False,
    }

    _run_route_features_cli(
        cli_runner,
        cli_error_message,
        config,
        tmp_path,
        "route_features_list_config.json",
    )

    outputs = _read_route_feature_outputs(tmp_path)
    assert len(outputs) == 2
    assert sorted(len(df) for df in outputs) == [1, 1]

    end_feat_ids = sorted(
        feat_id for df in outputs for feat_id in df["end_feat_id"].tolist()
    )
    assert end_feat_ids == [1, 2]


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_route_features_cli_accepts_pipeline_inputs(
    cli_runner,
    cli_error_message,
    monkeypatch,
    point_feature_dataset,
    tmp_path,
):
    """route-features CLI should resolve pipeline route-table inputs"""

    route_tables, feature_files = _write_route_feature_input_pairs(
        tmp_path, point_feature_dataset, tagged=True
    )
    previous_outputs = [
        str(fp)
        for pair in zip(route_tables, feature_files, strict=True)
        for fp in pair
    ]

    monkeypatch.setattr(
        "revrt.routing.cli.point_to_feature.parse_previous_status",
        lambda *_args, **_kwargs: previous_outputs,
    )

    out_dir = tmp_path / "pipeline_outputs"
    config = {
        "cost_fpath": str(point_feature_dataset["cost_fp"]),
        "route_table_fpath": "PIPELINE",
        "features_fpath": "PIPELINE",
        "routing_options": {
            "default": {
                "cost_layers": [{"layer_name": "tie_line_costs_400MW"}],
            }
        },
        "out_dir": str(out_dir),
        "job_name": "pipeline_cli_run",
        "save_paths": False,
    }

    _run_route_features_cli(
        cli_runner,
        cli_error_message,
        config,
        tmp_path,
        "route_features_pipeline_config.json",
    )

    outputs = _read_route_feature_outputs(tmp_path)
    assert len(outputs) == 2
    assert sorted(len(df) for df in outputs) == [1, 1]

    end_feat_ids = sorted(
        feat_id for df in outputs for feat_id in df["end_feat_id"].tolist()
    )
    assert end_feat_ids == [1, 2]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
