"""reVRt finalize CLI command unit tests"""

import os
import platform
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import LineString, Polygon
from rasterio.transform import xy

from revrt.constants import (
    SHORT_CUTOFF,
    MEDIUM_CUTOFF,
    SHORT_MULT,
    MEDIUM_MULT,
)
from revrt.routing.cli.finalize import RoutePostProcessor, RouteToFeatureMapper
from revrt.exceptions import revrtFileNotFoundError, revrtValueError


@pytest.fixture
def sample_cost_surface(tmp_path, revx_transmission_layers):
    """Subset transmission surface for finalize mapping tests"""

    cost_fp = tmp_path / "cost_subset.zarr"
    with xr.open_dataset(
        revx_transmission_layers, consolidated=False, engine="zarr"
    ) as ds:
        subset = ds.isel(y=slice(0, 4), x=slice(0, 4))
        subset.to_zarr(cost_fp, mode="w", zarr_format=3, consolidated=False)
        transform = subset.rio.transform()
        crs = subset.rio.crs

    return {
        "cost_fp": cost_fp,
        "transform": transform,
        "crs": crs,
        "cell_x": abs(transform.a),
        "cell_y": abs(transform.e),
    }


@pytest.fixture
def transmission_features(tmp_path, sample_cost_surface):
    """Create GeoPackage features overlapping known grid cells"""

    transform = sample_cost_surface["transform"]
    half_x = sample_cost_surface["cell_x"] / 2
    half_y = sample_cost_surface["cell_y"] / 2

    def _cell_polygon(row, col):
        x_coord, y_coord = xy(transform, row, col)
        return Polygon(
            [
                (x_coord - half_x, y_coord - half_y),
                (x_coord + half_x, y_coord - half_y),
                (x_coord + half_x, y_coord + half_y),
                (x_coord - half_x, y_coord + half_y),
            ]
        )

    features = gpd.GeoDataFrame(
        {
            "trans_gid": [101, 101, 202],
            "a_test_col": ["a", "a", "c"],
            "another_test_col": [40, 40, 60],
        },
        geometry=[
            _cell_polygon(1, 1),
            _cell_polygon(1, 2),
            _cell_polygon(2, 2),
        ],
        crs=sample_cost_surface["crs"],
    )

    fp = tmp_path / "transmission_features.gpkg"
    features.to_file(fp, driver="GPKG")
    return {"fp": fp, "matches": {(1, 2): 101, (2, 2): 202}}


def test_merge_routes_no_files(tmp_path):
    """RoutePostProcessor should raise when no files match collect pattern"""
    with pytest.raises(
        revrtFileNotFoundError, match="No files found using collect pattern:"
    ):
        RoutePostProcessor(
            collect_pattern="dne*.csv",
            project_dir=tmp_path,
            out_dir=tmp_path,
            job_name="test",
        ).process()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_finalize_routes_merges_csv(
    run_gaps_cli_with_expected_file, tmp_path
):
    """finalize-routes CLI should merge CSV chunk outputs"""

    chunk_dir = tmp_path / "outputs"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_frames = [
        pd.DataFrame(
            {
                "route_id": ["chunk0_route0", "chunk0_route1"],
                "start_row": [0, 1],
                "start_col": [0, 1],
                "end_row": [2, 3],
                "end_col": [2, 3],
                "cost": [10.0, 20.0],
            }
        ),
        pd.DataFrame(
            {
                "route_id": ["chunk1_route0", "chunk1_route1"],
                "start_row": [4, 5],
                "start_col": [4, 5],
                "end_row": [6, 7],
                "end_col": [6, 7],
                "cost": [30.0, 40.0],
            }
        ),
    ]

    for idx, frame in enumerate(chunk_frames):
        frame.to_csv(chunk_dir / f"routes_part_{idx}.csv", index=False)

    chunk_fp = chunk_dir / "routes_part_999.csv"
    pd.DataFrame(columns=["route_id"]).to_csv(chunk_fp, index=False)

    config = {
        "collect_pattern": "outputs/routes_part_*.csv",
        "chunk_size": 1,
    }

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes", config, tmp_path
    )

    merged_df = pd.read_csv(merged_fp)
    expected_df = pd.concat(chunk_frames, ignore_index=True)
    pd.testing.assert_frame_equal(
        merged_df.sort_values("route_id").reset_index(drop=True),
        expected_df.sort_values("route_id").reset_index(drop=True),
    )

    chunk_files_dir = tmp_path / "chunk_files"
    assert chunk_files_dir.exists()

    for idx in range(len(chunk_frames)):
        original_fp = chunk_dir / f"routes_part_{idx}.csv"
        relocated_fp = chunk_files_dir / f"routes_part_{idx}.csv"
        assert not original_fp.exists()
        assert relocated_fp.exists()

    original_fp = chunk_dir / "routes_part_999.csv"
    relocated_fp = chunk_files_dir / "routes_part_999.csv"
    assert not original_fp.exists()
    assert relocated_fp.exists()


@pytest.mark.parametrize("tol", [None, 0.01])
@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_finalize_routes_merges_gpkg(
    run_gaps_cli_with_expected_file, tmp_path, tol
):
    """finalize-routes CLI should merge GeoPackage chunk outputs"""

    chunk_dir = tmp_path / "geoms"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_geometries = []
    for idx in range(2):
        gdf = gpd.GeoDataFrame(
            {
                "route_id": [
                    f"chunk{idx}_route0",
                    f"chunk{idx}_route1",
                ],
                "start_row": [idx, idx + 10],
                "start_col": [idx, idx + 10],
                "end_row": [idx + 20, idx + 30],
                "end_col": [idx + 20, idx + 30],
            },
            geometry=[
                LineString(
                    [
                        (idx, idx),
                        (idx + 0.05, idx + 0.02),
                        (idx + 1.0, idx + 0.8),
                    ]
                ),
                LineString(
                    [
                        (idx + 1.0, idx),
                        (idx + 1.2, idx + 0.3),
                        (idx + 2.0, idx + 1.0),
                    ]
                ),
            ],
            crs="EPSG:4326",
        )

        chunk_fp = chunk_dir / f"segment_{idx}.gpkg"
        gdf.to_file(chunk_fp, driver="GPKG")
        chunk_geometries.append(gdf)

    chunk_fp = chunk_dir / "segment_999.gpkg"
    gpd.GeoDataFrame(columns=["route_id"]).to_file(chunk_fp, driver="GPKG")

    config = {
        "collect_pattern": "geoms/segment_*.gpkg",
        "chunk_size": 1,
        "simplify_geo_tolerance": tol,
        "purge_chunks": True,
    }

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes", config, tmp_path
    )

    merged_gdf = gpd.read_file(merged_fp)
    expected_count = sum(len(gdf) for gdf in chunk_geometries)
    assert len(merged_gdf) == expected_count
    assert set(merged_gdf["route_id"]) == {
        "chunk0_route0",
        "chunk0_route1",
        "chunk1_route0",
        "chunk1_route1",
    }
    assert all(geom.geom_type == "LineString" for geom in merged_gdf.geometry)

    for idx in range(len(chunk_geometries)):
        assert not (chunk_dir / f"segment_{idx}.gpkg").exists()
    assert not (chunk_dir / "segment_999.gpkg").exists()
    assert not (chunk_dir / "chunk_files").exists()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_finalize_single_csv(run_gaps_cli_with_expected_file, tmp_path):
    """finalize-routes CLI should accept single CSV chunk"""

    chunk_dir = tmp_path / "outputs"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "route_id": ["chunk0_route0", "chunk0_route1"],
            "start_row": [0, 1],
            "start_col": [0, 1],
            "end_row": [2, 3],
            "end_col": [2, 3],
            "cost": [10.0, 20.0],
        }
    )

    frame.to_csv(chunk_dir / "routes.csv", index=False)

    config = {"collect_pattern": "outputs/routes.csv"}

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes", config, tmp_path
    )

    merged_df = pd.read_csv(merged_fp)
    pd.testing.assert_frame_equal(
        merged_df.sort_values("route_id").reset_index(drop=True),
        frame.sort_values("route_id").reset_index(drop=True),
    )


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_finalize_strips_required_path_whitespace(
    run_gaps_cli_with_expected_file, tmp_path
):
    """finalize-routes CLI strips whitespace on required path inputs"""

    chunk_dir = tmp_path / "trimmed_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "route_id": ["chunk0_route0"],
            "start_row": [0],
            "start_col": [0],
            "end_row": [2],
            "end_col": [2],
            "cost": [10.0],
        }
    ).to_csv(chunk_dir / "routes.csv", index=False)

    config = {
        "collect_pattern": "  trimmed_chunks/routes.csv  ",
        "project_dir": f"  {tmp_path}  ",
    }

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes",
        config,
        tmp_path,
        glob_pattern="*_finalize_routes.*",
    )

    merged_df = pd.read_csv(merged_fp)
    assert merged_df["route_id"].tolist() == ["chunk0_route0"]


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_finalize_min_length(run_gaps_cli_with_expected_file, tmp_path):
    """CLI finalize should enforce minimum lengths and rescale cost"""

    chunk_dir = tmp_path / "min_length_cli"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "route_id": ["very_short", "on_threshold"],
            "start_row": [0, 1],
            "start_col": [0, 1],
            "end_row": [2, 3],
            "end_col": [2, 3],
            "length_km": [0.5, 2.0],
            "cost": [5.0, 30.0],
        }
    ).to_csv(chunk_dir / "routes_part_0.csv", index=False)

    config = {
        "collect_pattern": "min_length_cli/routes_part_*.csv",
        "chunk_size": 1,
        "min_line_length": 2.0,
    }

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes", config, tmp_path
    )

    merged_df = pd.read_csv(merged_fp).set_index("route_id").sort_index()

    assert merged_df.loc["very_short", "length_km"] == pytest.approx(2.0)
    assert merged_df.loc["very_short", "cost"] == pytest.approx(20.0)
    assert merged_df.loc["on_threshold", "length_km"] == pytest.approx(2.0)
    assert merged_df.loc["on_threshold", "cost"] == pytest.approx(30.0)


def test_process_csv_applies_linear_length_multiplier(tmp_path):
    """Linear length multipliers should adjust cost and add raw_cost"""

    chunk_dir = tmp_path / "segments"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    original = pd.DataFrame(
        {
            "route_id": [
                "linear_long",
                "linear_medium",
                "linear_short",
            ],
            "start_row": [0, 2, 4],
            "start_col": [1, 3, 5],
            "end_row": [10, 12, 14],
            "end_col": [11, 13, 15],
            "length_km": [
                MEDIUM_CUTOFF + 1,
                MEDIUM_CUTOFF - 1,
                SHORT_CUTOFF / 2,
            ],
            "cost": [300.0, 200.0, 100.0],
        }
    )

    original.to_csv(chunk_dir / "routes.csv", index=False)

    processor = RoutePostProcessor(
        collect_pattern="segments/routes.csv",
        project_dir=tmp_path,
        job_name="linear",
        length_mult_kind="linear",
        chunk_size=2,
    )

    out_fp = Path(processor.process())
    processed = pd.read_csv(out_fp)

    slope = (1 - SHORT_MULT) / (MEDIUM_CUTOFF - SHORT_CUTOFF / 2)
    processed = processed.set_index("route_id").sort_index()
    original = original.set_index("route_id").sort_index()

    expected_mult = [
        1,
        1 - slope,
        1 - slope * (MEDIUM_CUTOFF - SHORT_CUTOFF / 2),
    ]

    assert np.allclose(processed["length_mult"], expected_mult)
    assert np.allclose(processed["raw_cost"].values, original["cost"])
    scaled = original["cost"] * expected_mult
    assert np.allclose(processed["cost"], scaled)


def test_process_csv_applies_step_length_multiplier(tmp_path):
    """Step multipliers should apply configured bucketed adjustments"""

    chunk_dir = tmp_path / "step_segments"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    original = pd.DataFrame(
        {
            "route_id": [
                "step_short",
                "step_medium",
                "step_long",
            ],
            "start_row": [0, 2, 4],
            "start_col": [1, 3, 5],
            "end_row": [6, 8, 10],
            "end_col": [7, 9, 11],
            "length_km": [
                SHORT_CUTOFF - 0.1,
                MEDIUM_CUTOFF - 0.1,
                MEDIUM_CUTOFF + 1,
            ],
            "cost": [50.0, 60.0, 70.0],
        }
    )

    original.to_csv(chunk_dir / "routes.csv", index=False)

    processor = RoutePostProcessor(
        collect_pattern="step_segments/routes.csv",
        project_dir=tmp_path,
        job_name="step",
        length_mult_kind="step",
        chunk_size=3,
    )

    out_fp = Path(processor.process())
    processed = pd.read_csv(out_fp).set_index("route_id").sort_index()

    original = original.set_index("route_id").sort_index()
    expected_mult = {
        "step_long": 1.0,
        "step_medium": MEDIUM_MULT,
        "step_short": SHORT_MULT,
    }

    for route_id, expected in expected_mult.items():
        assert processed.loc[route_id, "length_mult"] == pytest.approx(
            expected
        )
        assert processed.loc[route_id, "raw_cost"] == pytest.approx(
            original.loc[route_id, "cost"]
        )
        assert processed.loc[route_id, "cost"] == pytest.approx(
            original.loc[route_id, "cost"] * expected
        )


def test_process_csv_enforces_min_length(tmp_path):
    """Minimum line length floor should adjust length and scale cost"""

    chunk_dir = tmp_path / "min_length_csv"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    original = pd.DataFrame(
        {
            "route_id": ["short", "long"],
            "length_km": [1.5, 5.5],
            "cost": [100.0, 200.0],
        }
    )

    original.to_csv(chunk_dir / "routes.csv", index=False)

    processor = RoutePostProcessor(
        collect_pattern="min_length_csv/routes.csv",
        project_dir=tmp_path,
        job_name="min_length",
        min_line_length=3.0,
        chunk_size=2,
    )

    out_fp = Path(processor.process())

    processed = pd.read_csv(out_fp).set_index("route_id").sort_index()
    original = original.set_index("route_id").sort_index()

    assert processed.loc["short", "length_km"] == pytest.approx(3.0)
    assert processed.loc["short", "cost"] == pytest.approx(200.0)
    assert processed.loc["long", "length_km"] == pytest.approx(
        original.loc["long", "length_km"]
    )
    assert processed.loc["long", "cost"] == pytest.approx(
        original.loc["long", "cost"]
    )


def test_collect_geo_files_apply_length_multiplier(tmp_path):
    """GeoPackage collection should apply length multipliers"""

    chunk_dir = tmp_path / "geo_segments"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.GeoDataFrame(
        {
            "route_id": ["geo_short", "geo_long"],
            "start_row": [0, 1],
            "start_col": [0, 1],
            "end_row": [2, 3],
            "end_col": [2, 3],
            "length_km": [SHORT_CUTOFF - 0.1, MEDIUM_CUTOFF + 1],
            "cost": [80.0, 120.0],
        },
        geometry=[
            LineString([(0, 0), (0.5, 0.5)]),
            LineString([(1, 1), (1.5, 1.5)]),
        ],
        crs="EPSG:4326",
    )

    chunk_fp = chunk_dir / "segment.gpkg"
    gdf.to_file(chunk_fp, driver="GPKG")

    processor = RoutePostProcessor(
        collect_pattern="geo_segments/*.gpkg",
        project_dir=tmp_path,
        job_name="geo_merged",
        length_mult_kind="step",
        chunk_size=1,
    )

    out_fp = Path(processor.process())
    processed = gpd.read_file(out_fp).set_index("route_id").sort_index()
    original = gdf.set_index("route_id").sort_index()

    expected_mult = {
        "geo_long": 1.0,
        "geo_short": SHORT_MULT,
    }

    for route_id, multiplier in expected_mult.items():
        assert processed.loc[route_id, "length_mult"] == pytest.approx(
            multiplier
        )
        assert processed.loc[route_id, "raw_cost"] == pytest.approx(
            original.loc[route_id, "cost"]
        )
        assert processed.loc[route_id, "cost"] == pytest.approx(
            original.loc[route_id, "cost"] * multiplier
        )


def test_collect_geo_files_enforces_min_length(tmp_path):
    """GeoPackage collection should enforce minimum segment lengths"""

    chunk_dir = tmp_path / "geo_min_length"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.GeoDataFrame(
        {
            "route_id": ["short_geo", "long_geo"],
            "length_km": [2.0, 4.0],
            "cost": [10.0, 20.0],
        },
        geometry=[
            LineString([(0, 0), (0.02, 0.01)]),
            LineString([(1, 1), (1.04, 1.03)]),
        ],
        crs="EPSG:4326",
    )

    gdf.to_file(chunk_dir / "segment.gpkg", driver="GPKG")

    processor = RoutePostProcessor(
        collect_pattern="geo_min_length/*.gpkg",
        project_dir=tmp_path,
        job_name="geo_min",
        min_line_length=5.0,
        chunk_size=1,
    )

    out_fp = Path(processor.process())
    processed = gpd.read_file(out_fp).set_index("route_id").sort_index()

    assert processed.loc["short_geo", "length_km"] == pytest.approx(5.0)
    assert processed.loc["short_geo", "cost"] == pytest.approx(25.0)
    assert processed.loc["long_geo", "length_km"] == pytest.approx(5.0)
    assert processed.loc["long_geo", "cost"] == pytest.approx(25.0)


def test_process_csv_requires_length_column(tmp_path):
    """Length multipliers should error if length_km is missing"""

    chunk_dir = tmp_path / "missing_length"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "route_id": ["no_length"],
            "cost": [10.0],
        }
    )

    frame.to_csv(chunk_dir / "routes.csv", index=False)

    with pytest.raises(revrtValueError, match="length_km"):
        RoutePostProcessor(
            collect_pattern="missing_length/routes.csv",
            project_dir=tmp_path,
            job_name="invalid",
            length_mult_kind="linear",
        ).process()


def test_process_csv_requires_length_for_min_floor(tmp_path):
    """Minimum line length enforcement should require length_km column"""

    chunk_dir = tmp_path / "min_length_missing"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "route_id": ["no_length"],
            "cost": [10.0],
        }
    ).to_csv(chunk_dir / "routes.csv", index=False)

    with pytest.raises(revrtValueError, match="length_km"):
        RoutePostProcessor(
            collect_pattern="min_length_missing/routes.csv",
            project_dir=tmp_path,
            job_name="invalid_min_length",
            min_line_length=1.0,
        ).process()


def test_process_csv_rejects_unknown_length_kind(tmp_path):
    """Invalid length multiplier kind should raise an error"""

    chunk_dir = tmp_path / "unknown_kind"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "route_id": ["unknown"],
            "length_km": [SHORT_CUTOFF],
            "cost": [15.0],
        }
    )

    frame.to_csv(chunk_dir / "routes.csv", index=False)

    with pytest.raises(
        revrtValueError, match="Unknown length computation kind"
    ):
        RoutePostProcessor(
            collect_pattern="unknown_kind/routes.csv",
            project_dir=tmp_path,
            job_name="invalid_kind",
            length_mult_kind="quadratic",
        ).process()


def test_file_suffix_requires_matching_types(tmp_path):
    """Process should raise when collect pattern spans multiple suffixes"""

    (tmp_path / "mixed").mkdir()
    (tmp_path / "mixed/routes.csv").write_text("route_id,cost\nfoo,1\n")
    (tmp_path / "mixed/routes.txt").write_text("route_id,cost\nbar,2\n")

    processor = RoutePostProcessor(
        collect_pattern="mixed/routes.*",
        project_dir=tmp_path,
        job_name="mixed",
    )

    with pytest.raises(revrtValueError, match="Multiple file types"):
        processor.process()


def test_route_to_feature_mapper_maps_features(
    sample_cost_surface, transmission_features
):
    """Mapper should attach transmission IDs for reachable endpoints"""

    start_x, start_y = xy(sample_cost_surface["transform"], 0, 0)
    first_target = xy(sample_cost_surface["transform"], 1, 2)
    second_target = xy(sample_cost_surface["transform"], 2, 2)
    missing_target = xy(sample_cost_surface["transform"], 0, 3)

    routes = gpd.GeoDataFrame(
        {
            "route_id": ["dup_match", "single_match", "no_feature"],
            "start_row": [0, 0, 0],
            "start_col": [0, 0, 0],
            "end_row": [1, 2, 0],
            "end_col": [2, 2, 3],
            "length_km": [1.0, 2.0, 3.0],
            "cost": [10.0, 20.0, 30.0],
        },
        geometry=[
            LineString([(start_x, start_y), first_target]),
            LineString([(start_x, start_y), second_target]),
            LineString([(start_x, start_y), missing_target]),
        ],
        crs=sample_cost_surface["crs"],
    )

    mapper = RouteToFeatureMapper(sample_cost_surface["cost_fp"])
    mapped = mapper.process(routes, transmission_features["fp"])
    mapped = mapped.set_index("route_id")
    assert len(mapped) == 3
    assert "a_test_col" in mapped
    assert "another_test_col" in mapped

    assert (
        mapped.loc["dup_match", "trans_gid"]
        == transmission_features["matches"][(1, 2)]
    )
    assert (
        mapped.loc["single_match", "trans_gid"]
        == transmission_features["matches"][(2, 2)]
    )

    missing_value = mapped.loc["no_feature", "trans_gid"]
    assert pd.isna(missing_value)
    assert (None if pd.isna(missing_value) else missing_value) is None


def test_route_post_processor_merges_features_csv(
    tmp_path, sample_cost_surface, transmission_features
):
    """CSV collection should merge transmission feature identifiers"""

    chunk_dir = tmp_path / "csv_routes"
    chunk_dir.mkdir()

    pd.DataFrame(
        {
            "route_id": ["dup_match", "single_match", "no_feature"],
            "end_row": [1, 2, 0],
            "end_col": [2, 2, 3],
            "length_km": [1.0, 2.0, 3.0],
            "cost": [10.0, 20.0, 30.0],
        }
    ).to_csv(chunk_dir / "routes_part_0.csv", index=False)

    processor = RoutePostProcessor(
        collect_pattern="csv_routes/routes_part_*.csv",
        project_dir=tmp_path,
        job_name="csv_with_features",
        chunk_size=1,
        cost_fpath=sample_cost_surface["cost_fp"],
        features_fpath=transmission_features["fp"],
        transmission_feature_id_col="trans_gid",
        purge_chunks=True,
    )

    out_fp = Path(processor.process())
    merged = pd.read_csv(out_fp).set_index("route_id")
    assert len(merged) == 3
    assert "a_test_col" in merged
    assert "another_test_col" in merged

    assert (
        merged.loc["dup_match", "trans_gid"]
        == transmission_features["matches"][(1, 2)]
    )
    assert (
        merged.loc["single_match", "trans_gid"]
        == transmission_features["matches"][(2, 2)]
    )

    missing_value = merged.loc["no_feature", "trans_gid"]
    assert pd.isna(missing_value)
    assert (None if pd.isna(missing_value) else missing_value) is None
    assert not (chunk_dir / "routes_part_0.csv").exists()


def test_route_post_processor_merges_features_gpkg(
    tmp_path, sample_cost_surface, transmission_features
):
    """GeoPackage collection should merge transmission feature IDs"""

    transform = sample_cost_surface["transform"]

    start = xy(transform, 0, 0)
    first_target = xy(transform, 1, 2)
    second_target = xy(transform, 2, 2)
    missing_target = xy(transform, 0, 3)

    chunk_dir = tmp_path / "gpkg_routes"
    chunk_dir.mkdir()

    routes = gpd.GeoDataFrame(
        {
            "route_id": ["dup_match", "single_match", "no_feature"],
            "start_row": [0, 0, 0],
            "start_col": [0, 0, 0],
            "end_row": [1, 2, 0],
            "end_col": [2, 2, 3],
            "length_km": [1.0, 2.0, 3.0],
            "cost": [10.0, 20.0, 30.0],
        },
        geometry=[
            LineString([start, first_target]),
            LineString([start, second_target]),
            LineString([start, missing_target]),
        ],
        crs=sample_cost_surface["crs"],
    )

    chunk_fp = chunk_dir / "routes_chunk.gpkg"
    routes.to_file(chunk_fp, driver="GPKG")

    processor = RoutePostProcessor(
        collect_pattern="gpkg_routes/*.gpkg",
        project_dir=tmp_path,
        job_name="routes_with_features",
        chunk_size=1,
        cost_fpath=sample_cost_surface["cost_fp"],
        features_fpath=transmission_features["fp"],
        transmission_feature_id_col="trans_gid",
    )

    out_fp = Path(processor.process())
    merged = gpd.read_file(out_fp).set_index("route_id")
    assert len(merged) == 3
    assert "a_test_col" in merged
    assert "another_test_col" in merged

    assert (
        merged.loc["dup_match", "trans_gid"]
        == transmission_features["matches"][(1, 2)]
    )
    assert (
        merged.loc["single_match", "trans_gid"]
        == transmission_features["matches"][(2, 2)]
    )

    missing_value = merged.loc["no_feature", "trans_gid"]
    assert pd.isna(missing_value)
    assert (None if pd.isna(missing_value) else missing_value) is None

    chunk_files_dir = tmp_path / "chunk_files"
    assert chunk_files_dir.exists()
    assert (chunk_files_dir / chunk_fp.name).exists()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_finalize_routes_merges_features_csv(
    run_gaps_cli_with_expected_file,
    tmp_path,
    sample_cost_surface,
    transmission_features,
):
    """CLI finalize should merge features for CSV chunks"""

    chunk_dir = tmp_path / "csv_routes"
    chunk_dir.mkdir()

    pd.DataFrame(
        {
            "route_id": ["dup_match", "single_match", "no_feature"],
            "end_row": [1, 2, 0],
            "end_col": [2, 2, 3],
            "length_km": [1.0, 2.0, 3.0],
            "cost": [10.0, 20.0, 30.0],
        }
    ).to_csv(chunk_dir / "routes_part_0.csv", index=False)

    config = {
        "collect_pattern": "csv_routes/routes_part_*.csv",
        "chunk_size": 1,
        "purge_chunks": True,
        "cost_fpath": str(sample_cost_surface["cost_fp"]),
        "features_fpath": str(transmission_features["fp"]),
        "transmission_feature_id_col": "trans_gid",
    }

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes", config, tmp_path
    )

    merged = pd.read_csv(merged_fp).set_index("route_id")
    assert len(merged) == 3
    assert "a_test_col" in merged
    assert "another_test_col" in merged

    assert (
        merged.loc["dup_match", "trans_gid"]
        == transmission_features["matches"][(1, 2)]
    )
    assert (
        merged.loc["single_match", "trans_gid"]
        == transmission_features["matches"][(2, 2)]
    )

    missing_value = merged.loc["no_feature", "trans_gid"]
    assert pd.isna(missing_value)
    assert (None if pd.isna(missing_value) else missing_value) is None
    assert not (chunk_dir / "routes_part_0.csv").exists()


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_finalize_routes_merges_features_gpkg(
    run_gaps_cli_with_expected_file,
    tmp_path,
    sample_cost_surface,
    transmission_features,
):
    """CLI finalize should merge features for GeoPackage chunks"""

    transform = sample_cost_surface["transform"]

    start = xy(transform, 0, 0)
    first_target = xy(transform, 1, 2)
    second_target = xy(transform, 2, 2)
    missing_target = xy(transform, 0, 3)

    chunk_dir = tmp_path / "gpkg_routes"
    chunk_dir.mkdir()

    routes = gpd.GeoDataFrame(
        {
            "route_id": ["dup_match", "single_match", "no_feature"],
            "start_row": [0, 0, 0],
            "start_col": [0, 0, 0],
            "end_row": [1, 2, 0],
            "end_col": [2, 2, 3],
            "length_km": [1.0, 2.0, 3.0],
            "cost": [10.0, 20.0, 30.0],
        },
        geometry=[
            LineString([start, first_target]),
            LineString([start, second_target]),
            LineString([start, missing_target]),
        ],
        crs=sample_cost_surface["crs"],
    )

    chunk_fp = chunk_dir / "routes_chunk.gpkg"
    routes.to_file(chunk_fp, driver="GPKG")

    config = {
        "collect_pattern": "gpkg_routes/*.gpkg",
        "chunk_size": 1,
        "cost_fpath": str(sample_cost_surface["cost_fp"]),
        "features_fpath": str(transmission_features["fp"]),
        "transmission_feature_id_col": "trans_gid",
    }

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes", config, tmp_path
    )

    merged = gpd.read_file(merged_fp).set_index("route_id")
    assert len(merged) == 3
    assert "a_test_col" in merged
    assert "another_test_col" in merged

    assert (
        merged.loc["dup_match", "trans_gid"]
        == transmission_features["matches"][(1, 2)]
    )
    assert (
        merged.loc["single_match", "trans_gid"]
        == transmission_features["matches"][(2, 2)]
    )

    missing_value = merged.loc["no_feature", "trans_gid"]
    assert pd.isna(missing_value)
    assert (None if pd.isna(missing_value) else missing_value) is None

    chunk_files_dir = tmp_path / "chunk_files"
    assert chunk_files_dir.exists()
    assert (chunk_files_dir / chunk_fp.name).exists()


def test_route_post_processor_requires_features_inputs(
    tmp_path, sample_cost_surface
):
    """Processor should raise if only cost layer is provided"""

    (tmp_path / "missing_features").mkdir()
    pd.DataFrame(
        {
            "route_id": ["route_0"],
            "end_row": [0],
            "end_col": [0],
        }
    ).to_csv(
        tmp_path / "missing_features/routes.csv",
        index=False,
    )

    processor = RoutePostProcessor(
        collect_pattern="missing_features/routes.csv",
        project_dir=tmp_path,
        job_name="missing",
        cost_fpath=sample_cost_surface["cost_fp"],
    )

    with pytest.raises(
        revrtValueError,
        match="Both `cost_fpath` and `features_fpath` must be provided",
    ):
        processor.process()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
