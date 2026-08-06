"""reVRt base routing CLI unit tests"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError
from shapely.geometry import LineString
from rasterio.transform import from_origin

from revrt.exceptions import revrtConfigurationError, revrtKeyError
from revrt.models.routing import (
    RoutingCostLayer,
    RoutingFrictionLayer,
    RoutingOptionConfig,
    TrackedLayer,
    validate_routing_options,
)
from revrt.routing.cli.base import (
    update_multipliers,
    RoutingOptions,
    run_lcp,
    route_points_subset,
    split_routes,
    _get_row_multiplier,
    _get_polarity_multiplier,
    _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL,
)
from revrt.routing.cli.utilities import (
    _create_routing_layer_tmp_dir,
    _get_scratch_username,
)
from revrt.routing.cli.point_to_point import (
    PointToPointRouteDefinitionConverter,
)
from revrt.routing.utilities import map_to_costs
from revrt.utilities import LayeredFile


@pytest.fixture(scope="module")
def sample_layered_data(tmp_path_factory):
    """Create layered routing data mimicking point_to_point tests"""

    data_dir = tmp_path_factory.mktemp("routing_cli_data")

    layered_fp = data_dir / "test_layered.zarr"
    layer_file = LayeredFile(layered_fp)

    height, width = (7, 8)
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    layer_values = [
        np.array(
            [
                [
                    [7, 7, 8, 0, 9, 9, 9, 0],
                    [8, 1, 2, 2, 9, 9, 9, 0],
                    [9, 1, 3, 3, 9, 1, 2, 3],
                    [9, 1, 2, 1, 9, 1, 9, 0],
                    [9, 9, 9, 1, 9, 1, 9, 0],
                    [9, 9, 9, 1, 1, 1, 9, 0],
                    [9, 9, 9, 9, 9, 9, 9, 0],
                ]
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [
                    [8, 7, 6, 5, 5, 6, 7, 9],
                    [7, 1, 1, 2, 3, 3, 2, 8],
                    [6, 2, 9, 6, 5, 2, 1, 7],
                    [7, 3, 8, 1, 2, 3, 2, 6],
                    [8, 4, 7, 2, 8, 4, 3, 5],
                    [9, 5, 6, 3, 4, 4, 3, 4],
                    [9, 6, 7, 4, 5, 5, 4, 3],
                ]
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [
                    [6, 6, 6, 6, 6, 7, 8, 9],
                    [5, 2, 2, 3, 4, 5, 6, 8],
                    [4, 3, 7, 7, 6, 4, 5, 7],
                    [5, 4, 6, 2, 3, 4, 4, 6],
                    [6, 5, 5, 3, 7, 5, 5, 5],
                    [7, 6, 6, 4, 5, 5, 4, 4],
                    [8, 7, 7, 5, 6, 5, 4, 3],
                ]
            ],
            dtype=np.float32,
        ),
    ]

    for ind, routing_layer in enumerate(layer_values, start=1):
        da = xr.DataArray(
            routing_layer,
            dims=("band", "y", "x"),
            coords={"y": y_coords, "x": x_coords},
        )
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.write_transform(transform)

        geotiff_fp = data_dir / f"layer_{ind}.tif"
        da.rio.to_raster(geotiff_fp, driver="GTiff")

        layer_file.write_geotiff_to_file(
            geotiff_fp, f"layer_{ind}", overwrite=True
        )

    return layered_fp


def _build_route_table(layered_fp, rows_cols):
    """Helper to construct route tables with CRS-aligned coordinates"""

    with xr.open_dataset(layered_fp, consolidated=False, engine="zarr") as ds:
        latitudes = ds["latitude"].to_numpy()
        longitudes = ds["longitude"].to_numpy()

    records = []
    for idx, (start, end) in enumerate(rows_cols):
        s_row, s_col = start
        e_row, e_col = end
        records.append(
            {
                "route_id": f"route_{idx}",
                "start_lat": float(latitudes[s_row, s_col]),
                "start_lon": float(longitudes[s_row, s_col]),
                "end_lat": float(latitudes[e_row, e_col]),
                "end_lon": float(longitudes[e_row, e_col]),
                "voltage": 138,
                "polarity": "ac",
            }
        )

    return pd.DataFrame.from_records(records)


def test_run_lcp_with_save_paths_filters_existing_routes(
    sample_layered_data, tmp_path, monkeypatch
):
    """run_lcp should skip already processed routes and append geometries"""

    routes = _build_route_table(
        sample_layered_data,
        rows_cols=[((1, 1), (2, 2)), ((2, 2), (4, 4))],
    )

    with xr.open_dataset(
        sample_layered_data, consolidated=False, engine="zarr"
    ) as ds:
        mapped_routes = map_to_costs(
            routes.copy(), ds.rio.crs, ds.rio.transform(), ds.rio.shape
        )

    existing_tuple = (
        int(mapped_routes.iloc[0]["start_row"]),
        int(mapped_routes.iloc[0]["start_col"]),
        "default",
        int(mapped_routes.iloc[0]["end_row"]),
        int(mapped_routes.iloc[0]["end_col"]),
        "default",
        routes.iloc[0]["polarity"],
        str(routes.iloc[0]["voltage"]),
    )

    monkeypatch.setattr(
        "revrt.routing.cli.point_to_point."
        "PointToPointRouteDefinitionConverter.existing_routes",
        {existing_tuple},
    )

    saved_calls = []

    def fake_to_file(self, path, driver=None, mode=None, **_kwargs):
        saved_calls.append((path, driver, mode, self.copy(deep=True)))

    monkeypatch.setattr("geopandas.GeoDataFrame.to_file", fake_to_file)

    out_fp = tmp_path / "routes.gpkg"

    routes_to_compute = PointToPointRouteDefinitionConverter(
        cost_fpath=sample_layered_data,
        route_points=routes,
        out_fp=out_fp,
        routing_options={
            "default": {
                "cost_layers": [{"layer_name": "layer_1"}],
                "friction_layers": [
                    {"multiplier_layer": "layer_2", "apply_row_mult": True}
                ],
            }
        },
        transmission_config={
            "row_width": {"138": 1.0},
            "voltage_polarity_mult": {"138": {"ac": 1.0}},
        },
    )

    run_lcp(
        cost_fpath=sample_layered_data,
        out_fp=out_fp,
        routes_to_compute=routes_to_compute,
        tracked_layers=[{"layer_name": "layer_3", "agg_method": "max"}],
        invalid_costs_block_routing=True,
    )

    primary_calls = [call for call in saved_calls if call[0] == out_fp]
    option_calls = [
        call
        for call in saved_calls
        if call[0] == tmp_path / "routes_routing_options.gpkg"
    ]

    assert len(primary_calls) == 1
    assert len(option_calls) == 1

    saved_path, driver, mode, saved_gdf = primary_calls[0]
    assert saved_path == out_fp
    assert driver == "GPKG"
    assert mode == "a"
    assert len(saved_gdf) == 1
    assert saved_gdf["route_id"].iloc[0] == routes.iloc[1]["route_id"]

    expected = mapped_routes.iloc[1]
    assert int(saved_gdf["start_row"].iloc[0]) == int(expected["start_row"])
    assert int(saved_gdf["start_col"].iloc[0]) == int(expected["start_col"])
    assert int(saved_gdf["end_row"].iloc[0]) == int(expected["end_row"])
    assert int(saved_gdf["end_col"].iloc[0]) == int(expected["end_col"])

    cost_val = float(saved_gdf["cost"].iloc[0])
    objective_val = float(saved_gdf["optimized_objective"].iloc[0])
    length_val = float(saved_gdf["length_km"].iloc[0])

    assert cost_val > 0
    assert length_val > 0
    assert objective_val > cost_val

    geom = saved_gdf.geometry.iloc[0]
    assert isinstance(geom, LineString)
    assert len(geom.coords) >= 2


def test_run_lcp_returns_immediately_when_no_routes(tmp_path):
    """run_lcp should exit early when route_points is empty"""

    routes_to_compute = PointToPointRouteDefinitionConverter(
        cost_fpath="unused",
        route_points=pd.DataFrame(),
        out_fp=tmp_path / "unused.csv",
        routing_options={"default": {}},
    )

    run_lcp(
        cost_fpath="unused",  # cost file is ignored in this branch
        out_fp=tmp_path / "unused.csv",
        routes_to_compute=routes_to_compute,
    )


def test_route_points_subset_with_chunking(tmp_path):
    """route_points_subset should slice sorted features by chunk"""

    test_fp = tmp_path / "features.csv"
    features = pd.DataFrame(
        {
            "start_lat": [5.0, 1.0, 3.0, 7.0],
            "start_lon": [0.0, 1.0, 2.0, 3.0],
        }
    )

    features.to_csv(test_fp, index=False)

    first_chunk = route_points_subset(test_fp, (0, 2))
    assert first_chunk["start_lat"].tolist() == [1.0, 3.0]
    assert first_chunk["start_lon"].tolist() == [1.0, 2.0]

    second_chunk = route_points_subset(test_fp, (1, 2))
    assert second_chunk["start_lat"].tolist() == [5.0, 7.0]
    assert second_chunk["start_lon"].tolist() == [0.0, 3.0]


def test_paths_to_compute_inserts_missing_columns(tmp_path):
    """_paths_to_compute should back-fill per-option value columns"""

    route_points = pd.DataFrame(
        {
            "start_row": [0],
            "start_col": [1],
            "end_row": [2],
            "end_col": [3],
        }
    )

    route_generator = PointToPointRouteDefinitionConverter(
        cost_fpath=None,
        route_points=route_points,
        out_fp=tmp_path / "not_there.csv",
        routing_options={
            "default": {
                "cost_layers": None,
                "friction_layers": None,
            }
        },
        transmission_config=None,
    )

    groups = list(route_generator._paths_to_compute)
    assert groups
    route_option_values, grouped_routes = groups[0]
    assert route_option_values == {
        "default": {"polarity": "unknown", "voltage": "unknown"}
    }
    assert grouped_routes.iloc[0]["polarity_default"] == "unknown"
    assert grouped_routes.iloc[0]["voltage_default"] == "unknown"
    assert "polarity" not in grouped_routes.columns
    assert "voltage" not in grouped_routes.columns
    assert grouped_routes.iloc[0]["start_row"] == 0


def test_paths_to_compute_fills_missing_columns_from_shared_values(tmp_path):
    """Shared values should back-fill missing per-option columns"""

    route_points = pd.DataFrame(
        {
            "start_row": [0],
            "start_col": [1],
            "end_row": [2],
            "end_col": [3],
            "polarity": ["ac"],
            "voltage": [138],
            "polarity_overhead": ["dc"],
            "voltage_overhead": [230],
        }
    )

    route_generator = PointToPointRouteDefinitionConverter(
        cost_fpath=None,
        route_points=route_points,
        out_fp=tmp_path / "not_there.csv",
        routing_options={
            "overhead": {
                "cost_layers": None,
                "friction_layers": None,
            },
            "underground": {
                "cost_layers": None,
                "friction_layers": None,
            },
        },
        transmission_config=None,
    )

    groups = list(route_generator._paths_to_compute)
    assert groups
    route_option_values, grouped_routes = groups[0]
    assert route_option_values == {
        "overhead": {"polarity": "dc", "voltage": 230},
        "underground": {"polarity": "ac", "voltage": 138},
    }
    assert grouped_routes.iloc[0]["polarity_overhead"] == "dc"
    assert grouped_routes.iloc[0]["voltage_overhead"] == 230
    assert grouped_routes.iloc[0]["polarity_underground"] == "ac"
    assert grouped_routes.iloc[0]["voltage_underground"] == 138


def test_paths_to_compute_fills_partial_per_option_columns(tmp_path):
    """Missing per-option route values should default to unknown"""

    route_points = pd.DataFrame(
        {
            "start_row": [0],
            "start_col": [1],
            "end_row": [2],
            "end_col": [3],
            "polarity_overhead": ["ac"],
            "voltage_overhead": [138],
        }
    )

    route_generator = PointToPointRouteDefinitionConverter(
        cost_fpath=None,
        route_points=route_points,
        out_fp=tmp_path / "not_there.csv",
        routing_options={
            "overhead": {
                "cost_layers": None,
                "friction_layers": None,
            },
            "underground": {
                "cost_layers": None,
                "friction_layers": None,
            },
        },
        transmission_config=None,
    )

    groups = list(route_generator._paths_to_compute)
    assert groups
    route_option_values, grouped_routes = groups[0]
    assert route_option_values == {
        "overhead": {"polarity": "ac", "voltage": 138},
        "underground": {"polarity": "unknown", "voltage": "unknown"},
    }
    assert grouped_routes.iloc[0]["polarity_underground"] == "unknown"
    assert grouped_routes.iloc[0]["voltage_underground"] == "unknown"


def test_split_routes_handles_local_and_cluster():
    """split_routes should configure chunking for local and cluster modes"""

    result_local = split_routes({}, 1)
    assert result_local["_split_params"] == [(0, 1)]

    result_cluster = split_routes({}, 3)
    assert result_cluster["_split_params"] == [(0, 3), (1, 3), (2, 3)]


def test_update_multipliers_applies_row_and_polarity():
    """update_multipliers should apply configured scalar adjustments"""

    layers = [
        {
            "layer_name": "layer_1",
            "multiplier_scalar": 2,
            "apply_row_mult": True,
        },
        {"layer_name": "layer_2", "apply_polarity_mult": True},
    ]

    transmission_config = {
        "row_width": {"138": 1.5},
        "voltage_polarity_mult": {"138": {"ac": 0.5}},
    }

    updated = update_multipliers(
        layers,
        polarity="ac",
        voltage=138,
        routing_option="overhead",
        transmission_config=transmission_config,
    )

    # original input remains unchanged
    assert layers[0]["apply_row_mult"] is True
    assert layers[1]["apply_polarity_mult"] is True
    assert layers[0]["multiplier_scalar"] == 2

    # output is updated
    assert updated[0]["multiplier_scalar"] == pytest.approx(3)
    assert updated[1]["multiplier_scalar"] == pytest.approx(
        0.5 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL
    )

    # Voltage marked as unknown should skip multiplier lookups
    unchanged = update_multipliers(
        [{"layer_name": "layer_3"}],
        "dc",
        "unknown",
        "overhead",
        transmission_config,
    )
    assert unchanged[0]["layer_name"] == "layer_3"


def test_update_multipliers_applies_shared_polarity_value_to_all_options():
    """Scalar polarity values should apply to every routing option"""

    routing_options = {
        option: {
            "cost_layers": [
                {"layer_name": f"{option}_cost", "apply_polarity_mult": True}
            ]
        }
        for option in ("overhead", "underground")
    }
    transmission_config = {"voltage_polarity_mult": {"500": {"ac": 2.5}}}

    updated = RoutingOptions(routing_options).update_from(
        transmission_config=transmission_config,
        pv_by_option={
            option: {"polarity": "ac", "voltage": 500}
            for option in routing_options
        },
    )

    for option in routing_options:
        assert updated[option]["cost_layers"][0][
            "multiplier_scalar"
        ] == pytest.approx(2.5 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL)


def test_update_multipliers_applies_per_option_row_values():
    """Option-map ROW values should scale cost and friction layers"""

    routing_options = {
        "overhead": {
            "cost_layers": [
                {
                    "layer_name": "overhead_cost",
                    "multiplier_scalar": 2,
                    "apply_row_mult": True,
                }
            ],
            "friction_layers": [
                {
                    "multiplier_layer": "overhead_friction",
                    "multiplier_scalar": 3,
                    "apply_row_mult": True,
                }
            ],
        },
        "underground": {
            "cost_layers": [
                {
                    "layer_name": "underground_cost",
                    "multiplier_scalar": 4,
                    "apply_row_mult": True,
                }
            ],
            "friction_layers": [
                {
                    "multiplier_layer": "underground_friction",
                    "multiplier_scalar": 5,
                    "apply_row_mult": True,
                }
            ],
        },
    }
    transmission_config = {
        "row_width": {"500": {"overhead": 1.5, "underground": 2}}
    }

    updated = RoutingOptions(routing_options).update_from(
        transmission_config=transmission_config,
        pv_by_option={
            option: {"polarity": "ac", "voltage": 500}
            for option in routing_options
        },
    )

    assert updated["overhead"]["cost_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(3)
    assert updated["overhead"]["friction_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(4.5)
    assert updated["underground"]["cost_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(8)
    assert updated["underground"]["friction_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(10)


def test_update_multipliers_applies_per_option_polarity_values():
    """Option maps should apply their values to cost and friction layers"""

    routing_options = {
        "overhead": {
            "cost_layers": [
                {
                    "layer_name": "overhead_cost",
                    "multiplier_scalar": 2,
                    "apply_polarity_mult": True,
                }
            ],
            "friction_layers": [
                {
                    "multiplier_layer": "overhead_friction",
                    "apply_polarity_mult": True,
                }
            ],
        },
        "underground": {
            "cost_layers": [
                {
                    "layer_name": "underground_cost",
                    "apply_polarity_mult": True,
                }
            ],
            "friction_layers": [
                {
                    "multiplier_layer": "underground_friction",
                    "multiplier_scalar": 3,
                    "apply_polarity_mult": True,
                }
            ],
        },
    }
    transmission_config = {
        "voltage_polarity_mult": {
            "500": {"ac": {"overhead": 2, "underground": 20}}
        }
    }

    updated = RoutingOptions(routing_options).update_from(
        transmission_config=transmission_config,
        pv_by_option={
            option: {"polarity": "ac", "voltage": 500}
            for option in routing_options
        },
    )

    assert updated["overhead"]["cost_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(4 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL)
    assert updated["overhead"]["friction_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(2 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL)
    assert updated["underground"]["cost_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(20 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL)
    assert updated["underground"]["friction_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(60 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL)


def test_update_multipliers_allows_mixed_shared_and_option_values():
    """Scalar and option-map polarity entries should resolve together"""

    config = {
        "voltage_polarity_mult": {
            "500": {
                "ac": 2,
                "dc": {"overhead": 3, "underground": 4},
            }
        }
    }

    assert _get_polarity_multiplier(config, "500", "ac", "overhead") == 2
    assert _get_polarity_multiplier(config, "500", "ac", "underground") == 2
    assert _get_polarity_multiplier(config, "500", "dc", "overhead") == 3
    assert _get_polarity_multiplier(config, "500", "dc", "underground") == 4


def test_update_multipliers_only_requires_option_values_when_applied():
    """Unused option-map entries should not be resolved"""

    layers = [{"layer_name": "overhead_cost"}]
    config = {"voltage_polarity_mult": {"500": {"ac": {}}}}

    updated = update_multipliers(layers, "ac", 500, "overhead", config)

    assert updated == layers


def test_get_polarity_multiplier_unknown_routing_option():
    """Option maps should report an option that lacks a multiplier"""

    config = {
        "voltage_polarity_mult": {
            "500": {"ac": {"overhead": 2, "underground": 20}}
        }
    }

    with pytest.raises(
        revrtKeyError,
        match=(
            r"routing option 'submarine' not found for voltage '500' and "
            r"polarity 'ac'. Available routing options: "
            r"\['overhead', 'underground'\]"
        ),
    ):
        _get_polarity_multiplier(config, "500", "ac", "submarine")


def test_update_route_options_requires_used_option_multiplier():
    """Used options should require an entry in an option-map multiplier"""

    routing_options = {
        "overhead": {
            "cost_layers": [
                {"layer_name": "overhead_cost", "apply_polarity_mult": True}
            ]
        },
        "underground": {
            "cost_layers": [
                {"layer_name": "underground_cost", "apply_polarity_mult": True}
            ]
        },
    }
    transmission_config = {
        "voltage_polarity_mult": {"500": {"ac": {"overhead": 2}}}
    }

    with pytest.raises(
        revrtKeyError,
        match=(
            r"routing option 'underground' not found for voltage '500' and "
            r"polarity 'ac'. Available routing options: \['overhead'\]"
        ),
    ):
        RoutingOptions(routing_options).update_from(
            transmission_config=transmission_config,
            pv_by_option={
                option: {"polarity": "ac", "voltage": 500}
                for option in routing_options
            },
        )


def test_update_route_options_updates_nested_layers_without_mutation():
    """Updated options should transform nested layer configs safely"""

    routing_options = {
        "overhead": {
            "cost_layers": [
                {
                    "layer_name": "layer_1",
                    "multiplier_scalar": 2,
                    "apply_row_mult": True,
                }
            ],
            "friction_layers": [
                {
                    "apply_polarity_mult": True,
                    "multiplier_layer": "layer_2",
                }
            ],
            "barrier_layers": [
                {
                    "layer_name": "layer_3",
                    "where": "==1",
                }
            ],
        }
    }
    transmission_config = {
        "row_width": {"138": 1.5},
        "voltage_polarity_mult": {"138": {"ac": 0.5}},
    }

    updated = RoutingOptions(routing_options).update_from(
        transmission_config=transmission_config,
        pv_by_option={
            "overhead": {"polarity": "ac", "voltage": 138},
        },
    )

    assert updated["overhead"]["cost_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(3)
    assert updated["overhead"]["friction_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(0.5 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL)
    assert updated["overhead"]["barrier_layers"] == [
        {"layer_name": "layer_3", "where": "==1"}
    ]
    assert (
        updated["overhead"]["barrier_layers"]
        is not routing_options["overhead"]["barrier_layers"]
    )

    assert routing_options["overhead"]["cost_layers"][0]["apply_row_mult"]
    assert (
        routing_options["overhead"]["cost_layers"][0]["multiplier_scalar"] == 2
    )
    assert routing_options["overhead"]["friction_layers"][0][
        "apply_polarity_mult"
    ]


def test_validate_routing_options_preserves_supported_schema():
    """Routing option validation should preserve accepted payload shape"""

    routing_options = {
        "overhead": {
            "cost_layers": [
                {
                    "layer_name": "layer_1",
                    "apply_row_mult": True,
                }
            ],
            "friction_layers": [
                {
                    "multiplier_layer": "layer_2",
                    "multiplier_scalar": 0.5,
                }
            ],
            "barrier_layers": [
                {
                    "layer_name": "layer_3",
                    "where": "==1",
                }
            ],
            "cost_multiplier_layer": "layer_4",
        }
    }

    assert validate_routing_options(routing_options) == {
        "overhead": {
            "cost_layers": [
                {
                    "layer_name": "layer_1",
                    "apply_row_mult": True,
                }
            ],
            "friction_layers": [
                {
                    "multiplier_layer": ["layer_2"],
                    "multiplier_scalar": 0.5,
                }
            ],
            "barrier_layers": [
                {
                    "layer_name": "layer_3",
                    "where": "==1",
                }
            ],
            "cost_multiplier_layer": ["layer_4"],
        }
    }


def test_multiplier_layers_accept_scalar_and_iterable_inputs():
    """Multiplier layer fields accept strings and finite iterables"""

    assert RoutingCostLayer(
        layer_name="cost", multiplier_layer="multiplier"
    ).multiplier_layer == ["multiplier"]
    assert RoutingFrictionLayer(
        multiplier_layer=("first", "second")
    ).multiplier_layer == ["first", "second"]
    assert TrackedLayer(
        layer_name="tracked",
        agg_method="mean",
        multiplier_layer=(name for name in ("first", "second")),
    ).multiplier_layer == ["first", "second"]
    assert RoutingOptionConfig(
        cost_multiplier_layer=("first", "second")
    ).cost_multiplier_layer == ["first", "second"]
    assert (
        RoutingCostLayer(
            layer_name="cost", multiplier_layer=[]
        ).multiplier_layer
        == []
    )


def test_multiplier_layers_reject_non_string_items():
    """Multiplier layer iterables must only contain layer names"""

    with pytest.raises(ValidationError):
        RoutingCostLayer(layer_name="cost", multiplier_layer=["layer", 1])


def test_validate_routing_options_normalizes_legacy_shapes():
    """Routing option validation should accept legacy routing payloads"""

    routing_options = {
        "default": {},
        "overhead": {
            "cost_layers": None,
            "friction_layers": None,
            "barrier_layers": None,
        },
    }

    out = validate_routing_options(routing_options)
    assert out == {"default": {}, "overhead": {}}


def test_validate_routing_options_raises_configuration_error():
    """Routing option validation should raise configuration errors"""

    routing_options = {
        "overhead": {
            "cost_layers": [
                {
                    "layer_name": "layer_1",
                    "unknown_field": True,
                }
            ]
        }
    }

    with pytest.raises(revrtConfigurationError, match="Extra inputs"):
        validate_routing_options(routing_options)


def test_update_route_options_applies_per_option_values():
    """Updated options should apply explicit values per option"""

    routing_options = {
        "overhead": {
            "cost_layers": [{"layer_name": "layer_1", "apply_row_mult": True}],
            "friction_layers": [
                {"multiplier_layer": "layer_2", "apply_polarity_mult": True}
            ],
        },
        "underground": {
            "cost_layers": [{"layer_name": "layer_3", "apply_row_mult": True}],
            "friction_layers": [
                {"multiplier_layer": "layer_4", "apply_polarity_mult": True}
            ],
        },
    }
    transmission_config = {
        "row_width": {"138": 1.5, "230": 2.0},
        "voltage_polarity_mult": {
            "138": {"ac": 0.5},
            "230": {"dc": 0.75},
        },
    }

    updated = RoutingOptions(routing_options).update_from(
        transmission_config=transmission_config,
        pv_by_option={
            "overhead": {"polarity": "ac", "voltage": 138},
            "underground": {"polarity": "dc", "voltage": 230},
        },
    )

    assert updated["overhead"]["cost_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(1.5)
    assert updated["overhead"]["friction_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(0.5 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL)
    assert updated["underground"]["cost_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(2.0)
    assert updated["underground"]["friction_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(0.75 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL)


def test_route_converter_updates_multi_option_layers(tmp_path):
    """Route converter should update nested routing-option multipliers"""

    route_points = pd.DataFrame(
        {
            "start_row": [0],
            "start_col": [1],
            "end_row": [2],
            "end_col": [3],
            "polarity": ["ac"],
            "voltage": [138],
        }
    )
    transmission_config = {
        "row_width": {"138": 1.5},
        "voltage_polarity_mult": {"138": {"ac": 0.5}},
    }

    converter = PointToPointRouteDefinitionConverter(
        cost_fpath=None,
        route_points=route_points,
        out_fp=tmp_path / "unused.csv",
        routing_options={
            "overhead": {
                "cost_layers": [
                    {
                        "layer_name": "layer_1",
                        "multiplier_scalar": 2,
                        "apply_row_mult": True,
                    }
                ],
                "friction_layers": [
                    {
                        "multiplier_layer": "layer_2",
                        "apply_polarity_mult": True,
                    }
                ],
                "barrier_layers": [
                    {
                        "layer_name": "layer_3",
                        "where": "==1",
                    }
                ],
            }
        },
        transmission_config=transmission_config,
        drivers={"default": {"overhead": 1}},
        transition_costs={"default": 0},
    )

    route_ro, route_definitions, route_attrs = next(iter(converter))

    assert route_ro["overhead"]["cost_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(3)
    assert route_ro["overhead"]["friction_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(0.5 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL)
    assert route_ro["overhead"]["barrier_layers"] == [
        {"layer_name": "layer_3", "where": "==1"}
    ]
    assert route_definitions == [
        (0, [(0, 1, "overhead")], [(2, 3, "overhead")])
    ]
    route_attr = route_attrs[(0, (0, 1, "overhead"))]
    assert route_attr["voltage_overhead"] == 138
    assert route_attr["polarity_overhead"] == "ac"
    assert route_attr["voltage"] == 138
    assert route_attr["polarity"] == "ac"


def test_route_converter_applies_different_polarities_per_option(tmp_path):
    """Route converter should apply each option's explicit polarity"""

    route_points = pd.DataFrame(
        {
            "start_row": [0],
            "start_col": [1],
            "end_row": [2],
            "end_col": [3],
            "start_option": ["overhead"],
            "end_option": ["underground"],
            "voltage_overhead": [500],
            "polarity_overhead": ["ac"],
            "voltage_underground": [500],
            "polarity_underground": ["dc"],
        }
    )
    transmission_config = {
        "voltage_polarity_mult": {
            "500": {
                "ac": {"overhead": 2},
                "dc": {"underground": 5},
            }
        }
    }

    converter = PointToPointRouteDefinitionConverter(
        cost_fpath=None,
        route_points=route_points,
        out_fp=tmp_path / "unused.csv",
        routing_options={
            "overhead": {
                "cost_layers": [
                    {
                        "layer_name": "overhead_cost",
                        "apply_polarity_mult": True,
                    }
                ]
            },
            "underground": {
                "cost_layers": [
                    {
                        "layer_name": "underground_cost",
                        "apply_polarity_mult": True,
                    }
                ]
            },
        },
        transmission_config=transmission_config,
    )

    route_options, route_definitions, route_attrs = next(iter(converter))

    assert route_options["overhead"]["cost_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(2 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL)
    assert route_options["underground"]["cost_layers"][0][
        "multiplier_scalar"
    ] == pytest.approx(5 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL)
    assert route_definitions == [
        (0, [(0, 1, "overhead")], [(2, 3, "underground")])
    ]
    route_attrs = route_attrs[(0, (0, 1, "overhead"))]
    assert route_attrs["polarity_overhead"] == "ac"
    assert route_attrs["polarity_underground"] == "dc"


def test_get_row_multiplier_missing_config():
    """_get_row_multiplier should raise when configuration keys are absent"""

    with pytest.raises(
        revrtKeyError,
        match=(
            r"`apply_row_mult` was set to `True`, but 'row_width' not found "
            r"in transmission config"
        ),
    ):
        _get_row_multiplier({}, "138", "overhead")


def test_get_row_multiplier_unknown_voltage():
    """_get_row_multiplier should surface available voltages on failure"""

    config = {"row_width": {"230": 1.2}}
    with pytest.raises(
        revrtKeyError,
        match=(
            r"`apply_row_mult` was set to `True`, but voltage '\s*138' not "
            r"found in transmission config 'row_width' settings. "
            r"Available voltages: \['230'\]"
        ),
    ):
        _get_row_multiplier(config, "138", "overhead")


@pytest.mark.parametrize("routing_option", ["overhead", "underground"])
def test_get_row_multiplier_shared_value_applies_to_every_option(
    routing_option,
):
    """A scalar ROW value should apply to every routing option"""

    config = {"row_width": {"138": 1.2}}

    assert _get_row_multiplier(config, "138", routing_option) == pytest.approx(
        1.2
    )


def test_get_row_multiplier_allows_mixed_shared_and_option_values():
    """ROW configuration should allow scalar and option-map voltages"""

    config = {
        "row_width": {
            "138": 1.2,
            "500": {"overhead": 1.5, "underground": 2},
        }
    }

    assert _get_row_multiplier(config, "138", "overhead") == pytest.approx(1.2)
    assert _get_row_multiplier(config, "138", "underground") == pytest.approx(
        1.2
    )
    assert _get_row_multiplier(config, "500", "overhead") == pytest.approx(1.5)
    assert _get_row_multiplier(config, "500", "underground") == 2


def test_get_row_multiplier_unknown_option():
    """ROW option maps should require a multiplier for applied options"""

    config = {"row_width": {"500": {"overhead": 1.5}}}

    with pytest.raises(
        revrtKeyError,
        match=(
            r"`apply_row_mult` was set to `True`, but routing option "
            r"'underground' not found for voltage '500'. Available "
            r"routing options: \['overhead'\]"
        ),
    ):
        _get_row_multiplier(config, "500", "underground")


def test_get_polarity_multiplier_missing_config():
    """_get_polarity_multiplier should raise when multiplier section missing"""

    with pytest.raises(
        revrtKeyError,
        match=(
            r"`apply_polarity_mult` was set to `True`, but "
            r"'voltage_polarity_mult' not found in transmission config"
        ),
    ):
        _get_polarity_multiplier({}, "138", "ac", "default")


def test_get_polarity_multiplier_unknown_voltage():
    """_get_polarity_multiplier should guard against unknown voltages"""

    config = {"voltage_polarity_mult": {"230": {"ac": 1.0}}}
    with pytest.raises(
        revrtKeyError,
        match=(
            r"`apply_polarity_mult` was set to `True`, but voltage '\s*138' "
            r"not found in polarity config. Available voltages: \['230'\]"
        ),
    ):
        _get_polarity_multiplier(config, "138", "ac", "default")


def test_get_polarity_multiplier_unknown_polarity():
    """_get_polarity_multiplier should guard against unknown polarities"""

    config = {"voltage_polarity_mult": {"138": {"dc": 1.0}}}
    with pytest.raises(
        revrtKeyError,
        match=(
            r"`apply_polarity_mult` was set to `True`, but polarity '\s*ac' "
            r"not found in voltage config. Available polarities: \['dc'\]"
        ),
    ):
        _get_polarity_multiplier(config, "138", "ac", "default")


def test_get_scratch_username_prefers_environment(monkeypatch):
    """_get_scratch_username should prefer USERNAME environment value"""

    for env_name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("USERNAME", "runner admin")

    assert _get_scratch_username() == "runner_admin"


def test_create_routing_layer_tmp_dir_handles_getpass_failure(
    tmp_path, monkeypatch
):
    """_create_routing_layer_tmp_dir should not fail when getpass fails"""

    for env_name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(env_name, raising=False)

    def _raise_getuser():
        msg = "No module named 'pwd'"
        raise ModuleNotFoundError(msg)

    monkeypatch.setattr(
        "revrt.routing.cli.utilities.getpass.getuser",
        _raise_getuser,
    )
    monkeypatch.setattr(
        "revrt.routing.cli.utilities.Path.home",
        classmethod(lambda _cls: Path("/tmp/fallback-user")),  # ruff:ignore[hardcoded-temp-file]
    )
    monkeypatch.setattr(
        "revrt.routing.cli.utilities.tempfile.gettempdir",
        lambda: str(tmp_path),
    )

    out_dir = _create_routing_layer_tmp_dir()

    assert out_dir == tmp_path / "scratch" / "fallback-user"
    assert out_dir.exists()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
