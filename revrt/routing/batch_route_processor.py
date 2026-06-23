"""reVRt batch routing logic"""

import json
import time
import logging
from pathlib import Path
from warnings import warn
from functools import cached_property
from itertools import pairwise

import rasterio
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import MultiLineString
from shapely.geometry.linestring import LineString

from revrt import RouteFinder, simplify_using_slopes
from revrt.routing.base import RoutingLayerManager, RouteMetrics
from revrt.routing.utilities import compute_lens
from revrt.utilities.handlers import IncrementalWriter
from revrt.utilities.monitoring import log_runtime
from revrt.exceptions import (
    revrtKeyError,
    revrtLeastCostPathNotFoundError,
    revrtRustError,
)
from revrt.warn import revrtWarning


logger = logging.getLogger(__name__)
_ROUTE_SOLUTION_LEN = 3


class _IncrementalRouteWriter(IncrementalWriter):
    """Stream results to disk by appending each new result to a file

    A new file is created if one does not exist.
    """

    def __init__(self, out_fp, crs=None):
        """

        Parameters
        ----------
        out_fp : path-like
            Path to output file.
        crs : rasterio.crs.CRS or dict, optional
            Coordinate reference system for geometries when saving to
            GeoPackage. By default, ``None``.
        """
        super().__init__(out_fp)
        self.crs = crs

    def preprocess_chunk(self, result):
        """Turn result into a dataframe chunk

        Parameters
        ----------
        result : dict
            Route result dictionary as built by
            ``RouteMetrics.compute()``.

        Returns
        -------
        pandas.DataFrame or geopandas.GeoDataFrame
            A dataframe holding the route result.
        """
        if "geometry" in result:
            return gpd.GeoDataFrame(
                [result], geometry="geometry", crs=self.crs
            )
        return pd.DataFrame([result])


class BatchRouteProcessor:
    """Class to manage batches of route computations"""

    def __init__(
        self,
        routing_scenario,
        route_definitions,
        route_attrs=None,
        mem_limit_gb=4,
    ):
        """

        Parameters
        ----------
        routing_scenario : RoutingScenario
            Scenario describing the cost layers and routing options.
        route_definitions : Iterable
            Sequence of ``(start_points, end_points, option)`` tuples
            defining which points to route between. Each of
            ``start_points`` and ``end_points`` should be a list of
            ``(row, col, option_name)`` tuples.
        route_attrs : dict, optional
            Mapping of tuples of the form (int, (int, int, option))
            where the first integer represents the route ID and the
            tuple of integers + str represents the starting index to
            additional attributes to include in the output for that
            route. By default, ``None``.
        mem_limit_gb : int or float, default=4
            Memory limit in gigabytes for routing computations.
            By default, ``4``.
        """
        self.routing_scenario = routing_scenario
        self._route_attrs = route_attrs or {}
        self.mem_limit_gb = mem_limit_gb
        self.__rd = route_definitions

    @cached_property
    def default_attrs(self):
        """dict: Default attributes for all routes"""
        keys = set().union(*[set(x) for x in self._route_attrs.values()])
        return dict.fromkeys(keys)

    @cached_property
    def route_attrs(self):
        """dict: Mapping of frozen route node pair sets to attributes"""
        return {
            k: {**self.default_attrs, **v}
            for k, v in self._route_attrs.items()
        }

    @cached_property
    def routing_layers(self):
        """RoutingLayerManager: Built routing layers for the scenario"""
        return RoutingLayerManager(self.routing_scenario).build()

    @cached_property
    def route_definitions(self):
        """list: Validated route definitions for computation"""
        return _RouteDefinitionFormatter(
            self.__rd, self.routing_layers, self.routing_scenario
        ).route_definitions

    def process(self, out_fp, save_paths=False, routing_layer_out_fp=None):
        """Compute all routes and save to disk

        Parameters
        ----------
        out_fp : path-like
            Path to output file. If ``save_paths=True``, a GeoPackage
            will be created (recommend to pass in a filepath ending in
            ".gpkg"). Otherwise, a CSV file will be created (recommend
            to pass in a filepath ending in ".csv").
        save_paths : bool, default=False
            Include shapely geometries in the output when ``True``.
            By default, ``False``.
        routing_layer_out_fp : path-like, optional
            Optional output path for Rust routing-layer cache data.
            By default, ``None``.
        """
        if not self.route_definitions:
            return

        with log_runtime(
            f"Routing for {len(self.route_definitions)} route definitions"
        ):
            try:
                self._compute_routes(
                    out_fp, save_paths=save_paths, rl=routing_layer_out_fp
                )
            finally:
                self._reset_routing_layers()

    def _compute_routes(self, out_fp, save_paths, rl=None):
        """Evaluate route definitions and build result records"""

        result_writer = _RouteResultWriter(
            out_fp,
            save_paths,
            self.routing_layers.cost_crs,
            self.routing_layers.transform,
        )

        for indices, optimized_objective, attrs in self._route_results(rl):
            metrics = RouteMetrics(
                self.routing_layers,
                indices,
                optimized_objective,
                add_geom=save_paths,
                attrs=attrs,
            )
            route_result = metrics.compute()
            result_writer.persist(route_result, indices)

    def _route_results(self, routing_layer_out_fp=None):
        """Generator yielding route results from Rust computations"""
        if not self.route_definitions:
            return

        logger.debug(
            "Setting memory limit to %.2f GB for Rust computations",
            self.mem_limit_gb,
        )
        route_results = RouteFinder(
            zarr_fp=self.routing_scenario.cost_fpath,
            cost_function=self.routing_scenario.cost_function_json,
            route_definitions=[
                (rid, sp, ep)
                for rid, (sp, ep) in self.route_definitions.items()
            ],
            mem_limit_bytes=int(self.mem_limit_gb * 1_000_000_000),
            algorithm=self.routing_scenario.algorithm,
            log_level=logging.getLogger("revrt").level or None,
            routing_layer_out_fp=routing_layer_out_fp,
        )
        yield from self._skip_failed_routes(route_results)

    def _skip_failed_routes(self, routing_results):
        """Yield only successfully computed routes from Rust results"""

        results_iter = iter(routing_results)
        num_complete = 0
        ts = time.monotonic()
        while True:
            num_complete += 1
            try:
                route_id, solutions = next(results_iter)
                yield from self._formatted_solutions(solutions, route_id)
                time_elapsed = f"{(time.monotonic() - ts) / 60:.2f} minute(s)"
                logger.info(
                    "%d/%d (%.2f%%) route definitions processed in %s",
                    num_complete,
                    len(self.route_definitions),
                    (num_complete / len(self.route_definitions)) * 100,
                    time_elapsed,
                )
            except revrtRustError:  # pragma: no cover
                logger.exception("Rust error when computing route")
                continue
            except StopIteration:
                logger.info("Routing complete")
                break

    def _formatted_solutions(self, solutions, route_id):
        """Format reVRt output solutions and log any failures"""
        start_points, end_points = self.route_definitions[route_id]
        if not solutions:
            msg = (
                f"Unable to find route from {start_points} to any of "
                f"{end_points} (route ID: {route_id}). Please verify "
                "that the start and end points are not separated by "
                "hard barriers or invalid cost cells."
            )
            logger.error(msg)
            return

        logger.debug(
            "Got result from Rust for route_id %d. Processing..."
            "\n\t- Start points: %r\n\t- End points: %r",
            route_id,
            start_points,
            end_points,
        )
        for solution in solutions:
            if len(solution) == _ROUTE_SOLUTION_LEN:
                indices, optimized_objective, dbl = solution
            else:  # pragma: no cover
                msg = f"Unexpected route solution payload: {solution!r}"
                raise revrtKeyError(msg)

            attrs_key = (route_id, indices[0])
            attrs = {
                **self.route_attrs.get(attrs_key, self.default_attrs),
                "dropped_barrier_layers": json.dumps(dbl),
            }
            yield indices, optimized_objective, attrs

    def _reset_routing_layers(self):
        """Close handler and remove built routing layers from memory"""
        self.routing_layers.close()
        del self.routing_layers
        del self.route_definitions


class _RouteDefinitionFormatter:
    """Validate route definitions against routing layers"""

    def __init__(self, route_definitions, routing_layers, routing_scenario):
        self._route_definitions = route_definitions
        self.routing_layers = routing_layers
        self.routing_scenario = routing_scenario

    @cached_property
    def route_definitions(self):
        """list: Validated route definitions for computation"""
        return self._compile_valid_route_definitions()

    def _compile_valid_route_definitions(self):
        """Filter route definitions to those with valid route nodes"""
        if not self._route_definitions:
            return {}

        sample_definition = self._route_definitions[0]
        if len(sample_definition) == 2:  # noqa: PLR2004
            self._route_definitions = _add_route_ids(self._route_definitions)

        routes_to_compute = {}
        for route_id, start_points, end_points in self._route_definitions:
            filtered_start_points = self._validate_start_points(start_points)
            if not filtered_start_points:
                msg = (
                    f"All start points are invalid for route with ID "
                    f"{route_id}: {start_points}\nSkipping..."
                )
                warn(msg, revrtWarning)
                continue

            try:
                filtered_end_points = self._validate_end_points(end_points)
            except revrtLeastCostPathNotFoundError:
                continue

            if not filtered_end_points:
                msg = (
                    f"All end points are invalid for route with ID "
                    f"{route_id}: {end_points}\nSkipping..."
                )
                warn(msg, revrtWarning)
                continue

            routes_to_compute[route_id] = (
                filtered_start_points,
                filtered_end_points,
            )

        return routes_to_compute

    def _validate_start_points(self, points):
        """Validate start points by removing cells invalid cost"""
        points = _get_valid_points(
            points, self.routing_layers.full_shape, point_type="start"
        )
        if not points or not self.routing_scenario.invalid_costs_block_routing:
            return points

        routing_options = {point[-1] for point in points}
        bad_point_inds = set()
        for r_o in routing_options:
            rows, cols = np.array(
                [point[:2] for point in points if point[-1] == r_o]
            ).T
            costs = self.routing_layers.costs[r_o].isel(
                y=xr.DataArray(rows, dims="points"),
                x=xr.DataArray(cols, dims="points"),
            )

            cost_values = costs.compute()
            bad_point_inds |= set(
                np.where(np.isnan(cost_values) | (cost_values <= 0))[0]
            )

        if not bad_point_inds:
            return points

        invalid_points = {points[i] for i in bad_point_inds}
        msg = (
            f"One or more of the start points have an invalid cost "
            f"(must be > 0): {invalid_points}\n"
            "Dropping these from consideration..."
        )
        warn(msg, revrtWarning)

        if not points:
            all_invalid_points_msg = (
                "None of the start points have a valid cost (must be > 0): "
                f"{points}"
            )
            raise revrtLeastCostPathNotFoundError(all_invalid_points_msg)

        return [p for p in points if p not in invalid_points]

    def _validate_end_points(self, points):
        """Filter out invalid endpoints; raise if all are invalid"""
        points = _get_valid_points(
            points, self.routing_layers.full_shape, point_type="end"
        )
        if not points or not self.routing_scenario.invalid_costs_block_routing:
            return points

        routing_options = {point[-1] for point in points}
        bad_point_inds = set()
        for r_o in routing_options:
            rows, cols = np.array(
                [point[:2] for point in points if point[-1] == r_o]
            ).T
            costs = self.routing_layers.costs[r_o].isel(
                y=xr.DataArray(rows, dims="points"),
                x=xr.DataArray(cols, dims="points"),
            )

            cost_values = costs.compute()
            bad_point_inds |= set(
                np.where(np.isnan(cost_values) | (cost_values <= 0))[0]
            )
        if not bad_point_inds:
            return points

        invalid_points = {points[i] for i in bad_point_inds}
        msg = (
            f"One or more of the end points have an invalid cost "
            f"(must be > 0): {invalid_points}\n"
            "Dropping these from consideration..."
        )
        warn(msg, revrtWarning)
        points = [p for p in points if p not in invalid_points]

        if not points:
            all_invalid_points_msg = (
                "None of the end points have a valid cost (must be > 0): "
                f"{sorted(invalid_points)}"
            )
            raise revrtLeastCostPathNotFoundError(all_invalid_points_msg)

        return points


class _RouteResultWriter:
    """Class to manage output of route results"""

    def __init__(self, out_fp, save_paths, cost_crs, transform):
        out_fp = _validate_out_fp(out_fp, save_paths)
        self._writer = _IncrementalRouteWriter(out_fp, crs=cost_crs)
        self._transform = transform
        self._option_writer = (
            None
            if not save_paths
            else _IncrementalRouteWriter(
                _routing_options_output_fp(out_fp), crs=cost_crs
            )
        )

    def persist(self, route_result, indices):
        """Persist route result and any routing-option pieces to disk

        Parameters
        ----------
        route_result : dict
            Route result dictionary as built by
            ``RouteMetrics.compute()``.
        indices : list
            List of route indices as returned by the Rust routing
            engine.
        """
        self._writer.save(route_result)
        if self._option_writer is None:
            return

        for option_result in self._routing_option_results(
            indices, route_result
        ):
            self._option_writer.save(option_result)

    def _routing_option_results(self, indices, route_result):
        """Yield aggregated geometries for each routing option used"""

        segments_by_option = {}
        current_option = None
        current_segment = []
        for start_p, end_p in pairwise(indices):
            if start_p == end_p:
                continue

            start = tuple(start_p[:2])
            end = tuple(end_p[:2])
            start_point_option = start_p[-1]
            end_point_option = end_p[-1]

            if current_option != start_point_option:
                if len(current_segment) > 1:
                    segments_by_option.setdefault(current_option, []).append(
                        current_segment
                    )
                current_option = start_point_option
                current_segment = [start]

            if current_segment[-1] != start:
                current_segment.append(start)

            if start_point_option == end_point_option:
                if current_segment[-1] != end:
                    current_segment.append(end)
                continue

            midpoint = _midpoint(start, end)
            if current_segment[-1] != midpoint:
                current_segment.append(midpoint)

            if len(current_segment) > 1:
                segments_by_option.setdefault(current_option, []).append(
                    current_segment
                )
            current_option = end_point_option
            current_segment = [midpoint]
            if current_segment[-1] != end:
                current_segment.append(end)

        if len(current_segment) > 1:
            segments_by_option.setdefault(current_option, []).append(
                current_segment
            )

        cell_size = abs(self._transform.a)
        results = []
        for option, segments in segments_by_option.items():
            geoms = [self._component_geometry(segment) for segment in segments]
            if not geoms:
                continue

            geometry = (
                geoms[0]
                if len(geoms) == 1
                else MultiLineString([list(geom.coords) for geom in geoms])
            )
            length_km = sum(
                compute_lens(segment, cell_size)[1] for segment in segments
            )
            results.append(
                {
                    **{
                        key: value
                        for key, value in route_result.items()
                        if key
                        not in {
                            "geometry",
                            "cost",
                            "optimized_objective",
                            "length_km",
                        }
                    },
                    "routing_option": option,
                    "length_km": length_km,
                    "geometry": geometry,
                }
            )

        return results

    def _component_geometry(self, route):
        """Build geometry for one contiguous routing-option segment"""
        rows, cols = np.array(route).T
        x, y = rasterio.transform.xy(self._transform, rows, cols)
        if len(route) == 1:
            return Point(x, y)

        return LineString(simplify_using_slopes(list(zip(x, y, strict=True))))


def _validate_out_fp(out_fp, save_paths):
    """Validate output filepath extension"""
    out_fp = Path(out_fp)

    if save_paths and out_fp.suffix.lower() != ".gpkg":
        msg = (
            "When saving paths, the output file should have a '.gpkg' "
            f"extension to ensure proper format! Got input file: '{out_fp}'. "
            "Adding '.gpkg' extension... "
        )
        warn(msg, revrtWarning)
        out_fp = out_fp.with_suffix(".gpkg")
    elif not save_paths and out_fp.suffix.lower() != ".csv":
        msg = (
            "When not saving paths, the output file should have a '.csv' "
            f"extension to ensure proper format! Got input file: '{out_fp}'. "
            "Adding '.csv' extension... "
        )
        warn(msg, revrtWarning)
        out_fp = out_fp.with_suffix(".csv")

    logger.debug("Validated output filepath: %s", out_fp)
    return out_fp


def _routing_options_output_fp(out_fp):
    """pathlib.Path: Companion output path for routing-option pieces"""
    return out_fp.with_name(f"{out_fp.stem}_routing_options{out_fp.suffix}")


def _get_valid_points(points, arr_shape, point_type):
    """Get only points that are within array bounds"""
    valid_points = []
    invalid_points = []
    for point in points:
        if _is_valid_point(point, arr_shape):
            valid_points.append(point)
        else:
            invalid_points.append(point)

    if invalid_points:
        msg = (
            f"One or more of the {point_type} points are out of bounds for an "
            f"array of shape {arr_shape}: {invalid_points}\n"
            "Dropping these from consideration..."
        )
        warn(msg, revrtWarning)

    return valid_points


def _is_valid_point(point, arr_shape):
    """Check if point is within array bounds"""
    row, col, *__ = point
    return 0 <= row < arr_shape[0] and 0 <= col < arr_shape[1]


def _midpoint(start, end):
    """tuple: Midpoint between two route indices"""
    return (
        (start[0] + end[0]) / 2,
        (start[1] + end[1]) / 2,
    )


def _add_route_ids(route_definitions):
    """Add route IDs to route definitions missing them"""
    logger.info(
        "Route ID's missing from route definitions - adding definition "
        "index as route ID..."
    )
    return [
        (ind, start_points, end_points)
        for ind, (start_points, end_points) in enumerate(route_definitions)
    ]
