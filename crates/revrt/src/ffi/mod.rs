mod py_tracing;
mod simplify_path;

use std::path::PathBuf;
use std::sync::mpsc;

use pyo3::exceptions::{PyException, PyIOError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::error::{Error, Result};
use crate::routing::RouteDefinition;
use crate::{ArrayIndex, RevrtRoutingSolutions, Solution, resolve, resolve_generator};

type PyRoutePoint = (u64, u64);
type PyPossibleRouteNodes = Vec<PyRoutePoint>;
type PyRouteResult = (Vec<PyRoutePoint>, f32, Vec<String>);
type PyRoutingSolutions = Vec<PyRouteResult>;
type PyRouteYield = PyResult<Option<(u32, PyRoutingSolutions)>>;
type PyRouteDefinition = (u32, PyPossibleRouteNodes, PyPossibleRouteNodes);

impl From<&PyRouteDefinition> for RouteDefinition {
    fn from(route: &PyRouteDefinition) -> RouteDefinition {
        let (id, start_points, end_points) = route;
        RouteDefinition {
            route_id: *id,
            start_inds: start_points
                .iter()
                .map(|(i, j)| ArrayIndex { i: *i, j: *j })
                .collect(),
            end_inds: end_points
                .iter()
                .map(|(i, j)| ArrayIndex { i: *i, j: *j })
                .collect(),
        }
    }
}

impl From<Solution<ArrayIndex, f32>> for PyRouteResult {
    fn from(solution: Solution<ArrayIndex, f32>) -> Self {
        let Solution {
            route,
            total_cost,
            dropped_barrier_layers,
        } = solution;
        let path = route.into_iter().map(Into::into).collect();
        (path, total_cost, dropped_barrier_layers)
    }
}

pyo3::create_exception!(_rust, revrtRustError, PyException);

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        match err {
            Error::IO(msg) => PyIOError::new_err(msg),
            object_store_error @ Error::ObjectStore(_) => {
                PyIOError::new_err(object_store_error.to_string())
            }
            Error::ZarrsArrayCreate(e) => PyIOError::new_err(e.to_string()),
            Error::ZarrsArray(e) => PyIOError::new_err(e.to_string()),
            Error::ZarrsStorage(e) => PyIOError::new_err(e.to_string()),
            Error::ZarrsGroupCreate(e) => PyIOError::new_err(e.to_string()),
            invalid_data_type @ Error::UnsupportedDataType(_, _) => {
                PyTypeError::new_err(invalid_data_type.to_string())
            }
            Error::Undefined(msg) => revrtRustError::new_err(msg),
            invalid_dataset_shape @ Error::InvalidDatasetShape { .. } => {
                PyValueError::new_err(invalid_dataset_shape.to_string())
            }
            invalid_algorithm @ Error::InvalidAlgorithm { .. } => {
                PyValueError::new_err(invalid_algorithm.to_string())
            }
        }
    }
}

/// A Python module implemented in Rust
#[pymodule]
fn _rust(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_paths, m)?)?;
    m.add_function(wrap_pyfunction!(simplify_using_slopes, m)?)?;
    m.add("revrtRustError", py.get_type::<revrtRustError>())?;
    m.add_class::<RouteFinder>()?;
    Ok(())
}

/// Simplify path geometry by removing points on the same slope.
///
/// This function removes all points that fall on the same slope,
/// only keeping the points required to define new segments (i.e.
/// changes in slope). This effectively reduces the storage
/// requirement for a shapely LineString without changing the
/// geometry at all.
///
/// Parameters
/// ----------
/// path : list of tuple
///     List of two-tuples containing floating-point numbers or integers
///     representing the points making up the route. This function works
///     best if these values represent array indices instead of values
///     in a CRS.
/// slope_tolerance : float, default=0.5
///     Tolerance value for determining if two slopes are the same.
///     For typical 8-direction routing, any value 1 and below should
///     suffice. By default, ``0.5``.
///
/// Returns
/// -------
/// list of tuple
///     Input path with redundant points removed.
#[pyfunction]
#[pyo3(signature = (path, slope_tolerance=0.5))]
fn simplify_using_slopes(path: Vec<(f64, f64)>, slope_tolerance: f64) -> Vec<(f64, f64)> {
    simplify_path::simplify_path(path, slope_tolerance)
}

/// Find least-cost paths for one or more starting points.
///
/// This function determined the least cost path for one or more starting
/// points to one or more ending points. A unique path is returned for
/// every starting point, but each route terminates when any of the ending
/// points are found. To ensure that a path is found to every end point,
/// call this function N times if you have N end points and pass a single
/// end point each time.
///
/// Parameters
/// ----------
/// zarr_fp : path-like
///     Path to zarr file containing cost layers.
/// cost_function : str
///     JSON string representation of the cost function. The following
///     keys are allowed in the cost function: "cost_layers",
///     "friction_layers", "barrier_layers", and
///     "ignore_invalid_costs". See the documentation of the cost
///     function for details on each of these inputs.
/// start : list of tuple
///     List of two-tuples containing non-negative integers representing
///     the indices in the array for the pixel from which routing should
///     begin. A unique path will be returned for each of the starting
///     points.
/// end : list of tuple
///     List of two-tuples containing non-negative integers representing
///     the indices in the array for the any allowed final pixel.
///     When the algorithm reaches any of these points, the routing
///     is terminated and the final path + cost is returned.
/// routing_layer_out_fp : path-like, optional
///    Optional path to a cost zarr file that will be used to store the routing
///    cost layers. If not given, the routing layers will be kept in a temporary
///    directory and deleted after the routing is done. By default, `None`.
/// log_level : int, optional
///     Logging level for Rust tracing emitted to stderr. Roughly follows the
///     Python logging module levels, where 0 = TRACE, 10 = DEBUG, 20 = INFO,
///     30 = WARN, and 40 = ERROR. If None is given, no logging is set up.
///     By default, `None`.
/// mem_limit_bytes : int, default=250_000_000
///     Overall memory limit to use for routing computations, in bytes.
///     By default, `250,000,000` (250MB).
/// algorithm : str, default="long_range_dijkstra"
///     Routing algorithm implementation to use. Supported values are
///     ``"astar"``, ``"long_range_astar"``, ``"long_range_dijkstra"``,
///     ``"bidirectional_long_range_dijkstra"``, and ``"dijkstra"``.
///     ``"astar"`` and ``"dijkstra"`` are faster in-memory
///     implementations but do not respect the `mem_limit_bytes` input.
///     Prefer a long-range option unless you know for a fact that your route
///     computations will not need much memory and speed is very important to you.
///     By default, ``"long_range_dijkstra"``.
///
/// Returns
/// -------
/// list of tuple
///     List of path routing results. Each result is a tuple
///     where the first element is a list of points that the
///     route goes through and the second element is the final
///     route cost.
#[pyfunction]
#[pyo3(signature = (zarr_fp, cost_function, start, end, algorithm="long_range_dijkstra", routing_layer_out_fp=None, mem_limit_bytes=250_000_000, log_level=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn find_paths(
    zarr_fp: PathBuf,
    cost_function: String,
    start: Vec<(u64, u64)>,
    end: Vec<(u64, u64)>,
    algorithm: &str,
    routing_layer_out_fp: Option<PathBuf>,
    mem_limit_bytes: u64,
    log_level: Option<u8>,
) -> PyResult<PyRoutingSolutions> {
    py_tracing::configure(log_level).map_err(PyErr::from)?;
    let start: Vec<ArrayIndex> = start
        .into_iter()
        .map(|(i, j)| ArrayIndex { i, j })
        .collect();
    let end: Vec<ArrayIndex> = end.into_iter().map(|(i, j)| ArrayIndex { i, j }).collect();
    let paths = resolve(
        zarr_fp,
        &cost_function,
        algorithm,
        &start,
        end,
        routing_layer_out_fp,
        mem_limit_bytes,
    )
    .map_err(PyErr::from)?;
    Ok(paths.into_iter().map(Into::into).collect())
}

/// Find least-cost paths for one or more starting points in parallel.
///
/// Parameters
/// ----------
/// zarr_fp : path-like
///     Path to zarr file containing cost layers.
/// cost_function : str
///     JSON string representation of the cost function. The following
///     keys are allowed in the cost function: "cost_layers",
///     "friction_layers", "barrier_layers", and
///     "ignore_invalid_costs". See the documentation of the cost
///     function for details on each of these inputs.
/// route_definitions : list of tuple
///     List of tuples containing path definitions. Each path definition
///     tuple should be of the form (int, list, list). The int input is
///     a route ID (non-negative) that you can use to link results to
///     input route definitions. The first list contains the starting
///     points and the second list contains the ending points. Each point
///     is represented as a two-tuple of non-negative integers representing
///     the indices in the array for the pixel indicating where routing should
///     begin/end. A unique path will be returned for each of the starting
///     points in each of the path definition tuples (assuming a valid path
///     exists).
/// routing_layer_out_fp : path-like, optional
///    Optional path to a cost zarr file that will be used to store the routing
///    cost layers. If not given, the routing layers will be kept in a temporary
///    directory and deleted after the routing is done. By default, `None`.
/// mem_limit_bytes : int, default=250_000_000
///     Overall memory limit to use for routing computations, in bytes.
///     By default, `250,000,000` (250MB).
/// algorithm : str, default="long_range_dijkstra"
///     Routing algorithm implementation to use. Supported values are
///     ``"astar"``, ``"long_range_astar"``, ``"long_range_dijkstra"``,
///     ``"bidirectional_long_range_dijkstra"``, and ``"dijkstra"``.
///     ``"astar"`` and ``"dijkstra"`` are faster in-memory
///     implementations but do not respect the `mem_limit_bytes` input.
///     Prefer a long-range option unless you know for a fact that your route
///     computations will not need much memory and speed is very important to you.
///     By default, ``"long_range_dijkstra"``.
/// log_level : int, optional
///     Logging level for Rust tracing emitted to stderr. Roughly follows the
///     Python logging module levels, where 0 = TRACE, 10 = DEBUG, 20 = INFO,
///     30 = WARN, and 40 = ERROR. If None is given, no logging is set up.
///     By default, `None`.
///
/// Yields
/// ------
/// tuple
///     A tuple representing the route finding result for a single
///     path definition. The first element is the route definition ID
///     (as given in the input) and the second element is a list of path
///     routing results. Each result is a tuple where the first element
///     is a list of points that the route goes through and the second
///     element is the final route cost. The third and fourth elements of
///     each routing result are lists containing the names and importance
///     ranks of any soft barriers dropped to obtain that specific path.
///     The result list will contain multiple tuples if the path definition
///     had multiple starting points. An empty list will be returned if no
///     paths were found from any of the starting points to any of the
///     ending points. This generator will yield one tuple per path
///     definition. Order is not guaranteed, so use the route ID input to
///     match results to inputs.
#[pyclass]
struct RouteFinder {
    /// Path to the Zarr file containing the cost layers
    #[pyo3(get)]
    zarr_fp: PathBuf,
    /// JSON string representation of the cost function configuration
    #[pyo3(get)]
    cost_function: String,
    /// Route definitions as ``(route_id, start_points, end_points)`` tuples
    #[pyo3(get)]
    route_definitions: Vec<PyRouteDefinition>,
    /// Cache size used during routing, in bytes
    #[pyo3(get)]
    mem_limit_bytes: u64,
    /// Optional output path where routing layers are persisted
    #[pyo3(get)]
    routing_layer_out_fp: Option<PathBuf>,
    /// Algorithm used for routing
    #[pyo3(get)]
    algorithm: String,
}

#[pymethods]
impl RouteFinder {
    #[new]
    #[pyo3(signature = (zarr_fp, cost_function, route_definitions, algorithm="long_range_dijkstra", routing_layer_out_fp=None, mem_limit_bytes=250_000_000, log_level=None))]
    fn new(
        zarr_fp: PathBuf,
        cost_function: String,
        route_definitions: Vec<PyRouteDefinition>,
        algorithm: &str,
        routing_layer_out_fp: Option<PathBuf>,
        mem_limit_bytes: u64,
        log_level: Option<u8>,
    ) -> PyResult<Self> {
        py_tracing::configure(log_level).map_err(PyErr::from)?;

        Ok(Self {
            zarr_fp,
            cost_function,
            route_definitions,
            routing_layer_out_fp,
            mem_limit_bytes,
            algorithm: algorithm.to_string(),
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<RouteOutputIter>> {
        let iter = RouteOutputIter::new(&slf)?;
        Py::new(slf.py(), iter)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "`RouteFinder` instance for {} routes",
            self.route_definitions.len()
        ))
    }
}

#[pyclass(unsendable)]
struct RouteOutputIter {
    receiver: Option<mpsc::Receiver<(u32, RevrtRoutingSolutions)>>,
    finished: bool,
}

impl RouteOutputIter {
    fn new(user_input: &RouteFinder) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        resolve_generator(
            &user_input.zarr_fp,
            &user_input.cost_function,
            user_input
                .route_definitions
                .iter()
                .map(Into::into)
                .collect::<Vec<_>>(),
            tx,
            user_input.routing_layer_out_fp.clone(),
            user_input.mem_limit_bytes,
            &user_input.algorithm,
        )?;
        Ok(Self {
            receiver: Some(rx),
            finished: false,
        })
    }
}

#[pymethods]
impl RouteOutputIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyRouteYield {
        if slf.finished {
            return Ok(None);
        }

        let receiver = match slf.receiver.take() {
            Some(receiver) => receiver,
            None => {
                slf.finished = true;
                return Ok(None);
            }
        };

        let py = slf.py();
        let (recv_result, receiver) = py.detach(move || {
            let result = receiver.recv();
            (result, receiver)
        });
        slf.receiver = Some(receiver);

        match recv_result {
            Ok((id, solutions)) => Ok(Some((id, solutions.into_iter().map(Into::into).collect()))),
            // Ok(Err(err)) => Err(err.into()),
            Err(_) => {
                slf.finished = true;
                Ok(None)
            }
        }
    }
}
