"""reVRt utilities"""

from .base import (
    buffer_routes,
    check_geotiff,
    delete_data_file,
    expand_dim_if_needed,
    file_full_path,
    load_data_using_layer_file_profile,
    features_to_route_table,
    save_data_using_layer_file_profile,
    save_data_using_custom_props,
    strip_path_keys,
    region_mapper,
    transform_xy,
)
from .handlers import (
    LayeredFile,
    IncrementalWriter,
    chunked_read_gpkg,
    num_feats_in_gpkg,
    gpkg_crs,
)
from .monitoring import (
    close_dask_client,
    dask_performance_report,
    log_mem,
    log_array_backend,
    log_runtime,
    elapsed_time_as_str,
)
from .raster import rasterize, rasterize_shape_file, integer_dimension_window
