"""Test reVRt cost layer creator"""

from pathlib import Path
from itertools import chain, combinations

import pytest
import rioxarray
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import Affine, from_origin

from revrt.constants import METERS_IN_MILE
from revrt.models.cost_layers import LayerBuildConfig, RangeConfig, Rasterize
from revrt.costs.layer_creator import LayerCreator
from revrt.costs.masks import Masks
from revrt.utilities import LayeredFile
from revrt.exceptions import revrtAttributeError, revrtValueError
from revrt.warn import revrtWarning


_MUTEX_CONFIG_PARAMS = [
    {"global_value": 1},
    {"map": {1: 2}},
    {"bins": [RangeConfig(min=0, max=10, value=5)]},
    {"pass_through": True},
]
_NO_FI_PARAMS = [*_MUTEX_CONFIG_PARAMS[:-1], {"rasterize": Rasterize(value=1)}]


@pytest.fixture(scope="module")
def tiff_layers_for_testing(tmp_path_factory):
    """Test TIFF layers for testing LayerCreator"""
    layer_dir = tmp_path_factory.mktemp("data")

    width = height = 3
    cell_size = 1
    x0 = y0 = 0
    transform = from_origin(x0, y0, cell_size, cell_size)

    layers = {
        "friction_1.tif": np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        "fi_1.tif": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
        "fi_2.tif": np.array([[0, 0, 0], [0, 0, 0], [2, 2, 2]]),
    }

    for layer_fn, data in layers.items():
        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={
                "x": x0 + np.arange(width) * cell_size + cell_size / 2,
                "y": y0 - np.arange(height) * cell_size - cell_size / 2,
            },
            name="test_band",
        )

        da = da.rio.write_crs("EPSG:4326")
        da.rio.write_transform(transform)
        da.rio.to_raster(layer_dir / layer_fn, driver="GTiff")

    return layer_dir, layers


@pytest.fixture(scope="module")
def lf_instance(tmp_path_factory, tiff_layers_for_testing):
    """LayeredFile instance for testing"""
    test_fp = tmp_path_factory.mktemp("zarr_data") / "test.zarr"
    lf = LayeredFile(test_fp)
    layer_dir, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))
    lf.create_new(layer_dir / layer_fn)
    return lf


@pytest.fixture(scope="module")
def mask_instance():
    """Mask instance for testing"""
    masks = Masks(
        shape=(3, 3),
        crs="ESRI:102008",
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
    )
    masks._dry_mask = np.array(
        [
            [False, False, True],
            [False, False, True],
            [False, False, True],
        ]
    )

    masks._wet_mask = np.array(
        [
            [True, False, False],
            [True, False, False],
            [True, False, False],
        ]
    )

    masks._landfall_mask = np.array(
        [
            [False, True, False],
            [False, True, False],
            [False, True, False],
        ]
    )
    return masks


@pytest.fixture(scope="module")
def builder_instance(
    tmp_path_factory, lf_instance, mask_instance, tiff_layers_for_testing
):
    """LayerCreator instance for testing"""
    layer_dir, _ = tiff_layers_for_testing
    out_dir = tmp_path_factory.mktemp("out_data")
    return LayerCreator(
        lf_instance,
        mask_instance,
        input_layer_dir=layer_dir,
        output_tiff_dir=out_dir,
    )


def test_no_write(tmp_path, mask_instance, tiff_layers_for_testing):
    """Test build layer but don't writer to file"""
    layer_dir, layers = tiff_layers_for_testing

    config = {
        "fi_1.tif": LayerBuildConfig(
            extent="wet+",
            forced_inclusion=True,
        ),
        "friction_1.tif": LayerBuildConfig(
            extent="all", map={1: 1, 2: 2, 3: 3}
        ),
        "fi_2.tif": LayerBuildConfig(
            extent="dry+",
            forced_inclusion=True,
        ),
    }

    out_dir = tmp_path / "out_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_fp = tmp_path / "zarr_data" / "test.zarr"
    test_fp.parent.mkdir(parents=True, exist_ok=True)

    lf = LayeredFile(test_fp)
    layer_fn = next(iter(layers))
    lf.create_new(layer_dir / layer_fn)

    tiff_fp = out_dir / "friction.tif"
    assert not tiff_fp.exists()

    builder = LayerCreator(
        lf,
        mask_instance,
        input_layer_dir=layer_dir,
        output_tiff_dir=out_dir,
    )
    builder.build("friction", config, write_to_file=False)
    assert tiff_fp.exists()

    with rioxarray.open_rasterio(tiff_fp, chunks="auto") as ds:
        assert np.allclose(np.array([[[1, 1, 1], [0, 0, 2], [3, 0, 0]]]), ds)

    with xr.open_dataset(lf.fp, consolidated=False, engine="zarr") as ds:
        assert "friction" not in ds


def test_forced_inclusion(lf_instance, builder_instance):
    """Test forced inclusions"""
    config = {
        "fi_1.tif": LayerBuildConfig(
            extent="wet+",
            forced_inclusion=True,
        ),
        "friction_1.tif": LayerBuildConfig(
            extent="all", map={1: 1, 2: 2, 3: 3}
        ),
        "fi_2.tif": LayerBuildConfig(
            extent="dry+",
            forced_inclusion=True,
        ),
    }
    builder_instance.build("friction", config, write_to_file=True)

    with xr.open_dataset(
        lf_instance.fp, consolidated=False, engine="zarr"
    ) as ds:
        assert np.allclose(
            np.array([[[1, 1, 1], [0, 0, 2], [3, 0, 0]]]), ds["friction"]
        )


def test_forced_inclusion_all(lf_instance, builder_instance):
    """Test forced inclusions for 'all' extent"""

    config = {
        "friction_1.tif": LayerBuildConfig(
            extent="all", map={1: 1, 2: 2, 3: 3}
        ),
        "fi_1.tif": LayerBuildConfig(extent="all", forced_inclusion=True),
    }
    builder_instance.build("friction_all", config, write_to_file=True)

    with xr.open_dataset(
        lf_instance.fp, consolidated=False, engine="zarr"
    ) as ds:
        assert np.allclose(
            np.array([[1, 1, 1], [0, 0, 0], [3, 3, 3]]), ds["friction_all"]
        )


def test_global_value(builder_instance):
    """Test global_value key in LayerBuildConfig"""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    config = LayerBuildConfig(extent="dry", global_value=-1)
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1]])).all()

    config = LayerBuildConfig(extent="all", global_value=25)
    result = builder_instance._process_raster_data(data, config)
    assert (
        result == np.array([[25, 25, 25], [25, 25, 25], [25, 25, 25]])
    ).all()

    config = LayerBuildConfig(extent="landfall", global_value=100)
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[0, 100, 0], [0, 100, 0], [0, 100, 0]])).all()


def test_bins(builder_instance):
    """Test bins key in LayerBuildConfig"""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    config = LayerBuildConfig(
        extent="wet+", bins=[RangeConfig(min=1, max=5, value=4)]
    )
    with pytest.warns(revrtWarning, match="Gap detected"):
        result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[4, 4, 0], [4, 0, 0], [0, 0, 0]])).all()

    config = LayerBuildConfig(
        extent="dry+", bins=[RangeConfig(min=5, max=9, value=5)]
    )
    with pytest.warns(revrtWarning, match="Gap detected"):
        result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[0, 0, 0], [0, 5, 5], [0, 5, 0]])).all()

    config = LayerBuildConfig(
        extent="all", bins=[RangeConfig(min=2, max=9, value=5)]
    )
    with pytest.warns(revrtWarning, match="Gap detected"):
        result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[0, 5, 5], [5, 5, 5], [5, 5, 0]])).all()


def test_complex_bins(builder_instance):
    """Test bins key with multiple bins in LayerBuildConfig"""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    config = LayerBuildConfig(
        extent="wet+",
        bins=[
            RangeConfig(min=1, max=5, value=4),
            RangeConfig(min=4, max=10, value=10),
        ],
    )
    with pytest.warns(revrtWarning):
        result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[4, 4, 0], [14, 10, 0], [10, 10, 0]])).all()

    config = LayerBuildConfig(
        extent="all",
        bins=[
            RangeConfig(min=1, max=6, value=5),
            RangeConfig(min=4, max=9, value=1),
        ],
    )
    with pytest.warns(revrtWarning):
        result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[5, 5, 5], [6, 6, 1], [1, 1, 0]])).all()


def test_map(builder_instance):
    """Test map key in LayerBuildConfig"""
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    config = LayerBuildConfig(
        extent="wet",
        map={1: 5, 3: 9},
    )
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[5, 0, 0], [0, 0, 0], [9, 0, 0]])).all()

    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    config = LayerBuildConfig(
        extent="landfall",
        map={1: 5, 3: 9},
    )
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[0, 5, 0], [0, 0, 0], [0, 9, 0]])).all()


def test_pass_through(builder_instance):
    """Test pass through key in LayerBuildConfig"""
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    config = LayerBuildConfig(extent="wet", pass_through=True)
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])).all()

    config = LayerBuildConfig(extent="landfall", pass_through=True)
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]])).all()

    config = LayerBuildConfig(extent="all", pass_through=True)
    result = builder_instance._process_raster_data(data, config)
    assert (result == data).all()


def test_bin_config_sanity_checking(builder_instance, tiff_layers_for_testing):
    """Test cost binning config sanity checking"""
    __, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))

    reverse_bins = [RangeConfig(min=10, max=0, value=1)]
    config = LayerBuildConfig(extent="all", bins=reverse_bins)
    with pytest.raises(
        revrtAttributeError, match="Min is greater than max for bin config"
    ):
        builder_instance._process_raster_layer(layer_fn, config)

    bin_config = [
        RangeConfig(min=1, max=2, value=3),
        RangeConfig(min=2, max=5, value=4),
    ]
    good_config = LayerBuildConfig(extent="all", bins=bin_config)
    with pytest.warns(revrtWarning, match="Gap detected"):
        builder_instance._process_raster_layer(layer_fn, good_config)


def test_bin_config_warns_full_range(
    builder_instance, tiff_layers_for_testing
):
    """Test cost binning config warns for full range"""
    __, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))

    reverse_bins = [RangeConfig(value=1)]
    config = LayerBuildConfig(extent="all", bins=reverse_bins)
    with pytest.warns(
        revrtWarning,
        match=(
            "Bin covers all possible values, did you forget to set min or max?"
        ),
    ):
        builder_instance._process_raster_layer(layer_fn, config)


def test_tiff_config_rasterize(builder_instance, tiff_layers_for_testing):
    """Test error if including `rasterize` for TIFF"""
    __, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))

    config = LayerBuildConfig(rasterize=Rasterize(value=1))
    with pytest.raises(
        revrtValueError,
        match=f"'rasterize' is only for vectors. Found in {layer_fn!r} config",
    ):
        builder_instance._process_raster_layer(layer_fn, config)


@pytest.mark.parametrize(
    "mutex_params",
    chain.from_iterable(
        combinations(_MUTEX_CONFIG_PARAMS, r)
        for r in range(2, len(_MUTEX_CONFIG_PARAMS) + 1)
    ),
)
def test_tiff_config_mutually_exclusive_entries(
    builder_instance, mutex_params, tiff_layers_for_testing
):
    """Test error for mutually exclusive entries in LayerBuildConfig"""
    __, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))

    config_params = {k: v for param in mutex_params for k, v in param.items()}
    config = LayerBuildConfig(**config_params)
    with pytest.raises(
        revrtValueError,
        match=f"Keys .* are mutually exclusive .* {layer_fn!r} raster config",
    ):
        builder_instance._process_raster_layer(layer_fn, config)


def test_tiff_config_missing_entries(
    builder_instance, tiff_layers_for_testing
):
    """Test error for missing required entries in LayerBuildConfig"""
    __, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))

    config = LayerBuildConfig()
    with pytest.raises(
        revrtValueError,
        match=(
            f"Either .* must be specified for a raster, .* {layer_fn!r} config"
        ),
    ):
        builder_instance._process_raster_layer(layer_fn, config)


def test_bad_filetype_input(builder_instance):
    """Test a bad file type input in the config"""
    config = {
        "fi_1.txt": LayerBuildConfig(extent="wet+", forced_inclusion=False)
    }
    with pytest.raises(
        revrtValueError,
        match=r"Unsupported file extension on 'fi_1.txt'",
    ):
        builder_instance.build("friction", config, write_to_file=True)


def test_bad_filetype_input_forced_inclusion(builder_instance):
    """Test a bad file type input in the config"""
    config = {
        "fi_1.txt": LayerBuildConfig(extent="wet+", forced_inclusion=True)
    }
    with pytest.raises(
        revrtValueError,
        match=r"Forced inclusion file 'fi_1.txt' does not end with .tif",
    ):
        builder_instance.build("friction", config, write_to_file=True)


@pytest.mark.parametrize("bad_input", _NO_FI_PARAMS)
def test_bad_config_input_forced_inclusion(builder_instance, bad_input):
    """Test a bad file type input in the config"""
    config = {
        "fi_1.tif": LayerBuildConfig(
            extent="wet+", forced_inclusion=True, **bad_input
        )
    }
    with pytest.raises(
        revrtValueError, match=r".* are not allowed .* 'fi_1.tif'"
    ):
        builder_instance.build("friction", config, write_to_file=True)


def test_getting_bad_mask(builder_instance):
    """Test getting bad mask"""

    with pytest.raises(
        revrtAttributeError, match="Mask for extent of 'all' is unnecessary"
    ):
        builder_instance._get_mask("all")

    with pytest.raises(revrtAttributeError, match="Unknown mask type: 'DNE'"):
        builder_instance._get_mask("DNE")


def test_bad_config_no_rasterize_vector(builder_instance):
    """Test a bad file type input in the config"""
    config = {"fi_1.gpkg": LayerBuildConfig(extent="wet+")}
    with pytest.raises(
        revrtValueError,
        match=r'is a vector but the config is missing key "rasterize"',
    ):
        builder_instance.build("friction", config, write_to_file=False)


@pytest.mark.parametrize("all_touched", [True, False])
@pytest.mark.parametrize("extent", ["all", "dry"])
@pytest.mark.parametrize("cpm", [True, False])
@pytest.mark.parametrize("reproject", [True, False])
@pytest.mark.parametrize("buffer", [None, 20])
def test_rasterizing_shape_file(
    tmp_path, mask_instance, all_touched, extent, cpm, reproject, buffer
):
    """Test rasterizing a basic shapefile"""
    fn = "test_basic_shape.gpkg"
    vector_file = tmp_path / fn
    tiff_fp = tmp_path / "friction.tif"
    assert not tiff_fp.exists()

    template_tiff = tmp_path / "template.tif"
    crs = "ESRI:102008"
    transform = Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5)

    width = height = 3
    cell_size = 5
    x0, y0 = -12.5, 12.5

    da = xr.DataArray(
        np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        dims=("y", "x"),
        coords={
            "x": x0 + np.arange(width) * cell_size + cell_size / 2,
            "y": y0 - np.arange(height) * cell_size - cell_size / 2,
        },
        name="test_band",
    )

    da = da.rio.write_crs(crs)
    da.rio.write_transform(transform)
    da.rio.to_raster(template_tiff, driver="GTiff")

    basic_shape = gpd.GeoDataFrame(geometry=[box(-5, -20, 20, 20)], crs=crs)
    basic_shape.to_file(vector_file, driver="GPKG")

    test_fp = tmp_path / "test.zarr"
    lf = LayeredFile(test_fp).create_new(template_tiff)

    config = {
        fn: LayerBuildConfig(
            extent=extent,
            rasterize=Rasterize(
                value=1000,
                buffer=buffer,
                reproject=reproject,
                all_touched=all_touched,
            ),
        )
    }
    builder = LayerCreator(
        lf, mask_instance, input_layer_dir=tmp_path, output_tiff_dir=tmp_path
    )
    builder.build(
        "friction", config, write_to_file=False, values_are_costs_per_mile=cpm
    )

    assert tiff_fp.exists()
    value = 1000 / METERS_IN_MILE * cell_size if cpm else 1000
    with rioxarray.open_rasterio(tiff_fp, chunks="auto") as ds:
        if extent != "dry" and buffer:
            assert np.allclose(np.full(fill_value=value, shape=(3, 3)), ds), (
                f"{np.array(ds.values)}"
            )
        elif extent != "dry" and all_touched:
            assert np.allclose(
                np.array(
                    [[[0, value, value], [0, value, value], [0, value, value]]]
                ),
                ds,
            ), f"{np.array(ds.values)}"
        else:
            assert np.allclose(
                np.array([[[0, 0, value], [0, 0, value], [0, 0, value]]]), ds
            ), f"{np.array(ds.values)}"


def test_cost_binning_results(builder_instance):
    """Test results of creating cost raster using bins"""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    bins = [
        RangeConfig(max=2, value=1),
        RangeConfig(min=2, max=4, value=2),
        RangeConfig(min=4, max=8, value=3),
        RangeConfig(min=8, value=4),
    ]
    config = LayerBuildConfig(extent="all", bins=bins)
    output = builder_instance._process_raster_data(data, config)
    assert (output == np.array([[1, 2, 2], [3, 3, 3], [3, 4, 4]])).all()

    bins = [
        RangeConfig(min=2, max=4, value=2),
        RangeConfig(min=4, max=8, value=3),
    ]
    config = LayerBuildConfig(extent="all", bins=bins)
    output = builder_instance._process_raster_data(data, config)
    assert (output == np.array([[0, 2, 2], [3, 3, 3], [3, 0, 0]])).all()

    data = np.array([[-600, -400, -50], [-700, -250, 70], [-500, -150, -70]])
    bins = [
        RangeConfig(max=-500, value=999),
        RangeConfig(min=-500, max=-300, value=666),
        RangeConfig(min=-300, max=-100, value=333),
        RangeConfig(min=-100, value=111),
    ]
    config = LayerBuildConfig(extent="all", bins=bins)
    output = builder_instance._process_raster_data(data, config)
    assert (
        output == np.array([[999, 666, 111], [999, 333, 111], [666, 333, 111]])
    ).all()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
