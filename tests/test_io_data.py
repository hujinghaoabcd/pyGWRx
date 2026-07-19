from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

import pygwrx.io as io
import pygwrx.io.data as data_module
from pygwrx.io import from_geodataframe, load_data, save_results, to_geodataframe


def test_public_api_exports_only_supported_data_functions():
    for name in ("load_data", "to_geodataframe", "from_geodataframe", "save_results"):
        assert name in io.__all__
        assert hasattr(io, name)

    assert not hasattr(data_module, "create_spatial_weights_matrix")
    assert not hasattr(data_module, "split_spatial_data")


def test_load_csv_filters_invalid_rows_and_keeps_arrays_aligned(tmp_path: Path):
    path = tmp_path / "observations.csv"
    pd.DataFrame(
        {
            "x1": [1.0, 2.0, np.inf, 4.0],
            "x2": [10.0, 20.0, 30.0, 40.0],
            "target": [5.0, 6.0, 7.0, np.nan],
            "east": [100.0, 101.0, 102.0, 103.0],
            "north": [30.0, 31.0, 32.0, 33.0],
        }
    ).to_csv(path, index=False)

    X, y, coords = load_data(
        path,
        x_cols=["x1", "x2"],
        y_col="target",
        coord_cols=("east", "north"),
    )

    assert X.shape == (2, 2)
    assert y is not None and y.shape == (2,)
    assert coords.shape == (2, 2)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert np.isfinite(coords).all()
    np.testing.assert_array_equal(X[:, 0], [1.0, 2.0])


def test_load_csv_can_auto_select_numeric_features(tmp_path: Path):
    path = tmp_path / "observations.csv"
    pd.DataFrame(
        {
            "x1": [1.0, 2.0],
            "label": ["a", "b"],
            "target": [3.0, 4.0],
            "east": [10.0, 11.0],
            "north": [20.0, 21.0],
        }
    ).to_csv(path, index=False)

    X, y, coords = load_data(
        path,
        y_col="target",
        coord_cols=("east", "north"),
    )

    assert X.shape == (2, 1)
    np.testing.assert_array_equal(X[:, 0], [1.0, 2.0])
    assert y is not None
    assert coords.shape == (2, 2)


def test_load_spatial_file_uses_point_geometry(tmp_path: Path):
    path = tmp_path / "points.geojson"
    gdf = gpd.GeoDataFrame(
        {"x1": [1.0, 2.0], "target": [3.0, 4.0]},
        geometry=gpd.points_from_xy([100.0, 101.0], [30.0, 31.0]),
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON", index=False)

    X, y, coords = load_data(path, x_cols=["x1"], y_col="target")

    assert X.shape == (2, 1)
    assert y is not None
    np.testing.assert_allclose(coords, [[100.0, 30.0], [101.0, 31.0]])


def test_load_data_requires_coordinates_for_csv(tmp_path: Path):
    path = tmp_path / "table.csv"
    pd.DataFrame({"x": [1.0], "y": [2.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="coord_cols"):
        load_data(path, x_cols=["x"])


def test_to_and_from_geodataframe_round_trip():
    X = np.array([[1.0, 10.0], [2.0, 20.0]])
    y = np.array([5.0, 6.0])
    coords = np.array([[100.0, 30.0], [101.0, 31.0]])

    gdf = to_geodataframe(
        X,
        y,
        coords,
        feature_names=["x1", "x2"],
        target_name="response",
        crs="EPSG:4326",
    )
    X2, y2, coords2 = from_geodataframe(
        gdf,
        x_cols=["x1", "x2"],
        y_col="response",
    )

    assert str(gdf.crs) == "EPSG:4326"
    np.testing.assert_allclose(X2, X)
    assert y2 is not None
    np.testing.assert_allclose(y2, y)
    np.testing.assert_allclose(coords2, coords)


def test_to_geodataframe_supports_no_target_and_no_default_crs():
    gdf = to_geodataframe(
        np.array([[1.0], [2.0]]),
        None,
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        feature_names=["prediction"],
    )
    assert list(gdf.columns) == ["prediction", "geometry"]
    assert gdf.crs is None


def test_to_geodataframe_validates_feature_names():
    X = np.ones((3, 2))
    coords = np.ones((3, 2))

    with pytest.raises(ValueError, match="length"):
        to_geodataframe(X, None, coords, feature_names=["only_one"])
    with pytest.raises(ValueError, match="duplicate"):
        to_geodataframe(X, None, coords, feature_names=["x", "x"])
    with pytest.raises(ValueError, match="target_name"):
        to_geodataframe(
            X,
            np.ones(3),
            coords,
            feature_names=["target", "x2"],
        )


def test_from_geodataframe_filters_invalid_rows():
    gdf = gpd.GeoDataFrame(
        {"x1": [1.0, np.inf, 3.0], "target": [4.0, 5.0, np.nan]},
        geometry=[
            gpd.points_from_xy([0.0], [0.0])[0],
            gpd.points_from_xy([1.0], [1.0])[0],
            gpd.points_from_xy([2.0], [2.0])[0],
        ],
    )
    X, y, coords = from_geodataframe(gdf, x_cols=["x1"], y_col="target")
    assert X.shape == (1, 1)
    assert y is not None and y.shape == (1,)
    assert coords.shape == (1, 2)


def test_save_numpy_and_dataframe_to_csv(tmp_path: Path):
    array_path = save_results(
        np.array([[1.0, 2.0], [3.0, 4.0]]), tmp_path / "array.csv"
    )
    frame_path = save_results(pd.DataFrame({"value": [1, 2]}), tmp_path / "frame.csv")

    assert array_path.exists()
    assert frame_path.exists()
    assert list(pd.read_csv(array_path).columns) == ["result_0", "result_1"]
    assert list(pd.read_csv(frame_path).columns) == ["value"]


def test_save_results_adds_suffix_when_format_is_explicit(tmp_path: Path):
    path = save_results(np.array([1.0, 2.0]), tmp_path / "predictions", format="csv")
    assert path.name == "predictions.csv"
    assert path.exists()


def test_save_geodataframe_to_geojson(tmp_path: Path):
    gdf = to_geodataframe(
        np.array([[1.0], [2.0]]),
        None,
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        feature_names=["coefficient"],
        crs="EPSG:4326",
    )
    path = save_results(gdf, tmp_path / "coefficients.geojson")
    loaded = gpd.read_file(path)
    assert len(loaded) == 2
    assert "coefficient" in loaded.columns


def test_spatial_output_requires_geodataframe(tmp_path: Path):
    with pytest.raises(TypeError, match="GeoDataFrame"):
        save_results(pd.DataFrame({"value": [1]}), tmp_path / "output.geojson")
