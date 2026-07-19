# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Data loading, conversion, and result-saving utilities.

The functions in this module convert user files and GeoDataFrames into aligned model arrays and save tabular or spatial results.

Author:
    Jinghao Hu
"""

from __future__ import annotations

__author__ = "Jinghao Hu"
__license__ = "MIT"

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pygwrx._optional import import_required_dependency

if TYPE_CHECKING:
    import geopandas as gpd

PathLike = Union[str, Path]


def _optional_geopandas() -> Optional[ModuleType]:
    """Return GeoPandas when installed, otherwise ``None``."""
    try:
        return import_required_dependency("geopandas", purpose="geospatial data I/O")
    except ImportError:
        return None


def _require_geopandas(*, purpose: str) -> ModuleType:
    """Return GeoPandas or raise an actionable optional-dependency error."""
    return import_required_dependency("geopandas", purpose=purpose)


_SUPPORTED_INPUT_SUFFIXES = {
    ".csv",
    ".shp",
    ".geojson",
    ".json",
    ".gpkg",
    ".parquet",
    ".pq",
}

_FORMAT_ALIASES = {
    "csv": "csv",
    "parquet": "parquet",
    "pq": "parquet",
    "shapefile": "shapefile",
    "shp": "shapefile",
    "geojson": "geojson",
    "json": "geojson",
    "gpkg": "gpkg",
    "geopackage": "gpkg",
}

_FORMAT_SUFFIXES = {
    "csv": {".csv"},
    "parquet": {".parquet", ".pq"},
    "shapefile": {".shp"},
    "geojson": {".geojson", ".json"},
    "gpkg": {".gpkg"},
}

_DEFAULT_SUFFIX = {
    "csv": ".csv",
    "parquet": ".parquet",
    "shapefile": ".shp",
    "geojson": ".geojson",
    "gpkg": ".gpkg",
}


def _normalize_column_names(
    columns: Optional[Sequence[str]],
    *,
    parameter_name: str,
) -> Optional[List[str]]:
    """Validate and normalize a sequence of column names."""
    if columns is None:
        return None
    if isinstance(columns, str):
        raise TypeError(
            f"{parameter_name} must be a sequence of column names, not a string."
        )

    normalized = list(columns)
    if not normalized:
        raise ValueError(f"{parameter_name} cannot be empty.")
    if not all(isinstance(column, str) and column.strip() for column in normalized):
        raise TypeError(f"Every entry in {parameter_name} must be a non-empty string.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{parameter_name} contains duplicate column names.")
    return normalized


def _validate_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, role: str
) -> None:
    """Raise a clear error when requested columns are missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing {role} column(s): {missing}.")


def _read_data_file(path: Path) -> pd.DataFrame:
    """Read a supported tabular or spatial file."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".shp", ".geojson", ".json", ".gpkg"}:
        geopandas = _require_geopandas(purpose="reading spatial files")
        return geopandas.read_file(path)
    if suffix in {".parquet", ".pq"}:
        # GeoPandas preserves geometry/CRS for GeoParquet when available.
        optional_geopandas = _optional_geopandas()
        if optional_geopandas is not None:
            try:
                return optional_geopandas.read_parquet(path)
            except (ValueError, TypeError):
                pass
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported file format {suffix!r}. Supported extensions are: "
        f"{sorted(_SUPPORTED_INPUT_SUFFIXES)}."
    )


def _extract_point_coordinates(gdf: "gpd.GeoDataFrame") -> np.ndarray:
    """Extract finite x/y values from the active point geometry column."""
    geopandas = _require_geopandas(purpose="GeoDataFrame coordinate extraction")
    if not isinstance(gdf, geopandas.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame.")
    if gdf.geometry.name not in gdf.columns:
        raise ValueError("The GeoDataFrame has no active geometry column.")

    geometry = gdf.geometry
    non_null = geometry.notna() & ~geometry.is_empty
    non_point = non_null & (geometry.geom_type != "Point")
    if bool(non_point.any()):
        found = sorted(set(geometry.loc[non_point].geom_type.astype(str)))
        raise ValueError(
            "GeoDataFrame geometry must contain only Point geometries; "
            f"found {found}. Convert polygons or lines to explicit representative points first."
        )

    x = pd.Series(np.nan, index=gdf.index, dtype=float)
    y = pd.Series(np.nan, index=gdf.index, dtype=float)
    if bool(non_null.any()):
        x.loc[non_null] = geometry.loc[non_null].x.to_numpy(dtype=float)
        y.loc[non_null] = geometry.loc[non_null].y.to_numpy(dtype=float)
    return np.column_stack([x.to_numpy(), y.to_numpy()])


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert selected columns to numeric values, coercing invalid values to NaN."""
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")


def _finite_row_mask(
    X: np.ndarray,
    y: Optional[np.ndarray],
    coords: np.ndarray,
) -> np.ndarray:
    """Create one shared finite-value mask for X, y, and coordinates."""
    mask = np.isfinite(X).all(axis=1) & np.isfinite(coords).all(axis=1)
    if y is not None:
        mask &= np.isfinite(y)
    return mask


def load_data(
    filepath: PathLike,
    x_cols: Optional[Sequence[str]] = None,
    y_col: Optional[str] = None,
    coord_cols: Optional[Tuple[str, str]] = None,
    *,
    dropna: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Load a user data file and extract model features, target, and coordinates. Extract
    predictors, an optional response, and coordinates from a user data file.

    Supported inputs are CSV, Shapefile, GeoJSON, GeoPackage, and Parquet.
    For spatial files, coordinates are read from Point geometry when
    ``coord_cols`` is omitted.

    Args:
        filepath: Input data file.
        x_cols: Feature columns. When omitted, all numeric columns except the target,
            coordinate columns, and active geometry column are used.
        y_col: Target column. When omitted, ``y`` is returned as ``None``.
        coord_cols: Explicit x/y or longitude/latitude columns. Required for non-spatial
            tables such as CSV unless the input is a GeoDataFrame-backed format.
        dropna: Remove rows containing NaN or infinite values in X, y, or coordinates.
            One shared row mask is used, so returned arrays always remain aligned.

    Returns:
        X: Feature matrix with floating-point dtype.
        y: Target values, or ``None`` when ``y_col`` is omitted.
        coords: Coordinate matrix.

    Examples:
        Load a CSV file with explicit coordinate columns:

        >>> from pygwrx.io import load_data
        >>> X, y, coords = load_data(
        ...     "observations.csv",
        ...     x_cols=["income", "population"],
        ...     y_col="house_price",
        ...     coord_cols=("x", "y"),
        ... )

        Load a point Shapefile using its geometry:

        >>> X, y, coords = load_data(
        ...     "observations.shp",
        ...     x_cols=["income", "population"],
        ...     y_col="house_price",
        ... )
    """
    if not isinstance(filepath, (str, Path)):
        raise TypeError("filepath must be a string or pathlib.Path.")
    if not isinstance(dropna, (bool, np.bool_)):
        raise TypeError("dropna must be a boolean.")
    if y_col is not None and (not isinstance(y_col, str) or not y_col.strip()):
        raise TypeError("y_col must be a non-empty string or None.")

    path = Path(filepath).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"filepath must point to a file: {path}")
    if path.suffix.lower() not in _SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(
            f"Unsupported file format {path.suffix!r}. Supported extensions are: "
            f"{sorted(_SUPPORTED_INPUT_SUFFIXES)}."
        )

    frame = _read_data_file(path)
    x_columns = _normalize_column_names(x_cols, parameter_name="x_cols")

    if coord_cols is not None:
        if not isinstance(coord_cols, (tuple, list)) or len(coord_cols) != 2:
            raise TypeError("coord_cols must contain exactly two column names.")
        coord_columns = _normalize_column_names(coord_cols, parameter_name="coord_cols")
        assert coord_columns is not None  # for type checkers
    else:
        coord_columns = None

    if y_col is not None:
        _validate_columns(frame, [y_col], role="target")
    if coord_columns is not None:
        _validate_columns(frame, coord_columns, role="coordinate")

    geometry_name: Optional[str] = None
    geopandas = _optional_geopandas()
    if geopandas is not None and isinstance(frame, geopandas.GeoDataFrame):
        geometry_name = frame.geometry.name

    if x_columns is None:
        excluded = {column for column in ([y_col] if y_col else [])}
        if coord_columns is not None:
            excluded.update(coord_columns)
        if geometry_name is not None:
            excluded.add(geometry_name)
        x_columns = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column not in excluded
        ]
        if not x_columns:
            raise ValueError(
                "No numeric feature columns were found. Provide x_cols explicitly."
            )
    else:
        _validate_columns(frame, x_columns, role="feature")

    if y_col is not None and y_col in x_columns:
        raise ValueError("y_col cannot also appear in x_cols.")
    if coord_columns is not None:
        overlap = sorted(set(x_columns).intersection(coord_columns))
        if overlap:
            raise ValueError(
                f"Coordinate columns cannot also be feature columns: {overlap}."
            )

    X = _numeric_frame(frame, x_columns).to_numpy(dtype=float)
    y = (
        pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
        if y_col is not None
        else None
    )

    if coord_columns is not None:
        coords = _numeric_frame(frame, coord_columns).to_numpy(dtype=float)
    elif geopandas is not None and isinstance(frame, geopandas.GeoDataFrame):
        coords = _extract_point_coordinates(frame)
    else:
        raise ValueError(
            "coord_cols must be provided for a non-spatial table such as CSV."
        )

    if X.shape[0] != coords.shape[0] or (y is not None and y.shape[0] != X.shape[0]):
        raise RuntimeError(
            "Loaded features, target, and coordinates are not row-aligned."
        )

    if dropna:
        mask = _finite_row_mask(X, y, coords)
        if not bool(mask.any()):
            raise ValueError(
                "No valid rows remain after removing NaN and infinite values."
            )
        X = X[mask]
        coords = coords[mask]
        if y is not None:
            y = y[mask]

    return X, y, coords


def to_geodataframe(
    X: np.ndarray,
    y: Optional[np.ndarray],
    coords: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    target_name: str = "target",
    crs: Optional[Union[str, int]] = None,
) -> gpd.GeoDataFrame:
    """Convert aligned arrays into a point GeoDataFrame.

    ``crs`` defaults to ``None`` deliberately: assigning EPSG:4326 to unknown
    projected coordinates would mislabel rather than transform the data.

    Args:
        X: Feature or result matrix.
        y: Optional target/result vector.
        coords: x/y or longitude/latitude coordinates.
        feature_names: Names of X columns. Defaults to ``feature_0``, ``feature_1``, ...
        target_name: Name used for y when y is supplied.
        crs: Coordinate reference system, for example ``"EPSG:32650"``.

    Returns:
        GeoDataFrame: Point GeoDataFrame containing the supplied columns.

    Examples:
        >>> from pygwrx.io import to_geodataframe
        >>> gdf = to_geodataframe(
        ...     X,
        ...     y,
        ...     coords,
        ...     feature_names=["income", "population"],
        ...     target_name="house_price",
        ...     crs="EPSG:32650",
        ... )
    """
    X_array = np.asarray(X)
    coords_array = np.asarray(coords)

    if X_array.ndim == 1:
        X_array = X_array.reshape(-1, 1)
    if X_array.ndim != 2:
        raise ValueError("X must be a one- or two-dimensional array.")
    if coords_array.ndim != 2 or coords_array.shape[1] != 2:
        raise ValueError("coords must have shape (n_samples, 2).")
    if X_array.shape[0] != coords_array.shape[0]:
        raise ValueError("X and coords must contain the same number of rows.")

    try:
        coords_float = coords_array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("coords must contain numeric values.") from exc
    if not np.isfinite(coords_float).all():
        raise ValueError("coords must contain only finite numeric values.")

    names = _normalize_column_names(feature_names, parameter_name="feature_names")
    if names is None:
        names = [f"feature_{index}" for index in range(X_array.shape[1])]
    if len(names) != X_array.shape[1]:
        raise ValueError(
            "feature_names length must equal the number of columns in X: "
            f"expected {X_array.shape[1]}, got {len(names)}."
        )

    reserved_geometry_name = "geometry"
    if reserved_geometry_name in names:
        raise ValueError("feature_names cannot contain the reserved name 'geometry'.")

    if not isinstance(target_name, str) or not target_name.strip():
        raise TypeError("target_name must be a non-empty string.")
    if y is not None:
        if target_name == reserved_geometry_name:
            raise ValueError("target_name cannot be 'geometry'.")
        if target_name in names:
            raise ValueError("target_name cannot duplicate a feature name.")

    data = {name: X_array[:, index] for index, name in enumerate(names)}

    if y is not None:
        y_array = np.asarray(y)
        if y_array.ndim == 2 and y_array.shape[1] == 1:
            y_array = y_array[:, 0]
        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional or a single-column array.")
        if y_array.shape[0] != X_array.shape[0]:
            raise ValueError("X, y, and coords must contain the same number of rows.")
        data[target_name] = y_array

    geopandas = _require_geopandas(purpose="GeoDataFrame conversion")
    geometry = geopandas.points_from_xy(coords_float[:, 0], coords_float[:, 1])
    return geopandas.GeoDataFrame(data, geometry=geometry, crs=crs)


def from_geodataframe(
    gdf: "gpd.GeoDataFrame",
    x_cols: Optional[Sequence[str]] = None,
    y_col: Optional[str] = None,
    *,
    dropna: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Extract aligned arrays from a point GeoDataFrame.

    Args:
        gdf: Input GeoDataFrame with Point geometry.
        x_cols: Feature columns. When omitted, all numeric columns except y are used.
        y_col: Target column. When omitted, y is returned as ``None``.
        dropna: Remove rows containing NaN or infinite values in X, y, or coordinates.

    Returns:
        X, y, coords: Floating-point feature matrix, optional target vector, and coordinates.

    Examples:
        >>> from pygwrx.io import from_geodataframe
        >>> X, y, coords = from_geodataframe(
        ...     gdf,
        ...     x_cols=["income", "population"],
        ...     y_col="house_price",
        ... )
    """
    geopandas = _require_geopandas(purpose="GeoDataFrame conversion")
    if not isinstance(gdf, geopandas.GeoDataFrame):
        raise TypeError("gdf must be a geopandas.GeoDataFrame.")
    if not isinstance(dropna, (bool, np.bool_)):
        raise TypeError("dropna must be a boolean.")
    if y_col is not None and (not isinstance(y_col, str) or not y_col.strip()):
        raise TypeError("y_col must be a non-empty string or None.")

    x_columns = _normalize_column_names(x_cols, parameter_name="x_cols")
    if y_col is not None:
        _validate_columns(gdf, [y_col], role="target")

    if x_columns is None:
        geometry_name = gdf.geometry.name
        excluded = {geometry_name}
        if y_col is not None:
            excluded.add(y_col)
        x_columns = [
            column
            for column in gdf.select_dtypes(include=[np.number]).columns
            if column not in excluded
        ]
        if not x_columns:
            raise ValueError(
                "No numeric feature columns were found. Provide x_cols explicitly."
            )
    else:
        _validate_columns(gdf, x_columns, role="feature")

    if y_col is not None and y_col in x_columns:
        raise ValueError("y_col cannot also appear in x_cols.")

    X = _numeric_frame(gdf, x_columns).to_numpy(dtype=float)
    y = (
        pd.to_numeric(gdf[y_col], errors="coerce").to_numpy(dtype=float)
        if y_col is not None
        else None
    )
    coords = _extract_point_coordinates(gdf)

    if dropna:
        mask = _finite_row_mask(X, y, coords)
        if not bool(mask.any()):
            raise ValueError(
                "No valid rows remain after removing NaN and infinite values."
            )
        X = X[mask]
        coords = coords[mask]
        if y is not None:
            y = y[mask]

    return X, y, coords


def _normalize_output_format(
    filepath: Path, output_format: Optional[str]
) -> Tuple[str, Path]:
    """Resolve output format and normalize a missing filename suffix."""
    if output_format is None:
        suffix = filepath.suffix.lower()
        for candidate, suffixes in _FORMAT_SUFFIXES.items():
            if suffix in suffixes:
                return candidate, filepath
        raise ValueError(
            "Cannot infer output format from filepath. Use one of these suffixes: "
            f"{sorted({suffix for values in _FORMAT_SUFFIXES.values() for suffix in values})}."
        )

    if not isinstance(output_format, str) or not output_format.strip():
        raise TypeError("format must be a non-empty string or None.")
    key = output_format.strip().lower()
    if key not in _FORMAT_ALIASES:
        raise ValueError(
            f"Unsupported output format {output_format!r}. Supported formats are: "
            f"{sorted(_FORMAT_ALIASES)}."
        )
    normalized = _FORMAT_ALIASES[key]

    if not filepath.suffix:
        filepath = filepath.with_suffix(_DEFAULT_SUFFIX[normalized])
    elif filepath.suffix.lower() not in _FORMAT_SUFFIXES[normalized]:
        raise ValueError(
            f"File suffix {filepath.suffix!r} does not match format {normalized!r}."
        )
    return normalized, filepath


def _array_to_dataframe(results: np.ndarray) -> pd.DataFrame:
    """Convert a one- or two-dimensional result array to a DataFrame."""
    array = np.asarray(results)
    if array.ndim == 1:
        return pd.DataFrame({"result": array})
    if array.ndim == 2:
        return pd.DataFrame(
            array,
            columns=[f"result_{index}" for index in range(array.shape[1])],
        )
    raise ValueError("NumPy results must be one- or two-dimensional.")


def save_results(
    results: Union[np.ndarray, pd.DataFrame, "gpd.GeoDataFrame"],
    filepath: PathLike,
    format: Optional[str] = None,
) -> Path:
    """Save model results to CSV, Parquet, Shapefile, GeoJSON, or GeoPackage.

    Args:
        results: Results to save. Spatial formats require a GeoDataFrame.
        filepath: Output path. Parent directories are created automatically.
        format: Explicit output format. When omitted, it is inferred from the suffix.
            Accepted values include ``csv``, ``parquet``, ``shapefile``/``shp``,
            ``geojson``, and ``gpkg``.

    Returns:
        pathlib.Path: The file path written to disk.

    Examples:
        Save a table:

        >>> from pygwrx.io import save_results
        >>> output = save_results(results_df, "outputs/gwr_results.csv")

        Save mapped coefficients:

        >>> output = save_results(coef_gdf, "outputs/gwr_coefficients.geojson")
    """
    if not isinstance(filepath, (str, Path)):
        raise TypeError("filepath must be a string or pathlib.Path.")

    path = Path(filepath).expanduser()
    output_format, path = _normalize_output_format(path, format)
    path.parent.mkdir(parents=True, exist_ok=True)

    geopandas = _optional_geopandas()
    if isinstance(results, np.ndarray):
        table: Union[pd.DataFrame, "gpd.GeoDataFrame"] = _array_to_dataframe(results)
    elif isinstance(results, pd.DataFrame) or (
        geopandas is not None and isinstance(results, geopandas.GeoDataFrame)
    ):
        table = results
    else:
        raise TypeError("results must be a NumPy array, DataFrame, or GeoDataFrame.")

    if output_format == "csv":
        table.to_csv(path, index=False)
    elif output_format == "parquet":
        table.to_parquet(path, index=False)
    else:
        geopandas = _require_geopandas(purpose=f"writing {output_format} files")
        if not isinstance(table, geopandas.GeoDataFrame):
            raise TypeError(
                f"The {output_format} format requires a GeoDataFrame with geometry."
            )
        if table.geometry.name not in table.columns:
            raise ValueError("The GeoDataFrame has no active geometry column.")
        if output_format == "shapefile":
            table.to_file(path, driver="ESRI Shapefile", index=False)
        elif output_format == "geojson":
            table.to_file(path, driver="GeoJSON", index=False)
        elif output_format == "gpkg":
            table.to_file(path, driver="GPKG", index=False)
        else:  # pragma: no cover - protected by format normalization
            raise RuntimeError(f"Unhandled output format: {output_format}")

    if not path.exists():
        raise OSError(f"The output file was not created: {path}")
    return path


__all__ = [
    "load_data",
    "to_geodataframe",
    "from_geodataframe",
    "save_results",
]


# Deferred APIs
# The following draft functions were intentionally removed from the public I/O
# module until their responsibilities and APIs are designed in the appropriate
# packages:
#
# - create_spatial_weights_matrix(...)
#   Spatial-neighbour/weight construction is computational functionality, not
#   data I/O. A future implementation should live in ``pygwrx.core.weights``.
#
# - split_spatial_data(...)
#   Spatial validation and train/test splitting belong in a future
#   ``pygwrx.model_selection`` package.
