# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Built-in example datasets distributed with pyGWRx.

The registry-backed loaders provide reproducible datasets for examples, documentation, and automated testing.

Author:
    Jinghao Hu
"""

__author__ = "Jinghao Hu"
__license__ = "MIT"

import os
import warnings
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import geopandas as gpd

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

import pandas as pd

# --------------------------------------------------------------------------------------- #
# Dataset registry — one entry per dataset. `features`/`response`/`coords` are the         #
# canonical modelling columns; see each data/<name>/README.md for full field descriptions. #
# --------------------------------------------------------------------------------------- #
_DATASETS: Dict[str, Dict[str, Any]] = {
    "dublin_voter": {
        "name": "Dublin Voter Turnout",
        "name_zh": "都柏林选民投票率",
        "format": "shapefile",
        "path": ("DubVoter", "Dub.voter.shp"),
        "coords": ("X", "Y"),
        "response": "GenEl2004",
        "features": [
            "DiffAdd",
            "LARent",
            "SC1",
            "Unempl",
            "LowEduc",
            "Age18_24",
            "Age25_44",
            "Age45_64",
        ],
        "spatial_unit": "Electoral Division (ED)",
        "study_area": "Greater Dublin, Ireland",
        "n_samples": 322,
        "crs": "EPSG:29902 (Irish National Grid)",
        "reference": (
            "Gollini, I., Lu, B., Charlton, M., Brunsdon, C., & Harris, P. (2015). "
            "GWmodel: An R Package for Exploring Spatial Heterogeneity Using "
            "Geographically Weighted Models. JSS, 63(17)."
        ),
        "license": "GPL-2.0-or-later",
        "source_url": "https://cran.r-project.org/package=GWmodel",
        "processing": "Bundled from the GWmodel example data; CRS metadata normalized to EPSG:29902.",
        "aliases": ["dublin", "dubvoter", "dub_voter", "dub"],
    },
    "hiv": {
        "name": "County-level HIV prevalence",
        "name_zh": "县级 HIV 发病率",
        "format": "csv",
        "path": ("HIV", "HIV.csv"),
        "coords": ("X", "Y"),
        "response": "Rate_per_100000",
        "features": ["Walkability", "PopulationFemale", "NoHealthInsurance", "X.Black"],
        "spatial_unit": "County",
        "study_area": "USA",
        "n_samples": 2526,
        "crs": None,
        "reference": (
            "Lessani, M. N., & Li, Z. (2025). Enhancing the computational efficiency "
            "of the SGWR model... Annals of GIS, 31(4), 635-650."
        ),
        "license": "MIT",
        "source_url": "https://github.com/Lessani252/FastSGWR",
        "processing": "Bundled from the FastSGWR research-code repository without changing modelling values.",
        "aliases": [],
    },
    "crime": {
        "name": "County-level crime rate",
        "name_zh": "县级犯罪率",
        "format": "csv",
        "path": ("Crime", "Crime.csv"),
        "coords": ("X", "Y"),
        "response": "Five_ave_crime",
        "features": [
            "PopulationDensity",
            "PopulationFemale",
            "X.Black",
            "Neighbor_Disadvantage",
            "Casinos",
        ],
        "spatial_unit": "County",
        "study_area": "USA",
        "n_samples": 2841,
        "crs": None,
        "reference": (
            "Lessani, M. N., & Li, Z. (2025). Enhancing the computational efficiency "
            "of the SGWR model... Annals of GIS, 31(4), 635-650."
        ),
        "license": "MIT",
        "source_url": "https://github.com/Lessani252/FastSGWR",
        "processing": "Bundled from the FastSGWR research-code repository without changing modelling values.",
        "aliases": [],
    },
    "housing": {
        "name": "Neighborhood house prices",
        "name_zh": "社区房价",
        "format": "csv",
        "path": ("Housing", "Housing.csv"),
        "coords": ("x_coor", "y_coor"),
        "response": "price",
        "features": [
            "bedrooms",
            "bathrooms",
            "sqft_lot",
            "grade",
            "sqft_living15",
            "sqft_lot15",
        ],
        "spatial_unit": "Neighborhood",
        "study_area": "King County, USA",
        "n_samples": 20832,
        "crs": None,
        "reference": (
            "Lessani, M. N., & Li, Z. (2025). Enhancing the computational efficiency "
            "of the SGWR model... Annals of GIS, 31(4), 635-650."
        ),
        "license": "MIT",
        "source_url": "https://github.com/Lessani252/FastSGWR",
        "processing": "Bundled from the FastSGWR research-code repository without changing modelling values.",
        "aliases": [],
    },
    "columbus": {
        "name": "Columbus (OH) neighborhood crime",
        "name_zh": "哥伦布市社区犯罪",
        "format": "csv",
        "path": ("Columbus", "columbus.csv"),
        "coords": ("X", "Y"),
        "response": "CRIME",
        "features": ["INC", "HOVAL"],
        "spatial_unit": "Neighborhood",
        "study_area": "Columbus, Ohio, USA",
        "n_samples": 49,
        "crs": None,
        "reference": (
            "Anselin, L. (1988). Spatial Econometrics: Methods and Models. "
            "Kluwer Academic Publishers."
        ),
        "license": "CC0-1.0",
        "source_url": "https://cran.r-project.org/package=spData",
        "processing": "Bundled as a tabular extract for reproducible GWR examples.",
        "aliases": [],
    },
    "ewhp": {
        "name": "England & Wales house prices",
        "name_zh": "英格兰与威尔士房价",
        "format": "csv",
        "path": ("EWHP", "EWHP.csv"),
        "coords": ("Easting", "Northing"),
        "response": "PurPrice",
        "features": [
            "BldIntWr",
            "BldPostW",
            "Bld60s",
            "Bld70s",
            "Bld80s",
            "TypDetch",
            "TypSemiD",
            "TypFlat",
            "FlrArea",
        ],
        "spatial_unit": "Property",
        "study_area": "England & Wales, 2001",
        "n_samples": 519,
        "crs": None,
        "reference": (
            "Fotheringham, A. S., Brunsdon, C., & Charlton, M. E. (2002). "
            "Geographically Weighted Regression. Chichester: Wiley. "
            "(GWmodel R package.)"
        ),
        "license": "GPL-2.0-or-later",
        "source_url": "https://cran.r-project.org/package=GWmodel",
        "processing": "Bundled from the GWmodel example data without changing modelling values.",
        "aliases": ["england_wales_hp"],
    },
    "georgia": {
        "name": "Georgia educational attainment",
        "name_zh": "佐治亚州教育程度",
        "format": "shapefile",
        "path": ("GeorgiaEduc", "GeorgiaEduc.shp"),
        "coords": ("X", "Y"),
        "response": "PctBach",
        "features": ["PctRural", "PctPov", "PctBlack", "PctEld", "PctFB"],
        "spatial_unit": "County",
        "study_area": "Georgia, USA",
        "n_samples": 159,
        "crs": "EPSG:32616",
        "reference": (
            "Fotheringham, A. S., Brunsdon, C., & Charlton, M. E. (2002). "
            "Geographically Weighted Regression. Chichester: Wiley."
        ),
        "license": "GPL-2.0-or-later",
        "source_url": "https://cran.r-project.org/package=GWmodel",
        "processing": "Bundled from the GWmodel example data; duplicate AREAKEY geometries are dissolved and coordinates refreshed from projected centroids.",
        "aliases": ["georgia_educ", "gedu", "georgiaeduc"],
    },
}

# Release-facing source provenance. Keeping this evidence separate from the modelling
# registry makes the exact snapshot fields easy to audit without duplicating them in
# every dataset definition.
_PROVENANCE: Dict[str, Dict[str, Optional[str]]] = {
    "dublin_voter": {
        "source_version": "GWmodel 2.4-1",
        "source_release_date": "2024-09-07",
        "source_revision": "GWmodel_2.4-1.tar.gz",
        "source_path": "DubVoter / Dub.voter",
        "evidence_date": "2026-07-19",
        "integrity": "Local SHA-256 manifest; CRAN release pin (no fresh archive byte comparison claimed).",
    },
    "hiv": {
        "source_version": "FastSGWR",
        "source_release_date": None,
        "source_revision": "b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6",
        "source_path": "Data/HIV.csv",
        "evidence_date": "2026-07-19",
        "integrity": "Local SHA-256 plus pinned upstream Git blob SHA-1 after reversible CSV line-ending normalization.",
    },
    "crime": {
        "source_version": "FastSGWR",
        "source_release_date": None,
        "source_revision": "b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6",
        "source_path": "Data/Crime.csv",
        "evidence_date": "2026-07-19",
        "integrity": "Local SHA-256 plus pinned upstream Git blob SHA-1 after reversible CSV line-ending normalization.",
    },
    "housing": {
        "source_version": "FastSGWR",
        "source_release_date": None,
        "source_revision": "b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6",
        "source_path": "Data/Housing.csv",
        "evidence_date": "2026-07-19",
        "integrity": "Local SHA-256 plus pinned upstream Git blob SHA-1 after reversible CSV line-ending normalization.",
    },
    "columbus": {
        "source_version": "spData 2.3.5",
        "source_release_date": "2026-05-04",
        "source_revision": "spData_2.3.5.tar.gz",
        "source_path": "data/columbus.rda (object: columbus)",
        "evidence_date": "2026-07-19",
        "integrity": "Local SHA-256 manifest; CRAN release pin (no fresh archive byte comparison claimed).",
    },
    "ewhp": {
        "source_version": "GWmodel 2.4-1",
        "source_release_date": "2024-09-07",
        "source_revision": "GWmodel_2.4-1.tar.gz",
        "source_path": "EWHP and EWOutline",
        "evidence_date": "2026-07-19",
        "integrity": "Local SHA-256 manifest; CRAN release pin (no fresh archive byte comparison claimed).",
    },
    "georgia": {
        "source_version": "GWmodel 2.4-1",
        "source_release_date": "2024-09-07",
        "source_revision": "GWmodel_2.4-1.tar.gz",
        "source_path": "Georgia / Georgia counties example data",
        "evidence_date": "2026-07-19",
        "integrity": "Local SHA-256 manifest; CRAN release pin (no fresh archive byte comparison claimed).",
    },
}

for _dataset_key, _provenance in _PROVENANCE.items():
    _DATASETS[_dataset_key].update(_provenance)


# --------------------------------------------------------------------------------------- #
# Internal helpers                                                                         #
# --------------------------------------------------------------------------------------- #
def _get_data_dir(
    data_dir: Optional[Union[str, PathLike[str]]] = None,
) -> Path:
    """Resolve the directory containing the bundled datasets.

    Resolution priority is: explicit ``data_dir`` argument, the
    ``PYGWRX_DATA_DIR`` environment variable, then ``src/pygwrx/data``.
    """
    if data_dir is not None:
        if not isinstance(data_dir, (str, os.PathLike)):
            raise TypeError("data_dir must be a path-like object or None.")
        return Path(data_dir).expanduser()

    env = os.environ.get("PYGWRX_DATA_DIR")
    if env:
        return Path(env).expanduser()

    return Path(__file__).resolve().parents[1] / "data"


def _resolve_name(name: str) -> str:
    """Map a dataset name or alias to its canonical registry key."""
    if not isinstance(name, str):
        raise TypeError("name must be a dataset name or alias string.")

    key = name.strip().lower().replace("-", "_")
    if not key:
        raise ValueError("name cannot be empty.")

    if key in _DATASETS:
        return key

    for canonical, spec in _DATASETS.items():
        aliases = {
            str(alias).strip().lower().replace("-", "_")
            for alias in spec.get("aliases", [])
        }
        if key in aliases:
            return canonical

    available = ", ".join(sorted(_DATASETS))
    raise ValueError(f"Unknown dataset: {name!r}. Available: {available}")


def _normalize_return_type(return_type: str) -> str:
    """Validate and normalize a dataset return type."""
    if not isinstance(return_type, str):
        raise TypeError("return_type must be a string.")

    value = return_type.strip().lower()
    aliases = {
        "frame": "frame",
        "dataframe": "frame",
        "geodataframe": "frame",
        "arrays": "arrays",
        "dict": "dict",
        "path": "path",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            f"Invalid return_type: {return_type!r}. "
            "Choose from: 'frame', 'arrays', 'dict', 'path'."
        ) from exc


def _validate_dropna(dropna: bool) -> bool:
    """Validate the row-filtering option."""
    if not isinstance(dropna, (bool, np.bool_)):
        raise TypeError("dropna must be a boolean.")
    return bool(dropna)


def _read_frame(spec: Dict[str, Any], path: Path) -> pd.DataFrame:
    """Read a raw dataset into a pandas or GeoPandas frame."""
    if spec["format"] == "shapefile":
        if not HAS_GEOPANDAS:
            raise ImportError(
                "geopandas is required for this dataset. "
                "Install it with: pip install geopandas"
            )
        return gpd.read_file(path)

    if spec["format"] == "csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported dataset format: {spec['format']!r}.")


def _extract_coords(frame: pd.DataFrame, spec: Dict[str, Any]) -> np.ndarray:
    """Extract an ``(n_samples, 2)`` floating-point coordinate array.

    Named coordinate columns are preferred. Non-numeric column values are
    converted to ``NaN`` so that ``dropna=True`` can remove them consistently.
    If coordinate columns are unavailable, centroids of a GeoDataFrame's active
    geometry are used as a fallback.
    """
    cx, cy = spec["coords"]
    if cx in frame.columns and cy in frame.columns:
        coord_frame = frame[[cx, cy]].apply(pd.to_numeric, errors="coerce")
        coords = coord_frame.to_numpy(dtype=float)
    elif HAS_GEOPANDAS and isinstance(frame, gpd.GeoDataFrame):
        geometry = frame.geometry
        if geometry.isna().any():
            raise ValueError("Geometry contains missing values.")
        if geometry.is_empty.any():
            raise ValueError("Geometry contains empty values.")
        if frame.crs is not None and getattr(frame.crs, "is_geographic", False):
            warnings.warn(
                "Centroids are being computed in a geographic CRS. "
                "Project the data before distance-based modelling when possible.",
                UserWarning,
                stacklevel=2,
            )
        centroids = geometry.centroid
        coords = np.column_stack(
            [centroids.x.to_numpy(dtype=float), centroids.y.to_numpy(dtype=float)]
        )
    else:
        raise ValueError(
            f"Coordinate columns {spec['coords']} were not found and no "
            "GeoDataFrame geometry is available."
        )

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"Extracted coordinates must have shape (n_samples, 2), got {coords.shape}."
        )
    if coords.shape[0] != len(frame):
        raise ValueError("Coordinate count does not match the number of frame rows.")
    return coords


def _frame_crs(frame: pd.DataFrame) -> Optional[str]:
    """Return a frame's CRS as text when available."""
    crs = getattr(frame, "crs", None)
    return None if crs is None else str(crs)


# --------------------------------------------------------------------------------------- #
# Generic loader                                                                           #
# --------------------------------------------------------------------------------------- #
def load_dataset(
    name: str,
    return_type: str = "frame",
    data_dir: Optional[Union[str, PathLike[str]]] = None,
    dropna: bool = True,
) -> Any:
    """Load a bundled example dataset by name.

    Load an example dataset distributed with pyGWRx.

    Args:
        name: Canonical dataset name or registered alias.
        return_type: ``'frame'`` returns the raw frame; ``'arrays'`` returns
            ``(X, y, coords)``; ``'dict'`` returns aligned modelling arrays,
            metadata and both filtered/raw frames; ``'path'`` returns the absolute
            file path without reading the dataset.
        data_dir: Override the bundled data directory.
        dropna: For ``'arrays'`` and ``'dict'``, remove rows containing non-finite
            modelling values or coordinates. This option does not alter the raw
            frame returned by ``return_type='frame'``.

    Returns:
        DataFrame | GeoDataFrame | tuple | dict | str: The requested dataset representation.
    """
    key = _resolve_name(name)
    normalized_return_type = _normalize_return_type(return_type)
    dropna = _validate_dropna(dropna)

    spec = _DATASETS[key]
    path = _get_data_dir(data_dir).joinpath(*spec["path"]).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{key}' was not found at:\n  {path}\n"
            "The example datasets are normally bundled with pyGWRx. If this "
            "file is missing, set PYGWRX_DATA_DIR or pass data_dir=... .\n"
            f"Source: {spec['reference']}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path is not a file: {path}")

    if normalized_return_type == "path":
        # Preserve the original public contract: path results are strings.
        return str(path)

    frame = _read_frame(spec, path)

    if normalized_return_type == "frame":
        return frame

    needed: List[str] = list(spec["features"]) + [spec["response"]]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Dataset '{key}' is missing expected columns {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    coords_all = _extract_coords(frame, spec)
    numeric = frame[needed].apply(pd.to_numeric, errors="coerce")
    numeric_values = numeric.to_numpy(dtype=float)

    if dropna:
        valid_mask = np.isfinite(numeric_values).all(axis=1)
        valid_mask &= np.isfinite(coords_all).all(axis=1)
    else:
        valid_mask = np.ones(len(frame), dtype=bool)

    if not np.any(valid_mask):
        raise ValueError(
            f"Dataset '{key}' contains no usable rows after applying dropna={dropna}."
        )

    model_frame = frame.loc[valid_mask].copy()
    model_numeric = numeric.loc[valid_mask]
    coords = np.asarray(coords_all[valid_mask], dtype=float)
    X = model_numeric[spec["features"]].to_numpy(dtype=float)
    y = model_numeric[spec["response"]].to_numpy(dtype=float)

    if normalized_return_type == "arrays":
        return X, y, coords

    file_crs = _frame_crs(frame)
    return {
        "data": X,
        "target": y,
        "coords": coords,
        "feature_names": list(spec["features"]),
        "target_name": spec["response"],
        # frame is deliberately aligned with data/target/coords.
        "frame": model_frame,
        "raw_frame": frame,
        "row_index": model_frame.index.to_numpy(copy=True),
        "description": spec["name"],
        "description_zh": spec["name_zh"],
        "n_samples": int(len(y)),
        "registered_n_samples": int(spec["n_samples"]),
        "n_features": len(spec["features"]),
        "filepath": str(path),
        "spatial_unit": spec["spatial_unit"],
        "study_area": spec["study_area"],
        # Keep the historical `crs` key as the registry/declaration value and
        # expose the CRS actually read from a spatial file separately.
        "crs": spec["crs"],
        "declared_crs": spec["crs"],
        "file_crs": file_crs,
        "reference": spec["reference"],
        "license": spec["license"],
        "source_url": spec["source_url"],
        "processing": spec["processing"],
        "source_version": spec["source_version"],
        "source_release_date": spec["source_release_date"],
        "source_revision": spec["source_revision"],
        "source_path": spec["source_path"],
        "evidence_date": spec["evidence_date"],
        "integrity": spec["integrity"],
    }


# --------------------------------------------------------------------------------------- #
# Metadata helpers                                                                         #
# --------------------------------------------------------------------------------------- #
def get_dataset_info(dataset_name: str = "dublin_voter") -> Dict[str, Any]:
    """Return registry metadata for a dataset without loading its data file."""
    key = _resolve_name(dataset_name)
    spec = _DATASETS[key]
    relative_path = Path(*spec["path"])
    return {
        "key": key,
        "name": spec["name"],
        "name_zh": spec["name_zh"],
        "format": spec["format"],
        "n_samples": spec["n_samples"],
        "n_features": len(spec["features"]),
        "feature_names": list(spec["features"]),
        "target_name": spec["response"],
        "coords": tuple(spec["coords"]),
        "spatial_unit": spec["spatial_unit"],
        "study_area": spec["study_area"],
        "crs": spec["crs"],
        "declared_crs": spec["crs"],
        "aliases": list(spec.get("aliases", [])),
        "relative_path": str(relative_path),
        "reference": spec["reference"],
        "license": spec["license"],
        "source_url": spec["source_url"],
        "processing": spec["processing"],
        "source_version": spec["source_version"],
        "source_release_date": spec["source_release_date"],
        "source_revision": spec["source_revision"],
        "source_path": spec["source_path"],
        "evidence_date": spec["evidence_date"],
        "integrity": spec["integrity"],
        "loader_function": f"load_{key}",
        "readme": f"data/{spec['path'][0]}/README.md",
    }


def list_datasets(verbose: bool = True) -> List[str]:
    """List available built-in datasets and optionally print their metadata."""
    if not isinstance(verbose, (bool, np.bool_)):
        raise TypeError("verbose must be a boolean.")
    names = list(_DATASETS)
    if verbose:
        print("Available pyGWRx datasets:")
        print("=" * 70)
        for i, key in enumerate(names, 1):
            s = _DATASETS[key]
            print(f"{i}. {key}  —  {s['name']} / {s['name_zh']}")
            print(
                f"   n={s['n_samples']}, {len(s['features'])} features, "
                f"y='{s['response']}', {s['format']}  |  load_{key}()"
            )
        print("\nUse: load_dataset('<name>', return_type='arrays')")
    return names


# --------------------------------------------------------------------------------------- #
# Per-dataset convenience loaders                                                          #
# --------------------------------------------------------------------------------------- #
def load_dublin_voter(return_type: str = "frame", **kwargs: Any) -> Any:
    """Load the Dublin voter turnout dataset. See :func:`load_dataset`."""
    return load_dataset("dublin_voter", return_type=return_type, **kwargs)


def load_hiv(return_type: str = "frame", **kwargs: Any) -> Any:
    """Load the county-level HIV prevalence dataset. See :func:`load_dataset`."""
    return load_dataset("hiv", return_type=return_type, **kwargs)


def load_crime(return_type: str = "frame", **kwargs: Any) -> Any:
    """Load the county-level crime dataset. See :func:`load_dataset`."""
    return load_dataset("crime", return_type=return_type, **kwargs)


def load_housing(return_type: str = "frame", **kwargs: Any) -> Any:
    """Load the neighborhood house-price dataset. See :func:`load_dataset`."""
    return load_dataset("housing", return_type=return_type, **kwargs)


def load_columbus(return_type: str = "frame", **kwargs: Any) -> Any:
    """Load the Columbus (OH) crime dataset. See :func:`load_dataset`."""
    return load_dataset("columbus", return_type=return_type, **kwargs)


def load_ewhp(return_type: str = "frame", **kwargs: Any) -> Any:
    """Load the England & Wales house-price dataset. See :func:`load_dataset`."""
    return load_dataset("ewhp", return_type=return_type, **kwargs)


def load_georgia(return_type: str = "frame", **kwargs: Any) -> Any:
    """Load the Georgia educational-attainment dataset. See :func:`load_dataset`."""
    return load_dataset("georgia", return_type=return_type, **kwargs)


# Backward-compatible aliases
get_dublin_voter = load_dublin_voter
load_dubvoter = load_dublin_voter
get_dubvoter = load_dublin_voter


__all__ = [
    "load_dataset",
    "load_dublin_voter",
    "load_hiv",
    "load_crime",
    "load_housing",
    "load_columbus",
    "load_ewhp",
    "load_georgia",
    "get_dublin_voter",
    "load_dubvoter",
    "get_dubvoter",
    "get_dataset_info",
    "list_datasets",
]
