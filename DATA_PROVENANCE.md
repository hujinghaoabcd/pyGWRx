# Bundled Dataset Provenance and Integrity Evidence

This file records the exact upstream snapshot used to identify each bundled dataset,
the local processing applied by pyGWRx, and the integrity evidence shipped with the
release. Source provenance and local file integrity are deliberately recorded as
separate claims:

- **Provenance pin** identifies an upstream package release or repository commit.
- **Local integrity** uses `DATA_HASHES.sha256` to identify every redistributed file.
- **Byte identity** is claimed only where it was directly verified. For the three
  FastSGWR CSV files, the local Git blob SHA-1 exactly matches the blob stored at the
  pinned commit. For CRAN-derived datasets, this audit fixes the package release and
  records processing, but does not claim a fresh byte-for-byte comparison with the
  compressed CRAN source archive.

Evidence was reviewed on **2026-07-19**.

## Snapshot summary

| Dataset | Upstream snapshot | Upstream object/path | pyGWRx processing | Byte-level evidence |
|---|---|---|---|---|
| Columbus | `spData` 2.3.5, released 2026-05-04 | R object `columbus` (`data/columbus.rda`; documentation also points to `inst/shapes/columbus.gpkg`) | Tabular extract for examples; non-tabular geometry/neighbour objects are not bundled | Local SHA-256 only |
| Dublin Voter | `GWmodel` 2.4-1, released 2024-09-07 | `DubVoter` / `Dub.voter` example data | ESRI Shapefile sidecars retained; CRS declaration normalized to EPSG:29902 | Local SHA-256 only |
| EWHP | `GWmodel` 2.4-1, released 2024-09-07 | `EWHP` and `EWOutline` example data | Modelling values retained in CSV form | Local SHA-256 only |
| Georgia Educ | `GWmodel` 2.4-1, released 2024-09-07 | `Georgia` / Georgia counties example data | Duplicate county keys dissolved during preparation; projected centroids and coordinate fields refreshed; final release contains 159 counties in EPSG:32616 | Local SHA-256 only |
| Crime | FastSGWR commit `b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6` | `Data/Crime.csv` | No modelling-value changes | Git blob SHA-1 and local SHA-256 |
| HIV | FastSGWR commit `b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6` | `Data/HIV.csv` | No modelling-value changes | Git blob SHA-1 and local SHA-256 |
| Housing | FastSGWR commit `b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6` | `Data/Housing.csv` | No modelling-value changes | Git blob SHA-1 and local SHA-256 |

## Exact source records

### Columbus

- Package: `spData`
- Version: `2.3.5`
- CRAN publication date: `2026-05-04`
- Package page: <https://cran.r-project.org/package=spData>
- Source archive name: `spData_2.3.5.tar.gz`
- Upstream object: `columbus`
- Upstream package locations: `data/columbus.rda`, with spatial geometry also
  documented at `inst/shapes/columbus.gpkg`
- Local file: `src/pygwrx/data/Columbus/columbus.csv`
- Processing: a tabular extract was retained for reproducible regression examples;
  polygon geometry, neighbour lists, and R row names were not included.
- Integrity: local SHA-256 is recorded in `DATA_HASHES.sha256`.

### Dublin Voter, EWHP, and Georgia Educ

- Package: `GWmodel`
- Version: `2.4-1`
- CRAN publication date: `2024-09-07`
- Package page: <https://cran.r-project.org/package=GWmodel>
- Archived source: <https://cran.r-project.org/src/contrib/Archive/GWmodel/GWmodel_2.4-1.tar.gz>
- Upstream objects: `DubVoter` / `Dub.voter`, `EWHP`, `EWOutline`, and `Georgia`
- Local processing:
  - Dublin Voter: shapefile components retained and CRS metadata normalized to
    EPSG:29902.
  - EWHP: modelling table and map outline retained in CSV form without changes to
    modelling values.
  - Georgia Educ: preparation resolved duplicate county keys, refreshed projected
    centroids/coordinate columns, and produced the canonical 159-county release
    shapefile in EPSG:32616.
- Integrity: each local file and sidecar has a SHA-256 entry in
  `DATA_HASHES.sha256`.

### FastSGWR CSV files

- Repository: <https://github.com/Lessani252/FastSGWR>
- Commit: `b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6`
- Commit evidence date: `2026-07-19`
- License: upstream MIT notice preserved as `licenses/FastSGWR-MIT.txt`

| Local file | Upstream path | Upstream/local Git blob SHA-1 | Local SHA-256 |
|---|---|---|---|
| `src/pygwrx/data/Crime/Crime.csv` | `Data/Crime.csv` | `ac8ac10e020232a5293e7984c9e90ac440f91414` | `0cacb7f155bc93293ce85847b01073b3bcdfdb6b279967afc9e1c3d6db20daf0` |
| `src/pygwrx/data/HIV/HIV.csv` | `Data/HIV.csv` | `cbe28a992be30dab5f7913f277d87672d5865d13` | `88236793c8d927a72b98cc945a5e15682d424d3201508e359fa4f9d561f80e25` |
| `src/pygwrx/data/Housing/Housing.csv` | `Data/Housing.csv` | `35f4a3e7f8fea05d8f34a0c2bd03312afe74559e` | `c2e8469da73b8e041364b93ca39fdf87ac4a6d56b19227ba803370cbf3aa010a` |

The Git blob SHA-1 was recomputed locally as
`SHA1("blob " + byte_length + NUL + file_bytes)` and matched the blob identifier
reported by GitHub at the pinned commit for all three files.

## Reproducing the local integrity manifest

From the repository root:

```bash
python tools/update_data_hashes.py
python tools/verify_data_provenance.py
```

The first command regenerates `DATA_HASHES.sha256` deterministically. The second
checks every listed SHA-256 and the pinned FastSGWR Git blob identities without
network access.
