# Bundled Dataset Licenses

pyGWRx source code is distributed under the MIT License. The datasets bundled in
`src/pygwrx/data/` are third-party works and retain their own licences; they are
not relicensed as pyGWRx source code.

| Dataset | Exact upstream snapshot | Retained licence | pyGWRx processing |
|---|---|---|---|
| Columbus | CRAN `spData` 2.3.5 (2026-05-04) | CC0-1.0 | Tabular extract for reproducible examples |
| Dublin Voter | CRAN `GWmodel` 2.4-1 (2024-09-07) | GPL-2.0-or-later | CRS metadata normalized to EPSG:29902 |
| EWHP | CRAN `GWmodel` 2.4-1 (2024-09-07) | GPL-2.0-or-later | Modelling values retained in CSV form |
| Georgia Educ | CRAN `GWmodel` 2.4-1 (2024-09-07) | GPL-2.0-or-later | County keys consolidated; projected centroids refreshed |
| Crime | FastSGWR `b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6` | MIT | Modelling values unchanged |
| HIV | FastSGWR `b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6` | MIT | Modelling values unchanged |
| Housing | FastSGWR `b63064938a2ba5a1eb27cc7bdb642eaa62cb5de6` | MIT | Modelling values unchanged |

Complete licence texts are included under `licenses/`. Exact source versions,
commit identifiers, dates, upstream paths, processing, and byte-level evidence are
recorded in `DATA_PROVENANCE.md`. Release-file hashes are listed in
`DATA_HASHES.sha256`.

This record supports reproducible redistribution; it is not legal advice.
