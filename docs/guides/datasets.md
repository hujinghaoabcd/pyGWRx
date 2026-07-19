# Datasets

```python
from pygwrx.io import list_datasets, get_dataset_info, load_dataset

print(list_datasets())
print(get_dataset_info("georgia"))
X, y, coords = load_dataset("georgia", return_type="arrays")
```

Supported return modes include `frame`, `arrays`, `dict`, and `path` where applicable.

For reproducibility, record dataset name, package version, original citation, response and predictors, CRS, filtering, missing-data handling, and transformations.

!!! warning "Licensing"
    The current development tree contains third-party example data. Citation, academic-use permission, and redistribution permission are not equivalent. Verify every dataset's original licence before redistributing a wheel or data bundle.


## Licensing, provenance, and integrity

`get_dataset_info()` and `load_dataset(..., return_type="dict")` report each
dataset's `license`, `source_url`, and pyGWRx `processing` record. The repository
also includes `DATA_LICENSES.md`, `THIRD_PARTY_NOTICES.md`, and
`DATA_HASHES.sha256`. Bundled data are third-party works and are not relicensed
as MIT-licensed pyGWRx code.
