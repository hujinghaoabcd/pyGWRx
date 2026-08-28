# Release procedure

This procedure is a blocking release gate. Record command output in the release
validation report; do not convert configured-but-unrun checks into pass claims.

## 1. Prepare metadata and generated sources

- update `CHANGELOG.md`, `CITATION.cff`, and the versioned release notes;
- update `DATA_PROVENANCE.md` when any bundled data byte or source snapshot changes;
- regenerate and verify data hashes;
- regenerate API and example documentation.

```bash
python tools/update_data_hashes.py
python tools/verify_data_provenance.py
python tools/generate_example_docs.py
python tools/generate_api_docs.py
python examples/validate_coverage.py
```

## 2. Run local quality and validation gates

```bash
python -m black --check src tests tools examples
python -m isort --check-only src tests tools examples
python -m ruff check src tests tools examples
python -m mypy
python -m pytest -q -m "not reference"
python -m pytest -q -m reference
python tools/run_coverage.py --batch 1
python tools/run_coverage.py --batch 2
python tools/run_coverage.py --batch 3
python tools/run_coverage.py --combine
python -m mkdocs build --strict --clean --site-dir site
python examples/run_all.py
```

## 3. Build and inspect distributions

```bash
rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*
python tools/verify_distributions.py dist
```

Install the wheel and sdist in separate clean virtual environments. Each
environment must run `tools/smoke_installed_distribution.py` outside the source
tree and pass `pip check`. The smoke test verifies `py.typed`, loads bundled data,
fits an internal `MGTWR`, and rejects any installed external top-level `mgtwr`
package.

## 4. Security evidence

```bash
AUDIT_SITE=$(.audit-env/bin/python -c "import site; print(site.getsitepackages()[0])")
python -m pip_audit --strict --progress-spinner=off --path "$AUDIT_SITE"
cyclonedx-py environment .audit-env --pyproject pyproject.toml --mc-type library \
  --output-reproducible --output-format JSON --output-file SBOM.cdx.json
sha256sum dist/* SBOM.cdx.json > SHA256SUMS
```

Attach the SBOM, checksums, and validation report to the GitHub Release.

## 5. TestPyPI Trusted Publishing

In the TestPyPI project settings, register a pending or existing trusted publisher
for this repository with:

- owner: `hujinghaoabcd`
- repository: `pyGWRx`
- workflow: `publish-testpypi.yml`
- environment: `testpypi`

Protect the GitHub `testpypi` environment as desired, then manually dispatch the
workflow. Inspect project metadata and install the exact uploaded version from
TestPyPI in a fresh environment before publishing to production.

## 6. PyPI and GitHub Release

In the PyPI project settings, configure the trusted publisher for:

- owner: `hujinghaoabcd`
- repository: `pyGWRx`
- workflow: `release.yml`
- environment: `pypi`

Create an annotated tag matching `pygwrx.__version__`, for example `v0.1.2`, and
push it. The release workflow re-runs blocking quality, provenance, test, strict
documentation, and all 12 operating-system/Python compatibility jobs; verifies
the tag/version pair; builds the distributions once;
publishes the wheel and sdist with OIDC; generates a wheel-runtime SBOM; and
creates a GitHub Release with documentation, release notes, the validation report,
the handoff record, and checksums. `SHA256SUMS` and all non-distribution assets
remain outside the PyPI upload directory.

## Supported CI matrix

The main CI and tag-triggered release gate define Windows, Linux, and macOS jobs
for Python 3.11, 3.12, 3.13, and 3.14. They also contain separate blocking jobs
for quality, coverage,
independent numerical references, declared minimum dependencies, and isolated
wheel/sdist installation. The release report should link to the successful remote
run once it exists.
