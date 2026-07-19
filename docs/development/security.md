# Security, dependency audit, and SBOM

The repository security policy is published in `SECURITY.md`. Suspected
vulnerabilities should be reported privately rather than through a public issue.

## Local audit

Use an isolated target environment so the result represents the release
installation rather than unrelated developer tooling:

```bash
python -m pip install pip-audit cyclonedx-bom
python -m venv .audit-env
.audit-env/bin/python -m pip install -e ".[all]"
.audit-env/bin/python -m pip check
AUDIT_SITE=$(.audit-env/bin/python -c "import site; print(site.getsitepackages()[0])")
python -m pip_audit --strict --progress-spinner=off --path "$AUDIT_SITE"
cyclonedx-py environment .audit-env \
  --pyproject pyproject.toml \
  --mc-type library \
  --output-reproducible \
  --output-format JSON \
  --output-file SBOM.cdx.json
.audit-env/bin/python tools/finalize_sbom.py SBOM.cdx.json
```

`pip check` verifies installed dependency consistency. `pip-audit` checks the
resolved Python environment against published vulnerability data. The CycloneDX
file records the resolved software components used for the audit.

## Automation

`.github/workflows/security.yml` runs on pull requests, protected branches, a
weekly schedule, and manual dispatch. A vulnerability finding or audit-service
failure blocks that job. The generated `SBOM.cdx.json` is uploaded as a workflow
artifact.

The release report must distinguish locally executed checks from the configured
cross-platform GitHub Actions matrix. A workflow definition is not evidence that
a remote run passed until GitHub records a successful run for the release commit.
