# Security Policy

## Supported versions

Security fixes are applied to the current `0.1.x` release line. Development
snapshots and archived source trees are not supported release channels.

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| < 0.1 | No |

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability-reporting or Security Advisory interface for
`hujinghaoabcd/pyGWRx` when it is available. If private reporting is unavailable,
email the maintainer at `hujinghao20@mails.ucas.ac.cn` with the subject
`pyGWRx security report`.

Include, where possible:

- affected pyGWRx version and installation method;
- operating system and Python version;
- a minimal reproducer or proof of concept;
- the expected and observed security impact;
- whether the report may be shared with dependency maintainers.

The maintainer will acknowledge receipt, investigate privately, coordinate a fix
and disclosure where appropriate, and credit reporters who request attribution.
Please avoid publishing exploit details before a coordinated release is available.

## Release security controls

Release workflows include:

- dependency consistency checks with `pip check`;
- vulnerability scanning with `pip-audit`;
- a CycloneDX JSON software bill of materials;
- isolated wheel and source-distribution installation tests;
- SHA-256 checksums for release artifacts;
- PyPI Trusted Publishing through GitHub Actions OIDC;
- GitHub-hosted release artifacts and PyPI publication attestations.

A clean audit records the environment tested at release time. It does not imply
that future vulnerabilities cannot be disclosed in transitive dependencies.
