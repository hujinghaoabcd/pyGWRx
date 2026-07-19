# Typing and `py.typed` policy

pyGWRx ships `pygwrx/py.typed`, so type checkers may consume the inline
annotations installed with the package. The marker is a packaging promise that
annotations are distributed; it is not a claim that every legacy implementation
module already satisfies `mypy --strict`.

## Blocking typed-API gate

The release CI runs a strict, blocking mypy gate over a deliberately named typed
surface in `pyproject.toml`. The current surface includes package exports,
optional-dependency handling, summary formatting, kernels, solvers, diagnostic
helpers, dataset/data I/O, plotting validation, and the self-contained MGTWR
implementation.

```bash
python -m mypy
```

The configuration requires complete function annotations, rejects implicit
optionals, checks strict equality, and reports unused ignores. Imports outside the
selected surface are skipped so that third-party stub quality and unrelated
legacy modules do not weaken this gate.

## Expansion rule

A module is added to the blocking set only after:

1. its public inputs and return types are explicit;
2. its implementation passes the configured mypy gate without broad ignores;
3. runtime tests cover the typed contract;
4. its public documentation is regenerated when signatures change.

New public modules should enter the strict surface immediately. Existing modules
are migrated incrementally rather than hiding hundreds of known issues behind a
non-blocking whole-package command.

## Consumer expectations

Users can rely on distributed annotations for editor assistance and static
analysis, but runtime validation remains authoritative for numerical shape,
finite-value, coordinate, and fitted-state constraints. Type annotations do not
encode all array dimensions or statistical preconditions.
