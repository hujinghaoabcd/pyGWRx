"""Every name advertised in ``pygwrx.__all__`` must be importable."""

import importlib

import pygwrx


def test_version_present():
    assert isinstance(pygwrx.__version__, str)
    assert pygwrx.__version__


def test_all_names_are_exported():
    for name in pygwrx.__all__:
        assert hasattr(pygwrx, name), f"{name} listed in __all__ but not importable"


def test_all_model_modules_import():
    """Import every module under pygwrx.models to catch syntax/import errors."""
    import pkgutil

    import pygwrx.models as models_pkg

    for mod in pkgutil.iter_modules(models_pkg.__path__):
        importlib.import_module(f"pygwrx.models.{mod.name}")
