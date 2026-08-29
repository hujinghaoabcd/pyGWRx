from pathlib import Path
import re

UTILS = Path("src/pygwrx/core/utils.py")
GWR = Path("src/pygwrx/models/gwr.py")
BANDWIDTH = Path("src/pygwrx/core/bandwidth.py")
GWR_TEST = Path("tests/test_gwr_distance_streaming.py")
BW_TEST = Path("tests/test_bandwidth_distance_streaming.py")

helper = '''

_DEFAULT_DISTANCE_BLOCK_ROWS = 128


def _iter_distance_blocks(
    target_coords: CoordinateInput,
    source_coords: Optional[CoordinateInput] = None,
    *,
    distance_metric: str = "euclidean",
    block_rows: int = _DEFAULT_DISTANCE_BLOCK_ROWS,
) -> Iterator[np.ndarray]:
    """Yield bounded target-to-source pairwise distance blocks."""
    if source_coords is None:
        source_coords = target_coords
    targets, sources = _validate_coordinate_pair(target_coords, source_coords)
    if isinstance(block_rows, (bool, np.bool_)) or not isinstance(block_rows, Integral):
        raise TypeError("block_rows must be a positive integer.")
    block_rows_int = int(block_rows)
    if block_rows_int <= 0:
        raise ValueError("block_rows must be greater than zero.")

    n_sources = sources.shape[0]
    for start, stop in chunked_computation(targets.shape[0], block_rows_int):
        block = np.asarray(
            compute_distance_matrix(
                targets[start:stop],
                sources,
                metric=distance_metric,
            ),
            dtype=float,
        )
        expected_shape = (stop - start, n_sources)
        if block.shape != expected_shape:
            raise ValueError("The distance implementation returned an unexpected block shape.")
        if not np.all(np.isfinite(block)) or np.any(block < 0.0):
            raise ValueError("The distance implementation returned invalid distances.")
        yield block


def _iter_distance_rows(
    target_coords: CoordinateInput,
    source_coords: Optional[CoordinateInput] = None,
    *,
    distance_metric: str = "euclidean",
    block_rows: int = _DEFAULT_DISTANCE_BLOCK_ROWS,
) -> Iterator[np.ndarray]:
    """Yield target-to-source distance rows from bounded-size blocks."""
    for block in _iter_distance_blocks(
        target_coords,
        source_coords,
        distance_metric=distance_metric,
        block_rows=block_rows,
    ):
        yield from block
'''

utils_text = UTILS.read_text(encoding="utf-8")
assert "def _iter_distance_blocks(" not in utils_text
UTILS.write_text(utils_text.rstrip() + helper + "\n", encoding="utf-8")


gwr_text = GWR.read_text(encoding="utf-8")
old_import = "from pygwrx.core.utils import add_intercept, compute_distance_matrix, validate_coords"
new_import = '''from pygwrx.core.utils import (
    _iter_distance_rows as _iter_core_distance_rows,
    add_intercept,
    validate_coords,
)'''
assert gwr_text.count(old_import) == 1
gwr_text = gwr_text.replace(old_import, new_import)
assert gwr_text.count("_DISTANCE_BLOCK_ROWS = 128\n\n\n") == 1
gwr_text = gwr_text.replace("_DISTANCE_BLOCK_ROWS = 128\n\n\n", "", 1)
pattern = re.compile(
    r'''    def _iter_distance_rows\(self, target_coords: np\.ndarray\) -> Iterator\[np\.ndarray\]:\n        """Yield target-to-training distance rows from bounded-size blocks\."""\n        if self\.coords_train_ is None:\n            raise RuntimeError\("Training coordinates are unavailable\."\)\n        targets = np\.asarray\(target_coords, dtype=float\)\n        if targets\.ndim != 2:\n            raise ValueError\("target_coords must be a two-dimensional array\."\)\n\n        n_train = self\.coords_train_\.shape\[0\]\n        for start in range\(0, targets\.shape\[0\], _DISTANCE_BLOCK_ROWS\):\n            stop = min\(start \+ _DISTANCE_BLOCK_ROWS, targets\.shape\[0\]\)\n            block = np\.asarray\(\n                compute_distance_matrix\(\n                    targets\[start:stop\],\n                    self\.coords_train_,\n                    metric=self\.distance_metric,\n                \),\n                dtype=float,\n            \)\n            expected_shape = \(stop - start, n_train\)\n            if block\.shape != expected_shape:\n                raise ValueError\(\n                    "The distance implementation returned an unexpected block shape\."\n                \)\n            for distance_row in block:\n                yield distance_row\n'''
)
replacement = '''    def _iter_distance_rows(self, target_coords: np.ndarray) -> Iterator[np.ndarray]:
        """Yield target-to-training distance rows from the shared bounded backend."""
        if self.coords_train_ is None:
            raise RuntimeError("Training coordinates are unavailable.")
        targets = np.asarray(target_coords, dtype=float)
        if targets.ndim != 2:
            raise ValueError("target_coords must be a two-dimensional array.")
        return _iter_core_distance_rows(
            targets,
            self.coords_train_,
            distance_metric=self.distance_metric,
        )
'''
gwr_text, count = pattern.subn(replacement, gwr_text, count=1)
assert count == 1
GWR.write_text(gwr_text, encoding="utf-8")


bw_text = BANDWIDTH.read_text(encoding="utf-8")
assert bw_text.count("from typing import Callable, Iterator, Optional, Tuple, Union") == 1
bw_text = bw_text.replace(
    "from typing import Callable, Iterator, Optional, Tuple, Union",
    "from typing import Callable, Optional, Tuple, Union",
)
assert bw_text.count("from pygwrx.core.utils import compute_distance_matrix") == 1
bw_text = bw_text.replace(
    "from pygwrx.core.utils import compute_distance_matrix",
    "from pygwrx.core.utils import _iter_distance_rows",
)
assert bw_text.count("_DISTANCE_BLOCK_ROWS = 128\n") == 1
bw_text = bw_text.replace("_DISTANCE_BLOCK_ROWS = 128\n", "", 1)
pattern = re.compile(
    r'''\n\ndef _iter_distance_rows\(\n    coords: np\.ndarray,\n    \*,\n    distance_metric: str,\n\) -> Iterator\[np\.ndarray\]:\n    """Yield coordinate-to-coordinate distance rows from bounded-size blocks\."""\n    n_samples = coords\.shape\[0\]\n    for start in range\(0, n_samples, _DISTANCE_BLOCK_ROWS\):\n        stop = min\(start \+ _DISTANCE_BLOCK_ROWS, n_samples\)\n        block = np\.asarray\(\n            compute_distance_matrix\(\n                coords\[start:stop\],\n                coords,\n                metric=distance_metric,\n            \),\n            dtype=float,\n        \)\n        expected_shape = \(stop - start, n_samples\)\n        if block\.shape != expected_shape:\n            raise ValueError\("The computed distance block has an invalid shape\."\)\n        if not np\.all\(np\.isfinite\(block\)\) or np\.any\(block < 0\):\n            raise ValueError\("The computed distance block contains invalid distances\."\)\n        for distance_row in block:\n            yield distance_row\n'''
)
bw_text, count = pattern.subn("", bw_text, count=1)
assert count == 1
BANDWIDTH.write_text(bw_text, encoding="utf-8")


gwr_test = GWR_TEST.read_text(encoding="utf-8")
gwr_test = gwr_test.replace(
    'gwr_module = importlib.import_module("pygwrx.models.gwr")\n',
    'core_utils = importlib.import_module("pygwrx.core.utils")\n',
)
gwr_test = gwr_test.replace(
    "    original = gwr_module.compute_distance_matrix\n",
    "    original = core_utils.compute_distance_matrix\n",
)
gwr_test = gwr_test.replace(
    "        if first.shape[0] > gwr_module._DISTANCE_BLOCK_ROWS:\n",
    "        if first.shape[0] > core_utils._DEFAULT_DISTANCE_BLOCK_ROWS:\n",
)
gwr_test = gwr_test.replace(
    '    monkeypatch.setattr(gwr_module, "compute_distance_matrix", tracked)\n',
    '    monkeypatch.setattr(core_utils, "compute_distance_matrix", tracked)\n',
)
gwr_test = gwr_test.replace(
    "    assert max(rows for rows, _ in calls) <= gwr_module._DISTANCE_BLOCK_ROWS\n",
    "    assert max(rows for rows, _ in calls) <= core_utils._DEFAULT_DISTANCE_BLOCK_ROWS\n",
)
GWR_TEST.write_text(gwr_test, encoding="utf-8")


bw_test = BW_TEST.read_text(encoding="utf-8")
bw_test = bw_test.replace(
    'bandwidth_module = importlib.import_module("pygwrx.core.bandwidth")\n',
    'core_utils = importlib.import_module("pygwrx.core.utils")\n',
)
bw_test = bw_test.replace(
    "    original = bandwidth_module.compute_distance_matrix\n",
    "    original = core_utils.compute_distance_matrix\n",
)
bw_test = bw_test.replace(
    '    monkeypatch.setattr(bandwidth_module, "compute_distance_matrix", tracked)\n',
    '    monkeypatch.setattr(core_utils, "compute_distance_matrix", tracked)\n',
)
bw_test = bw_test.replace(
    "    assert max(left_rows for left_rows, _ in calls) <= 128\n",
    "    assert (\n        max(left_rows for left_rows, _ in calls)\n        <= core_utils._DEFAULT_DISTANCE_BLOCK_ROWS\n    )\n",
)
bw_test = bw_test.replace(
    "    assert any(left_rows < 128 for left_rows, _ in calls)\n",
    "    assert any(\n        left_rows < core_utils._DEFAULT_DISTANCE_BLOCK_ROWS for left_rows, _ in calls\n    )\n",
)
BW_TEST.write_text(bw_test, encoding="utf-8")
