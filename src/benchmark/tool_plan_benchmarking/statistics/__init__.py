"""Statistics calculation helpers for tool-plan benchmarking.

IMPORTANT:
This directory name (`statistics`) shadows Python's stdlib module `statistics` when
`src/benchmark/tool_plan_benchmarking/run.py` is executed as a script (Python adds
that directory to `sys.path`). Some third-party libraries (e.g. seaborn) import
`statistics.NormalDist` from the stdlib.

To avoid breaking those imports, we dynamically load the stdlib `statistics.py`
under an alternate name and re-export the expected symbols.
"""

from __future__ import annotations

import importlib.util
import sysconfig

from pathlib import Path
from types import ModuleType


def _load_stdlib_statistics() -> ModuleType:
    stdlib_dir = Path(sysconfig.get_paths()["stdlib"])  # e.g. .../lib/python3.12
    statistics_path = stdlib_dir / "statistics.py"

    spec = importlib.util.spec_from_file_location("_stdlib_statistics", statistics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load stdlib statistics module from {statistics_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_statistics = _load_stdlib_statistics()

# Re-export the subset that external libraries commonly import.
NormalDist = _stdlib_statistics.NormalDist
mean = _stdlib_statistics.mean
median = _stdlib_statistics.median
stdev = _stdlib_statistics.stdev
pstdev = _stdlib_statistics.pstdev
variance = _stdlib_statistics.variance
pvariance = _stdlib_statistics.pvariance

__all__ = [
    "NormalDist",
    "mean",
    "median",
    "stdev",
    "pstdev",
    "variance",
    "pvariance",
]
