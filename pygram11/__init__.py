from .histogram import uniform1d, nonuniform1d
from ._core import _HAS_OPENMP

__all__ = ["uniform1d", "nonuniform1d"]

OPENMP = _HAS_OPENMP()
