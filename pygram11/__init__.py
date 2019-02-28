from .histogram import uniform1d, nonuniform1d, uniform2d
from ._core import _HAS_OPENMP

__all__ = ["uniform1d", "nonuniform1d", "uniform2d"]

OPENMP = _HAS_OPENMP()
