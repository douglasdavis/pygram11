from .histogram import uniform1d, nonuniform1d
from ._core import _OPENMP

__all__ = ["uniform1d", "nonuniform1d"]

OPENMP = _OPENMP()
