from .histogram import fix1d, var1d, fix2d, var2d
from ._core import _HAS_OPENMP

__all__ = ["fix1d", "var1d", "fix2d", "var2d"]

OPENMP = _HAS_OPENMP()
