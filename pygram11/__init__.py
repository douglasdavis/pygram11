from .hist import fix1d, var1d, fix2d, var2d
from .hist import histogram, histogram2d
from ._core import _HAS_OPENMP

__version__ = "0.3.0"

__all__ = ["fix1d", "var1d", "fix2d", "var2d", "histogram", "histogram2d"]

OPENMP = _HAS_OPENMP()
version_info = tuple(__version__.split("."))
