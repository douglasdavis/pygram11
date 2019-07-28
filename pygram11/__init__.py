from __future__ import absolute_import

from .hist import fix1d, var1d, fix2d, var2d
from .hist import histogram, histogram2d
from ._core import _HAS_OPENMP as OPENMP

__version__ = "0.5.0a1"

__all__ = ["fix1d", "var1d", "fix2d", "var2d", "histogram", "histogram2d"]

version_info = tuple(__version__.split("."))
