from __future__ import absolute_import

from .hist import fix1d, fix1dmw, var1d, var1dmw, fix2d, var2d
from .hist import histogram, histogram2d
from ._core import _HAS_OPENMP as OPENMP

__version__ = "0.5.0"

__all__ = [
    "fix1d",
    "fix1dmw",
    "var1d",
    "var1dmw",
    "fix2d",
    "var2d",
    "histogram",
    "histogram2d",
]

version_info = tuple(__version__.split("."))
