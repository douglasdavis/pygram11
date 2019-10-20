from .hist import fix1d, fix1dmw, var1d, var1dmw, fix2d, var2d
from .hist import histogram, histogram2d
from ._core import _max_threads, _has_openmp


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

__version__ = "0.6.1"
version_info = tuple(__version__.split("."))

def omp_available() -> bool:
    """query the existence of OpenMP acceleration

    Returns
    -------
    bool
       whether or not OpenMP is available

    """
    return _has_openmp()

def omp_max_threads() -> int:
    """query the number of threads available to OpenMP

    Returns
    -------
    int
       the number of available threads

    """
    return _max_threads()

# to be removed in future release
OPENMP = omp_available()
