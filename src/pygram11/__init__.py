"""Simple and fast histogramming in Python.

MIT License

Copyright (c) 2021 Douglas Davis

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from .version import version as __version__  # noqa
from pygram11._backend import _omp_get_max_threads
from .hist import fix1d, fix1dmw, var1d, var1dmw
from .hist import fix2d, var2d
from .hist import histogram, histogram2d

version_info = tuple(__version__.split("."))

__all__ = [
    "fix1d",
    "fix1dmw",
    "var1d",
    "var1dmw",
    "fix2d",
    "var2d",
    "histogram",
    "histogram2d",
    "omp_get_max_threads",
]

FIXED_WIDTH_PARALLEL_THRESHOLD: int = 10_000
"""int: Threshold for running OpenMP accelerated loop for fixed width histograms."""

VARIABLE_WIDTH_PARALLEL_THRESHOLD: int = 5_000
"""int: Threshold for running OpenMP accelerated loop for variable width histograms."""


def omp_get_max_threads() -> int:
    """Get the number of threads available to OpenMP.

    This returns the result of calling the OpenMP C API function `of
    the same name
    <https://www.openmp.org/spec-html/5.0/openmpsu112.html>`_.

    Returns
    -------
    int
        the maximum number of available threads

    """
    return _omp_get_max_threads()
