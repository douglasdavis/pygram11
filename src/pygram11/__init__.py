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
from ._hist import (
    bin_centers,
    bin_edges,
    fix1d,
    fix1dmw,
    var1d,
    var1dmw,
    fix2d,
    var2d,
    histogram,
    histogram2d,
)
from ._misc import force_omp, disable_omp, omp_get_max_threads


version_info = tuple(__version__.split("."))

FIXED_WIDTH_PARALLEL_THRESHOLD_1D: int = 10_000
"""int: Threshold for running OpenMP acceleration for fixed width histograms in 1D."""

FIXED_WIDTH_MW_PARALLEL_THRESHOLD_1D: int = 10_000
"""int: Threshold for running OpenMP acceleration for multiweight fixed width histograms."""

FIXED_WIDTH_PARALLEL_THRESHOLD_2D: int = 10_000
"""int: Threshold for running OpenMP acceleration for fixed with histograms in 2D."""

VARIABLE_WIDTH_PARALLEL_THRESHOLD_1D: int = 5_000
"""int: Threshold for running OpenMP acceleration for variable width histograms in 1D."""

VARIABLE_WIDTH_MW_PARALLEL_THRESHOLD_1D: int = 5_000
"""int: Threshold for running OpenMP acceleration for multiweight variable width histograms."""

VARIABLE_WIDTH_PARALLEL_THRESHOLD_2D: int = 5_000
"""int: Threshold for running OpenMP acceleration for variable width histograms in 2D."""
