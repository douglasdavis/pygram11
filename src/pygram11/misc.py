"""pygram11 miscellaneous API."""

# MIT License
#
# Copyright (c) 2021 Douglas Davis
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import pygram11
from pygram11._backend import _omp_get_max_threads


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


def disable_omp() -> None:
    """Disable OpenMP acceleration by maximizing the parallel thresholds.

    The default behavior is to avoid OpenMP acceleration for input
    data with length below about 10,000 for fixed with histograms and
    5,000 for variable width histograms. This function forces all
    thresholds to be the ``sys.maxsize`` (never use OpenMP
    acceleration).

    """
    # fixed
    pygram11.FIXED_WIDTH_PARALLEL_THRESHOLD_1D = sys.maxsize
    pygram11.FIXED_WIDTH_PARALLEL_THRESHOLD_2D = sys.maxsize
    # parallel
    pygram11.VARIABLE_WIDTH_PARALLEL_THRESHOLD_1D = sys.maxsize
    pygram11.VARIABLE_WIDTH_PARALLEL_THRESHOLD_2D = sys.maxsize
    # multiweight fixed
    pygram11.FIXED_WIDTH_MW_PARALLEL_THRESHOLD_1D = sys.maxsize
    # multiweight variable
    pygram11.VARIABLE_WIDTH_MW_PARALLEL_THRESHOLD_1D = sys.maxsize


def force_omp() -> None:
    """Force OpenMP acceleration by minimizing the parallel thresholds.

    The default behavior is to avoid OpenMP acceleration for input
    data with length below about 10,000 for fixed with histograms and
    5,000 for variable width histograms. This function forces all
    thresholds to be the 1 (always use OpenMP acceleration).

    """
    # fixed
    pygram11.FIXED_WIDTH_PARALLEL_THRESHOLD_1D = 0
    pygram11.FIXED_WIDTH_PARALLEL_THRESHOLD_2D = 0
    # parallel
    pygram11.VARIABLE_WIDTH_PARALLEL_THRESHOLD_1D = 0
    pygram11.VARIABLE_WIDTH_PARALLEL_THRESHOLD_2D = 0
    # multiweight fixed
    pygram11.FIXED_WIDTH_MW_PARALLEL_THRESHOLD_1D = 0
    # multiweight variable
    pygram11.VARIABLE_WIDTH_MW_PARALLEL_THRESHOLD_1D = 0
