"""Simple and fast histogramming in Python.

MIT License

Copyright (c) 2025 Douglas Davis

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

__version__ = "0.15.0"

from ._hist import (  # noqa
    bin_centers,
    bin_edges,
    fix1d,
    fix1dmw,
    var1d,
    var1dmw,
    fix2d,
    var2d,
)

from ._numpy import histogram, histogram2d  # noqa

from ._misc import (  # noqa
    omp_get_max_threads,
    force_omp,
    disable_omp,
    default_omp,
    omp_disabled,
    omp_forced,
    without_omp,
    with_omp,
)


version_info = tuple(__version__.split("."))
