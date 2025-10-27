# MIT License
#
# Copyright (c) 2025 Douglas Davis
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

import functools
import contextlib
import sys
from typing import Iterator, Optional
import pygram11.config
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


def default_omp() -> None:
    """Set OpenMP acceleration thresholds to the default values."""
    pygram11.config.set("thresholds.fix1d", 10_000)
    pygram11.config.set("thresholds.fix1dmw", 10_000)
    pygram11.config.set("thresholds.fix2d", 10_000)
    pygram11.config.set("thresholds.var1d", 5_000)
    pygram11.config.set("thresholds.var1dmw", 5_000)
    pygram11.config.set("thresholds.var2d", 5_000)


def disable_omp() -> None:
    """Disable OpenMP acceleration by maximizing all thresholds."""
    for k in pygram11.config.threshold_keys():
        pygram11.config.set(k, sys.maxsize)


def force_omp() -> None:
    """Force OpenMP acceleration by nullifying all thresholds."""
    for k in pygram11.config.threshold_keys():
        pygram11.config.set(k, 0)


def without_omp(*args, **kwargs):
    """Wrap a function to disable OpenMP while it's called.

    If a specific key is defined, only that threshold will be modified
    to turn OpenMP off.

    The settings of the pygram11 OpenMP threshold configurations will
    be restored to their previous values at the end of the function
    that is being wrapped.

    Parameters
    ----------
    key : str, optional
        Specific threshold key to turn off.

    Examples
    --------
    Writing a function with this decorator:

    >>> import numpy as np
    >>> from pygram11 import histogram, without_omp
    >>> @without_omp
    ... def single_threaded_histogram():
    ...     data = np.random.standard_normal(size=(1000,))
    ...     return pygram11.histogram(data, bins=10, range=(-5, 5), flow=True)

    Defining a specific `key`:

    >>> import pygram11.config
    >>> previous = pygram11.config.get("thresholds.var1d")
    >>> @without_omp(key="thresholds.var1d")
    ... def single_threaded_histogram2():
    ...     print(f"in function threshold: {pygram11.config.get('thresholds.var1d')}")
    ...     data = np.random.standard_normal(size=(1000,))
    ...     return pygram11.histogram(data, bins=[-2, -1, 1.5, 3.2])
    >>> result = single_threaded_histogram2()
    in function threshold: 9223372036854775807
    >>> previous
    5000
    >>> previous == pygram11.config.get("thresholds.var1d")
    True
    >>> result[0].shape
    (3,)

    """
    func = None
    if len(args) == 1 and callable(args[0]):
        func = args[0]
    if func:
        key = None
    if not func:
        key = kwargs.get("key")

    def cable(func):
        @functools.wraps(func)
        def decorator(*args, **kwargs):
            with omp_disabled(key=key):
                res = func(*args, **kwargs)
            return res

        return decorator

    return cable(func) if func else cable


def with_omp(*args, **kwargs):
    """Wrap a function to always enable OpenMP while it's called.

    If a specific key is defined, only that threshold will be modified
    to turn OpenMP on.

    The settings of the pygram11 OpenMP threshold configurations will
    be restored to their previous values at the end of the function
    that is being wrapped.

    Parameters
    ----------
    key : str, optional
        Specific threshold key to turn on.

    Examples
    --------
    Writing a function with this decorator:

    >>> import numpy as np
    >>> from pygram11 import histogram, with_omp
    >>> @with_omp
    ... def multi_threaded_histogram():
    ...     data = np.random.standard_normal(size=(1000,))
    ...     return pygram11.histogram(data, bins=10, range=(-5, 5), flow=True)

    Defining a specific `key`:

    >>> import pygram11.config
    >>> previous = pygram11.config.get("thresholds.var1d")
    >>> @with_omp(key="thresholds.var1d")
    ... def multi_threaded_histogram2():
    ...     print(f"in function threshold: {pygram11.config.get('thresholds.var1d')}")
    ...     data = np.random.standard_normal(size=(1000,))
    ...     return pygram11.histogram(data, bins=[-2, -1, 1.5, 3.2])
    >>> result = multi_threaded_histogram2()
    in function threshold: 0
    >>> previous
    5000
    >>> previous == pygram11.config.get("thresholds.var1d")
    True
    >>> result[0].shape
    (3,)

    """
    func = None
    if len(args) == 1 and callable(args[0]):
        func = args[0]
    if func:
        key = None
    if not func:
        key = kwargs.get("key")

    def cable(func):
        @functools.wraps(func)
        def decorator(*args, **kwargs):
            with omp_forced(key=key):
                res = func(*args, **kwargs)
            return res

        return decorator

    return cable(func) if func else cable


@contextlib.contextmanager
def omp_disabled(*, key: Optional[str] = None) -> Iterator[None]:
    """Context manager to disable OpenMP.

    Parameters
    ----------
    key : str, optional
        Specific threshold key to turn off.

    Examples
    --------
    Using a specific key:

    >>> import pygram11
    >>> import numpy as np
    >>> with pygram11.omp_disabled(key="thresholds.var1d"):
    ...     data = np.random.standard_normal(size=(200,))
    ...     result = pygram11.histogram(data, bins=[-2, -1, 1.5, 3.2])
    >>> result[0].shape
    (3,)

    Disable all thresholds:

    >>> import pygram11
    >>> import numpy as np
    >>> with pygram11.omp_disabled():
    ...     data = np.random.standard_normal(size=(200,))
    ...     result = pygram11.histogram(data, bins=12, range=(-3, 3))
    >>> result[0].shape
    (12,)

    """
    if key is not None:
        try:
            prev = pygram11.config.get(key)
            pygram11.config.set(key, sys.maxsize)
            yield
        finally:
            pygram11.config.set(key, prev)

    else:
        previous = {k: pygram11.config.get(k) for k in pygram11.config.threshold_keys()}
        try:
            disable_omp()
            yield
        finally:
            for k, v in previous.items():
                pygram11.config.set(k, v)


@contextlib.contextmanager
def omp_forced(*, key: Optional[str] = None) -> Iterator[None]:
    """Context manager to force enable OpenMP.

    Parameters
    ----------
    key : str, optional
        Specific threshold key to turn on.

    Examples
    --------
    Using a specific key:

    >>> import pygram11
    >>> import numpy as np
    >>> with pygram11.omp_forced(key="thresholds.var1d"):
    ...     data = np.random.standard_normal(size=(200,))
    ...     result = pygram11.histogram(data, bins=[-2, -1, 1.5, 3.2])
    >>> result[0].shape
    (3,)

    Enable all thresholds:

    >>> import pygram11
    >>> import numpy as np
    >>> with pygram11.omp_forced():
    ...     data = np.random.standard_normal(size=(200,))
    ...     result = pygram11.histogram(data, bins=10, range=(-3, 3))
    >>> result[0].shape
    (10,)

    """
    if key is not None:
        try:
            prev = pygram11.config.get(key)
            pygram11.config.set(key, 0)
            yield
        finally:
            pygram11.config.set(key, prev)
    else:
        previous = {k: pygram11.config.get(k) for k in pygram11.config.threshold_keys()}
        try:
            force_omp()
            yield
        finally:
            for k, v in previous.items():
                pygram11.config.set(k, v)
