"""pygram11 Histogram API."""

# MIT License
#
# Copyright (c) 2020 Douglas Davis
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

import numpy as np
import numbers

from pygram11._backend1d import _v1dw, _f1dw, _f1dmw, _v1dmw
from pygram11._backend2d import _f2dw, _v2dw


def _likely_uniform_bins(edges):
    """Test if bin edges describe a set of fixed width bins."""
    diffs = np.ediff1d(edges)
    ones = np.ones_like(diffs)
    max_close = np.allclose(ones, diffs / np.amax(diffs))
    min_close = np.allclose(ones, diffs / np.amin(diffs))
    return max_close and min_close


def fix1d(x, bins=10, range=None, weights=None, density=False, flow=False):
    r"""Histogram data with fixed (uniform) bin widths.

    Parameters
    ----------
    x : array_like
        Data to histogram.
    bins : int
        The number of bins.
    range : (float, float), optional
        The minimum and maximum of the histogram axis.
    weights : array_like, optional
        The weights for each element of ``x``.
    density : bool
        If True, normalize histogram bins as value of PDF such that
        the integral over the range is one.
    flow : bool
        If True, the under and overflow bin contents are added to the
        first and last bins, respectively.

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`
        The standard error of each bin count, :math:`\sqrt{\sum_i w_i^2}`.

    Examples
    --------
    A histogram of ``x`` with 20 bins between 0 and 100:

    >>> h, __ = fix1d(x, bins=20, range=(0, 100))

    The same data, now histogrammed with weights:

    >>> w = np.abs(np.random.randn(x.shape[0]))
    >>> h, h_err = fix1d(x, bins=20, range=(0, 100), weights=w)

    """
    x = np.ascontiguousarray(x)
    if weights is not None:
        weights = np.ascontiguousarray(weights)
    else:
        weights = np.ones_like(x, order="C")
        if not (weights.dtype == np.float32 or weights.dtype == np.float64):
            weights = weights.astype(np.float64)

    if range is not None:
        start, stop = range[0], range[1]
    else:
        start, stop = np.amin(x), np.amax(x)

    return _f1dw(x, weights, bins, start, stop, flow, density, True)


def fix1dmw(x, weights, bins=10, range=None, flow=False):
    r"""Histogram data with multiple weight variations and fixed width bins.

    Parameters
    ----------
    x : array_like
        data to histogram.
    weights : array_like
        The weight variations for the elements of ``x``, first
        dimension is the length of ``x``, second dimension is the
        number of weights variations.
    bins : int
        The number of bins.
    range : (float, float), optional
        The minimum and maximumm of the histogram axis.
    flow : bool
        If True, the under and overflow bin contents are added to the
        first and last bins, respectively.

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`
        The standard error of each bin count, :math:`\sqrt{\sum_i w_i^2}`.

    Examples
    --------
    Multiple histograms of ``x`` with 50 bins between 0 and 100; using
    20 different weight variations:

    >>> x = np.random.randn(10000)
    >>> twenty_weights = np.random.rand(x.shape[0], 20)
    >>> h, err = fix1dmw(x, w, bins=50, range=(-3, 3))

    ``h`` and ``err`` are now shape ``(50, 20)``. Each column
    represents the histogram of the data using its respective weight.

    """
    x = np.ascontiguousarray(x)
    weights = np.ascontiguousarray(weights)
    if not (weights.dtype == np.float32 or weights.dtype == np.float64):
        weights = weights.astype(np.float64)

    if range is not None:
        start, stop = range[0], range[1]
    else:
        start, stop = np.amin(x), np.amax(x)

    return _f1dmw(x, weights, bins, start, stop, flow, True)


def var1d(x, bins, weights=None, density=False, flow=False):
    r"""Histogram data with variable bin widths.

    Parameters
    ----------
    x : array_like
        data to histogram
    bins : array_like
        bin edges
    weights : array_like, optional
        weight for each element of ``x``
    density : bool
        normalize histogram bins as value of PDF such that the integral
        over the range is 1.
    flow : bool
        if ``True`` the under and overflow bin contents are added to the first
        and last bins, respectively

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`
        The standard error of each bin count, :math:`\sqrt{\sum_i w_i^2}`.

    Examples
    --------
    A simple histogram with variable width bins:

    >>> x = np.random.randn(10000)
    >>> bin_edges = [-3.0, -2.5, -1.5, -0.25, 0.25, 2.0, 3.0]
    >>> h, __ = var1d(x, bin_edges)

    """
    x = np.ascontiguousarray(x)
    if weights is not None:
        weights = np.ascontiguousarray(weights)
    else:
        weights = np.ones_like(x, order="C")
        if not (weights.dtype == np.float32 or weights.dtype == np.float64):
            weights = weights.astype(np.float64)

    bins = np.ascontiguousarray(bins)
    if not np.all(bins[1:] >= bins[:-1]):
        raise ValueError("bins sequence must monotonically increase")

    if _likely_uniform_bins(bins):
        return _f1dw(x, weights, len(bins) - 1, bins[0], bins[-1], flow, density, True)

    return _v1dw(x, weights, bins, flow, density, True)


def var1dmw(x, weights, bins, flow=False):
    r"""Histogram data with multiple weight variations and variable width bins.

    Parameters
    ----------
    x : array_like
        data to histogram
    bins : array_like
        bin edges
    weights : array_like
        weight variations for the elements of ``x``, first dimension
        is the shape of ``x``, second dimension is the number of weights.
    density : bool
        normalize histogram bins as value of PDF such that the integral
        over the range is 1.
    flow : bool
        if ``True`` the under and overflow bin contents are added to the first
        and last bins, respectively

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`
        The standard error of each bin count, :math:`\sqrt{\sum_i w_i^2}`.

    Examples
    --------
    Using three different weight variations:

    >>> x = np.random.randn(10000)
    >>> weights = np.abs(np.random.randn(x.shape[0], 3))
    >>> bin_edges = [-3.0, -2.5, -1.5, -0.25, 0.25, 2.0, 3.0]
    >>> h, err = var1dmw(x, weights, bin_edges)
    >>> h.shape
    (6, 3)
    >>> err.shape
    (6, 3)

    """
    x = np.ascontiguousarray(x)
    weights = np.ascontiguousarray(weights)
    if not (weights.dtype == np.float32 or weights.dtype == np.float64):
        weights = weights.astype(np.float64)

    bins = np.ascontiguousarray(bins)
    if not np.all(bins[1:] >= bins[:-1]):
        raise ValueError("bins sequence must monotonically increase")

    if _likely_uniform_bins(bins):
        return _f1dmw(x, weights, len(bins) - 1, bins[0], bins[-1], flow, True)

    return _v1dmw(x, weights, bins, flow, True)


def histogram(x, bins=10, range=None, weights=None, density=False, flow=False):
    r"""Histogram data in one dimension.

    Parameters
    ----------
    x : array_like
        data to histogram.
    bins : int or array_like
        if int: the number of bins; if array_like: the bin edges.
    range : tuple(float, float), optional
        the definition of the edges of the bin range (start, stop).
    weights : array_like, optional
        a set of weights associated with the elements of ``x``. This
        can also be a two dimensional set of multiple weights
        varitions with shape (len(x), n_weight_variations).
    density : bool
        normalize counts such that the integral over the range is
        equal to 1. If ``weights`` is two dimensional this argument is
        ignored.
    flow : bool
        if ``True``, include under/overflow in the first/last bins.

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`
        The standard error of each bin count, :math:`\sqrt{\sum_i w_i^2}`.

    Examples
    --------
    A simple fixed width histogram:

    >>> h, __ = histogram(x, bins=20, range=(0, 100))

    And with variable width histograms and weights:

    >>> h, err = histogram(x, bins=[-3, -2, -1.5, 1.5, 3.5], weights=w)

    """
    # fixed bins
    if isinstance(bins, numbers.Integral):
        if weights is not None:
            if weights.shape != x.shape:
                return fix1dmw(x, weights, bins=bins, range=range, flow=flow)
        return fix1d(
            x, weights=weights, bins=bins, range=range, density=density, flow=flow
        )

    # variable bins
    else:
        if range is not None:
            raise TypeError("range must be None if bins is non-int")
        if weights is not None:
            if weights.shape != x.shape:
                return var1dmw(x, weights, bins=bins, flow=flow)
        return var1d(x, weights=weights, bins=bins, density=density, flow=flow)


def fix2d(x, y, bins=10, range=None, weights=None):
    r"""Histogram the ``x``, ``y`` data with fixed (uniform) binning.

    Parameters
    ----------
    x : array_like
       first entries in data pairs to histogram
    y : array_like
       second entries in data pairs to histogram
    bins : int or iterable
       if int, both dimensions will have that many bins,
       if iterable, the number of bins for each dimension
    range : iterable, optional
       axis limits to histogram over in the form [(xmin, xmax), (ymin, ymax)]
    weights : array_like, optional
       weight for each :math:`(x_i, y_i)` pair.

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`
        The standard error of each bin count, :math:`\sqrt{\sum_i w_i^2}`.

    Examples
    --------
    A histogram of (``x``, ``y``) with 20 bins between 0 and 100 in
    the ``x`` dimention and 10 bins between 0 and 50 in the ``y``
    dimension:

    >>> h, __ = fix2d(x, y, bins=(20, 10), range=((0, 100), (0, 50)))

    The same data, now histogrammed weighted (via ``w``):

    >>> h, err = fix2d(x, y, bins=(20, 10), range=((0, 100), (0, 50)), weights=w)

    """
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must be the same shape")
    if weights is None:
        weights = np.ones_like(x, dtype=np.float64)
    else:
        weights = np.ascontiguousarray(weights)

    if isinstance(bins, numbers.Integral):
        nx = ny = bins
    else:
        nx, ny = bins

    if range is None:
        range = [(x.min(), x.max()), (y.min(), y.max())]
    (xmin, xmax), (ymin, ymax) = range

    return _f2dw(x, y, weights, nx, xmin, xmax, ny, ymin, ymax, False, True)


def var2d(x, y, xbins, ybins, weights=None):
    r"""Histogram the ``x``, ``y`` data with variable width binning.

    Parameters
    ----------
    x : array_like
       first entries in the data pairs to histogram
    y : array_like
       second entries in the data pairs to histogram
    xbins : array_like
       bin edges for the ``x`` dimension
    ybins : array_like
       bin edges for the ``y`` dimension
    weights : array_like, optional
       weights for each :math:`(x_i, y_i)` pair.

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`
        The standard error of each bin count, :math:`\sqrt{\sum_i w_i^2}`.

    Examples
    --------
    A histogram of (``x``, ``y``) where the edges are defined by a
    :func:`numpy.logspace` in both dimensions:

    >>> bins = numpy.logspace(0.1, 1.0, 10, endpoint=True)
    >>> h, __ = var2d(x, y, bins, bins)

    """
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must be the same shape")
    xbins = np.ascontiguousarray(xbins)
    ybins = np.ascontiguousarray(ybins)
    if not np.all(xbins[1:] >= xbins[:-1]):
        raise ValueError("xbins sequence must monotonically increase")
    if not np.all(ybins[1:] >= ybins[:-1]):
        raise ValueError("ybins sequence must monotonically increase")

    if weights is None:
        weights = np.ones_like(x, dtype=np.float64)
    else:
        weights = np.ascontiguousarray(weights)

    return _v2dw(x, y, weights, xbins, ybins, False, True)


def histogram2d(x, y, bins=10, range=None, weights=None):
    r"""Histogram data in two dimensions.

    This function provides an API very simiar to
    :func:`numpy.histogram2d`. Keep in mind that the returns are
    different.

    Parameters
    ----------
    x: array_like
       Array representing the ``x`` coordinate of the data to histogram.
    y: array_like
       Array representing the ``y`` coordinate of the data to histogram.
    bins: int or array_like or [int, int] or [array, array], optional
       The bin specification:
          * If `int`, the number of bins for the two dimensions
            (``nx = ny = bins``).
          * If `array_like`, the bin edges for the two dimensions
            (``x_edges = y_edges = bins``).
          * If [int, int], the number of bins in each dimension
            (``nx, ny = bins``).
          * If [`array_like`, `array_like`], the bin edges in each
            dimension (``x_edges, y_edges = bins``).
    range: array_like, shape(2,2), optional
       The edges of this histogram along each dimension. If ``bins``
       is not integral, then this parameter is ignored. If None, the
       default is ``[[x.min(), x.max()], [y.min(), y.max()]]``.
    weights: array_like
       An array of weights associated to each element :math:`(x_i,
       y_i)` pair.  Each pair of the data will contribute its
       associated weight to the bin count.

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`
        The standard error of each bin count, :math:`\sqrt{\sum_i w_i^2}`.

    Examples
    --------
    >>> h, err = histogram2d(x, y, weights=w)

    """
    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1 and N != 2:
        return var2d(x, y, bins, bins, weights=weights)

    if N == 1:
        return fix2d(x, y, bins=bins, range=range, weights=weights)

    if N == 2:
        if isinstance(bins[0], numbers.Integral) and isinstance(
            bins[1], numbers.Integral
        ):
            return fix2d(x, y, bins=bins, range=range, weights=weights)
        else:
            return var2d(x, y, bins[0], bins[1], weights=weights)
