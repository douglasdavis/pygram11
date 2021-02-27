"""pygram11 Histogram API."""

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

# stdlib
from typing import Sequence, Tuple, Optional, Union

# third party
import numpy as np

# One dimensional backend functions
from pygram11._backend import _f1d, _v1d, _f1dw, _v1dw, _f1dmw, _v1dmw

# Two dimensional backend functions
from pygram11._backend import _f2d, _f2dw, _v2d, _v2dw


def _likely_uniform_bins(edges: np.ndarray) -> bool:
    """Test if bin edges describe a set of fixed width bins."""
    diffs = np.ediff1d(edges)
    return np.all(np.isclose(diffs, diffs[0]))


def _densify_fixed_counts(counts: np.ndarray, width: float) -> np.ndarray:
    """Convert fixed width histogram to unity integral over PDF."""
    return np.array(counts / (width * counts.sum()), dtype=np.float64)


def _densify_fixed_weighted_counts(
    raw: Tuple[np.ndarray, np.ndarray], width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert fixed width weighted histogram to unity integral over PDF."""
    counts = raw[0]
    integral = counts.sum()
    res0 = _densify_fixed_counts(counts, width)
    variances = raw[1] * raw[1]
    f1 = 1.0 / ((width * integral) ** 2)
    f2 = counts / integral
    res1 = f1 * (variances + (f2 * f2 * variances.sum()))
    return res0, np.sqrt(res1)


def _densify_variable_counts(
    counts: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Convert variable width histogram to unity integral over PDF."""
    widths = edges[1:] - edges[:-1]
    integral = float(np.sum(counts))
    return np.array(counts / widths / integral, dtype=np.float64)


def _densify_variable_weighted_counts(
    raw: Tuple[np.ndarray, np.ndarray], edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert variable width histogram to unity integral over PDF."""
    counts = raw[0]
    variances = raw[1] * raw[1]
    integral = counts.sum()
    widths = edges[1:] - edges[:-1]
    res0 = _densify_variable_counts(counts, edges)
    f1 = 1.0 / ((widths * integral) ** 2)
    f2 = counts / integral
    res1 = f1 * (variances + (f2 * f2 * variances.sum()))
    return res0, res1


def _get_limits_1d(
    x: np.ndarray, range: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    """Get bin limits given an optional range and 1D data."""
    if range is None:
        return (float(np.amin(x)), float(np.amax(x)))
    return float(range[0]), float(range[1])


def _get_limits_2d(
    x: np.ndarray,
    y: np.ndarray,
    range: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[float, float, float, float]:
    """Get bin limits given an optional range and 2D data."""
    if range is None:
        return (
            float(np.amin(x)),
            float(np.amax(x)),
            float(np.amin(y)),
            float(np.amin(y)),
        )
    else:
        return (
            float(range[0][0]),
            float(range[0][1]),
            float(range[1][0]),
            float(range[1][1]),
        )


def bin_edges(bins: int, range: Tuple[float, float]) -> np.ndarray:
    """Construct bin edges given number of bins and axis limits.

    Parameters
    ----------
    bins : int
        Total number of bins.
    range : (float, float)
        Minimum and maximum of the histogram axis.

    Returns
    -------
    numpy.ndarray
        Edges defined by the number of bins and axis limits.

    Examples
    --------
    >>> bin_edges(bins=8, range=(-2, 2))
    array([-2. , -1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])

    """
    return np.linspace(range[0], range[1], bins + 1)


def bin_centers(
    bins: Union[int, Sequence[float]],
    range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Construct array of center values for each bin.

    Parameters
    ----------
    bins : int or array_like
        Number of bins or bin edges array.
    range : (float, float), optional
        The minimum and maximum of the histogram axis.

    Returns
    -------
    numpy.ndarray
        Array of bin centers.

    Raises
    ------
    ValueError
        If ``bins`` is an integer and range is undefined (``None``).

    Examples
    --------
    The centers given the number of bins and max/min:

    >>> bin_centers(10, range=(-3, 3))
    array([-2.7, -2.1, -1.5, -0.9, -0.3,  0.3,  0.9,  1.5,  2.1,  2.7])

    Or given bin edges:

    >>> bin_centers([0, 1, 2, 3, 4])
    array([0.5, 1.5, 2.5, 3.5])

    """
    if isinstance(bins, int):
        if range is None:
            raise ValueError("Integer bins requires defining range")
        bins = bin_edges(bins, range=range)
    bins = np.asarray(bins)
    return 0.5 * (bins[1:] + bins[:-1])


def fix1d(
    x: np.ndarray,
    bins: int = 10,
    range: Optional[Tuple[float, float]] = None,
    weights: Optional[np.ndarray] = None,
    density: bool = False,
    flow: bool = False,
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    r"""Histogram data with fixed (uniform) bin widths.

    Parameters
    ----------
    x : numpy.ndarray
        Data to histogram.
    bins : int
        The number of bins.
    range : (float, float), optional
        The minimum and maximum of the histogram axis. If ``None``,
        min and max of ``x`` will be used.
    weights : numpy.ndarray, optional
        The weights for each element of ``x``. If weights are absent,
        the second return type will be ``None``.
    density : bool
        Normalize histogram counts as value of PDF such that the
        integral over the range is unity.
    flow : bool
        Include under/overflow in the first/last bins.

    Raises
    ------
    ValueError
        If ``x`` and ``weights`` have incompatible shapes.
    TypeError
        If ``x`` or ``weights`` are unsupported types

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The resulting histogram bin counts.
    :py:obj:`numpy.ndarray`, optional
        The standard error of each bin count, :math:`\sqrt{\sum_i
        w_i^2}`. The return is ``None`` if weights are not used.

    Examples
    --------
    A histogram of ``x`` with 20 bins between 0 and 100:

    >>> h, __ = fix1d(x, bins=20, range=(0, 100))

    When weights are absent the second return is ``None``. The same
    data, now histogrammed with weights and over/underflow included:

    >>> rng = np.random.default_rng(123)
    >>> w = rng.uniform(0.1, 0.9, x.shape[0]))
    >>> h, stderr = fix1d(x, bins=20, range=(0, 100), weights=w, flow=True)

    """
    xmin, xmax = _get_limits_1d(x, range)

    if weights is None:
        result = _f1d(x, bins, xmin, xmax, flow)
        if density:
            width = (xmax - xmin) / bins
            result = _densify_fixed_counts(result, width)
        return result, None

    if np.shape(x) != np.shape(weights):
        raise ValueError("x and weights must have the same shape")

    result = _f1dw(x, weights, int(bins), xmin, xmax, flow)
    if density:
        width = (xmax - xmin) / bins
        result = _densify_fixed_weighted_counts(result, width)
    return result


def fix1dmw(
    x: np.ndarray,
    weights: np.ndarray,
    bins: int = 10,
    range: Optional[Tuple[float, float]] = None,
    flow: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Histogram data with multiple weight variations and fixed width bins.

    The weights array must have a total number of rows equal to the
    length of the input data. The number of columns in the weights
    array is equal to the number of weight variations. (The weights
    array must be an M x N matrix where M is the length of x and N is
    the number of weight variations).

    Parameters
    ----------
    x : numpy.ndarray
        Data to histogram.
    weights : numpy.ndarray
        The weight variations for the elements of ``x``, first
        dimension is the length of ``x``, second dimension is the
        number of weights variations.
    bins : int
        The number of bins.
    range : (float, float), optional
        The minimum and maximum of the histogram axis. If ``None``,
        min and max of ``x`` will be used.
    flow : bool
        Include under/overflow in the first/last bins.

    Raises
    ------
    ValueError
        If ``x`` and ``weights`` have incompatible shapes (if
        ``x.shape[0] != weights.shape[0]``).
    ValueError
        If ``weights`` is not a two dimensional array.
    TypeError
        If ``x`` or ``weights`` are unsupported types

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`
        The standard error of each bin count, :math:`\sqrt{\sum_i w_i^2}`.

    Examples
    --------
    Multiple histograms of ``x`` using 20 different weight variations:

    >>> rng = np.random.default_rng(123)
    >>> x = rng.standard_normal(10000)
    >>> twenty_weights = np.abs(rng.standard_normal((x.shape[0], 20)))
    >>> h, err = fix1dmw(x, twenty_weights, bins=50, range=(-3, 3))

    ``h`` and ``err`` are now shape ``(50, 20)``. Each column
    represents the histogram of the data using its respective weight.

    """
    if len(np.shape(weights)) != 2:
        raise ValueError("weights must be a two dimensional array.")

    if np.shape(x)[0] != np.shape(weights)[0]:
        raise ValueError("x and weights have incompatible shapes.")

    xmin, xmax = _get_limits_1d(x, range)
    return _f1dmw(x, weights, int(bins), xmin, xmax, flow)


def var1d(
    x: np.ndarray,
    bins: np.ndarray,
    weights: Optional[np.ndarray] = None,
    density: bool = False,
    flow: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    r"""Histogram data with variable bin widths.

    Parameters
    ----------
    x : numpy.ndarray
        Data to histogram
    bins : numpy.ndarray
        Bin edges
    weights : numpy.ndarray, optional
        The weights for each element of ``x``. If weights are absent,
        the second return type will be ``None``.
    density : bool
        Normalize histogram counts as value of PDF such that the
        integral over the range is unity.
    flow : bool
        Include under/overflow in the first/last bins.

    Raises
    ------
    ValueError
        If the array of bin edges is not monotonically increasing.
    ValueError
        If ``x`` and ``weights`` have incompatible shapes.
    TypeError
        If ``x`` or ``weights`` are unsupported types

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`, optional
        The standard error of each bin count, :math:`\sqrt{\sum_i
        w_i^2}`. The return is ``None`` if weights are not used.

    Examples
    --------
    A simple histogram with variable width bins:

    >>> rng = np.random.default_rng(123)
    >>> x = rng.standard_normal(1000)
    >>> edges = np.array([-3.0, -2.5, -1.5, -0.25, 0.25, 2.0, 3.0])
    >>> h, __ = var1d(x, edges)

    """
    if not np.all(bins[1:] >= bins[:-1]):
        raise ValueError("bins sequence must monotonically increase")

    if _likely_uniform_bins(bins):
        nbins = np.shape(bins)[0] - 1
        return fix1d(
            x,
            bins=nbins,
            weights=weights,
            range=(bins[0], bins[-1]),
            flow=flow,
            density=density,
        )

    bins = np.array(bins, dtype=np.float64, copy=False)
    if weights is None:
        result = _v1d(x, bins, flow)
        if density:
            result = _densify_variable_counts(result, bins)
        return result, None

    if np.shape(x) != np.shape(weights):
        raise ValueError("x and weights have incompatible shapes.")

    result = _v1dw(x, weights, bins, flow)
    if density:
        result = _densify_variable_weighted_counts(result, bins)
    return result


def var1dmw(
    x: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    flow: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Histogram data with multiple weight variations and variable width bins.

    The weights array must have a total number of rows equal to the
    length of the input data. The number of columns in the weights
    array is equal to the number of weight variations. (The weights
    array must be an M x N matrix where M is the length of x and N is
    the number of weight variations).

    Parameters
    ----------
    x : numpy.ndarray
        Data to histogram.
    weights : numpy.ndarray
        Weight variations for the elements of ``x``, first dimension
        is the shape of ``x``, second dimension is the number of weights.
    bins : numpy.ndarray
        Bin edges.
    flow : bool
        Include under/overflow in the first/last bins.

    Raises
    ------
    ValueError
        If the array of bin edges is not monotonically increasing.
    ValueError
        If ``x`` and ``weights`` have incompatible shapes.
    ValueError
        If ``weights`` is not a two dimensional array.
    TypeError
        If ``x`` or ``weights`` are unsupported types

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`
        The standard error of each bin count, :math:`\sqrt{\sum_i w_i^2}`.

    Examples
    --------
    Using three different weight variations:

    >>> rng = np.random.default_rng(123)
    >>> x = rng.standard_normal(10000)
    >>> weights = nb.abs(rng.standard_normal((x.shape[0], 3)))
    >>> edges = np.array([-3.0, -2.5, -1.5, -0.25, 0.25, 2.0, 3.0])
    >>> h, err = var1dmw(x, weights, edges)
    >>> h.shape
    (6, 3)
    >>> err.shape
    (6, 3)

    """
    if len(np.shape(weights)) != 2:
        raise ValueError("weights must be a two dimensional array.")
    if np.shape(x)[0] != np.shape(weights)[0]:
        raise ValueError("x and weights have incompatible shapes.")
    if not np.all(bins[1:] >= bins[:-1]):
        raise ValueError("bins sequence must monotonically increase.")

    if _likely_uniform_bins(bins):
        return fix1dmw(
            x,
            weights,
            bins=(len(bins) - 1),
            range=(bins[0], bins[-1]),
            flow=flow,
        )

    return _v1dmw(x, weights, bins, flow)


def histogram(x, bins=10, range=None, weights=None, density=False, flow=False):
    r"""Histogram data in one dimension.

    Parameters
    ----------
    x : array_like
        Data to histogram.
    bins : int or array_like
        If int: the number of bins; if array_like: the bin edges.
    range : (float, float), optional
        The minimum and maximum of the histogram axis. If ``None``
        with integer ``bins``, min and max of ``x`` will be used. If
        ``bins`` is an array this is expected to be ``None``.
    weights : array_like, optional
        Weight variations for the elements of ``x``. For single weight
        histograms the shape must be the same shape as ``x``. For
        multiweight histograms the first dimension is the length of
        ``x``, second dimension is the number of weights variations.
    density : bool
        Normalize histogram counts as value of PDF such that the
        integral over the range is unity.
    flow : bool
        Include under/overflow in the first/last bins.

    Raises
    ------
    ValueError
        If ``bins`` defines edges while ``range`` is also not
        ``None``.
    ValueError
        If the array of bin edges is not monotonically increasing.
    ValueError
        If ``x`` and ``weights`` have incompatible shapes.
    ValueError
        If multiweight histogramming is detected and ``weights`` is
        not a two dimensional array.
    TypeError
        If ``x`` or ``weights`` are unsupported types

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`, optional
        The standard error of each bin count, :math:`\sqrt{\sum_i
        w_i^2}`. The return is ``None`` if weights are not used.

    See Also
    --------
    fix1d
        Used for no weight or single weight fixed bin width histograms
    fix1dmw
        Used for multiweight fixed bin width histograms.
    var1d
        Used for no weight or single weight variable bin width
        histograms.
    var1dmw
        Used for multiweight variable bin width histograms.

    Examples
    --------
    A simple fixed width histogram:

    >>> h, __ = histogram(x, bins=20, range=(0, 100))

    And with variable width histograms and weights:

    >>> h, err = histogram(x, bins=[-3, -2, -1.5, 1.5, 3.5], weights=w)

    """
    # make sure x and weight data are NumPy arrays.
    x = np.array(x)
    if weights is not None:
        weights = np.asarray(weights)
        is_multiweight = np.shape(weights) != np.shape(x)
        if is_multiweight and len(np.shape(weights)) != 2:
            raise ValueError("weight must be a 2D array for multiweight histograms.")

    # fixed bins
    if isinstance(bins, int):
        if weights is not None:
            if is_multiweight:
                return fix1dmw(x, weights, bins=bins, range=range, flow=flow)
        return fix1d(
            x, weights=weights, bins=bins, range=range, density=density, flow=flow
        )

    # variable bins
    else:
        bins = np.asarray(bins)
        if range is not None:
            raise ValueError("range must be None if bins is non-int")
        if weights is not None:
            if is_multiweight:
                return var1dmw(x, weights, bins=bins, flow=flow)
        return var1d(x, weights=weights, bins=bins, density=density, flow=flow)


def fix2d(
    x: np.ndarray,
    y: np.ndarray,
    bins: Union[int, Tuple[int, int]] = 10,
    range: Optional[Sequence[Tuple[float, float]]] = None,
    weights: Optional[np.ndarray] = None,
    flow: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    r"""Histogram the ``x``, ``y`` data with fixed (uniform) binning.

    The two input arrays (``x`` and ``y``) must be the same length
    (shape).

    Parameters
    ----------
    x : numpy.ndarray
        First entries in data pairs to histogram.
    y : numpy.ndarray
        Second entries in data pairs to histogram.
    bins : int or (int, int)
        If int, both dimensions will have that many bins;
        if tuple, the number of bins for each dimension
    range : Sequence[Tuple[float, float]], optional
        Axis limits in the form ``[(xmin, xmax), (ymin, ymax)]``. If
        ``None`` the input data min and max will be used.
    weights : array_like, optional
        The weights for data element. If weights are absent, the
        second return type will be ``None``.
    flow : bool
        Include over/underflow.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have incompatible shapes.
    ValueError
        If the shape of ``weights`` is incompatible with ``x`` and ``y``
    TypeError
        If ``x``, ``y``, or ``weights`` are unsupported types

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`, optional
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
    if np.shape(x) != np.shape(y):
        raise ValueError("x and y must be the same shape.")
    if weights is not None:
        if np.shape(weights) != np.shape(x):
            raise ValueError("data and weights must be the same shape.")

    if isinstance(bins, int):
        nx = ny = bins
    else:
        nx, ny = bins

    xmin, xmax, ymin, ymax = _get_limits_2d(x, y, range)

    if weights is None:
        result = _f2d(x, y, int(nx), xmin, xmax, int(ny), ymin, ymax, flow)
        return result, None

    return _f2dw(x, y, weights, int(nx), xmin, xmax, int(ny), ymin, ymax, flow)


def var2d(
    x: np.ndarray,
    y: np.ndarray,
    xbins: np.ndarray,
    ybins: np.ndarray,
    weights: Optional[np.ndarray] = None,
    flow: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    r"""Histogram the ``x``, ``y`` data with variable width binning.

    The two input arrays (``x`` and ``y``) must be the same length
    (shape).

    Parameters
    ----------
    x : numpy.ndarray
        First entries in data pairs to histogram.
    y : numpy.ndarray
        Second entries in data pairs to histogram.
    xbins : numpy.ndarray
        Bin edges for the ``x`` dimension.
    ybins : np.ndarray
        Bin edges for the ``y`` dimension.
    weights : array_like, optional
        The weights for data element. If weights are absent, the
        second return type will be ``None``.
    flow : bool
        Include under/overflow.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different shape.
    ValueError
        If either bin edge definition is not monotonically increasing.
    TypeError
        If ``x``, ``y``, or ``weights`` are unsupported types

    Returns
    -------
    :py:obj:`numpy.ndarray`
        The bin counts.
    :py:obj:`numpy.ndarray`, optional
        The standard error of each bin count, :math:`\sqrt{\sum_i w_i^2}`.

    Examples
    --------
    A histogram of (``x``, ``y``) where the edges are defined by a
    :func:`numpy.logspace` in both dimensions:

    >>> bins = numpy.logspace(0.1, 1.0, 10, endpoint=True)
    >>> h, __ = var2d(x, y, bins, bins)

    """
    if np.shape(x) != np.shape(y):
        raise ValueError("x and y must be the same shape.")
    if not np.all(xbins[1:] >= xbins[:-1]):
        raise ValueError("xbins sequence must monotonically increase.")
    if not np.all(ybins[1:] >= ybins[:-1]):
        raise ValueError("ybins sequence must monotonically increase.")
    if weights is not None:
        weights = np.asarray(weights)
        if np.shape(weights) != np.shape(x):
            raise ValueError("data and weights must be the same shape.")

    if weights is None:
        result = _v2d(x, y, xbins, ybins, flow)
        return result, None

    return _v2dw(x, y, weights, xbins, ybins, flow)


def histogram2d(x, y, bins=10, range=None, weights=None, flow=False):
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
           * If int, the number of bins for the two dimensions
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
    flow : bool
        Include over/underflow.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different shape or either bin edge definition
        is not monotonically increasing.
    ValueError
        If the shape of ``weights`` is not compatible with ``x`` and ``y``.
    TypeError
        If ``x``, ``y``, or ``weights`` are unsupported types

    See Also
    --------
    fix2d
        Used for no weight or single weight fixed bin width histograms
    var2d
        Used for no weight or single weight variable bin width histograms.

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

    x = np.asarray(x)
    y = np.asarray(y)
    if weights is not None:
        weights = np.asarray(weights)
    if N != 1 and N != 2:
        bins = np.asarray(bins)
        return var2d(x, y, bins, bins, weights=weights, flow=flow)

    elif N == 1:
        return fix2d(x, y, bins=bins, range=range, weights=weights, flow=flow)

    elif N == 2:
        if isinstance(bins[0], int) and isinstance(bins[1], int):
            return fix2d(x, y, bins=bins, range=range, weights=weights, flow=flow)
        else:
            b1 = np.asarray(bins[0])
            b2 = np.asarray(bins[1])
            return var2d(x, y, b1, b2, weights=weights, flow=flow)

    else:
        raise ValueError("bins argument is not compatible")
