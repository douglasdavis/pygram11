from ._core import _f1d_f4, _f1d_f8
from ._core import _f1dw_f4, _f1dw_f8
from ._core import _f1dmw_f4, _f1dmw_f8

from ._core import _v1d_f4, _v1d_f8
from ._core import _v1dw_f4, _v1dw_f8
from ._core import _v1dmw_f4, _v1dmw_f8

from ._core import _fix2d_f4
from ._core import _fix2d_f8
from ._core import _fix2d_weighted_f4
from ._core import _fix2d_weighted_f8

from ._core import _var2d_f4
from ._core import _var2d_f8
from ._core import _var2d_weighted_f4
from ._core import _var2d_weighted_f8

from .utils import densify1d

import numpy as np
import numbers


def fix1d(x, bins=10, range=None, weights=None, density=False, flow=False, omp="auto"):
    """histogram ``x`` with fixed (uniform) binning over a range
    [xmin, xmax).

    Parameters
    ----------
    x: array_like
        data to histogram
    bins: int or str, optional
        number of bins or str
    range: (float, float), optional
        axis limits to histogram over
    weights: array_like, optional
        weight for each element of ``x``.
    density: bool
        normalize histogram bins as value of PDF such that the integral
        over the range is 1.
    flow: bool
        if ``True`` the under and overflow bin contents are added to the first
        and last bins, respectively
    omp: bool or str
        if ``True``, use OpenMP if available; if "auto" (and OpenMP is available),
        enables OpenMP if len(x) > 10^4

    Returns
    -------
    :obj:`numpy.ndarray`
        bin counts (heights)
    :obj:`numpy.ndarray`
        Poisson uncertainty on counts (``None`` if weights are absent)

    Examples
    --------
    A histogram of ``x`` with 20 bins between 0 and 100.

    >>> h = fix1d(x, bins=20, range=(0, 100))

    The same data, now histogrammed weighted & accelerated with
    OpenMP.

    >>> h, h_err = fix1d(x, bins=20, range=(0, 100), omp=True)

    """
    x = np.asarray(x)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != x.shape:
            raise ValueError("weights must be the same shape as the data")

    if omp == "auto":
        use_omp = len(x) > 1e4
    elif type(omp) == bool:
        use_omp = omp
    else:
        raise TypeError("omp should be 'auto' or a boolean value")

    weighted_func = _f1dw_f8
    unweight_func = _f1d_f8
    if x.dtype == np.float32:
        weighted_func = _f1dw_f4
        unweight_func = _f1d_f4

    if range is None:
        range = (x.min(), x.max())
    if range[0] > range[1]:
        raise ValueError("range=(a, b) must satisfy b > a")

    if weights is not None:
        result, sw2 = weighted_func(x, weights, bins, range[0], range[1], use_omp)
    else:
        result = unweight_func(x, bins, range[0], range[1], use_omp)

    if flow:
        result[1] += result[0]
        result[-2] += result[-1]
        if weights is not None:
            sw2[1] += sw2[0]
            sw2[-1] += sw2[-2]

    result = result[1:-1]
    if weights is not None:
        sw2 = sw2[1:-1]

    if density:
        if weights is None:
            sw2 = None
        return densify1d(result, range, sumw2=sw2)

    if weights is None:
        return (result, None)

    return (result, np.sqrt(sw2))


def fix1dmw(x, weights, bins=10, range=None, flow=False, omp="auto"):
    """histogram ``x`` with fixed (uniform) binning over a range
    [xmin, xmax) using multiple weight variations.

    Parameters
    ----------
    x: array_like
        data to histogram
    weights: array_like
        weight variations for the elements of ``x``, first dimension
        is the shape of ``x``, second dimension is the number of weights.
    bins: int or str, optional
        number of bins or str
    range: (float, float), optional
        axis limits to histogram over
    flow: bool
        if ``True`` the under and overflow bin contents are added to the first
        and last bins, respectively
    omp: bool or str
        if ``True``, use OpenMP if available; if "auto" (and OpenMP is available),
        enables OpenMP if len(x) > 10^4

    Returns
    -------
    :obj:`numpy.ndarray`
        bin counts (heights) for each variation (shape == [n_bins, n_weight_variations])
    :obj:`numpy.ndarray`
        Poisson uncertainty on counts, shape == [n_bins, n_weight_variations],
        (``None`` if weights are absent)

    Examples
    --------
    Multiple histograms of ``x`` with 50 bins between 0 and 100; using
    20 different weight variations:

    >>> x = np.random.randn(10000)
    >>> twenty_weights = np.random.rand(x.shape[0], 20)
    >>> h, err = fix1dmw(x, w, bins=50, range=(-3, 3), omp=True)

    ``h`` and ``err`` are now shape ``(50, 20)``. Each column
    represents the histogram of the data with the respective weight.

    """
    x = np.asarray(x)
    weights = np.asarray(weights)
    if weights.ndim != 2:
        raise ValueError("weights must be the same shape as the data")
    if weights.shape[0] != x.shape[0]:
        raise ValueError("x and weights must have equal shape[0]")

    if omp == "auto":
        use_omp = len(x) > 1e4
    elif type(omp) == bool:
        use_omp = omp
    else:
        raise TypeError("omp should be 'auto' or a boolean value")

    if range is None:
        range = (x.min(), x.max())
    if range[0] > range[1]:
        raise ValueError("range=(a, b) must satisfy b > a")

    hfunc = _f1dmw_f8
    if weights.dtype == np.float32:
        hfunc = _f1dmw_f4

    count, sumw2 = hfunc(x, weights, bins, range[0], range[1], use_omp)
    if flow:
        count[-2, :] += count[-1, :]
        sumw2[-2, :] += sumw2[-1, :]
        count[1, :] += count[0, :]
        sumw2[1, :] += sumw2[0, :]

    return (count[1:-1, :], np.sqrt(sumw2[1:-1, :]))


def var1d(x, bins, weights=None, density=False, flow=False, omp="auto"):
    """histogram ``x`` with variable (non-uniform) binning over a range
    [bins[0], bins[-1]).

    Parameters
    ----------
    x: array_like
        data to histogram
    bins: array_like
        bin edges
    weights: array_like, optional
        weight for each element of ``x``
    density: bool
        normalize histogram bins as value of PDF such that the integral
        over the range is 1.
    flow: bool
        if ``True`` the under and overflow bin contents are added to the first
        and last bins, respectively
    omp: bool or str
        if ``True``, use OpenMP if available; if "auto" (and OpenMP is available),
        enables OpenMP if len(x) > 10^3

    Returns
    -------
    :obj:`numpy.ndarray`
        bin counts (heights)
    :obj:`numpy.ndarray`
        Poisson uncertainty on counts (``None`` if weights are absent)

    Examples
    --------
    A histogram of ``x`` where the edges are defined by the list
    ``[1, 5, 10, 12]``:

    >>> h = var1d(x, [1, 5, 10, 12])

    The same data, now weighted and accelerated with OpenMP:

    >>> h = var1d(x, [1, 5, 10, 12], weights=w, omp=True)

    """
    x = np.asarray(x)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != x.shape:
            raise ValueError("weights must be same shape as data")

    if omp == "auto":
        use_omp = len(x) > 1e3
    elif type(omp) == bool:
        use_omp = omp
    else:
        raise TypeError("omp should be 'auto' or a boolean value")

    bins = np.asarray(bins)
    if not np.all(bins[1:] >= bins[:-1]):
        raise ValueError("bins sequence must monotonically increase")

    weighted_func = _v1dw_f8
    unweight_func = _v1d_f8
    if x.dtype == np.float32:
        weighted_func = _v1dw_f4
        unweight_func = _v1d_f4

    if weights is not None:
        result, sw2 = weighted_func(x, weights, bins, use_omp)
    else:
        result = unweight_func(x, bins, use_omp)

    if flow:
        result[1] += result[0]
        result[-2] += result[-1]
        if weights is not None:
            sw2[1] += sw2[0]
            sw2[-1] += sw2[-2]

    result = result[1:-1]
    if weights is not None:
        sw2 = sw2[1:-1]

    if density:
        if weights is None:
            sw2 = None
        return densify1d(result, [bins[-1], bins[0]], binw=np.diff(bins), sumw2=sw2)

    if weights is None:
        return (result, None)
    return (result, np.sqrt(sw2))


def var1dmw(x, weights, bins, flow=False, omp="auto"):
    """histogram ``x`` with fixed (uniform) binning over a range
    [xmin, xmax) using multiple weight variations.

    Parameters
    ----------
    x: array_like
        data to histogram
    weights: array_like
        weight variations for the elements of ``x``, first dimension
        is the shape of ``x``, second dimension is the number of weights.
    bins: int or str, optional
        number of bins or str
    flow: bool
        if ``True`` the under and overflow bin contents are added to the first
        and last bins, respectively
    omp: bool or str
        if ``True``, use OpenMP if available; if "auto" (and OpenMP is available),
        enables OpenMP if len(x) > 10^4

    Returns
    -------
    :obj:`numpy.ndarray`
        bin counts (heights) for each variation (shape == [n_bins, n_weight_variations])
    :obj:`numpy.ndarray`
        Poisson uncertainty on counts, shape == [n_bins, n_weight_variations]
        (``None`` if weights are absent)

    Examples
    --------
    Multiple histograms of ``x`` (with 9 variable width bins); using
    20 different weight variations (also shifting under and overflow
    contents):

    >>> x = np.random.randn(10000)
    >>> bins = [-2, -1.5, -1.2, -1, -0.8, 0, 1.2, 1.5, 3.0]
    >>> twenty_weights = np.random.rand(x.shape[0], 20)
    >>> h, err = fix1dmw(x, w, bins, flow, True, omp=True)

    ``h`` and ``err`` are now shape ``(9, 20)``. Each column
    represents the histogram of the data with the respective weight.

    """
    x = np.asarray(x)
    weights = np.asarray(weights)
    if weights.ndim != 2:
        raise ValueError("weights must be the same shape as the data")
    if weights.shape[0] != x.shape[0]:
        raise ValueError("x and weights must have equal shape[0]")

    if omp == "auto":
        use_omp = len(x) > 1e4
    elif type(omp) == bool:
        use_omp = omp
    else:
        raise TypeError("omp should be 'auto' or a boolean value")

    bins = np.asarray(bins)
    if not np.all(bins[1:] >= bins[:-1]):
        raise ValueError("bins sequence must monotonically increase")

    hfunc = _v1dmw_f8
    if weights.dtype == np.float32:
        hfunc = _v1dmw_f4

    count, sumw2 = hfunc(x, weights, bins, use_omp)
    if flow:
        count[-2, :] += count[-1, :]
        sumw2[-2, :] += sumw2[-1, :]
        count[1, :] += count[0, :]
        sumw2[1, :] += sumw2[0, :]

    return (count[1:-1, :], np.sqrt(sumw2[1:-1, :]))


def fix2d(x, y, bins=10, range=None, weights=None, omp=False):
    """histogram the ``x``, ``y`` data with fixed (uniform) binning in
    two dimensions over the ranges [xmin, xmax), [ymin, ymax).

    Parameters
    ----------
    x: array_like
       first entries in data pairs to histogram
    y: array_like
       second entries in data pairs to histogram
    bins: int or iterable
       if int, both dimensions will have that many bins,
       if iterable, the number of bins for each dimension
    range: iterable, optional
        axis limits to histogram over in the form [(xmin, xmax), (ymin, ymax)]
    weights: array_like, optional
        weight for each :math:`(x_i, y_i)` pair.
    omp: bool
        use OpenMP if available

    Returns
    -------
    :obj:`numpy.ndarray`
        bin counts (heights)
    :obj:`numpy.ndarray`
        Poisson uncertainty on counts (``None`` if weights are absent)

    Examples
    --------

    A histogram of (``x``, ``y``) with 20 bins between 0 and 100 in
    the ``x`` dimention and 10 bins between 0 and 50 in the ``y``
    dimension.

    >>> h = fix2d(x, y, bins=(20, 10), range=((0, 100), (0, 50)))

    The same data, now histogrammed weighted (via ``w``) & accelerated
    with OpenMP.

    >>> h, err = fix2d(x, y, bins=(20, 10), range=((0, 100), (0, 50)),
    ...                weights=w, omp=True)

    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must be the same shape")

    weighted_func = _fix2d_weighted_f8
    unweight_func = _fix2d_f8
    if x.dtype == np.float32 and y.dtype == np.float32:
        weighted_func = _fix2d_weighted_f4
        unweight_func = _fix2d_f4

    if isinstance(bins, numbers.Integral):
        nx = ny = bins
    else:
        nx, ny = bins

    if range is None:
        range = [(x.min(), x.max()), (y.min(), y.max())]
    (xmin, xmax), (ymin, ymax) = range

    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != x.shape:
            raise ValueError("weights must be the same shape as the data")
        count, sumw2 = weighted_func(x, y, weights, nx, xmin, xmax, ny, ymin, ymax, omp)
        return (count, np.sqrt(sumw2))
    else:
        count = unweight_func(x, y, nx, xmin, xmax, ny, ymin, ymax, omp)
        return (count, None)


def var2d(x, y, xbins, ybins, weights=None, omp=False):
    """histogram the ``x`` and ``y`` data with variable width binning in
    two dimensions over the range [xbins[0], xbins[-1]), [ybins[0], ybins[-1])

    Parameters
    ----------
    x: array_like
       first entries in the data pairs to histogram
    y: array_like
       second entries in the data pairs to histogram
    xbins: array_like
       bin edges for the ``x`` dimension
    ybins: array_like
       bin edges for the ``y`` dimension
    weights: array_like, optional
       weights for each :math:`(x_i, y_i)` pair.
    omp: bool
       use OpenMP if available

    Returns
    -------
    :obj:`numpy.ndarray`
        bin counts (heights)
    :obj:`numpy.ndarray`
        Poisson uncertainty on counts (``None`` if weights are absent)

    Examples
    --------
    A histogram of (``x``, ``y``) where the edges are defined by a
    :func:`numpy.logspace` in both dimensions, accelerated with
    OpenMP.

    >>> bins = numpy.logspace(0.1, 1.0, 10, endpoint=True)
    >>> h = var2d(x, y, bins, bins, omp=True)

    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must be the same shape")
    xbins = np.asarray(xbins)
    ybins = np.asarray(ybins)
    if not np.all(xbins[1:] >= xbins[:-1]):
        raise ValueError("xbins sequence must monotonically increase")
    if not np.all(ybins[1:] >= ybins[:-1]):
        raise ValueError("ybins sequence must monotonically increase")

    weighted_func = _var2d_weighted_f8
    unweight_func = _var2d_f8
    if x.dtype == np.float32 and y.dtype == np.float32:
        weighted_func = _var2d_weighted_f4
        unweight_func = _var2d_f4

    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != x.shape:
            raise ValueError("weights must be the same shape as data")
        count, sumw2 = weighted_func(x, y, weights, xbins, ybins, omp)
        return (count, np.sqrt(sumw2))
    else:
        count = unweight_func(x, y, xbins, ybins, omp)
        return (count, None)


def histogram(
    x, bins=10, range=None, weights=None, density=False, flow=False, omp="auto"
):
    """Compute the histogram for the data ``x``.

    This function provides an API very simiar to
    :func:`numpy.histogram`. Keep in mind that the returns are
    different.


    Parameters
    ----------
    x: array_like
       Data to histogram.
    bins: int or sequence of scalars, optional
       If bins is an int, that many equal-width bins will be used to
       construct the histogram in the given range. If bins is a
       sequence, it must define a monotonically increasing array of
       bin edges. This allows for nonuniform bin widths.
    range: (float, float), optional
       The range over which the histogram is constructed. If a range
       is not provided then the default is (x.min(), x.max()). Values
       outside of the range are ignored. If bins is a sequence, this
       options is ignored.
    weights: array_like, optional
       An array of weights associated to each element of ``x``. Each
       value of ``x`` will contribute its associated weight to the
       bin count. If argument is two-dimensional then the function will
       return histogram data for all weight variations. The first
       dimension of the shape of the weights (if multiple variations)
       must be the shape of ``x``.
    density: bool
        normalize histogram bins as value of PDF such that the integral
        over the range is 1.
    flow: bool
        if ``True`` the under and overflow bin contents are added to the first
        and last bins, respectively
    omp: bool or str
        if ``True``, use OpenMP if available; if "auto" (and OpenMP is available),
        enables OpenMP if len(x) > 10^4 for fixed width and > 10^3 for variaxsble
        width bins.

    Returns
    -------
    :obj:`numpy.ndarray`
        bin counts (heights)
    :obj:`numpy.ndarray`,
        Poisson uncertainty on each bin count (``None`` if weights are absent)


    Examples
    --------
    A histogram of ``x`` with 20 bins between 0 and 100.

    >>> h = fix1d(x, bins=20, range=(0, 100))

    The same data, now histogrammed weighted & accelerated with
    OpenMP.

    >>> h, h_err = histogram(x, bins=20, range=(0, 100), omp=True)

    Multiple histograms of ``x`` with 50 bins between 0 and 100; using
    20 different weight variations:

    >>> x = np.random.randn(10000)
    >>> twenty_weights = np.random.rand(x.shape[0], 20)
    >>> h, err = histogram(x, bins=50, range=(-3, 3), weights=twenty_weights, omp=True)

    ``h`` and ``err`` are now shape ``(50, 20)``. Each column
    represents the histogram of the data with the respective weight.

    A histogram of ``x`` where the edges are defined by the list
    ``[1, 5, 10, 12]``:

    >>> h = var1d(x, bins=[1, 5, 10, 12])

    The same data, now weighted and accelerated with OpenMP:

    >>> h, err = histogram(x, bin=[1, 5, 10, 12], weights=w, omp=True)

    Multiple histograms of ``x`` (with 9 variable width bins); using
    20 different weight variations (also shifting under and overflow
    contents):

    >>> x = np.random.randn(10000)
    >>> bins = [-2, -1.5, -1.2, -1, -0.8, 0, 1.2, 1.5, 3.0]
    >>> twenty_weights = np.random.rand(x.shape[0], 20)
    >>> h, err = histogram(x, bins=bins, weights=twenty_weights, flow=True, omp=True)

    ``h`` and ``err`` are now shape ``(9, 20)``. Each column
    represents the histogram of the data with the respective weight.

    """
    if isinstance(bins, numbers.Integral):
        if weights is not None:
            if weights.ndim == 2:
                return fix1dmw(x, weights, bins=bins, range=range, flow=flow, omp=omp)
        return fix1d(
            x,
            bins=bins,
            range=range,
            weights=weights,
            density=density,
            flow=flow,
            omp=omp,
        )
    else:
        if weights is not None:
            if weights.ndim == 2:
                return var1dmw(x, weights, bins, flow=flow, omp=omp)
        return var1d(x, bins, weights=weights, density=density, flow=flow, omp=omp)


def histogram2d(x, y, bins=10, range=None, weights=None, omp=False):
    """Compute the two-dimensional histogram for the data (``x``, ``y``).

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
       An array of weights associated to each element :math:`(x_i, y_i)` pair.
       Each pair of the the data will contribute its associated weight to the
       bin count.
    omp: bool
       Use OpenMP if available

    Returns
    -------
    :obj:`numpy.ndarray`:
        bin counts (heights)
    :obj:`numpy.ndarray`:
        Poisson uncertainty on each bin count (``None`` if weights are absent)

    """
    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1 and N != 2:
        return var2d(x, y, bins, bins, weights=weights, omp=omp)

    if N == 1:
        return fix2d(x, y, bins=bins, range=range, weights=weights, omp=omp)

    if N == 2:
        if isinstance(bins[0], numbers.Integral) and isinstance(
            bins[1], numbers.Integral
        ):
            return fix2d(x, y, bins=bins, range=range, weights=weights, omp=omp)
        else:
            return var2d(x, y, bins[0], bins[1], weights=weights, omp=omp)
