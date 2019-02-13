from ._core import _uniform1d_f4
from ._core import _uniform1d_f8
from ._core import _uniform1d_weighted_f4
from ._core import _uniform1d_weighted_f8
import numpy as np


def uniform1d(x, bins=10, range=None, weights=None, omp=False):
    """histogram ``x`` with uniform binning

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
    omp: bool
        use OpenMP if available

    Returns
    -------
    :obj:`numpy.ndarray`
        bin counts
    :obj:`numpy.ndarray`
        sum of weights squared (if ``weights`` is not None)

    Examples
    --------

    >>> h, w = fixed_histogram1d([1,2,3], weights=[.1, .2, .3])

    """
    x = np.asarray(x)
    if x.dtype == np.float64:
        weighted_func = _uniform1d_weighted_f8
        unweight_func = _uniform1d_f8
    elif x.dtype == np.float32:
        weighted_func = _uniform1d_weighted_f4
        unweight_func = _uniform1d_f4
    if range is None:
        range = (x.min(), x.max())
    if weights is not None:
        weights = np.asarray(weights)
        assert weights.shape == x.shape, "weights must be the same shape as the data"
        return weighted_func(x, weights, bins, range[0], range[1], omp)
    else:
        return unweight_func(x, bins, range[0], range[1], omp)


def nonuniform1d(x, bins, weights=None):
    """histogram ``x`` with non-uniform binning

    Parameters
    ----------
    x: array_like
        data to histogram
    bins: array_like
        bin edges
    weights: array_like, optional
        weight for each element of ``x``

    """
    raise NotImplementedError
