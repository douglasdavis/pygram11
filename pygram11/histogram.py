from ._core import _uniform1d
from ._core import _uniform1d_weighted
import numpy as np


def uniform1d(x, bins=10, range=None, weights=None):
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

    Returns
    -------
    (:obj:`numpy.ndarray`, :obj:`numpy.ndarray`)
        bin counts and the statistical uncertainty on each bin

    Examples
    --------

    >>> h, w = fixed_histogram1d([1,2,3], weights=[.1, .2, .3])

    """
    x = np.asarray(x)
    if range is None:
        range = (x.min(), x.max())
    if weights is not None:
        weights = np.asarray(weights)
        assert weights.shape == x.shape, "weights must be the same shape as the data"
        count, sumw2 = _uniform1d_weighted(x, weights, bins, range[0], range[1])
        return count, np.sqrt(sumw2)
    else:
        count = _uniform1d(x, bins, range[0], range[1])
        uncertainty = np.sqrt(count)
        return count, uncertainty


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
