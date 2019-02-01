from ._core import _histogram1d_uniform
from ._core import _histogram1d_uniform_weighted
import numpy as np


def histogram(x, nbins=10, xlim=(0, 10), weights=None):
    """
    histogram ``x`` given based on uniform binning

    Parameters
    ----------
    x: :obj:`numpy.ndarray`
      array of data to histogram
    nbins: int
      number of bins
    xlim: Tuple[float, float]
      axis limits to histogram over
    weights: Optional[:obj:`numpy.ndarray`]
      weight for each element of ``x``.

    Returns
    -------
    Tuple[:obj:`numpy.ndarray`, :obj:`numpy.ndarray`]
      bin counts and the statistical uncertainty on each bin

    """
    if weights is not None:
        assert weights.shape == x.shape, "weights must be the same shape as the data"
        count, sumw2 = _histogram1d_uniform_weighted(
            x, weights, nbins, xlim[0], xlim[1]
        )
        return count, np.sqrt(sumw2)
    else:
        count = _histogram1d_uniform(x, nbins, xlim[0], xlim[1])
        uncertainty = np.sqrt(count)
        return count, uncertainty
