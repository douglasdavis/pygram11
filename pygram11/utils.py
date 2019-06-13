import numpy as np


def densify1d(h, range, binw=None, sumw2=None):
    """normalize histogram as a PDF such that the integral over the range
    is equal to 1.

    This function is used by the histogramming functions if the
    ``density`` argument is true.

    Parameters
    ----------
    h: array_like
       histogram bin heights
    range: (float, float)
       histogram axis limits
    binw: array_like, optional
       array of bin widths if variable width bins (``None`` assumes
       fixed width bins).
    sumw2: array_like
       array representing sum of weights squared in each bin

    Returns
    -------
    :obj:`numpy.ndarray`
        bin counts normalized
    :obj:`numpy.ndarray`
        uncertainty on the bin heights (only if ``sumw2`` is not None)

    """
    raw_integral = h.sum()

    if binw is None:
        binwidth = float(range[1] - range[0]) / len(h)
    else:
        binwidth = binw

    result_normed = h / binwidth / raw_integral

    if sumw2 is not None:
        result_normed_unc = np.sqrt(
            sumw2 + np.power(h / raw_integral, 2) * np.sum(sumw2)
        ) / (binwidth * raw_integral)
        return (result_normed, result_normed_unc)

    return (result_normed, None)
