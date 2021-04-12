import numpy as np
from ._hist import fix1d, fix1dmw, fix2d, var1d, var1dmw, var2d


def histogram(
    x,
    bins=10,
    range=None,
    weights=None,
    density=False,
    flow=False,
    cons_var=False,
):
    r"""Histogram data in one dimension.

    Parameters
    ----------
    x : array_like
        Data to histogram.
    bins : int or array_like
        If int: the number of bins; if array_like: the bin edges.
    range : (float, float), optional
        The minimum and maximum of the histogram axis. If ``None``
        with integer `bins`, min and max of `x` will be used. If
        `bins` is an array this is expected to be ``None``.
    weights : array_like, optional
        Weight variations for the elements of `x`. For single weight
        histograms the shape must be the same shape as `x`. For
        multiweight histograms the first dimension is the length of
        `x`, second dimension is the number of weights variations.
    density : bool
        Normalize histogram counts as value of PDF such that the
        integral over the range is unity.
    flow : bool
        Include under/overflow in the first/last bins.
    cons_var : bool
        If ``True``, conserve the variance rather than return the
        standard error (square root of the variance).

    Raises
    ------
    ValueError
        If `bins` defines edges while `range` is also not ``None``.
    ValueError
        If the array of bin edges is not monotonically increasing.
    ValueError
        If `x` and `weights` have incompatible shapes.
    ValueError
        If multiweight histogramming is detected and `weights` is
        not a two dimensional array.
    TypeError
        If `x` or `weights` are unsupported types

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

    >>> rng = np.random.default_rng(123)
    >>> x = rng.standard_normal(2000)
    >>> h, __ = histogram(x, bins=20, range=(-3, 3))

    And with variable width histograms and weights:

    >>> w = rng.uniform(0.3, 1.1, size=x.shape)
    >>> h, err = histogram(x, bins=[-3, -2, -1.5, 1.5, 3.5], weights=w)

    """

    # make sure x and weight data are NumPy arrays.
    x = np.array(x)
    is_multiweight = False
    if weights is not None:
        weights = np.asarray(weights)
        is_multiweight = np.shape(weights) != np.shape(x)
        if is_multiweight and len(np.shape(weights)) != 2:
            raise ValueError("weight must be a 2D array for multiweight histograms.")

    # fixed bins
    if isinstance(bins, int):
        if is_multiweight:
            return fix1dmw(
                x,
                weights,
                bins=bins,
                range=range,
                flow=flow,
                cons_var=cons_var,
            )
        return fix1d(
            x,
            weights=weights,
            bins=bins,
            range=range,
            density=density,
            flow=flow,
            cons_var=cons_var,
        )

    # variable bins
    else:
        bins = np.asarray(bins)
        if range is not None:
            raise ValueError("range must be None if bins is non-int")
        if is_multiweight:
            return var1dmw(x, weights, bins=bins, flow=flow, cons_var=cons_var)
        return var1d(
            x,
            weights=weights,
            bins=bins,
            density=density,
            flow=flow,
            cons_var=cons_var,
        )


def histogram2d(x, y, bins=10, range=None, weights=None, flow=False, cons_var=False):
    r"""Histogram data in two dimensions.

    This function provides an API very simiar to
    :func:`numpy.histogram2d`. Keep in mind that the returns are
    different.

    Parameters
    ----------
    x: array_like
        Array representing the `x` coordinate of the data to histogram.
    y: array_like
        Array representing the `y` coordinate of the data to histogram.
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
    cons_var : bool
        If ``True``, conserve the variance rather than return the
        standard error (square root of the variance).

    Raises
    ------
    ValueError
        If `x` and `y` have different shape or either bin edge definition
        is not monotonically increasing.
    ValueError
        If the shape of `weights` is not compatible with `x` and `y`.
    TypeError
        If `x`, `y`, or `weights` are unsupported types

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
    Gaussian distributions in 2D with automatic bin ranges:

    >>> rng = np.random.default_rng(123)
    >>> x = rng.standard_normal(size=(1000,))
    >>> y = rng.standard_normal(size=(1000,))
    >>> w = rng.uniform(0.3, 0.4, size=x.shape)
    >>> h, err = histogram2d(x, y, bins=[10, 10], weights=w)
    >>> h.shape
    (10, 10)

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
        return var2d(x, y, bins, bins, weights=weights, flow=flow, cons_var=cons_var)

    elif N == 1:
        return fix2d(
            x,
            y,
            bins=bins,
            range=range,
            weights=weights,
            flow=flow,
            cons_var=cons_var,
        )

    elif N == 2:
        if isinstance(bins[0], int) and isinstance(bins[1], int):
            return fix2d(x, y, bins=bins, range=range, weights=weights, flow=flow)
        else:
            b1 = np.asarray(bins[0])
            b2 = np.asarray(bins[1])
            return var2d(x, y, b1, b2, weights=weights, flow=flow)

    else:
        raise ValueError("bins argument is not compatible")
