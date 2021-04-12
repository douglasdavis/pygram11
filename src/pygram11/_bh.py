from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def import_bh():
    """Try to import boost_histogram."""
    try:
        import boost_histogram
    except ImportError:
        raise ("Install boost-histogram to use out='bh' feature")
    return boost_histogram


bh = import_bh()


def f_axis(bins: int, range: Tuple[float, float]) -> bh.axis.Axis:
    """Create fixed boost-histogram axis from bins and range."""
    return bh.axis.Regular(bins, range[0], range[1])


def v_axis(bins: Iterable[float]) -> bh.axis.Axis:
    """Create variable boost-histogram axis from edges."""
    return bh.axis.Variable(bins)


def f1d_to_boost(
    counts: np.ndarray,
    bins: int,
    range: Tuple[float, float],
    variances: Optional[np.ndarray] = None,
) -> bh.Histogram:
    """Create a 1D fixed width boost-histogram object."""
    if variances is None:
        h = bh.Histogram(f_axis(bins, range))
        h[...] = counts
    else:
        h = bh.Histogram(f_axis(bins, range), storage=bh.storage.Weight())
        view = h.view()
        view.value = counts
        view.variance = variances
    return h


def f2d_to_boost(
    counts: np.ndarray,
    bins: Tuple[int, int],
    range: Sequence[Tuple[float, float]],
    variances: Optional[np.ndarray] = None,
) -> bh.Histogram:
    """Create a 2D fixed width boost-histogram object."""
    if variances is None:
        h = bh.Histogram(f_axis(bins[0], range[0]), f_axis(bins[1], range[1]))
        h[...] = counts
    else:
        h = bh.Histogram(
            f_axis(bins[0], range[0]),
            f_axis(bins[1], range[1]),
            storage=bh.storage.Weight(),
        )
        view = h.view()
        view.value = counts
        view.variance = variances
    return h
